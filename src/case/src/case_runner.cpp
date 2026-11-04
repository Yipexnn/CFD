#include <algorithm>
#include <cfd/case/case_runner.hpp>
#include <cfd/mesh/gmsh_reader.hpp>
#include <cfd/mesh/vtu_writer.hpp>
#include <cfd/parallel/decomposition.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cfd::cases {

namespace {

constexpr char k_magic[8] = {'C', 'F', 'D', 'C', 'K', 'P', 'T', '1'};

template <typename T> void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T> T read_pod(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Checkpoint: unexpected end of file");
    }
    return value;
}

void write_array(std::ostream& out, const double* data, std::size_t n) {
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(n * sizeof(double)));
}

void read_array(std::istream& in, double* data, std::size_t n) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(n * sizeof(double)));
    if (!in) {
        throw std::runtime_error("Checkpoint: unexpected end of file");
    }
}

} // namespace

mesh::Mesh build_mesh(const CaseConfig& config) {
    switch (config.mesh.source) {
    case MeshSource::Gmsh:
        return mesh::read_gmsh(config.resolve(config.mesh.file));
    case MeshSource::Geometry: {
        geometry::GeometrySpec spec = config.mesh.geometry;
        for (auto& body : spec.bodies) {
            if (body.kind == geometry::Body::Kind::File) {
                body.file = config.resolve(body.file);
            }
        }
        const auto work_dir = config.resolve(config.output.directory) / "mesh";
        const parallel::Comm& comm = parallel::Comm::world();
        // Gmsh runs once (root); the other processes read its output.
        std::string error;
        if (comm.is_root()) {
            try {
                const std::string gmsh = geometry::find_gmsh();
                if (gmsh.empty()) {
                    throw std::runtime_error("Gmsh executable not found (install Gmsh and add "
                                             "it to PATH or set CFD_GMSH)");
                }
                if (!comm.parallel()) {
                    return geometry::generate_mesh(spec, work_dir, gmsh);
                }
                (void)geometry::generate_mesh(spec, work_dir, gmsh);
            } catch (const std::exception& e) {
                error = e.what();
                if (!comm.parallel()) {
                    throw;
                }
            }
        }
        comm.broadcast(error);
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
        return geometry::read_generated_mesh(spec, work_dir);
    }
    case MeshSource::Box:
        break;
    }
    return mesh::make_box_mesh(config.mesh.box);
}

namespace {

using core::Label;

// u = peak * prod_a 4 s (1 - s) over the bounding box of the (global) patch.
std::vector<Vec3> parabolic_profile(const mesh::Mesh& global, const mesh::Patch& global_patch,
                                    const mesh::Mesh& mesh, const mesh::Patch& patch,
                                    const Vec3& peak, const std::string& axes) {
    const auto& cf = mesh.face_centres();
    Vec3 lo = Vec3::Constant(std::numeric_limits<Scalar>::max());
    Vec3 hi = -lo;
    for (Label q = 0; q < global_patch.n_faces; ++q) {
        for (Label pt : global.face(global_patch.start_face + q)) {
            const Vec3& x = global.points()[static_cast<std::size_t>(pt)];
            lo = lo.cwiseMin(x);
            hi = hi.cwiseMax(x);
        }
    }
    std::vector<Vec3> u;
    u.reserve(static_cast<std::size_t>(patch.n_faces));
    for (Label q = 0; q < patch.n_faces; ++q) {
        const Vec3& x = cf[static_cast<std::size_t>(patch.start_face + q)];
        Scalar shape = 1.0;
        for (const char a : axes) {
            const auto d = static_cast<Eigen::Index>(a - 'x');
            const Scalar extent = hi[d] - lo[d];
            const Scalar s = extent > 0.0 ? (x[d] - lo[d]) / extent : 0.5;
            shape *= 4.0 * s * (1.0 - s);
        }
        u.push_back(peak * shape);
    }
    return u;
}

} // namespace

CaseRunner::CaseRunner(CaseConfig config) : config_(std::move(config)) {
    global_mesh_ = std::make_unique<mesh::Mesh>(build_mesh(config_));
    setup();
}

CaseRunner::CaseRunner(CaseConfig config, mesh::Mesh mesh)
    : config_(std::move(config)), global_mesh_(std::make_unique<mesh::Mesh>(std::move(mesh))) {
    setup();
}

const mesh::Mesh& CaseRunner::mesh() const {
    return local_ ? local_->mesh : *global_mesh_;
}

void CaseRunner::setup() {
    const parallel::Comm& comm = parallel::Comm::world();
    if (comm.parallel()) {
        const std::vector<int> parts = parallel::partition_rcb(*global_mesh_, comm.size());
        local_ = std::make_unique<parallel::LocalMesh>(
            parallel::decompose(*global_mesh_, parts, comm.rank(), comm));
    }
    const mesh::Mesh* const solver_mesh = &mesh();
    for (const auto& name : config_.forces.patches) {
        if (solver_mesh->find_patch(name) == nullptr) {
            throw std::runtime_error("forces: unknown patch '" + name + "'");
        }
    }
    flow_ = std::make_unique<fv::FlowSolver>(*solver_mesh, config_.boundaries, config_.solver.flow,
                                             local_ ? &local_->halo : nullptr);
    for (const auto& [name, profile] : config_.profiles) {
        const mesh::Patch* patch = solver_mesh->find_patch(name);
        const auto bc = config_.boundaries.find(name);
        if (patch == nullptr || bc == config_.boundaries.end()) {
            throw std::runtime_error("profile: unknown inlet patch '" + name + "'");
        }
        flow_->set_boundary_velocity_profile(
            name, parabolic_profile(*global_mesh_, *global_mesh_->find_patch(name), *solver_mesh,
                                    *patch, bc->second.velocity, profile.axes));
    }
    flow_->set_uniform_velocity(config_.initial.velocity);
    if (config_.turbulence == TurbulenceKind::KOmegaSst) {
        fv::SstSettings sst_settings;
        sst_settings.k_init = config_.initial.k;
        sst_settings.omega_init = config_.initial.omega;
        sst_ = std::make_unique<fv::KOmegaSst>(*flow_, sst_settings);
        flow_->set_turbulence_model(sst_.get());
    }
    if (config_.restart) {
        read_checkpoint(config_.resolve(*config_.restart));
    }
}

CaseRunner::~CaseRunner() = default;

std::filesystem::path CaseRunner::output_directory() const {
    return config_.resolve(config_.output.directory);
}

namespace {

const parallel::Comm& world() {
    return parallel::Comm::world();
}

// Owned cell values (ncomp per cell) in global cell order (root only in parallel).
std::vector<Scalar> gather_cells(const parallel::LocalMesh* lm, const Scalar* data,
                                 std::size_t ncomp, std::size_t n_global) {
    if (lm == nullptr) {
        return {data, data + n_global * ncomp};
    }
    const auto no = static_cast<std::size_t>(lm->halo.n_owned);
    const std::vector<Scalar> values(data, data + no * ncomp);
    const std::vector<Label> ids(lm->global_cell.begin(),
                                 lm->global_cell.begin() + static_cast<std::ptrdiff_t>(no));
    const std::vector<Scalar> all_values = world().gather(values);
    const std::vector<Label> all_ids = world().gather(ids);
    std::vector<Scalar> out;
    if (world().is_root()) {
        out.resize(n_global * ncomp);
        for (std::size_t i = 0; i < all_ids.size(); ++i) {
            const auto g = static_cast<std::size_t>(all_ids[i]);
            std::copy_n(all_values.begin() + static_cast<std::ptrdiff_t>(i * ncomp), ncomp,
                        out.begin() + static_cast<std::ptrdiff_t>(g * ncomp));
        }
    }
    return out;
}

// Face values (first n entries of local faces starting at `first`) in global face order;
// each global face is taken from the process owning its global owner cell.
std::vector<Scalar> gather_faces(const parallel::LocalMesh* lm, const mesh::Mesh& global,
                                 const std::vector<Scalar>& values, bool oriented,
                                 bool boundary_only) {
    const auto gnif = static_cast<std::size_t>(global.n_internal_faces());
    const std::size_t offset = boundary_only ? gnif : 0;
    const std::size_t n_out = static_cast<std::size_t>(global.n_faces()) - offset;
    if (lm == nullptr) {
        return values;
    }
    const auto& m = lm->mesh;
    const auto lnif = static_cast<std::size_t>(m.n_internal_faces());
    const std::size_t lfirst = boundary_only ? lnif : 0;
    std::vector<Scalar> v;
    std::vector<Label> ids;
    for (std::size_t f = lfirst; f < static_cast<std::size_t>(m.n_faces()); ++f) {
        const Label g = lm->global_face[f];
        if (g < 0) {
            continue;
        }
        const bool flipped = lm->face_flipped[f] != 0;
        const Label global_owner_local =
            flipped ? m.neighbour()[f] : m.owner()[f]; // flipped faces are internal
        if (global_owner_local >= lm->halo.n_owned) {
            continue;
        }
        const Scalar x = values[f - lfirst];
        v.push_back(oriented && flipped ? -x : x);
        ids.push_back(g - static_cast<Label>(offset));
    }
    const std::vector<Scalar> all_values = world().gather(v);
    const std::vector<Label> all_ids = world().gather(ids);
    std::vector<Scalar> out;
    if (world().is_root()) {
        out.assign(n_out, 0.0);
        for (std::size_t i = 0; i < all_ids.size(); ++i) {
            out[static_cast<std::size_t>(all_ids[i])] = all_values[i];
        }
    }
    return out;
}

std::vector<Scalar> scatter_cells(const parallel::LocalMesh* lm, const std::vector<Scalar>& global,
                                  std::size_t ncomp) {
    if (lm == nullptr) {
        return global;
    }
    std::vector<Scalar> local(lm->global_cell.size() * ncomp);
    for (std::size_t c = 0; c < lm->global_cell.size(); ++c) {
        const auto g = static_cast<std::size_t>(lm->global_cell[c]);
        std::copy_n(global.begin() + static_cast<std::ptrdiff_t>(g * ncomp), ncomp,
                    local.begin() + static_cast<std::ptrdiff_t>(c * ncomp));
    }
    return local;
}

std::vector<Scalar> scatter_faces(const parallel::LocalMesh* lm, const mesh::Mesh& global,
                                  const std::vector<Scalar>& values, bool oriented,
                                  bool boundary_only) {
    if (lm == nullptr) {
        return values;
    }
    const auto& m = lm->mesh;
    const std::size_t lfirst = boundary_only ? static_cast<std::size_t>(m.n_internal_faces()) : 0;
    const std::size_t gfirst =
        boundary_only ? static_cast<std::size_t>(global.n_internal_faces()) : 0;
    std::vector<Scalar> local(static_cast<std::size_t>(m.n_faces()) - lfirst, 0.0);
    for (std::size_t f = lfirst; f < static_cast<std::size_t>(m.n_faces()); ++f) {
        const Label g = lm->global_face[f];
        if (g >= 0) {
            const Scalar x = values[static_cast<std::size_t>(g) - gfirst];
            local[f - lfirst] = oriented && lm->face_flipped[f] != 0 ? -x : x;
        }
    }
    return local;
}

std::vector<Vec3> to_vec3(const std::vector<Scalar>& v) {
    std::vector<Vec3> out(v.size() / 3);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = Vec3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
    }
    return out;
}

} // namespace

void CaseRunner::write_results(const std::filesystem::path& file) const {
    const auto ng = static_cast<std::size_t>(global_mesh_->n_cells());
    const parallel::LocalMesh* lm = local_.get();
    const std::vector<Vec3> u = to_vec3(gather_cells(lm, flow_->velocity().data()->data(), 3, ng));
    const std::vector<Scalar> p = gather_cells(lm, flow_->pressure().data(), 1, ng);
    std::vector<std::pair<std::string, std::vector<Scalar>>> turbulence;
    if (sst_) {
        turbulence.emplace_back("k", gather_cells(lm, sst_->k().data(), 1, ng));
        turbulence.emplace_back("omega", gather_cells(lm, sst_->omega().data(), 1, ng));
        turbulence.emplace_back("nut", gather_cells(lm, sst_->nut().data(), 1, ng));
        turbulence.emplace_back("yPlus", gather_cells(lm, sst_->y_plus().data(), 1, ng));
    }
    if (!world().is_root()) {
        return;
    }
    std::vector<mesh::CellScalarData> scalars{{"p", &p}};
    std::vector<mesh::CellVectorData> vectors{{"U", &u}};
    for (const auto& [name, values] : turbulence) {
        scalars.push_back({name, &values});
    }
    mesh::write_vtu(file, *global_mesh_, scalars, vectors);
}

void CaseRunner::write_checkpoint(const std::filesystem::path& file) const {
    const mesh::Mesh& g = *global_mesh_;
    const auto nc = static_cast<std::int64_t>(g.n_cells());
    const auto nf = static_cast<std::int64_t>(g.n_faces());
    const auto ng = static_cast<std::size_t>(nc);
    const parallel::LocalMesh* lm = local_.get();
    const std::vector<Scalar> u = gather_cells(lm, flow_->velocity().data()->data(), 3, ng);
    const std::vector<Scalar> p = gather_cells(lm, flow_->pressure().data(), 1, ng);
    const std::vector<Scalar> phi = gather_faces(lm, g, flow_->face_flux(), true, false);
    std::vector<std::vector<Scalar>> turbulence;
    if (sst_) {
        turbulence.push_back(gather_cells(lm, sst_->k().data(), 1, ng));
        turbulence.push_back(gather_cells(lm, sst_->omega().data(), 1, ng));
        turbulence.push_back(gather_cells(lm, sst_->nut().data(), 1, ng));
        turbulence.push_back(gather_faces(lm, g, sst_->boundary_nut(), false, true));
    }
    if (!world().is_root()) {
        return;
    }
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Checkpoint: cannot write " + file.string());
    }
    out.write(k_magic, sizeof(k_magic));
    write_pod(out, nc);
    write_pod(out, nf);
    write_pod(out, static_cast<std::int64_t>(step_));
    write_pod(out, time_);
    write_pod(out, static_cast<std::uint8_t>(sst_ ? 1 : 0));
    write_array(out, u.data(), u.size());
    write_array(out, p.data(), p.size());
    write_array(out, phi.data(), phi.size());
    for (const auto& field : turbulence) {
        write_array(out, field.data(), field.size());
    }
    if (!out) {
        throw std::runtime_error("Checkpoint: write failed for " + file.string());
    }
}

CheckpointData read_checkpoint_file(const std::filesystem::path& file, Label n_cells, Label n_faces,
                                    Label n_boundary_faces) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Checkpoint: cannot open " + file.string());
    }
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, k_magic, sizeof(magic)) != 0) {
        throw std::runtime_error("Checkpoint: " + file.string() + " is not a checkpoint file");
    }
    const auto nc = read_pod<std::int64_t>(in);
    const auto nf = read_pod<std::int64_t>(in);
    if (nc != n_cells || nf != n_faces) {
        throw std::runtime_error("Checkpoint: mesh size mismatch (" + std::to_string(nc) +
                                 " cells in file, " + std::to_string(n_cells) + " in mesh)");
    }
    CheckpointData d;
    d.step = static_cast<int>(read_pod<std::int64_t>(in));
    d.time = read_pod<Scalar>(in);
    const auto has_turbulence = read_pod<std::uint8_t>(in);
    const auto ncs = static_cast<std::size_t>(nc);
    auto read_vector = [&](std::size_t n) {
        std::vector<Scalar> v(n);
        read_array(in, v.data(), n);
        return v;
    };
    d.u = to_vec3(read_vector(3 * ncs));
    d.p = read_vector(ncs);
    d.phi = read_vector(static_cast<std::size_t>(nf));
    if (has_turbulence != 0) {
        d.k = read_vector(ncs);
        d.omega = read_vector(ncs);
        d.nut = read_vector(ncs);
        d.boundary_nut = read_vector(static_cast<std::size_t>(n_boundary_faces));
    }
    return d;
}

void CaseRunner::read_checkpoint(const std::filesystem::path& file) {
    const mesh::Mesh& g = *global_mesh_;
    CheckpointData d = read_checkpoint_file(file, g.n_cells(), g.n_faces(), g.n_boundary_faces());
    const parallel::LocalMesh* lm = local_.get();
    if (!d.k.empty() && sst_) {
        sst_->set_fields(scatter_cells(lm, d.k, 1), scatter_cells(lm, d.omega, 1),
                         scatter_cells(lm, d.nut, 1),
                         scatter_faces(lm, g, d.boundary_nut, false, true));
    }
    std::vector<Scalar> u_flat(3 * d.u.size());
    for (std::size_t c = 0; c < d.u.size(); ++c) {
        for (int i = 0; i < 3; ++i) {
            u_flat[3 * c + static_cast<std::size_t>(i)] = d.u[c][i];
        }
    }
    flow_->set_state(to_vec3(scatter_cells(lm, u_flat, 3)), scatter_cells(lm, d.p, 1),
                     scatter_faces(lm, g, d.phi, true, false));
    step_ = d.step;
    time_ = d.time;
}

Vec3 CaseRunner::force(const std::string& patch) const {
    return config_.forces.density * flow_->patch_force(patch).total();
}

RunResult CaseRunner::run(const ProgressCallback& progress) {
    const std::filesystem::path dir = output_directory();
    const bool root = world().is_root();
    if (root) {
        std::filesystem::create_directories(dir);
    }
    world().barrier();
    const auto mode = step_ > 0 ? std::ios::app : std::ios::trunc;
    std::ofstream log;
    if (root) {
        log.open(dir / "residuals.csv", mode);
    }
    if (step_ == 0) {
        log << "step,time,Ux,Uy,Uz,p,k,omega,continuity\n";
    }
    log << std::setprecision(6) << std::scientific;
    const ForcesConfig& fc = config_.forces;
    std::ofstream forces_log;
    if (!fc.patches.empty()) {
        if (root) {
            forces_log.open(dir / "forces.csv", mode);
        }
        if (step_ == 0) {
            forces_log << "step,time";
            for (const auto& name : fc.patches) {
                for (const char* c : {"Fx", "Fy", "Fz", "Cx", "Cy", "Cz"}) {
                    forces_log << ',' << name << '_' << c;
                }
            }
            forces_log << '\n';
        }
        forces_log << std::setprecision(8) << std::scientific;
    }
    const Scalar dynamic_area =
        0.5 * fc.density * fc.reference_velocity * fc.reference_velocity * fc.reference_area;
    auto write_forces = [&]() {
        forces_log << step_ << ',' << time_;
        for (const auto& name : fc.patches) {
            const Vec3 f = force(name);
            forces_log << ',' << f.x() << ',' << f.y() << ',' << f.z() << ','
                       << f.x() / dynamic_area << ',' << f.y() / dynamic_area << ','
                       << f.z() / dynamic_area;
        }
        forces_log << '\n';
    };

    const SolverConfig& s = config_.solver;
    const OutputConfig& o = config_.output;
    const bool steady = s.mode == SolverMode::Steady;
    const int max_steps =
        steady ? s.max_iterations
               : static_cast<int>(std::ceil((s.end_time - time_) / s.dt - 1e-9)) + step_;

    RunResult result;
    while (step_ < max_steps) {
        fv::Residuals r;
        if (steady) {
            r = flow_->simple_iteration();
        } else {
            r = flow_->piso_step(s.dt);
            time_ += s.dt;
        }
        ++step_;
        result.final_residuals = r;
        log << step_ << ',' << time_ << ',' << r.u.x() << ',' << r.u.y() << ',' << r.u.z() << ','
            << r.p << ',' << r.k << ',' << r.omega << ',' << r.continuity << '\n';
        log.flush(); // followed live (e.g. by the GUI for parallel runs)

        if (o.write_interval > 0 && step_ % o.write_interval == 0) {
            std::ostringstream name;
            name << config_.name << '_' << std::setw(6) << std::setfill('0') << step_ << ".vtu";
            write_results(dir / name.str());
        }
        if (o.checkpoint_interval > 0 && step_ % o.checkpoint_interval == 0) {
            write_checkpoint(dir / "checkpoint.bin");
        }
        if (!fc.patches.empty() && (o.log_interval <= 0 || step_ % o.log_interval == 0)) {
            write_forces();
        }
        const bool keep_going = !progress || progress(step_, time_, r);
        if (steady && r.max() < s.tolerance) {
            result.converged = true;
            break;
        }
        if (!keep_going) {
            result.stopped = true;
            break;
        }
    }
    if (!steady && step_ >= max_steps) {
        result.converged = true;
    }

    if (!fc.patches.empty() && o.log_interval > 0 && step_ % o.log_interval != 0) {
        write_forces();
    }
    result.steps = step_;
    result.time = time_;
    result.result_file = dir / (config_.name + ".vtu");
    write_results(result.result_file);
    write_checkpoint(dir / "checkpoint.bin");
    return result;
}

} // namespace cfd::cases
