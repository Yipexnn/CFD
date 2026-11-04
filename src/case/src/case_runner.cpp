#include <cfd/case/case_runner.hpp>
#include <cfd/mesh/gmsh_reader.hpp>
#include <cfd/mesh/vtu_writer.hpp>
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
        const std::string gmsh = geometry::find_gmsh();
        if (gmsh.empty()) {
            throw std::runtime_error("Gmsh executable not found (install Gmsh and add it to "
                                     "PATH or set CFD_GMSH)");
        }
        return geometry::generate_mesh(spec, config.resolve(config.output.directory) / "mesh",
                                       gmsh);
    }
    case MeshSource::Box:
        break;
    }
    return mesh::make_box_mesh(config.mesh.box);
}

namespace {

using core::Label;

// u = peak * prod_a 4 s (1 - s) over the patch bounding box.
std::vector<Vec3> parabolic_profile(const mesh::Mesh& mesh, const mesh::Patch& patch,
                                    const Vec3& peak, const std::string& axes) {
    const auto& cf = mesh.face_centres();
    Vec3 lo = Vec3::Constant(std::numeric_limits<Scalar>::max());
    Vec3 hi = -lo;
    for (Label q = 0; q < patch.n_faces; ++q) {
        for (Label pt : mesh.face(patch.start_face + q)) {
            const Vec3& x = mesh.points()[static_cast<std::size_t>(pt)];
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
    mesh_ = std::make_unique<mesh::Mesh>(build_mesh(config_));
    setup();
}

CaseRunner::CaseRunner(CaseConfig config, mesh::Mesh mesh)
    : config_(std::move(config)), mesh_(std::make_unique<mesh::Mesh>(std::move(mesh))) {
    setup();
}

void CaseRunner::setup() {
    for (const auto& name : config_.forces.patches) {
        if (mesh_->find_patch(name) == nullptr) {
            throw std::runtime_error("forces: unknown patch '" + name + "'");
        }
    }
    flow_ = std::make_unique<fv::FlowSolver>(*mesh_, config_.boundaries, config_.solver.flow);
    for (const auto& [name, profile] : config_.profiles) {
        const mesh::Patch* patch = mesh_->find_patch(name);
        const auto bc = config_.boundaries.find(name);
        if (patch == nullptr || bc == config_.boundaries.end()) {
            throw std::runtime_error("profile: unknown inlet patch '" + name + "'");
        }
        flow_->set_boundary_velocity_profile(
            name, parabolic_profile(*mesh_, *patch, bc->second.velocity, profile.axes));
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

void CaseRunner::write_results(const std::filesystem::path& file) const {
    std::vector<mesh::CellScalarData> scalars{{"p", &flow_->pressure()}};
    std::vector<mesh::CellVectorData> vectors{{"U", &flow_->velocity()}};
    if (sst_) {
        scalars.push_back({"k", &sst_->k()});
        scalars.push_back({"omega", &sst_->omega()});
        scalars.push_back({"nut", &sst_->nut()});
        scalars.push_back({"yPlus", &sst_->y_plus()});
    }
    mesh::write_vtu(file, *mesh_, scalars, vectors);
}

void CaseRunner::write_checkpoint(const std::filesystem::path& file) const {
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Checkpoint: cannot write " + file.string());
    }
    const auto nc = static_cast<std::int64_t>(mesh_->n_cells());
    const auto nf = static_cast<std::int64_t>(mesh_->n_faces());
    out.write(k_magic, sizeof(k_magic));
    write_pod(out, nc);
    write_pod(out, nf);
    write_pod(out, static_cast<std::int64_t>(step_));
    write_pod(out, time_);
    write_pod(out, static_cast<std::uint8_t>(sst_ ? 1 : 0));
    write_array(out, flow_->velocity().data()->data(), static_cast<std::size_t>(nc) * 3);
    write_array(out, flow_->pressure().data(), static_cast<std::size_t>(nc));
    write_array(out, flow_->face_flux().data(), static_cast<std::size_t>(nf));
    if (sst_) {
        write_array(out, sst_->k().data(), static_cast<std::size_t>(nc));
        write_array(out, sst_->omega().data(), static_cast<std::size_t>(nc));
        write_array(out, sst_->nut().data(), static_cast<std::size_t>(nc));
        write_array(out, sst_->boundary_nut().data(), sst_->boundary_nut().size());
    }
    if (!out) {
        throw std::runtime_error("Checkpoint: write failed for " + file.string());
    }
}

void CaseRunner::read_checkpoint(const std::filesystem::path& file) {
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
    if (nc != mesh_->n_cells() || nf != mesh_->n_faces()) {
        throw std::runtime_error("Checkpoint: mesh size mismatch (" + std::to_string(nc) +
                                 " cells in file, " + std::to_string(mesh_->n_cells()) +
                                 " in mesh)");
    }
    const auto step = read_pod<std::int64_t>(in);
    const auto time = read_pod<Scalar>(in);
    const auto has_turbulence = read_pod<std::uint8_t>(in);

    std::vector<Vec3> u(static_cast<std::size_t>(nc));
    std::vector<Scalar> p(static_cast<std::size_t>(nc));
    std::vector<Scalar> phi(static_cast<std::size_t>(nf));
    read_array(in, u.data()->data(), u.size() * 3);
    read_array(in, p.data(), p.size());
    read_array(in, phi.data(), phi.size());
    if (has_turbulence != 0) {
        std::vector<Scalar> k(static_cast<std::size_t>(nc));
        std::vector<Scalar> omega(static_cast<std::size_t>(nc));
        read_array(in, k.data(), k.size());
        read_array(in, omega.data(), omega.size());
        std::vector<Scalar> nut(static_cast<std::size_t>(nc));
        std::vector<Scalar> nut_b(static_cast<std::size_t>(mesh_->n_boundary_faces()));
        read_array(in, nut.data(), nut.size());
        read_array(in, nut_b.data(), nut_b.size());
        if (sst_) {
            sst_->set_fields(std::move(k), std::move(omega), std::move(nut), std::move(nut_b));
        }
    }
    flow_->set_state(std::move(u), std::move(p), std::move(phi));
    step_ = static_cast<int>(step);
    time_ = time;
}

Vec3 CaseRunner::force(const std::string& patch) const {
    return config_.forces.density * flow_->patch_force(patch).total();
}

RunResult CaseRunner::run(const ProgressCallback& progress) {
    const std::filesystem::path dir = output_directory();
    std::filesystem::create_directories(dir);
    const auto mode = step_ > 0 ? std::ios::app : std::ios::trunc;
    std::ofstream log(dir / "residuals.csv", mode);
    if (step_ == 0) {
        log << "step,time,Ux,Uy,Uz,p,k,omega,continuity\n";
    }
    log << std::setprecision(6) << std::scientific;
    const ForcesConfig& fc = config_.forces;
    std::ofstream forces_log;
    if (!fc.patches.empty()) {
        forces_log.open(dir / "forces.csv", mode);
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
