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

const mesh::Mesh& CaseRunner::full_mesh() const {
    return local_ ? local_->mesh : *global_mesh_;
}

const mesh::Mesh& CaseRunner::mesh() const {
    return fluid_ ? fluid_->mesh : full_mesh();
}

std::vector<Scalar> CaseRunner::to_full_cells(const Scalar* flow_values, std::size_t ncomp) const {
    const auto n = static_cast<std::size_t>(mesh().n_cells());
    if (!fluid_) {
        return {flow_values, flow_values + n * ncomp};
    }
    std::vector<Scalar> full(static_cast<std::size_t>(full_mesh().n_cells()) * ncomp, 0.0);
    for (std::size_t c = 0; c < n; ++c) {
        const auto p = static_cast<std::size_t>(fluid_->parent_cell[c]);
        std::copy_n(flow_values + c * ncomp, ncomp,
                    full.begin() + static_cast<std::ptrdiff_t>(p * ncomp));
    }
    return full;
}

std::vector<Scalar> CaseRunner::from_full_cells(const std::vector<Scalar>& full,
                                                std::size_t ncomp) const {
    if (!fluid_) {
        return full;
    }
    const auto n = static_cast<std::size_t>(mesh().n_cells());
    std::vector<Scalar> values(n * ncomp);
    for (std::size_t c = 0; c < n; ++c) {
        const auto p = static_cast<std::size_t>(fluid_->parent_cell[c]);
        std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(p * ncomp), ncomp,
                    values.begin() + static_cast<std::ptrdiff_t>(c * ncomp));
    }
    return values;
}

// Face values (all faces, or boundary faces only) between flow and full meshes; faces
// without a counterpart (fluid/solid interface, solid faces) are zero.
std::vector<Scalar> CaseRunner::to_full_faces(const std::vector<Scalar>& flow_values, bool oriented,
                                              bool boundary_only) const {
    if (!fluid_) {
        return flow_values;
    }
    const mesh::Mesh& full = full_mesh();
    const auto fnif = static_cast<std::size_t>(full.n_internal_faces());
    const auto snif = static_cast<std::size_t>(fluid_->mesh.n_internal_faces());
    const std::size_t foff = boundary_only ? fnif : 0;
    const std::size_t soff = boundary_only ? snif : 0;
    std::vector<Scalar> out(static_cast<std::size_t>(full.n_faces()) - foff, 0.0);
    for (std::size_t sf = soff; sf < fluid_->parent_face.size(); ++sf) {
        const auto pf = static_cast<std::size_t>(fluid_->parent_face[sf]);
        const bool sub_internal = sf < snif;
        if ((sub_internal && pf < fnif) || (!sub_internal && pf >= fnif)) {
            if (pf >= foff) {
                const Scalar v = flow_values[sf - soff];
                out[pf - foff] = oriented && fluid_->face_flipped[sf] != 0 ? -v : v;
            }
        }
    }
    return out;
}

std::vector<Scalar> CaseRunner::from_full_faces(const std::vector<Scalar>& full_values,
                                                bool oriented, bool boundary_only) const {
    if (!fluid_) {
        return full_values;
    }
    const mesh::Mesh& full = full_mesh();
    const auto fnif = static_cast<std::size_t>(full.n_internal_faces());
    const auto snif = static_cast<std::size_t>(fluid_->mesh.n_internal_faces());
    const std::size_t foff = boundary_only ? fnif : 0;
    const std::size_t soff = boundary_only ? snif : 0;
    std::vector<Scalar> out(fluid_->parent_face.size() - soff, 0.0);
    for (std::size_t sf = soff; sf < fluid_->parent_face.size(); ++sf) {
        const auto pf = static_cast<std::size_t>(fluid_->parent_face[sf]);
        const bool sub_internal = sf < snif;
        if (((sub_internal && pf < fnif) || (!sub_internal && pf >= fnif)) && pf >= foff) {
            const Scalar v = full_values[pf - foff];
            out[sf - soff] = oriented && fluid_->face_flipped[sf] != 0 ? -v : v;
        }
    }
    return out;
}

void CaseRunner::setup() {
    const parallel::Comm& comm = parallel::Comm::world();
    if (comm.parallel()) {
        const std::vector<int> parts = parallel::partition_rcb(*global_mesh_, comm.size());
        local_ = std::make_unique<parallel::LocalMesh>(
            parallel::decompose(*global_mesh_, parts, comm.rank(), comm));
    }
    const parallel::Halo* full_halo = local_ ? &local_->halo : nullptr;
    if (!config_.solids.empty()) {
        if (!config_.energy) {
            throw std::runtime_error("solids need the energy equation");
        }
        // Zones of the global mesh: 0 = fluid, 1.. = solids.
        const mesh::Mesh& g = *global_mesh_;
        std::vector<std::string> names{"fluid"};
        std::vector<int> global_zone(static_cast<std::size_t>(g.n_cells()), 0);
        for (const SolidConfig& sc : config_.solids) {
            const int z = static_cast<int>(names.size());
            names.push_back(sc.name);
            int mesh_zone = -1;
            for (std::size_t k = 0; k < g.cell_zone_names().size(); ++k) {
                if (g.cell_zone_names()[k] == sc.name) {
                    mesh_zone = static_cast<int>(k);
                }
            }
            if (!sc.min && mesh_zone < 0) {
                throw std::runtime_error("solids: mesh has no zone '" + sc.name +
                                         "' (give min/max for a box selection)");
            }
            Label count = 0;
            for (std::size_t c = 0; c < global_zone.size(); ++c) {
                const Vec3& x = g.cell_centres()[c];
                const bool inside = sc.min ? (x.array() >= sc.min->array()).all() &&
                                                 (x.array() <= sc.max->array()).all()
                                           : g.cell_zones()[c] == mesh_zone;
                if (inside) {
                    global_zone[c] = z;
                    ++count;
                }
            }
            if (count == 0) {
                throw std::runtime_error("solids: no cells in '" + sc.name + "'");
            }
        }
        const mesh::Mesh& full = full_mesh();
        conjugate_ = std::make_unique<fv::ConjugateMesh>();
        conjugate_->zone_names = names;
        conjugate_->zone_of_cell.resize(static_cast<std::size_t>(full.n_cells()));
        for (std::size_t c = 0; c < conjugate_->zone_of_cell.size(); ++c) {
            conjugate_->zone_of_cell[c] =
                global_zone[local_ ? static_cast<std::size_t>(local_->global_cell[c]) : c];
        }
        for (const SolidConfig& sc : config_.solids) {
            conjugate_->solids.push_back({sc.name, sc.conductivity, sc.rho_cp});
        }
        fluid_ = std::make_unique<parallel::SubMesh>(
            parallel::extract_zone(full, conjugate_->zone_of_cell, names, 0, full_halo));
        full_geometry_ = std::make_unique<fv::FvGeometry>(full, full_halo);
        conjugate_->geometry = full_geometry_.get();
        conjugate_->fluid = fluid_.get();
    }
    const mesh::Mesh* const solver_mesh = &mesh();
    for (const auto& name : config_.forces.patches) {
        if (solver_mesh->find_patch(name) == nullptr) {
            throw std::runtime_error("forces: unknown patch '" + name + "'");
        }
    }
    const parallel::Halo* flow_halo = fluid_ && local_ ? &fluid_->halo : full_halo;
    flow_ = std::make_unique<fv::FlowSolver>(*solver_mesh, config_.boundaries, config_.solver.flow,
                                             flow_halo);
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
    for (const auto& name : config_.heat.patches) {
        const bool solid_zone = std::any_of(config_.solids.begin(), config_.solids.end(),
                                            [&](const SolidConfig& sc) { return sc.name == name; });
        if (full_mesh().find_patch(name) == nullptr && !solid_zone) {
            throw std::runtime_error("heat: unknown patch or solid '" + name + "'");
        }
    }
    if (config_.energy) {
        energy_ =
            conjugate_
                ? std::make_unique<fv::EnergyEquation>(*flow_, *conjugate_, *config_.energy,
                                                       config_.thermal)
                : std::make_unique<fv::EnergyEquation>(*flow_, *config_.energy, config_.thermal);
        energy_->set_uniform_temperature(
            config_.initial.temperature.value_or(config_.energy->t_ref));
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
    auto cells = [&](const Scalar* values, std::size_t ncomp) {
        return gather_cells(lm, to_full_cells(values, ncomp).data(), ncomp, ng);
    };
    const std::vector<Vec3> u = to_vec3(cells(flow_->velocity().data()->data(), 3));
    const std::vector<Scalar> p = cells(flow_->pressure().data(), 1);
    std::vector<std::pair<std::string, std::vector<Scalar>>> turbulence;
    if (sst_) {
        turbulence.emplace_back("k", cells(sst_->k().data(), 1));
        turbulence.emplace_back("omega", cells(sst_->omega().data(), 1));
        turbulence.emplace_back("nut", cells(sst_->nut().data(), 1));
        turbulence.emplace_back("yPlus", cells(sst_->y_plus().data(), 1));
    }
    if (energy_) {
        turbulence.emplace_back("T", gather_cells(lm, energy_->temperature().data(), 1, ng));
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
    auto cells = [&](const Scalar* values, std::size_t ncomp) {
        return gather_cells(lm, to_full_cells(values, ncomp).data(), ncomp, ng);
    };
    const std::vector<Scalar> u = cells(flow_->velocity().data()->data(), 3);
    const std::vector<Scalar> p = cells(flow_->pressure().data(), 1);
    const std::vector<Scalar> phi =
        gather_faces(lm, g, to_full_faces(flow_->face_flux(), true, false), true, false);
    const std::vector<Scalar> temperature =
        energy_ ? gather_cells(lm, energy_->temperature().data(), 1, ng) : std::vector<Scalar>{};
    std::vector<std::vector<Scalar>> turbulence;
    if (sst_) {
        turbulence.push_back(cells(sst_->k().data(), 1));
        turbulence.push_back(cells(sst_->omega().data(), 1));
        turbulence.push_back(cells(sst_->nut().data(), 1));
        turbulence.push_back(
            gather_faces(lm, g, to_full_faces(sst_->boundary_nut(), false, true), false, true));
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
    // Optional energy block (older checkpoints end here).
    write_pod(out, static_cast<std::uint8_t>(energy_ ? 1 : 0));
    write_array(out, temperature.data(), temperature.size());
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
    if (in.peek() != std::char_traits<char>::eof() && read_pod<std::uint8_t>(in) != 0) {
        d.t = read_vector(ncs);
    }
    return d;
}

void CaseRunner::read_checkpoint(const std::filesystem::path& file) {
    const mesh::Mesh& g = *global_mesh_;
    CheckpointData d = read_checkpoint_file(file, g.n_cells(), g.n_faces(), g.n_boundary_faces());
    const parallel::LocalMesh* lm = local_.get();
    auto cells = [&](const std::vector<Scalar>& global, std::size_t ncomp) {
        return from_full_cells(scatter_cells(lm, global, ncomp), ncomp);
    };
    if (!d.k.empty() && sst_) {
        sst_->set_fields(
            cells(d.k, 1), cells(d.omega, 1), cells(d.nut, 1),
            from_full_faces(scatter_faces(lm, g, d.boundary_nut, false, true), false, true));
    }
    std::vector<Scalar> u_flat(3 * d.u.size());
    for (std::size_t c = 0; c < d.u.size(); ++c) {
        for (int i = 0; i < 3; ++i) {
            u_flat[3 * c + static_cast<std::size_t>(i)] = d.u[c][i];
        }
    }
    flow_->set_state(to_vec3(cells(u_flat, 3)), cells(d.p, 1),
                     from_full_faces(scatter_faces(lm, g, d.phi, true, false), true, false));
    if (energy_ && !d.t.empty()) {
        energy_->set_temperature(scatter_cells(lm, d.t, 1));
    }
    step_ = d.step;
    time_ = d.time;
}

Scalar CaseRunner::heat_flow(const std::string& patch) const {
    if (!energy_) {
        throw std::runtime_error("heat_flow: the case has no energy equation");
    }
    return config_.heat.rho_cp * energy_->patch_heat_flow(*flow_, patch);
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
        log << "step,time,Ux,Uy,Uz,p,k,omega,continuity,T\n";
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

    const HeatConfig& hc = config_.heat;
    std::ofstream heat_log;
    if (energy_ && !hc.patches.empty()) {
        if (root) {
            heat_log.open(dir / "heat.csv", mode);
        }
        if (step_ == 0) {
            heat_log << "step,time";
            for (const auto& name : hc.patches) {
                heat_log << ',' << name << "_Q," << name << "_Nu";
            }
            heat_log << '\n';
        }
        heat_log << std::setprecision(8) << std::scientific;
    }
    auto write_heat = [&]() {
        if (!energy_ || hc.patches.empty()) {
            return;
        }
        heat_log << step_ << ',' << time_;
        const Scalar nu_scale = hc.reference_length / (energy_->settings().alpha *
                                                       hc.reference_delta_t * hc.reference_area);
        for (const auto& name : hc.patches) {
            const Scalar q = energy_->patch_heat_flow(*flow_, name);
            heat_log << ',' << hc.rho_cp * q << ',' << q * nu_scale;
        }
        heat_log << '\n';
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
        if (energy_ && energy_->buoyant()) {
            flow_->set_cell_acceleration(energy_->buoyancy());
        }
        if (steady) {
            r = flow_->simple_iteration();
        } else {
            r = flow_->piso_step(s.dt);
            time_ += s.dt;
        }
        if (energy_) {
            r.t = energy_->correct(*flow_, steady ? 0.0 : s.dt);
        }
        ++step_;
        result.final_residuals = r;
        log << step_ << ',' << time_ << ',' << r.u.x() << ',' << r.u.y() << ',' << r.u.z() << ','
            << r.p << ',' << r.k << ',' << r.omega << ',' << r.continuity << ',' << r.t << '\n';
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
        if (o.log_interval <= 0 || step_ % o.log_interval == 0) {
            write_heat();
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

    if (o.log_interval > 0 && step_ % o.log_interval != 0) {
        if (!fc.patches.empty()) {
            write_forces();
        }
        write_heat();
    }
    result.steps = step_;
    result.time = time_;
    result.result_file = dir / (config_.name + ".vtu");
    write_results(result.result_file);
    write_checkpoint(dir / "checkpoint.bin");
    return result;
}

} // namespace cfd::cases
