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
#include <tuple>

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

namespace {

// The global mesh in this process only (Gmsh called directly, no collective operations).
mesh::Mesh build_mesh_serial(const CaseConfig& config) {
    if (config.mesh.source != MeshSource::Geometry) {
        return config.mesh.source == MeshSource::Gmsh
                   ? mesh::read_gmsh(config.resolve(config.mesh.file))
                   : mesh::make_box_mesh(config.mesh.box);
    }
    geometry::GeometrySpec spec = config.mesh.geometry;
    for (auto& body : spec.bodies) {
        if (body.kind == geometry::Body::Kind::File) {
            body.file = config.resolve(body.file);
        }
    }
    const std::string gmsh = geometry::find_gmsh();
    if (gmsh.empty()) {
        throw std::runtime_error(
            "Gmsh executable not found (install Gmsh and add it to PATH or set CFD_GMSH)");
    }
    return geometry::generate_mesh(spec, config.resolve(config.output.directory) / "mesh", gmsh);
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
std::vector<Vec3> parabolic_profile(const mesh::Mesh& mesh, const mesh::Patch& patch,
                                    const Vec3& peak, const std::string& axes,
                                    const parallel::Comm& comm) {
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
    for (int d = 0; d < 3; ++d) { // extent of the whole (global) patch
        lo[d] = comm.min(lo[d]);
        hi[d] = comm.max(hi[d]);
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
    // Parallel runs: only the root process reads / generates the global mesh; setup()
    // distributes it (each process keeps only its part).
    const parallel::Comm& comm = parallel::Comm::world();
    if (!comm.parallel()) {
        global_mesh_ = std::make_unique<mesh::Mesh>(build_mesh(config_));
    } else if (!config_.decomposed_mesh) {
        std::string error;
        if (comm.is_root()) {
            try {
                global_mesh_ = std::make_unique<mesh::Mesh>(build_mesh_serial(config_));
            } catch (const std::exception& e) {
                error = e.what();
            }
        }
        comm.broadcast(error);
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
    }
    setup();
}

CaseRunner::CaseRunner(CaseConfig config, mesh::Mesh mesh) : config_(std::move(config)) {
    if (!parallel::Comm::world().parallel() || parallel::Comm::world().is_root()) {
        global_mesh_ = std::make_unique<mesh::Mesh>(std::move(mesh));
    }
    setup();
}

std::vector<Vec3> CaseRunner::velocity() const {
    if (compressible_) {
        return compressible_->velocity();
    }
    return multiphase_ ? multiphase_->velocity() : flow_->velocity();
}

std::vector<Scalar> CaseRunner::pressure() const {
    if (compressible_) {
        return compressible_->pressure();
    }
    return multiphase_ ? multiphase_->pressure() : flow_->pressure();
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

namespace {

constexpr char k_processor_magic[8] = {'C', 'F', 'D', 'P', 'R', 'O', 'C', '1'};

void write_processor_file(const std::filesystem::path& file, const parallel::LocalMesh& lm,
                          int n_parts, int rank, const std::array<Label, 3>& counts) {
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("decompose: cannot write " + file.string());
    }
    const std::vector<char> bytes = parallel::pack_local_mesh(lm);
    out.write(k_processor_magic, sizeof(k_processor_magic));
    const std::int32_t header[2] = {n_parts, rank};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(counts.data()), sizeof(Label) * counts.size());
    const auto n = static_cast<std::uint64_t>(bytes.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("decompose: write failed for " + file.string());
    }
}

} // namespace

static parallel::LocalMesh read_processor_file(const std::filesystem::path& file,
                                               const parallel::Comm& comm,
                                               std::array<Label, 3>& counts) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("decomposed mesh: cannot open " + file.string() +
                                 " (run `cfd decompose <case.json> <processes>` first)");
    }
    char magic[sizeof(k_processor_magic)];
    in.read(magic, sizeof(magic));
    std::int32_t header[2] = {0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    in.read(reinterpret_cast<char*>(counts.data()), sizeof(Label) * counts.size());
    std::uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in || std::memcmp(magic, k_processor_magic, sizeof(magic)) != 0) {
        throw std::runtime_error("decomposed mesh: " + file.string() + " is not a processor file");
    }
    if (header[0] != comm.size() || header[1] != comm.rank()) {
        throw std::runtime_error("decomposed mesh: " + file.string() + " was written for " +
                                 std::to_string(header[0]) + " processes");
    }
    std::vector<char> bytes(static_cast<std::size_t>(n));
    in.read(bytes.data(), static_cast<std::streamsize>(n));
    if (!in) {
        throw std::runtime_error("decomposed mesh: truncated " + file.string());
    }
    return parallel::unpack_local_mesh(bytes, comm);
}

std::filesystem::path decomposed_mesh_directory(const CaseConfig& config, int n_parts) {
    return config.resolve(config.output.directory) / ("processors" + std::to_string(n_parts));
}

std::filesystem::path decompose_case(const CaseConfig& config, int n_parts) {
    if (n_parts < 2) {
        throw std::invalid_argument("decompose: needs at least 2 processes");
    }
    const mesh::Mesh global = build_mesh_serial(config);
    const std::vector<int> parts = parallel::partition_rcb(global, n_parts);
    const auto dir = decomposed_mesh_directory(config, n_parts);
    std::filesystem::create_directories(dir);
    const std::array<Label, 3> counts{global.n_cells(), global.n_faces(),
                                      global.n_internal_faces()};
    for (int r = 0; r < n_parts; ++r) {
        const parallel::LocalMesh lm =
            parallel::decompose(global, parts, r, parallel::Comm::self());
        write_processor_file(dir / ("processor" + std::to_string(r) + ".bin"), lm, n_parts, r,
                             counts);
    }
    return dir;
}

void CaseRunner::setup() {
    const parallel::Comm& comm = parallel::Comm::world();
    std::array<Label, 3> counts{0, 0, 0};
    if (global_mesh_) {
        counts = {global_mesh_->n_cells(), global_mesh_->n_faces(),
                  global_mesh_->n_internal_faces()};
    }
    if (comm.parallel() && config_.decomposed_mesh) {
        // Every process reads only its own part (written by decompose_case).
        const auto file = decomposed_mesh_directory(config_, comm.size()) /
                          ("processor" + std::to_string(comm.rank()) + ".bin");
        local_ = std::make_unique<parallel::LocalMesh>(read_processor_file(file, comm, counts));
    } else if (comm.parallel()) {
        // Root partitions and sends each process its part (no global mesh elsewhere).
        local_ = std::make_unique<parallel::LocalMesh>(
            parallel::distribute(comm.is_root() ? global_mesh_.get() : nullptr, comm));
        for (Label& v : counts) {
            v = comm.sum(comm.is_root() ? v : 0);
        }
    }
    global_cells_ = counts[0];
    global_faces_ = counts[1];
    global_internal_faces_ = counts[2];
    const parallel::Halo* full_halo = local_ ? &local_->halo : nullptr;
    if (!config_.solids.empty()) {
        if (!config_.energy) {
            throw std::runtime_error("solids need the energy equation");
        }
        // Zones of this process's cells: 0 = fluid, 1.. = solids.
        const mesh::Mesh& full = full_mesh();
        std::vector<std::string> names{"fluid"};
        std::vector<int> zone(static_cast<std::size_t>(full.n_cells()), 0);
        for (const SolidConfig& sc : config_.solids) {
            const int z = static_cast<int>(names.size());
            names.push_back(sc.name);
            int mesh_zone = -1;
            for (std::size_t k = 0; k < full.cell_zone_names().size(); ++k) {
                if (full.cell_zone_names()[k] == sc.name) {
                    mesh_zone = static_cast<int>(k);
                }
            }
            if (!sc.min && mesh_zone < 0) {
                throw std::runtime_error("solids: mesh has no zone '" + sc.name +
                                         "' (give min/max for a box selection)");
            }
            Label count = 0;
            const Label n_owned = local_ ? local_->halo.n_owned : full.n_cells();
            for (std::size_t c = 0; c < zone.size(); ++c) {
                const Vec3& x = full.cell_centres()[c];
                const bool inside = sc.min ? (x.array() >= sc.min->array()).all() &&
                                                 (x.array() <= sc.max->array()).all()
                                           : full.cell_zones()[c] == mesh_zone;
                if (inside) {
                    zone[c] = z;
                    count += static_cast<Label>(c) < n_owned ? 1 : 0;
                }
            }
            if (comm.sum(count) == 0) {
                throw std::runtime_error("solids: no cells in '" + sc.name + "'");
            }
        }
        conjugate_ = std::make_unique<fv::ConjugateMesh>();
        conjugate_->zone_names = names;
        conjugate_->zone_of_cell = std::move(zone);
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
    if (config_.compressible) {
        if (!config_.forces.patches.empty() || !config_.profiles.empty()) {
            throw std::runtime_error(
                "compressible: forces and inlet profiles are not supported yet");
        }
        compressible_ = std::make_unique<fv::CompressibleSolver>(*solver_mesh, config_.boundaries,
                                                                 *config_.compressible, flow_halo);
        const Scalar r = config_.compressible->r_gas;
        const auto& in = config_.initial;
        std::vector<fv::GasState> states(
            static_cast<std::size_t>(solver_mesh->n_cells()),
            {in.pressure / (r * in.temperature_gas), in.velocity, in.pressure});
        for (std::size_t c = 0; c < states.size(); ++c) {
            const Vec3& x = solver_mesh->cell_centres()[c];
            for (const auto& g : in.regions) {
                if ((x.array() >= g.min.array()).all() && (x.array() <= g.max.array()).all()) {
                    states[c] = {g.pressure / (r * g.temperature), g.velocity, g.pressure};
                }
            }
        }
        compressible_->set_state(states);
        if (config_.restart) {
            read_checkpoint(config_.resolve(*config_.restart));
        }
        return;
    }
    if (config_.multiphase) {
        if (!config_.forces.patches.empty() || !config_.profiles.empty()) {
            throw std::runtime_error("multiphase: forces and inlet profiles are not supported yet");
        }
        multiphase_ = std::make_unique<fv::MultiphaseSolver>(*solver_mesh, config_.boundaries,
                                                             *config_.multiphase, flow_halo);
        for (const auto& [name, a] : config_.inlet_alpha) {
            multiphase_->set_inlet_alpha(name, a);
        }
        std::vector<Scalar> alpha(static_cast<std::size_t>(solver_mesh->n_cells()), 0.0);
        for (std::size_t c = 0; c < alpha.size(); ++c) {
            const Vec3& x = solver_mesh->cell_centres()[c];
            for (const auto& [lo, hi] : config_.initial.alpha_regions) {
                if ((x.array() >= lo.array()).all() && (x.array() <= hi.array()).all()) {
                    alpha[c] = 1.0;
                }
            }
        }
        multiphase_->set_alpha(alpha);
        multiphase_->set_velocity(std::vector<Vec3>(
            static_cast<std::size_t>(solver_mesh->n_cells()), config_.initial.velocity));
        if (config_.restart) {
            read_checkpoint(config_.resolve(*config_.restart));
        }
        return;
    }
    flow_ = std::make_unique<fv::FlowSolver>(*solver_mesh, config_.boundaries, config_.solver.flow,
                                             flow_halo);
    for (const auto& [name, profile] : config_.profiles) {
        const mesh::Patch* patch = solver_mesh->find_patch(name);
        const auto bc = config_.boundaries.find(name);
        if (patch == nullptr || bc == config_.boundaries.end()) {
            throw std::runtime_error("profile: unknown inlet patch '" + name + "'");
        }
        flow_->set_boundary_velocity_profile(
            name, parabolic_profile(*solver_mesh, *patch, bc->second.velocity, profile.axes, comm));
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
    if (energy_ && !energy_->settings().nu_law.is_constant()) {
        flow_->set_cell_viscosity(energy_->fluid_viscosity());
    }
    if (energy_ && config_.radiation) {
        radiation_ = std::make_unique<fv::P1Radiation>(*flow_, *config_.radiation);
        update_radiation();
    }
}

void CaseRunner::update_radiation() {
    // Solve G for the current temperature and pass the linearised radiative source (divided by
    // the fluid rho c_p of physics.energy) to the energy equation for its next solve.
    const Scalar rho_cp = energy_->settings().rho_cp;
    if (!fluid_) {
        const std::vector<Scalar>& t = energy_->temperature();
        (void)radiation_->correct(t, energy_->boundary_temperature());
        std::vector<Scalar> su;
        std::vector<Scalar> sp;
        radiation_->energy_source(t, su, sp);
        for (std::size_t c = 0; c < su.size(); ++c) {
            su[c] /= rho_cp;
            sp[c] /= rho_cp;
        }
        energy_->set_source(std::move(su), std::move(sp));
        return;
    }
    // Conjugate: radiation in the fluid zone; fluid/solid interfaces are opaque walls at the
    // interface temperature, and their net radiative flux is a surface heat sink of the solid
    // cells behind them.
    const mesh::Mesh& full = full_mesh();
    const mesh::Mesh& fm = fluid_->mesh;
    const auto fnif = static_cast<std::size_t>(full.n_internal_faces());
    const auto snif = static_cast<std::size_t>(fm.n_internal_faces());
    const std::vector<Scalar>& t_full = energy_->temperature();
    const std::vector<Scalar> tb_full = energy_->boundary_temperature();
    const std::vector<Scalar> t_fluid = energy_->fluid_temperature();
    std::vector<Scalar> tb(static_cast<std::size_t>(fm.n_boundary_faces()));
    for (std::size_t b = 0; b < tb.size(); ++b) {
        const auto pf = static_cast<std::size_t>(fluid_->parent_face[snif + b]);
        if (pf >= fnif) {
            tb[b] = tb_full[pf - fnif];
        } else {
            // Surface temperature of the interface: conductivity-weighted (continuity of the
            // conductive flux), i.e. the solid temperature next to a weakly conducting gas.
            const auto o = static_cast<std::size_t>(full.owner()[pf]);
            const auto n = static_cast<std::size_t>(full.neighbour()[pf]);
            const Vec3& cf = full.face_centres()[pf];
            const Vec3 nrm = full.face_area_vectors()[pf].normalized();
            auto conductance = [&](std::size_t c) {
                const int zone = conjugate_->zone_of_cell[c];
                const Scalar k =
                    zone > 0 ? conjugate_->solids[static_cast<std::size_t>(zone - 1)].conductivity /
                                   energy_->settings().rho_cp
                             : energy_->settings().alpha;
                return k / std::max(std::abs((cf - full.cell_centres()[c]).dot(nrm)), 1e-300);
            };
            const Scalar go = conductance(o);
            const Scalar gn = conductance(n);
            tb[b] = (go * t_full[o] + gn * t_full[n]) / (go + gn);
        }
    }
    (void)radiation_->correct(t_fluid, tb);
    std::vector<Scalar> su_fluid;
    std::vector<Scalar> sp_fluid;
    radiation_->energy_source(t_fluid, su_fluid, sp_fluid);
    std::vector<Scalar> su(t_full.size(), 0.0);
    std::vector<Scalar> sp(t_full.size(), 0.0);
    for (std::size_t c = 0; c < su_fluid.size(); ++c) {
        const auto p = static_cast<std::size_t>(fluid_->parent_cell[c]);
        su[p] = su_fluid[c] / rho_cp;
        sp[p] = sp_fluid[c] / rho_cp;
    }
    const std::vector<Scalar> q = radiation_->wall_heat_flux(); // into the gas [W/m^2]
    std::vector<char> is_fluid(t_full.size(), 0);
    for (const Label c : fluid_->parent_cell) {
        is_fluid[static_cast<std::size_t>(c)] = 1;
    }
    for (std::size_t b = 0; b < q.size(); ++b) {
        const auto pf = static_cast<std::size_t>(fluid_->parent_face[snif + b]);
        if (pf >= fnif || q[b] == 0.0) {
            continue;
        }
        const auto o = static_cast<std::size_t>(full.owner()[pf]);
        const auto n = static_cast<std::size_t>(full.neighbour()[pf]);
        const std::size_t solid = is_fluid[o] != 0 ? n : o;
        const Scalar area = fm.face_area_vectors()[snif + b].norm();
        su[solid] -= q[b] * area / full.cell_volumes()[solid] / rho_cp;
    }
    energy_->set_source(std::move(su), std::move(sp));
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
// Global face counts (the global mesh exists on the root process only).
struct FaceCounts {
    Label n_faces = 0;
    Label n_internal_faces = 0;
};

std::vector<Scalar> gather_faces(const parallel::LocalMesh* lm, const FaceCounts& global,
                                 const std::vector<Scalar>& values, bool oriented,
                                 bool boundary_only) {
    const auto gnif = static_cast<std::size_t>(global.n_internal_faces);
    const std::size_t offset = boundary_only ? gnif : 0;
    const std::size_t n_out = static_cast<std::size_t>(global.n_faces) - offset;
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

std::vector<Scalar> scatter_faces(const parallel::LocalMesh* lm, const FaceCounts& global,
                                  const std::vector<Scalar>& values, bool oriented,
                                  bool boundary_only) {
    if (lm == nullptr) {
        return values;
    }
    const auto& m = lm->mesh;
    const std::size_t lfirst = boundary_only ? static_cast<std::size_t>(m.n_internal_faces()) : 0;
    const std::size_t gfirst =
        boundary_only ? static_cast<std::size_t>(global.n_internal_faces) : 0;
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

const fv::KOmegaSst* CaseRunner::sst() const {
    if (sst_) {
        return sst_.get();
    }
    return multiphase_ ? multiphase_->sst() : nullptr;
}

fv::KOmegaSst* CaseRunner::mutable_sst() {
    if (sst_) {
        return sst_.get();
    }
    return multiphase_ ? multiphase_->sst() : nullptr;
}

void CaseRunner::write_results(const std::filesystem::path& file) const {
    const auto ng = static_cast<std::size_t>(global_cells_);
    const parallel::LocalMesh* lm = local_.get();
    // Without a global mesh (decomposed runs) every process writes its owned cells as a piece.
    const bool pieces = decomposed_run();
    if (pieces && !piece_) {
        const mesh::Mesh& full = full_mesh();
        std::vector<int> zone(static_cast<std::size_t>(full.n_cells()), 1);
        for (Label c = 0; c < lm->halo.n_owned; ++c) {
            zone[static_cast<std::size_t>(c)] = 0;
        }
        piece_ = std::make_unique<parallel::SubMesh>(
            parallel::extract_zone(full, zone, {"owned", "ghost"}, 0, nullptr));
    }
    auto collect = [&](const Scalar* full_values, std::size_t ncomp) {
        if (!pieces) {
            return gather_cells(lm, full_values, ncomp, ng);
        }
        std::vector<Scalar> out;
        out.reserve(piece_->parent_cell.size() * ncomp);
        for (const Label c : piece_->parent_cell) {
            for (std::size_t k = 0; k < ncomp; ++k) {
                out.push_back(full_values[static_cast<std::size_t>(c) * ncomp + k]);
            }
        }
        return out;
    };
    auto cells = [&](const Scalar* values, std::size_t ncomp) {
        return collect(to_full_cells(values, ncomp).data(), ncomp);
    };
    const std::vector<Vec3> u_cells = velocity();
    const std::vector<Scalar> p_cells = pressure();
    const std::vector<Vec3> u = to_vec3(cells(u_cells.data()->data(), 3));
    const std::vector<Scalar> p = cells(p_cells.data(), 1);
    std::vector<std::pair<std::string, std::vector<Scalar>>> turbulence;
    if (compressible_) {
        turbulence.emplace_back("rho", cells(compressible_->density().data(), 1));
        turbulence.emplace_back("T", cells(compressible_->temperature().data(), 1));
        turbulence.emplace_back("Mach", cells(compressible_->mach().data(), 1));
        if (compressible_->settings().turbulence) {
            std::vector<Scalar> nut = compressible_->eddy_viscosity();
            const std::vector<Scalar> rho = compressible_->density();
            for (std::size_t c = 0; c < nut.size(); ++c) {
                nut[c] /= rho[c];
            }
            turbulence.emplace_back("k", cells(compressible_->k().data(), 1));
            turbulence.emplace_back("omega", cells(compressible_->omega().data(), 1));
            turbulence.emplace_back("nut", cells(nut.data(), 1));
        }
    }
    if (multiphase_) {
        turbulence.emplace_back("alpha", cells(multiphase_->alpha().data(), 1));
        turbulence.emplace_back("p_rgh", cells(multiphase_->p_rgh().data(), 1));
        turbulence.emplace_back("rho", cells(multiphase_->density().data(), 1));
    }
    if (sst() != nullptr) {
        turbulence.emplace_back("k", cells(sst()->k().data(), 1));
        turbulence.emplace_back("omega", cells(sst()->omega().data(), 1));
        turbulence.emplace_back("nut", cells(sst()->nut().data(), 1));
        turbulence.emplace_back("yPlus", cells(sst()->y_plus().data(), 1));
    }
    if (energy_) {
        turbulence.emplace_back("T", collect(energy_->temperature().data(), 1));
        if (radiation_) {
            turbulence.emplace_back("G", cells(radiation_->incident_radiation().data(), 1));
        }
    }
    std::vector<mesh::CellScalarData> scalars{{"p", &p}};
    std::vector<mesh::CellVectorData> vectors{{"U", &u}};
    for (const auto& [name, values] : turbulence) {
        scalars.push_back({name, &values});
    }
    if (pieces) {
        const std::string stem = file.stem().string();
        const auto piece_dir = file.parent_path() / stem;
        std::filesystem::create_directories(piece_dir);
        const std::string piece_name = stem + "_" + std::to_string(world().rank()) + ".vtu";
        mesh::write_vtu(piece_dir / piece_name, piece_->mesh, scalars, vectors);
        world().barrier();
        if (world().is_root()) {
            std::vector<std::string> names;
            for (int r = 0; r < world().size(); ++r) {
                names.push_back(stem + "/" + stem + "_" + std::to_string(r) + ".vtu");
            }
            std::vector<std::string> scalar_names;
            for (const auto& sd : scalars) {
                scalar_names.push_back(sd.name);
            }
            mesh::write_pvtu(std::filesystem::path(file).replace_extension(".pvtu"), names,
                             scalar_names, {"U"});
        }
        return;
    }
    if (!world().is_root()) {
        return;
    }
    mesh::write_vtu(file, *global_mesh_, scalars, vectors);
}

void CaseRunner::write_checkpoint(const std::filesystem::path& file) const {
    const FaceCounts g{global_faces_, global_internal_faces_};
    const auto nc = static_cast<std::int64_t>(global_cells_);
    const auto nf = static_cast<std::int64_t>(global_faces_);
    const auto ng = static_cast<std::size_t>(nc);
    const parallel::LocalMesh* lm = local_.get();
    auto cells = [&](const Scalar* values, std::size_t ncomp) {
        return gather_cells(lm, to_full_cells(values, ncomp).data(), ncomp, ng);
    };
    const std::vector<Vec3> u_cells = velocity();
    const std::vector<Scalar> p_cells = multiphase_ ? multiphase_->p_rgh() : pressure();
    const std::vector<Scalar> u = cells(u_cells.data()->data(), 3);
    const std::vector<Scalar> p = cells(p_cells.data(), 1);
    const std::vector<Scalar> no_flux(static_cast<std::size_t>(mesh().n_faces()), 0.0);
    const std::vector<Scalar>& face_flux = multiphase_     ? multiphase_->face_flux()
                                           : compressible_ ? no_flux
                                                           : flow_->face_flux();
    std::vector<Scalar> conserved;
    if (compressible_) {
        std::vector<Scalar> flat;
        for (const auto& qc : compressible_->conserved()) {
            flat.insert(flat.end(), qc.data(), qc.data() + 5);
        }
        conserved = cells(flat.data(), 5);
    }
    const std::vector<Scalar> phi =
        gather_faces(lm, g, to_full_faces(face_flux, true, false), true, false);
    const std::vector<Scalar> alpha =
        multiphase_ ? cells(multiphase_->alpha().data(), 1) : std::vector<Scalar>{};
    const std::vector<Scalar> temperature =
        energy_ ? gather_cells(lm, energy_->temperature().data(), 1, ng) : std::vector<Scalar>{};
    std::vector<std::vector<Scalar>> turbulence;
    const bool compressible_turbulence = compressible_ && compressible_->settings().turbulence;
    if (compressible_turbulence) {
        // k, omega, mu_t; no boundary-face values (mu_t = 0 on walls, owner value elsewhere).
        turbulence.push_back(cells(compressible_->k().data(), 1));
        turbulence.push_back(cells(compressible_->omega().data(), 1));
        turbulence.push_back(cells(compressible_->eddy_viscosity().data(), 1));
        turbulence.emplace_back(static_cast<std::size_t>(g.n_faces - g.n_internal_faces), 0.0);
    }
    if (sst() != nullptr) {
        turbulence.push_back(cells(sst()->k().data(), 1));
        turbulence.push_back(cells(sst()->omega().data(), 1));
        turbulence.push_back(cells(sst()->nut().data(), 1));
        turbulence.push_back(
            gather_faces(lm, g, to_full_faces(sst()->boundary_nut(), false, true), false, true));
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
    write_pod(out, static_cast<std::uint8_t>(sst() != nullptr || compressible_turbulence ? 1 : 0));
    write_array(out, u.data(), u.size());
    write_array(out, p.data(), p.size());
    write_array(out, phi.data(), phi.size());
    for (const auto& field : turbulence) {
        write_array(out, field.data(), field.size());
    }
    // Optional energy block (older checkpoints end here).
    write_pod(out, static_cast<std::uint8_t>(energy_ ? 1 : 0));
    write_array(out, temperature.data(), temperature.size());
    // Optional multiphase block: p above is p_rgh.
    write_pod(out, static_cast<std::uint8_t>(multiphase_ ? 1 : 0));
    write_array(out, alpha.data(), alpha.size());
    // Optional compressible block: conserved variables (rho, rho u, rho E) per cell.
    write_pod(out, static_cast<std::uint8_t>(compressible_ ? 1 : 0));
    write_array(out, conserved.data(), conserved.size());
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
    if (in.peek() != std::char_traits<char>::eof() && read_pod<std::uint8_t>(in) != 0) {
        d.alpha = read_vector(ncs);
    }
    if (in.peek() != std::char_traits<char>::eof() && read_pod<std::uint8_t>(in) != 0) {
        d.conserved = read_vector(5 * ncs);
    }
    return d;
}

void CaseRunner::read_checkpoint(const std::filesystem::path& file) {
    const FaceCounts g{global_faces_, global_internal_faces_};
    CheckpointData d = read_checkpoint_file(file, global_cells_, global_faces_,
                                            global_faces_ - global_internal_faces_);
    const parallel::LocalMesh* lm = local_.get();
    auto cells = [&](const std::vector<Scalar>& global, std::size_t ncomp) {
        return from_full_cells(scatter_cells(lm, global, ncomp), ncomp);
    };
    if (!d.k.empty() && sst() != nullptr) {
        mutable_sst()->set_fields(
            cells(d.k, 1), cells(d.omega, 1), cells(d.nut, 1),
            from_full_faces(scatter_faces(lm, g, d.boundary_nut, false, true), false, true));
    }
    std::vector<Scalar> u_flat(3 * d.u.size());
    for (std::size_t c = 0; c < d.u.size(); ++c) {
        for (int i = 0; i < 3; ++i) {
            u_flat[3 * c + static_cast<std::size_t>(i)] = d.u[c][i];
        }
    }
    if (compressible_) {
        if (d.conserved.empty()) {
            throw std::runtime_error("Checkpoint: no conserved variables for a compressible case");
        }
        const std::vector<Scalar> flat = cells(d.conserved, 5);
        std::vector<fv::CompressibleSolver::Conserved> q(flat.size() / 5);
        for (std::size_t c = 0; c < q.size(); ++c) {
            q[c] = Eigen::Map<const fv::CompressibleSolver::Conserved>(flat.data() + 5 * c);
        }
        compressible_->set_conserved(std::move(q));
        if (compressible_->settings().turbulence && !d.k.empty()) {
            compressible_->set_turbulence(cells(d.k, 1), cells(d.omega, 1));
        }
        step_ = d.step;
        time_ = d.time;
        return;
    }
    if (multiphase_) {
        if (d.alpha.empty()) {
            throw std::runtime_error("Checkpoint: no volume fraction for a multiphase case");
        }
        multiphase_->set_state(
            to_vec3(cells(u_flat, 3)), cells(d.p, 1), cells(d.alpha, 1),
            from_full_faces(scatter_faces(lm, g, d.phi, true, false), true, false));
        step_ = d.step;
        time_ = d.time;
        return;
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
    const Scalar conduction = config_.heat.rho_cp * energy_->patch_heat_flow(*flow_, patch);
    return radiation_ ? conduction + radiation_->patch_heat_flow(patch) : conduction;
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
                // C = total coefficient, Cp / Cf = pressure / friction (viscous) parts.
                for (const char* c : {"Fx", "Fy", "Fz", "Cx", "Cy", "Cz", "Cpx", "Cpy", "Cpz",
                                      "Cfx", "Cfy", "Cfz"}) {
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
            const fv::PatchForce pf = flow_->patch_force(name);
            const Vec3 f = fc.density * pf.total();
            const Vec3 cp = fc.density * pf.pressure / dynamic_area;
            const Vec3 cv = fc.density * pf.viscous / dynamic_area;
            forces_log << ',' << f.x() << ',' << f.y() << ',' << f.z() << ','
                       << f.x() / dynamic_area << ',' << f.y() / dynamic_area << ','
                       << f.z() / dynamic_area << ',' << cp.x() << ',' << cp.y() << ',' << cp.z()
                       << ',' << cv.x() << ',' << cv.y() << ',' << cv.z();
        }
        forces_log << '\n';
        forces_log.flush(); // followed live; survives interrupted runs
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
    const bool adaptive = multiphase_ && s.max_courant > 0.0;

    RunResult result;
    const bool time_loop = multiphase_ || (compressible_ && !steady);
    while (time_loop ? time_ < s.end_time * (1.0 - 1e-12) : step_ < max_steps) {
        fv::Residuals r;
        if (compressible_) {
            if (steady) {
                r.p = compressible_->steady_iteration();
            } else {
                const Scalar dt = std::min({s.dt, compressible_->stable_dt(), s.end_time - time_});
                r.p = compressible_->step(dt);
                time_ += dt;
            }
            // Density residual relative to the largest of the first five iterations (a uniform
            // initial state can have an almost vanishing first residual).
            if (residual_count_ < 5) {
                residual0_ = std::max({residual0_, r.p, 1e-300});
                ++residual_count_;
            }
            r.p /= residual0_;
            if (compressible_->settings().turbulence) {
                std::tie(r.k, r.omega) = compressible_->turbulence_residuals();
            }
            ++step_;
            result.final_residuals = r;
            log << step_ << ',' << time_ << ",0,0,0," << r.p << ',' << r.k << ',' << r.omega
                << ",0,0\n";
            log.flush();
            if (o.write_interval > 0 && step_ % o.write_interval == 0) {
                std::ostringstream name;
                name << config_.name << '_' << std::setw(6) << std::setfill('0') << step_ << ".vtu";
                write_results(dir / name.str());
            }
            if (o.checkpoint_interval > 0 && step_ % o.checkpoint_interval == 0) {
                write_checkpoint(dir / "checkpoint.bin");
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
            continue;
        }
        if (multiphase_) {
            Scalar dt = s.dt;
            if (adaptive && dt_ > 0.0) {
                const Scalar co = multiphase_->courant(dt_);
                dt = std::min({s.dt, 1.2 * dt_, co > 0.0 ? dt_ * s.max_courant / co : s.dt});
            }
            dt = std::min(dt, s.end_time - time_);
            r = multiphase_->step(dt);
            dt_ = dt;
            time_ += dt;
            ++step_;
            result.final_residuals = r;
            log << step_ << ',' << time_ << ',' << r.u.x() << ',' << r.u.y() << ',' << r.u.z()
                << ',' << r.p << ',' << r.k << ',' << r.omega << ',' << r.continuity << ',' << r.t
                << '\n';
            log.flush();
            if (o.write_interval > 0 && step_ % o.write_interval == 0) {
                std::ostringstream name;
                name << config_.name << '_' << std::setw(6) << std::setfill('0') << step_ << ".vtu";
                write_results(dir / name.str());
            }
            if (o.checkpoint_interval > 0 && step_ % o.checkpoint_interval == 0) {
                write_checkpoint(dir / "checkpoint.bin");
            }
            if (progress && !progress(step_, time_, r)) {
                result.stopped = true;
                break;
            }
            continue;
        }
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
            if (!energy_->settings().nu_law.is_constant()) {
                flow_->set_cell_viscosity(energy_->fluid_viscosity());
            }
            if (radiation_) {
                update_radiation();
            }
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
    if (!steady && !result.stopped &&
        (time_loop ? time_ >= s.end_time * (1.0 - 1e-12) : step_ >= max_steps)) {
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
    if (decomposed_run()) {
        result.result_file.replace_extension(".pvtu");
    }
    write_checkpoint(dir / "checkpoint.bin");
    return result;
}

} // namespace cfd::cases
