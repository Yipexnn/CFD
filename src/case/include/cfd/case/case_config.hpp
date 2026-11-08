#pragma once

#include <array>
#include <cfd/fv/compressible.hpp>
#include <cfd/fv/energy.hpp>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/k_omega_sst.hpp>
#include <cfd/fv/multiphase.hpp>
#include <cfd/fv/radiation.hpp>
#include <cfd/geometry/geometry.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cfd::cases {

using core::Scalar;
using core::Vec3;

enum class MeshSource { Box, Gmsh, Geometry };
enum class TurbulenceKind { Laminar, KOmegaSst };
enum class SolverMode { Steady, Transient };

struct MeshConfig {
    MeshSource source = MeshSource::Box;
    mesh::BoxSpec box;               // source == Box (patch types from "patches")
    std::filesystem::path file;      // source == Gmsh (relative to the case file)
    geometry::GeometrySpec geometry; // source == Geometry (body files relative to the case)
};

// Parabolic velocity-inlet profile: u = velocity * prod_a 4 s_a (1 - s_a), with s_a the
// normalised coordinate across the patch bounding box along each listed axis.
struct InletProfile {
    std::string axes = "y"; // e.g. "y" (2D channel) or "yz" (rectangular duct)
};

// Force monitor: force on the listed patches, written to forces.csv.
struct ForcesConfig {
    std::vector<std::string> patches;
    Scalar density = 1.0;            // forces = density * kinematic force
    Scalar reference_velocity = 1.0; // coefficients C = F / (0.5 rho U^2 A)
    Scalar reference_area = 1.0;
};

struct InitialConfig {
    Vec3 velocity = Vec3::Zero();
    Scalar k = -1.0;                   // <= 0: derived by the model
    Scalar omega = -1.0;               // <= 0: derived by the model
    std::optional<Scalar> temperature; // default: energy t_ref
    // Multiphase: boxes filled with phase 1 (alpha = 1); everything else alpha = 0.
    std::vector<std::pair<Vec3, Vec3>> alpha_regions;
    // Compressible: initial gas state (pressure [Pa], temperature [K]) and boxes with their
    // own state (e.g. a shock tube).
    Scalar pressure = 1e5;
    Scalar temperature_gas = 300.0;
    struct GasRegion {
        Vec3 min;
        Vec3 max;
        Vec3 velocity;
        Scalar pressure;
        Scalar temperature;
    };
    std::vector<GasRegion> regions;
};

// Solid region for conjugate heat transfer: cells of the mesh zone `name` (Gmsh physical
// volume), or with their centre inside [min, max].
struct SolidConfig {
    std::string name;
    std::optional<Vec3> min;
    std::optional<Vec3> max;
    Scalar conductivity = 1.0; // k [W/(m K)]
    Scalar rho_cp = 1.0;       // [J/(m^3 K)]
};

// Heat-flow monitor (energy equation): heat.csv with the heat flow into the fluid per patch,
// Q = rho_cp * (kinematic heat flow) and Nu = Q_kin L / (alpha dT A).
struct HeatConfig {
    std::vector<std::string> patches;
    Scalar rho_cp = 1.0;
    Scalar reference_length = 1.0;
    Scalar reference_delta_t = 1.0;
    Scalar reference_area = 1.0;
};

struct SolverConfig {
    SolverMode mode = SolverMode::Steady;
    int max_iterations = 1000; // steady
    Scalar tolerance = 1e-6;   // steady convergence criterion (max initial residual)
    Scalar dt = 1e-3;          // transient
    Scalar end_time = 1.0;     // transient
    Scalar max_courant = 0.0;  // transient: > 0 adapts dt (at most `dt`) to this Courant number
    fv::FlowSettings flow;     // nu, body force, relaxation, schemes, correctors
};

struct OutputConfig {
    std::filesystem::path directory = "results"; // relative to the case file
    int write_interval = 0;      // iterations / time steps between VTU writes (0 = final only)
    int log_interval = 10;       // iterations / time steps between console log lines
    int checkpoint_interval = 0; // 0 = final only
};

struct CaseConfig {
    std::string name = "case";
    std::filesystem::path base_directory; // directory of the case file
    MeshConfig mesh;
    TurbulenceKind turbulence = TurbulenceKind::Laminar;
    std::map<std::string, fv::BoundaryCondition> boundaries;
    std::map<std::string, InletProfile> profiles;   // patch -> parabolic inlet profile
    std::optional<fv::EnergySettings> energy;       // energy equation enabled
    std::optional<fv::RadiationSettings> radiation; // P1 radiation (needs energy)
    std::map<std::string, fv::ThermalBoundary> thermal;
    HeatConfig heat;
    std::optional<fv::MultiphaseSettings> multiphase;     // VOF two-phase flow (transient)
    std::optional<fv::CompressibleSettings> compressible; // density-based compressible flow
    std::map<std::string, Scalar> inlet_alpha;            // multiphase: alpha at velocity inlets
    std::vector<SolidConfig> solids;                      // conjugate heat transfer (needs energy)
    ForcesConfig forces;
    InitialConfig initial;
    SolverConfig solver;
    OutputConfig output;
    std::optional<std::filesystem::path> restart; // checkpoint file to start from

    [[nodiscard]] std::filesystem::path resolve(const std::filesystem::path& p) const {
        return p.is_absolute() ? p : base_directory / p;
    }
};

// Parses a JSON case description (see docs/case_format.md).
// Throws std::runtime_error with a descriptive message on invalid input.
[[nodiscard]] CaseConfig parse_case(const std::string& json_text,
                                    const std::filesystem::path& base_directory = {});
[[nodiscard]] CaseConfig load_case(const std::filesystem::path& file);

// Serialises a CaseConfig to pretty-printed JSON (indent 2), the inverse of parse_case.
// base_directory is not written.
[[nodiscard]] std::string case_to_json(const CaseConfig& config);
// Throws std::runtime_error if the file cannot be written.
void save_case(const CaseConfig& config, const std::filesystem::path& file);

} // namespace cfd::cases
