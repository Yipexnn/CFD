#pragma once

#include <array>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/k_omega_sst.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cfd::cases {

using core::Scalar;
using core::Vec3;

enum class MeshSource { Box, Gmsh };
enum class TurbulenceKind { Laminar, KOmegaSst };
enum class SolverMode { Steady, Transient };

struct MeshConfig {
    MeshSource source = MeshSource::Box;
    mesh::BoxSpec box;          // source == Box (patch types from "patches")
    std::filesystem::path file; // source == Gmsh (relative to the case file)
};

struct InitialConfig {
    Vec3 velocity = Vec3::Zero();
    Scalar k = -1.0;     // <= 0: derived by the model
    Scalar omega = -1.0; // <= 0: derived by the model
};

struct SolverConfig {
    SolverMode mode = SolverMode::Steady;
    int max_iterations = 1000; // steady
    Scalar tolerance = 1e-6;   // steady convergence criterion (max initial residual)
    Scalar dt = 1e-3;          // transient
    Scalar end_time = 1.0;     // transient
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

} // namespace cfd::cases
