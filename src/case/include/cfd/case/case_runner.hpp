#pragma once

#include <cfd/case/case_config.hpp>
#include <filesystem>
#include <functional>
#include <memory>

namespace cfd::cases {

struct RunResult {
    int steps = 0;          // SIMPLE iterations (steady) or time steps (transient)
    Scalar time = 0.0;      // simulated time (transient)
    bool converged = false; // steady: tolerance reached; transient: end time reached
    bool stopped = false;   // progress callback requested a stop
    fv::Residuals final_residuals;
    std::filesystem::path result_file; // final VTU
};

// Called after every step; return false to stop the run (e.g. GUI "stop" button).
using ProgressCallback = std::function<bool(int step, Scalar time, const fv::Residuals& residuals)>;

// Builds mesh, flow solver and turbulence model from a CaseConfig and runs it.
//
// Output (in config.output.directory, created if needed):
//   <name>_<step>.vtu   intermediate results (write_interval > 0)
//   <name>.vtu          final result (cell data U, p and, with SST, k, omega, nut, yPlus)
//   residuals.csv       step, time, Ux, Uy, Uz, p, k, omega, continuity
//   checkpoint.bin      restart file (fields + face flux + step/time)
class CaseRunner {
public:
    explicit CaseRunner(CaseConfig config);
    ~CaseRunner();
    CaseRunner(const CaseRunner&) = delete;
    CaseRunner& operator=(const CaseRunner&) = delete;
    CaseRunner(CaseRunner&&) = delete;
    CaseRunner& operator=(CaseRunner&&) = delete;

    RunResult run(const ProgressCallback& progress = {});

    void write_results(const std::filesystem::path& file) const;
    void write_checkpoint(const std::filesystem::path& file) const;
    // Restores fields, step and time; throws if the checkpoint does not match the mesh.
    void read_checkpoint(const std::filesystem::path& file);

    [[nodiscard]] const CaseConfig& config() const { return config_; }
    [[nodiscard]] const mesh::Mesh& mesh() const { return *mesh_; }
    [[nodiscard]] const fv::FlowSolver& flow() const { return *flow_; }
    [[nodiscard]] const fv::KOmegaSst* sst() const { return sst_.get(); }
    [[nodiscard]] int step() const { return step_; }
    [[nodiscard]] Scalar time() const { return time_; }
    [[nodiscard]] std::filesystem::path output_directory() const;

private:
    CaseConfig config_;
    std::unique_ptr<mesh::Mesh> mesh_;
    std::unique_ptr<fv::FlowSolver> flow_;
    std::unique_ptr<fv::KOmegaSst> sst_;
    int step_ = 0;
    Scalar time_ = 0.0;
};

} // namespace cfd::cases
