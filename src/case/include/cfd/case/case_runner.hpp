#pragma once

#include <cfd/case/case_config.hpp>
#include <filesystem>
#include <functional>
#include <memory>

namespace cfd::parallel {
struct LocalMesh;
struct SubMesh;
} // namespace cfd::parallel

namespace cfd::cases {

using core::Label;

struct RunResult {
    int steps = 0;          // SIMPLE iterations (steady) or time steps (transient)
    Scalar time = 0.0;      // simulated time (transient)
    bool converged = false; // steady: tolerance reached; transient: end time reached
    bool stopped = false;   // progress callback requested a stop
    fv::Residuals final_residuals;
    std::filesystem::path result_file; // final VTU
};

// Builds the case mesh: box generator, Gmsh file or (MeshSource::Geometry) Gmsh run in
// <output directory>/mesh. Throws on failure.
[[nodiscard]] mesh::Mesh build_mesh(const CaseConfig& config);

// Contents of a checkpoint file (global cell / face order).
struct CheckpointData {
    int step = 0;
    Scalar time = 0.0;
    std::vector<Vec3> u;
    std::vector<Scalar> p;
    std::vector<Scalar> phi;           // face flux
    std::vector<Scalar> k, omega, nut; // empty without turbulence model
    std::vector<Scalar> boundary_nut;
    std::vector<Scalar> t;         // temperature (empty without energy equation)
    std::vector<Scalar> alpha;     // multiphase volume fraction (empty otherwise)
    std::vector<Scalar> conserved; // compressible: 5 values per cell (empty otherwise)
};
// Reads a checkpoint; n_cells / n_faces / n_boundary_faces must match the mesh.
[[nodiscard]] CheckpointData read_checkpoint_file(const std::filesystem::path& file, Label n_cells,
                                                  Label n_faces, Label n_boundary_faces);

// Called after every step; return false to stop the run (e.g. GUI "stop" button).
using ProgressCallback = std::function<bool(int step, Scalar time, const fv::Residuals& residuals)>;

// Builds mesh, flow solver and turbulence model from a CaseConfig and runs it.
//
// Output (in config.output.directory, created if needed):
//   <name>_<step>.vtu   intermediate results (write_interval > 0)
//   <name>.vtu          final result (cell data U, p and, with SST, k, omega, nut, yPlus)
//   residuals.csv       step, time, Ux, Uy, Uz, p, k, omega, continuity, T
//   heat.csv            step, time, per heat patch Q, Nu (if configured)
//   forces.csv          step, time, per force patch Fx, Fy, Fz, Cx, Cy, Cz (if configured)
//   checkpoint.bin      restart file (fields + face flux + step/time)
class CaseRunner {
public:
    explicit CaseRunner(CaseConfig config);
    // Uses a mesh built beforehand (must match the config's mesh, e.g. from build_mesh).
    CaseRunner(CaseConfig config, mesh::Mesh mesh);
    ~CaseRunner();
    CaseRunner(const CaseRunner&) = delete;
    CaseRunner& operator=(const CaseRunner&) = delete;
    CaseRunner(CaseRunner&&) = delete;
    CaseRunner& operator=(CaseRunner&&) = delete;

    RunResult run(const ProgressCallback& progress = {});

    void write_results(const std::filesystem::path& file) const;
    // Force (density * kinematic) on a patch, summed pressure + viscous.
    [[nodiscard]] Vec3 force(const std::string& patch) const;
    void write_checkpoint(const std::filesystem::path& file) const;
    // Restores fields, step and time; throws if the checkpoint does not match the mesh.
    void read_checkpoint(const std::filesystem::path& file);

    [[nodiscard]] const CaseConfig& config() const { return config_; }
    // Mesh the flow solver works on: the case mesh (its fluid zone for conjugate cases) in
    // serial runs, this process' part (owned + ghost cells) in MPI runs.
    [[nodiscard]] const mesh::Mesh& mesh() const;
    // Mesh of the energy equation / output: the case mesh incl. solid zones (this process'
    // part in MPI runs). Same as mesh() without solids.
    [[nodiscard]] const mesh::Mesh& full_mesh() const;
    // Flow cell values (ncomp per cell) on full_mesh() cells (zero in solids).
    [[nodiscard]] std::vector<Scalar> to_full_cells(const Scalar* flow_values,
                                                    std::size_t ncomp) const;
    // The undecomposed case mesh (same as mesh() in serial runs).
    [[nodiscard]] const mesh::Mesh& global_mesh() const { return *global_mesh_; }
    // Decomposition of an MPI run (nullptr in serial runs).
    [[nodiscard]] const parallel::LocalMesh* local_mesh() const { return local_.get(); }
    // Single-phase flow solver (not available for multiphase cases).
    [[nodiscard]] const fv::FlowSolver& flow() const { return *flow_; }
    [[nodiscard]] const fv::MultiphaseSolver* multiphase() const { return multiphase_.get(); }
    [[nodiscard]] const fv::CompressibleSolver* compressible() const { return compressible_.get(); }
    // Flow fields of either solver on mesh() cells (pressure: kinematic for single-phase,
    // static [Pa] for multiphase).
    [[nodiscard]] std::vector<Vec3> velocity() const;
    [[nodiscard]] std::vector<Scalar> pressure() const;
    // Active k-omega SST model (single-phase or multiphase mixture; nullptr if laminar).
    [[nodiscard]] const fv::KOmegaSst* sst() const;
    [[nodiscard]] const fv::EnergyEquation* energy() const { return energy_.get(); }
    // Kinematic heat flow into the fluid through a patch times heat.rho_cp.
    [[nodiscard]] Scalar heat_flow(const std::string& patch) const;
    [[nodiscard]] int step() const { return step_; }
    [[nodiscard]] Scalar time() const { return time_; }
    [[nodiscard]] std::filesystem::path output_directory() const;

private:
    void setup();
    [[nodiscard]] std::vector<Scalar> to_full_faces(const std::vector<Scalar>& flow_values,
                                                    bool oriented, bool boundary_only) const;
    [[nodiscard]] std::vector<Scalar> from_full_cells(const std::vector<Scalar>& full,
                                                      std::size_t ncomp) const;
    [[nodiscard]] std::vector<Scalar> from_full_faces(const std::vector<Scalar>& full,
                                                      bool oriented, bool boundary_only) const;

    CaseConfig config_;
    std::unique_ptr<mesh::Mesh> global_mesh_;
    std::unique_ptr<parallel::LocalMesh> local_;
    std::unique_ptr<parallel::SubMesh> fluid_;      // conjugate: fluid zone of full_mesh()
    std::unique_ptr<fv::FvGeometry> full_geometry_; // conjugate: energy geometry
    std::unique_ptr<fv::ConjugateMesh> conjugate_;
    std::unique_ptr<fv::FlowSolver> flow_;
    std::unique_ptr<fv::KOmegaSst> sst_;
    [[nodiscard]] fv::KOmegaSst* mutable_sst();
    std::unique_ptr<fv::EnergyEquation> energy_;
    std::unique_ptr<fv::MultiphaseSolver> multiphase_;
    std::unique_ptr<fv::CompressibleSolver> compressible_;
    Scalar residual0_ = 0.0; // compressible: reference density residual (steady convergence)
    int residual_count_ = 0;
    Scalar dt_ = 0.0; // last time step (adaptive time stepping)
    int step_ = 0;
    Scalar time_ = 0.0;
};

} // namespace cfd::cases
