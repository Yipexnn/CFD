#pragma once

#include <cfd/fv/fv_geometry.hpp>
#include <cfd/fv/ldu_matrix.hpp>
#include <map>
#include <string>
#include <vector>

namespace cfd::fv {

enum class FlowBoundary { VelocityInlet, Wall, PressureOutlet, Symmetry };

struct BoundaryCondition {
    FlowBoundary kind = FlowBoundary::Wall;
    Vec3 velocity = Vec3::Zero(); // inlet velocity or wall velocity
    Scalar pressure = 0.0;        // outlet (kinematic) pressure
};

enum class ConvectionScheme { Upwind, LinearUpwind };

struct FlowSettings {
    Scalar nu = 1e-3; // kinematic viscosity [m^2/s]
    ConvectionScheme convection = ConvectionScheme::LinearUpwind;
    Scalar alpha_u = 0.7; // SIMPLE velocity under-relaxation
    Scalar alpha_p = 0.3; // SIMPLE pressure under-relaxation
    int n_non_orth_correctors = 0;
    int n_piso_correctors = 2;
    LinearSolverSettings u_solver{LinearSolverType::BiCgStab, 1e-10, 0.1, 1000};
    LinearSolverSettings p_solver{LinearSolverType::ConjugateGradient, 1e-10, 0.01, 2000};
};

struct Residuals {
    Vec3 u = Vec3::Zero();   // initial normalised residual per velocity component
    Scalar p = 0.0;          // initial normalised pressure residual
    Scalar continuity = 0.0; // volume-weighted |div(phi)| / total volume

    [[nodiscard]] Scalar max() const { return std::max(u.maxCoeff(), p); }
};

// Incompressible laminar Navier-Stokes solver (collocated FV, SIMPLE / PISO).
// Pressure is kinematic (p / rho).
class FlowSolver {
public:
    // Boundary conditions are looked up by patch name. Patches without an entry use a
    // default derived from the mesh patch type (wall -> no-slip wall, symmetry ->
    // symmetry, outlet -> pressure outlet p = 0). Throws if a patch has no usable default.
    FlowSolver(const Mesh& mesh, std::map<std::string, BoundaryCondition> conditions,
               FlowSettings settings);

    // One steady SIMPLE iteration.
    Residuals simple_iteration();

    // SIMPLE until max residual < tolerance or max_iterations. Returns the number of
    // iterations performed; residual history is available via history().
    int solve_steady(Scalar tolerance, int max_iterations);

    // One transient PISO time step (implicit Euler).
    Residuals piso_step(Scalar dt);

    void set_uniform_velocity(const Vec3& u);

    // Replaces the cell fields (e.g. restart or initial condition); fluxes are recomputed.
    void set_state(std::vector<Vec3> u, std::vector<Scalar> p);

    [[nodiscard]] const std::vector<Vec3>& velocity() const { return u_; }
    [[nodiscard]] const std::vector<Scalar>& pressure() const { return p_; }
    [[nodiscard]] const std::vector<Scalar>& face_flux() const { return phi_; }
    [[nodiscard]] const std::vector<Residuals>& history() const { return history_; }
    [[nodiscard]] const FvGeometry& geometry() const { return geo_; }
    [[nodiscard]] const FlowSettings& settings() const { return settings_; }

    // Boundary-face values (indexed by f - n_internal_faces).
    [[nodiscard]] Vec3 boundary_velocity(Label face) const;
    [[nodiscard]] Scalar boundary_pressure(Label face) const;

    // Gauss-linear gradients of the current fields.
    [[nodiscard]] std::vector<Eigen::Matrix3d> velocity_gradient() const;
    [[nodiscard]] std::vector<Vec3> pressure_gradient() const;

    // Volume-weighted mean |div(phi)| / total volume.
    [[nodiscard]] Scalar continuity_error() const;

private:
    struct MomentumSystem {
        LduMatrix base;           // shared coefficients (diag, lower, upper)
        std::vector<Vec3> diag;   // per-component diagonal
        std::vector<Vec3> source; // per-component source (without grad p)
    };

    void assemble_momentum(MomentumSystem& sys, Scalar dt, bool relax) const;
    Vec3 solve_momentum(MomentumSystem& sys);
    Scalar pressure_correction(const MomentumSystem& sys, bool relax_p);
    void compute_initial_flux();

    const BoundaryCondition& bc(Label boundary_face_index) const;

    FvGeometry geo_;
    FlowSettings settings_;
    std::vector<BoundaryCondition> patch_bcs_;
    bool has_fixed_pressure_ = false;

    std::vector<Vec3> u_;
    std::vector<Vec3> u_old_;
    std::vector<Scalar> p_;
    std::vector<Scalar> phi_;
    std::vector<Residuals> history_;
};

} // namespace cfd::fv
