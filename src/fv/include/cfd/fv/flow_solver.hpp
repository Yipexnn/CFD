#pragma once

#include <algorithm>
#include <cfd/fv/fv_geometry.hpp>
#include <cfd/fv/ldu_matrix.hpp>
#include <cfd/fv/turbulence_model.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cfd::fv {

// Empty: zero flux and zero gradient; only valid for directions with one cell layer
// (quasi-1D/2D cases, e.g. a fully developed channel driven by a body force).
enum class FlowBoundary { VelocityInlet, Wall, PressureOutlet, Symmetry, Empty };

struct BoundaryCondition {
    FlowBoundary kind = FlowBoundary::Wall;
    Vec3 velocity = Vec3::Zero(); // inlet velocity or wall velocity
    Scalar pressure = 0.0;        // outlet (kinematic) pressure
    // Inlet turbulence: k = 1.5 (I |U|)^2, nu_t / nu = viscosity_ratio.
    Scalar turbulence_intensity = 0.05;
    Scalar viscosity_ratio = 10.0;
    Scalar temperature = 0.0; // compressible: inlet static temperature [K]
    // Multiphase walls: static contact angle [deg] between wall and interface, measured
    // through phase 1 (alpha = 1).
    Scalar contact_angle = 90.0;
    // Compressible velocity inlets: characteristic far field (Riemann invariants; free stream =
    // velocity / pressure / temperature) instead of a fixed state; works for in- and outflow.
    bool far_field = false;
};

// LimitedLinearUpwind: linear upwind with a gradient limiter (bounded, robust at high cell
// Peclet numbers). Lust: 75 % linear interpolation + 25 % linear upwind (low dissipation, for
// vortex shedding / LES; momentum only, scalars use linear upwind).
enum class ConvectionScheme { Upwind, LinearUpwind, LimitedLinearUpwind, Lust };

// Transient schemes: implicit Euler (1st order) or backward differencing (BDF2, 2nd order;
// the first step after a start / restart uses Euler).
enum class TimeScheme { Euler, Backward };

struct FlowSettings {
    Scalar nu = 1e-3;               // kinematic viscosity [m^2/s]
    Vec3 body_force = Vec3::Zero(); // acceleration [m/s^2], e.g. driving pressure gradient
    ConvectionScheme convection = ConvectionScheme::LinearUpwind;
    GradientScheme gradient = GradientScheme::Gauss;
    // LimitedLinearUpwind: limiter per velocity component; Venkatakrishnan threshold
    // eps = venkat_k * (global range of the component).
    Limiter limiter = Limiter::BarthJespersen;
    Scalar venkat_k = 0.05;
    Scalar alpha_u = 0.7; // SIMPLE velocity under-relaxation
    Scalar alpha_p = 0.3; // SIMPLE pressure under-relaxation
    int n_non_orth_correctors = 0;
    int n_piso_correctors = 2;
    TimeScheme time_scheme = TimeScheme::Euler;
    LinearSolverSettings u_solver{LinearSolverType::BiCgStab, 1e-10, 0.1, 1000};
    LinearSolverSettings p_solver{LinearSolverType::AmgCg, 1e-10, 0.01, 500};
};

// Kinematic force (per unit density) exerted by the fluid on a boundary patch.
struct PatchForce {
    Vec3 pressure = Vec3::Zero();
    Vec3 viscous = Vec3::Zero();
    [[nodiscard]] Vec3 total() const { return pressure + viscous; }
};

struct Residuals {
    Vec3 u = Vec3::Zero();   // initial normalised residual per velocity component
    Scalar p = 0.0;          // initial normalised pressure residual
    Scalar continuity = 0.0; // volume-weighted |div(phi)| / total volume
    Scalar k = 0.0;          // turbulence equations (0 for laminar)
    Scalar omega = 0.0;
    Scalar t = 0.0; // energy equation (0 without)

    [[nodiscard]] Scalar max() const { return std::max({u.maxCoeff(), p, k, omega, t}); }
};

// Incompressible laminar Navier-Stokes solver (collocated FV, SIMPLE / PISO).
// Pressure is kinematic (p / rho).
class FlowSolver {
public:
    // Boundary conditions are looked up by patch name. Patches without an entry use a
    // default derived from the mesh patch type (wall -> no-slip wall, symmetry ->
    // symmetry, outlet -> pressure outlet p = 0). Throws if a patch has no usable default.
    //
    // halo (optional, must outlive the solver): mesh is one part of a decomposed mesh; the
    // halo patch is handled internally and all norms, residuals and forces are global.
    FlowSolver(const Mesh& mesh, std::map<std::string, BoundaryCondition> conditions,
               FlowSettings settings, const parallel::Halo* halo = nullptr);

    // One steady SIMPLE iteration.
    Residuals simple_iteration();

    // SIMPLE until max residual < tolerance or max_iterations. Returns the number of
    // iterations performed; residual history is available via history().
    int solve_steady(Scalar tolerance, int max_iterations);

    // One transient PISO time step (implicit Euler).
    Residuals piso_step(Scalar dt);

    void set_uniform_velocity(const Vec3& u);

    // Additional acceleration per cell [m/s^2] (e.g. Boussinesq buoyancy), added to the
    // uniform body force. Empty clears it; otherwise the size must be the cell count.
    void set_cell_acceleration(std::vector<Vec3> acceleration);

    // Per-face velocity for a velocity-inlet or wall patch (e.g. a parabolic inlet
    // profile); values are ordered like the patch faces. Throws on size mismatch.
    void set_boundary_velocity_profile(const std::string& patch, std::vector<Vec3> velocities);

    // Couples an eddy-viscosity model (not owned; nullptr = laminar). The model is
    // corrected after every SIMPLE iteration / PISO step.
    void set_turbulence_model(TurbulenceModel* model) { turbulence_ = model; }
    [[nodiscard]] const TurbulenceModel* turbulence_model() const { return turbulence_; }

    // Boundary condition of a boundary face (indexed by f - n_internal_faces).
    // Variable laminar viscosity: per-cell nu (e.g. nu(T)); empty restores settings().nu.
    void set_cell_viscosity(std::vector<Scalar> nu);
    [[nodiscard]] const std::vector<Scalar>& cell_viscosity() const { return cell_nu_; }

    [[nodiscard]] const BoundaryCondition& boundary_condition(Label boundary_face_index) const {
        return bc(boundary_face_index);
    }

    // Replaces the cell fields (e.g. restart or initial condition); fluxes are recomputed.
    void set_state(std::vector<Vec3> u, std::vector<Scalar> p);
    // Restart: also restores the conservative face flux (size n_faces).
    void set_state(std::vector<Vec3> u, std::vector<Scalar> p, std::vector<Scalar> phi);

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

    // Force on a patch: pressure (linearly extrapolated to the face) plus viscous wall
    // shear nu_eff * dU/dn. Multiply by the density for the physical force. Throws if the
    // patch does not exist.
    [[nodiscard]] PatchForce patch_force(const std::string& patch) const;

    // Volume-weighted mean |div(phi)| / total volume.
    [[nodiscard]] Scalar continuity_error() const;

private:
    // Explicit FV parts (gradients, face corrections) on the accelerator kernels: GPU solvers
    // selected and a serial mesh.
    [[nodiscard]] bool device_fv() const;
    // Venkatakrishnan eps^2 per velocity component (global range, all processes).
    [[nodiscard]] Vec3 venkat_eps2() const;
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
    // Throws if the prescribed boundary flux is not balanced and no pressure outlet exists.
    void check_mass_balance() const;

    FvGeometry geo_;
    FlowSettings settings_;
    std::vector<Vec3> cell_acceleration_;
    std::vector<BoundaryCondition> patch_bcs_;
    std::vector<Vec3> fixed_velocity_; // per boundary face: inlet / wall velocity
    bool has_fixed_pressure_ = false;
    TurbulenceModel* turbulence_ = nullptr;
    std::shared_ptr<AmgCache> p_amg_ = std::make_shared<AmgCache>(); // pressure AMG reuse

    std::vector<Vec3> u_;
    std::vector<Scalar> cell_nu_; // optional variable laminar viscosity
    std::vector<Vec3> u_old_;
    std::vector<Vec3> u_old_old_; // BDF2: velocity two steps back (empty: not available)
    Scalar dt_old_ = 0.0;
    std::vector<Scalar> p_;
    std::vector<Scalar> phi_;
    std::vector<Residuals> history_;
};

} // namespace cfd::fv
