#pragma once

#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/properties.hpp>
#include <cfd/fv/scalar_transport.hpp>
#include <map>
#include <string>
#include <vector>

namespace cfd::fv {

// Slope limiter of the MUSCL reconstruction: Barth-Jespersen (strict, best for transient
// shocks) or Venkatakrishnan (smooth; steady problems converge to machine accuracy).

struct CompressibleSettings {
    Scalar gamma = 1.4;   // ratio of specific heats
    Scalar r_gas = 287.0; // specific gas constant [J/(kg K)]
    Scalar mu = 0.0;      // dynamic viscosity [Pa s] (0: inviscid Euler equations)
    // Variable viscosity mu(T) (e.g. Sutherland; mu above stays the reference/constant value
    // that switches viscous terms on). Heat conduction follows k = c_p mu(T) / Pr.
    PropertyLaw mu_law{};
    Scalar prandtl = 0.72; // heat conduction k = mu c_p / Pr
    Scalar cfl = 0.5;
    bool second_order = true; // MUSCL reconstruction
    Limiter limiter = Limiter::BarthJespersen;
    Scalar venkat_k = 0.1; // Venkatakrishnan threshold relative to the global variable range
    // Steady problems: implicit LU-SGS iterations (local time steps, CFL typically 5-100)
    // instead of explicit SSP-RK3 with local time steps.
    bool implicit = false;
    // Implicit: > 0 solves each linear system with Jacobian-free FGMRES (this Krylov dimension,
    // LU-SGS preconditioned) instead of a single LU-SGS sweep; needed for stretched wall grids.
    int krylov_dim = 0;
    Scalar krylov_tolerance = 0.1; // relative linear residual
    // k-omega SST RANS model (low-Re wall treatment: first cell y+ ~ 1 recommended).
    bool turbulence = false;
    Scalar prandtl_t = 0.9; // turbulent Prandtl number
};

// Primitive state of a cell.
struct GasState {
    Scalar rho = 1.0;
    Vec3 u = Vec3::Zero();
    Scalar p = 1.0;
};

// Density-based compressible flow solver (ideal gas): conservative variables
// (rho, rho u, rho E), HLLC Riemann fluxes, optional laminar viscous fluxes, explicit SSP-RK3
// time integration (global time step) or local time stepping for steady problems.
//
// Boundary conditions (FlowBoundary): VelocityInlet = fixed state (velocity, pressure,
// temperature; supersonic inlet); PressureOutlet = static pressure if subsonic, extrapolation
// if supersonic (pressure <= 0: always extrapolate); Wall = no-slip adiabatic (slip if mu = 0);
// Symmetry = slip; Empty.
//
// Turbulence (settings.turbulence): k-omega SST (Menter 2003) in density-weighted form,
// solved segregated after each mean-flow update (implicit, bounded); mu_t enters the viscous
// stresses and the heat flux (Pr_t). Inlet k / omega from turbulence_intensity and
// viscosity_ratio; walls: k zero gradient, omega in wall cells from the viscous / log-layer
// blend, mu_t = 0 on the wall faces.
class CompressibleSolver {
public:
    CompressibleSolver(const Mesh& mesh, std::map<std::string, BoundaryCondition> conditions,
                       CompressibleSettings settings, const parallel::Halo* halo = nullptr);

    void set_uniform(const GasState& state);
    void set_state(const std::vector<GasState>& states);

    // Stable global time step for the CFL number (collective).
    [[nodiscard]] Scalar stable_dt() const;
    // One SSP-RK3 step with the given time step; returns the density residual
    // (RMS of d(rho)/dt, global).
    Scalar step(Scalar dt);
    // One local-time-stepping iteration (steady problems; explicit RK3 or implicit LU-SGS);
    // returns the density residual.
    Scalar steady_iteration();

    [[nodiscard]] GasState state(std::size_t cell) const;
    [[nodiscard]] std::vector<Vec3> velocity() const;
    [[nodiscard]] std::vector<Scalar> pressure() const;
    [[nodiscard]] std::vector<Scalar> density() const;
    [[nodiscard]] std::vector<Scalar> temperature() const;
    [[nodiscard]] std::vector<Scalar> mach() const;
    [[nodiscard]] const CompressibleSettings& settings() const { return settings_; }
    [[nodiscard]] const FvGeometry& geometry() const { return geo_; }
    // Conservative variables per cell: rho, rho u (3), rho E.
    using Conserved = Eigen::Matrix<Scalar, 5, 1>;
    [[nodiscard]] const std::vector<Conserved>& conserved() const { return q_; }
    void set_conserved(std::vector<Conserved> q);

    // Turbulence fields (empty unless settings.turbulence).
    [[nodiscard]] const std::vector<Scalar>& k() const { return k_; }
    [[nodiscard]] const std::vector<Scalar>& omega() const { return omega_; }
    [[nodiscard]] const std::vector<Scalar>& eddy_viscosity() const { return mut_; } // mu_t
    [[nodiscard]] const std::vector<Scalar>& wall_distance() const { return y_; }
    // Restart: sets k and omega (mu_t = rho k / omega until the next update).
    void set_turbulence(std::vector<Scalar> k, std::vector<Scalar> omega);
    // Normalised initial residuals of the k and omega equations from the last update.
    [[nodiscard]] std::pair<Scalar, Scalar> turbulence_residuals() const { return turb_res_; }

private:
    [[nodiscard]] Conserved to_conserved(const GasState& s) const;
    [[nodiscard]] GasState to_primitive(const Conserved& q) const;
    [[nodiscard]] const BoundaryCondition& bc(std::size_t boundary_face) const;
    [[nodiscard]] GasState ghost(std::size_t face, const GasState& inside) const;
    // Residual R(q) = -sum of face fluxes per cell (per unit time, not divided by volume).
    [[nodiscard]] std::vector<Conserved> residual(const std::vector<Conserved>& q) const;
    [[nodiscard]] std::vector<Scalar> local_dt(const std::vector<Conserved>& q) const;
    Scalar lusgs_iteration();
    // Effective viscosity mu + mu_t on a face (laminar only on walls).
    [[nodiscard]] Scalar face_mut(std::size_t face) const;
    // Laminar viscosity of a cell / face (mu_law of the cell temperatures).
    [[nodiscard]] Scalar cell_mu(std::size_t cell) const;
    [[nodiscard]] Scalar face_mu(std::size_t face) const;
    void init_turbulence();
    void correct_turbulence(Scalar dt);

    FvGeometry geo_;
    CompressibleSettings settings_;
    std::vector<BoundaryCondition> patch_bcs_;
    std::vector<Conserved> q_;

    // Turbulence.
    std::vector<Scalar> k_;
    std::vector<Scalar> omega_;
    std::vector<Scalar> mut_;
    std::vector<Scalar> y_;
    std::vector<Scalar> k_old_;
    std::vector<Scalar> omega_old_;
    std::vector<Scalar> rho_old_;
    std::vector<char> is_wall_face_;
    std::vector<ScalarBoundary> k_bc_;
    std::vector<ScalarBoundary> omega_bc_;
    std::pair<Scalar, Scalar> turb_res_{0.0, 0.0};
};

} // namespace cfd::fv
