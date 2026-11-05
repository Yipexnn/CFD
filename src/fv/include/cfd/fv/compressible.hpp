#pragma once

#include <cfd/fv/flow_solver.hpp>
#include <map>
#include <string>
#include <vector>

namespace cfd::fv {

struct CompressibleSettings {
    Scalar gamma = 1.4;    // ratio of specific heats
    Scalar r_gas = 287.0;  // specific gas constant [J/(kg K)]
    Scalar mu = 0.0;       // dynamic viscosity [Pa s] (0: inviscid Euler equations)
    Scalar prandtl = 0.72; // heat conduction k = mu c_p / Pr
    Scalar cfl = 0.5;
    bool second_order = true; // MUSCL reconstruction with Barth-Jespersen limiter
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
    // One local-time-stepping iteration (steady problems); returns the density residual.
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

private:
    [[nodiscard]] Conserved to_conserved(const GasState& s) const;
    [[nodiscard]] GasState to_primitive(const Conserved& q) const;
    [[nodiscard]] const BoundaryCondition& bc(std::size_t boundary_face) const;
    [[nodiscard]] GasState ghost(std::size_t face, const GasState& inside) const;
    // Residual R(q) = -sum of face fluxes per cell (per unit time, not divided by volume).
    [[nodiscard]] std::vector<Conserved> residual(const std::vector<Conserved>& q) const;
    [[nodiscard]] std::vector<Scalar> local_dt(const std::vector<Conserved>& q) const;

    FvGeometry geo_;
    CompressibleSettings settings_;
    std::vector<BoundaryCondition> patch_bcs_;
    std::vector<Conserved> q_;
};

} // namespace cfd::fv
