#pragma once

#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/scalar_transport.hpp>
#include <map>
#include <string>
#include <vector>

namespace cfd::fv {

// Temperature boundary condition of a patch. Heat flux is kinematic: q / (rho c_p) [K m/s],
// positive into the fluid.
struct ThermalBoundary {
    enum class Kind { Adiabatic, FixedTemperature, HeatFlux };
    Kind kind = Kind::Adiabatic;
    Scalar value = 0.0;
};

struct EnergySettings {
    Scalar alpha = 1e-3 / 0.71;  // thermal diffusivity [m^2/s] (nu / Pr)
    Scalar prandtl_t = 0.85;     // turbulent Prandtl number
    Scalar beta = 0.0;           // thermal expansion coefficient [1/K]
    Scalar t_ref = 0.0;          // reference temperature [K]
    Vec3 gravity = Vec3::Zero(); // [m/s^2]
    Scalar relaxation = 0.9;     // steady under-relaxation
    LinearSolverSettings solver{LinearSolverType::BiCgStab, 1e-12, 0.01, 1000};
};

// Temperature transport ddt(T) + div(phi T) = div((alpha + nu_t / Pr_t) grad T) with
// Boussinesq buoyancy -beta (T - T_ref) g. Patches without a condition are adiabatic, except
// velocity inlets (fixed T_ref). Works on decomposed meshes like FlowSolver.
class EnergyEquation {
public:
    EnergyEquation(const FlowSolver& flow, EnergySettings settings,
                   const std::map<std::string, ThermalBoundary>& conditions);

    // Solves for T with the current face flux (dt > 0: implicit Euler step). Returns the
    // initial normalised residual.
    Scalar correct(const FlowSolver& flow, Scalar dt);

    // Buoyancy acceleration per cell, for FlowSolver::set_cell_acceleration.
    [[nodiscard]] std::vector<Vec3> buoyancy() const;
    [[nodiscard]] bool buoyant() const {
        return settings_.beta != 0.0 && settings_.gravity.norm() > 0.0;
    }

    // Heat flow into the fluid through a patch [K m^3/s] (kinematic, summed over processes).
    [[nodiscard]] Scalar patch_heat_flow(const FlowSolver& flow, const std::string& patch) const;

    [[nodiscard]] const std::vector<Scalar>& temperature() const { return t_; }
    void set_temperature(std::vector<Scalar> t);
    void set_uniform_temperature(Scalar t);
    [[nodiscard]] const EnergySettings& settings() const { return settings_; }
    // Boundary-face temperatures (indexed by f - n_internal_faces).
    [[nodiscard]] std::vector<Scalar> boundary_temperature() const;

private:
    [[nodiscard]] std::vector<Scalar> face_diffusivity(const FlowSolver& flow) const;
    void update_boundary(const FlowSolver& flow);

    const FvGeometry* geo_;
    EnergySettings settings_;
    std::vector<ThermalBoundary> face_bc_; // per boundary face
    std::vector<ScalarBoundary> bc_;       // per boundary face, for solve_scalar
    std::vector<Scalar> t_;
    std::vector<Scalar> t_old_;
};

} // namespace cfd::fv
