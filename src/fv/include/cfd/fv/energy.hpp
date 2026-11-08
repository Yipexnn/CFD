#pragma once

#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/properties.hpp>
#include <cfd/fv/scalar_transport.hpp>
#include <cfd/parallel/decomposition.hpp>
#include <map>
#include <string>
#include <vector>

namespace cfd::fv {

// Temperature boundary condition of a patch. Heat flux is kinematic: q / (rho c_p) [K m/s],
// positive into the domain.
struct ThermalBoundary {
    enum class Kind { Adiabatic, FixedTemperature, HeatFlux };
    Kind kind = Kind::Adiabatic;
    Scalar value = 0.0;
};

struct EnergySettings {
    Scalar alpha = 1e-3 / 0.71; // thermal diffusivity [m^2/s] (nu / Pr)
    // Variable properties (non-constant laws override alpha / the flow's nu): fluid thermal
    // diffusivity alpha(T), and nu(T) which the case runner applies to the flow solver.
    PropertyLaw alpha_law{};
    PropertyLaw nu_law{};
    Scalar prandtl_t = 0.85;     // turbulent Prandtl number
    Scalar beta = 0.0;           // thermal expansion coefficient [1/K]
    Scalar t_ref = 0.0;          // reference temperature [K]
    Vec3 gravity = Vec3::Zero(); // [m/s^2]
    Scalar rho_cp = 1.0;         // fluid volumetric heat capacity (solid properties relative)
    Scalar relaxation = 0.9;     // steady under-relaxation
    LinearSolverSettings solver{LinearSolverType::BiCgStab, 1e-12, 0.01, 1000};
};

// Solid region of a conjugate heat transfer case (a cell zone of the full mesh).
struct SolidRegion {
    std::string zone;
    Scalar conductivity = 1.0; // k [W/(m K)]
    Scalar rho_cp = 1.0;       // volumetric heat capacity [J/(m^3 K)]
};

// Conjugate setup: the energy equation lives on the full mesh (fluid + solid zones), the
// flow on its fluid zone (`fluid`, extracted with parallel::extract_zone).
struct ConjugateMesh {
    const FvGeometry* geometry = nullptr;     // full mesh
    const parallel::SubMesh* fluid = nullptr; // fluid zone of the full mesh
    std::vector<int> zone_of_cell;            // full mesh
    std::vector<std::string> zone_names;
    std::vector<SolidRegion> solids;
};

// Temperature transport (kinematic, fluid units):
//   C ddt(T) + div(phi T) = div(K grad T)
// fluid: C = 1, K = alpha + nu_t / Pr_t; solid: C = rho_cp_s / rho_cp, K = k_s / rho_cp
// (harmonic face interpolation across fluid/solid faces), with Boussinesq buoyancy
// -beta (T - T_ref) g in the fluid. Patches without a condition are adiabatic, except
// velocity inlets (fixed T_ref). Works on decomposed meshes like FlowSolver.
class EnergyEquation {
public:
    // Fluid only (the energy equation on the flow mesh).
    EnergyEquation(const FlowSolver& flow, EnergySettings settings,
                   const std::map<std::string, ThermalBoundary>& conditions);
    // Conjugate heat transfer (the ConjugateMesh must outlive the equation).
    EnergyEquation(const FlowSolver& flow, const ConjugateMesh& conjugate, EnergySettings settings,
                   const std::map<std::string, ThermalBoundary>& conditions);

    // Solves for T with the current face flux (dt > 0: implicit Euler step). Returns the
    // initial normalised residual.
    Scalar correct(const FlowSolver& flow, Scalar dt);

    // Buoyancy acceleration per flow cell, for FlowSolver::set_cell_acceleration.
    [[nodiscard]] std::vector<Vec3> buoyancy() const;
    [[nodiscard]] bool buoyant() const {
        return settings_.beta != 0.0 && settings_.gravity.norm() > 0.0;
    }

    // Heat flow into the domain through a patch of the energy mesh, or (conjugate) from a
    // solid zone into the fluid [K m^3/s, kinematic]; summed over processes.
    [[nodiscard]] Scalar patch_heat_flow(const FlowSolver& flow, const std::string& patch) const;

    // Temperature on the energy mesh (the full mesh for conjugate cases).
    [[nodiscard]] const std::vector<Scalar>& temperature() const { return t_; }
    // Temperature on the flow (fluid) cells.
    [[nodiscard]] std::vector<Scalar> fluid_temperature() const;
    // nu(T) on the fluid cells (settings().nu_law); empty if the law is constant.
    [[nodiscard]] std::vector<Scalar> fluid_viscosity() const;
    void set_temperature(std::vector<Scalar> t);
    void set_uniform_temperature(Scalar t);
    [[nodiscard]] const EnergySettings& settings() const { return settings_; }
    [[nodiscard]] const FvGeometry& geometry() const { return *geo_; }
    // Boundary-face temperatures (indexed by f - n_internal_faces of the energy mesh).
    [[nodiscard]] std::vector<Scalar> boundary_temperature() const;

private:
    void init(const FlowSolver& flow, const std::map<std::string, ThermalBoundary>& conditions);
    [[nodiscard]] std::vector<Scalar> face_flux(const FlowSolver& flow) const;
    [[nodiscard]] std::vector<Scalar> face_diffusivity(const FlowSolver& flow) const;
    void update_boundary(const FlowSolver& flow);

    const FvGeometry* geo_;
    const ConjugateMesh* conjugate_ = nullptr;
    EnergySettings settings_;
    std::vector<char> solid_;              // per energy cell
    std::vector<Scalar> capacity_;         // per energy cell (empty: 1)
    std::vector<Scalar> solid_k_;          // per energy cell (solid cells)
    std::vector<Label> flow_boundary_;     // energy boundary face -> flow boundary face (-1)
    std::vector<ThermalBoundary> face_bc_; // per boundary face
    std::vector<ScalarBoundary> bc_;       // per boundary face, for solve_scalar
    std::vector<Scalar> t_;
    std::vector<Scalar> t_old_;
};

} // namespace cfd::fv
