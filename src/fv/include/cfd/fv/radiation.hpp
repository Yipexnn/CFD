#pragma once

#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/scalar_transport.hpp>
#include <map>
#include <string>
#include <vector>

namespace cfd::fv {

struct RadiationSettings {
    Scalar absorption = 1.0; // absorption coefficient a [1/m]
    Scalar scattering = 0.0; // scattering coefficient sigma_s [1/m]
    Scalar anisotropy = 0.0; // linear-anisotropic phase function coefficient C (-1..1)
    Scalar emissivity = 1.0; // default wall emissivity
    std::map<std::string, Scalar> patch_emissivity;
    LinearSolverSettings solver{LinearSolverType::BiCgStab, 1e-12, 1e-4, 2000};
};

// P1 (spherical-harmonics) radiation model for a grey participating medium:
//   div(Gamma grad G) - a G + 4 a sigma T^4 = 0,  Gamma = 1 / (3 (a + sigma_s) - C sigma_s)
// with Marshak boundary conditions Gamma dG/dn = eps / (2 (2 - eps)) (4 sigma T_w^4 - G_w) on
// walls, inlets and outlets (the boundary temperature), zero radiative flux on symmetry /
// empty patches. Radiative heat source of the energy equation: a (G - 4 sigma T^4) [W/m^3].
class P1Radiation {
public:
    static constexpr Scalar k_sigma = 5.670374419e-8; // Stefan-Boltzmann [W/(m^2 K^4)]

    P1Radiation(const FlowSolver& flow, RadiationSettings settings);

    // Solves for G with cell temperatures t and boundary-face temperatures t_boundary [K];
    // returns the initial normalised residual.
    Scalar correct(const std::vector<Scalar>& t, const std::vector<Scalar>& t_boundary);

    // Linearised energy source  su + sp T  per unit volume [W/m^3] (sp <= 0).
    void energy_source(const std::vector<Scalar>& t, std::vector<Scalar>& su,
                       std::vector<Scalar>& sp) const;

    [[nodiscard]] const std::vector<Scalar>& incident_radiation() const { return g_; }
    void set_incident_radiation(std::vector<Scalar> g);
    // Net radiative heat flux into the domain per boundary face [W/m^2] (last correct()).
    [[nodiscard]] std::vector<Scalar> wall_heat_flux() const;
    // Net radiative heat flow into the domain through a patch [W] (global).
    [[nodiscard]] Scalar patch_heat_flow(const std::string& patch) const;
    [[nodiscard]] Scalar gamma() const { return gamma_; }
    [[nodiscard]] const RadiationSettings& settings() const { return settings_; }

private:
    const FvGeometry* geo_;
    RadiationSettings settings_;
    Scalar gamma_;
    std::vector<char> marshak_;      // per boundary face
    std::vector<Scalar> wall_coeff_; // eps / (2 (2 - eps)) per boundary face
    std::vector<ScalarBoundary> bc_;
    std::vector<Scalar> g_;
};

} // namespace cfd::fv
