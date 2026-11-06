#pragma once

#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/k_omega_sst.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cfd::fv {

// Two immiscible incompressible fluids (volume of fluid). alpha = volume fraction of phase 1.
struct MultiphaseSettings {
    Scalar rho1 = 1000.0; // phase 1 (alpha = 1), e.g. water [kg/m^3]
    Scalar nu1 = 1e-6;    // kinematic viscosity [m^2/s]
    Scalar rho2 = 1.0;    // phase 2 (alpha = 0), e.g. air
    Scalar nu2 = 1.48e-5;
    Scalar sigma = 0.0;              // surface tension [N/m]
    Vec3 gravity = Vec3::Zero();     // [m/s^2]
    Scalar compression = 1.0;        // interface compression coefficient c_alpha
    Scalar max_alpha_courant = 0.25; // alpha sub-cycling target
    int n_piso_correctors = 3;
    // Mixture k-omega SST (kinematic, volumetric flux, mixture viscosity; as interFoam).
    bool turbulence = false;
    SstSettings sst{};
    LinearSolverSettings u_solver{LinearSolverType::BiCgStab, 1e-10, 0.01, 1000};
    LinearSolverSettings p_solver{LinearSolverType::AmgCg, 1e-10, 0.001, 500};
};

// Transient interFoam-like solver: explicit bounded (Zalesak flux-corrected) alpha transport
// with interface compression, mixture density / viscosity, momentum with mass flux
// rhoPhi, PISO on p_rgh = p - rho g.x (gravity and CSF surface tension as face forces).
// Pressures are dynamic [Pa]. Boundary conditions as for FlowSolver; velocity inlets also
// fix alpha (`inlet_alpha`), other patches are zero-gradient for alpha.
class MultiphaseSolver {
public:
    MultiphaseSolver(const Mesh& mesh, std::map<std::string, BoundaryCondition> conditions,
                     MultiphaseSettings settings, const parallel::Halo* halo = nullptr);

    // One time step (alpha sub-cycles, PISO, turbulence). Residuals: u (initial momentum
    // residual), p (first pressure solve), continuity, k / omega; t holds the alpha bound error
    // (max |alpha - clip|).
    Residuals step(Scalar dt);

    void set_alpha(std::vector<Scalar> alpha);
    void set_inlet_alpha(const std::string& patch, Scalar alpha);
    void set_velocity(std::vector<Vec3> u);
    void set_state(std::vector<Vec3> u, std::vector<Scalar> p_rgh, std::vector<Scalar> alpha,
                   std::vector<Scalar> phi);

    [[nodiscard]] const std::vector<Vec3>& velocity() const { return u_; }
    [[nodiscard]] const std::vector<Scalar>& alpha() const { return alpha_; }
    [[nodiscard]] const std::vector<Scalar>& p_rgh() const { return p_rgh_; }
    [[nodiscard]] std::vector<Scalar> pressure() const; // p = p_rgh + rho g.x
    [[nodiscard]] std::vector<Scalar> density() const;
    [[nodiscard]] const std::vector<Scalar>& face_flux() const { return phi_; }
    [[nodiscard]] const FvGeometry& geometry() const { return geo_; }
    [[nodiscard]] const MultiphaseSettings& settings() const { return settings_; }
    // Volume of phase 1 (global) and the maximum Courant number of the last step.
    [[nodiscard]] Scalar phase1_volume() const;
    [[nodiscard]] Scalar courant(Scalar dt) const;
    // Turbulence model (nullptr unless settings.turbulence).
    [[nodiscard]] const KOmegaSst* sst() const { return sst_.get(); }
    [[nodiscard]] KOmegaSst* sst() { return sst_.get(); }

private:
    [[nodiscard]] const BoundaryCondition& bc(std::size_t boundary_face) const;
    [[nodiscard]] Vec3 boundary_velocity(std::size_t face) const;
    [[nodiscard]] std::vector<Vec3> alpha_gradient() const;
    [[nodiscard]] std::vector<Scalar> curvature_face() const; // per face
    void advance_alpha(Scalar dt);
    void solve_alpha(Scalar dt); // one explicit sub-step, updates rho_phi_ contribution
    [[nodiscard]] std::vector<Vec3> reconstruct(const std::vector<Scalar>& face_value) const;
    [[nodiscard]] std::vector<Eigen::Matrix3d> velocity_gradient() const;
    [[nodiscard]] std::vector<Scalar> mixture_nu() const;

    FvGeometry geo_;
    MultiphaseSettings settings_;
    std::vector<BoundaryCondition> patch_bcs_;
    std::vector<Vec3> fixed_velocity_; // per boundary face
    std::vector<Scalar> inlet_alpha_;  // per boundary face (velocity inlets)
    bool has_fixed_pressure_ = false;

    std::vector<Vec3> u_;
    std::vector<Scalar> p_rgh_;
    std::vector<Scalar> alpha_;
    std::vector<Scalar> phi_;     // volumetric face flux
    std::vector<Scalar> rho_phi_; // mass face flux
    std::vector<Scalar> gh_;      // g . x per cell
    std::vector<Scalar> ghf_;     // g . x per face
    std::unique_ptr<KOmegaSst> sst_;
};

} // namespace cfd::fv
