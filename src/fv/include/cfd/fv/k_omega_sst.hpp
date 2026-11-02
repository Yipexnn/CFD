#pragma once

#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/scalar_transport.hpp>
#include <cfd/fv/turbulence_model.hpp>
#include <vector>

namespace cfd::fv {

struct SstSettings {
    // Initial values; <= 0 derives them from the inlet conditions (or 1e-4 / 1 if none).
    Scalar k_init = -1.0;
    Scalar omega_init = -1.0;
    Scalar relax_k = 0.7;
    Scalar relax_omega = 0.7;
    LinearSolverSettings solver{LinearSolverType::BiCgStab, 1e-10, 0.1, 1000};
};

// k-omega SST (Menter et al. 2003, as in OpenFOAM's kOmegaSST) with automatic wall
// treatment: omega in wall cells blends the viscous and log-layer solutions, k has zero
// gradient at walls, and the wall-face nu_t / production follow the log law when
// y+ > y+_lam (~11.5).
class KOmegaSst final : public TurbulenceModel {
public:
    KOmegaSst(const FlowSolver& flow, SstSettings settings = {});

    std::pair<Scalar, Scalar> correct(const FlowSolver& flow, Scalar dt) override;

    [[nodiscard]] const std::vector<Scalar>& nut() const override { return nut_; }
    [[nodiscard]] const std::vector<Scalar>& boundary_nut() const override { return nut_b_; }

    // Restart: sets k and omega (nu_t = k / omega until the next correct()).
    void set_fields(std::vector<Scalar> k, std::vector<Scalar> omega);
    // Exact restart including cell and boundary-face nu_t.
    void set_fields(std::vector<Scalar> k, std::vector<Scalar> omega, std::vector<Scalar> nut,
                    std::vector<Scalar> boundary_nut);

    [[nodiscard]] const std::vector<Scalar>& k() const { return k_; }
    [[nodiscard]] const std::vector<Scalar>& omega() const { return omega_; }
    [[nodiscard]] const std::vector<Scalar>& wall_distance() const { return y_; }
    // y+ of wall-adjacent cells (0 elsewhere), based on the friction velocity.
    [[nodiscard]] const std::vector<Scalar>& y_plus() const { return y_plus_; }

    // Model constants.
    static constexpr Scalar k_alpha_k1 = 0.85;
    static constexpr Scalar k_alpha_k2 = 1.0;
    static constexpr Scalar k_alpha_omega1 = 0.5;
    static constexpr Scalar k_alpha_omega2 = 0.856;
    static constexpr Scalar k_beta1 = 0.075;
    static constexpr Scalar k_beta2 = 0.0828;
    static constexpr Scalar k_beta_star = 0.09;
    static constexpr Scalar k_gamma1 = 5.0 / 9.0;
    static constexpr Scalar k_gamma2 = 0.44;
    static constexpr Scalar k_a1 = 0.31;
    static constexpr Scalar k_b1 = 1.0;
    static constexpr Scalar k_c1 = 10.0;
    static constexpr Scalar k_kappa = 0.41;
    static constexpr Scalar k_e = 9.8;

private:
    void update_nut(const std::vector<Scalar>& s2, const std::vector<Scalar>& f2);

    const FvGeometry* geo_;
    SstSettings settings_;
    Scalar nu_;
    Scalar y_plus_lam_;

    std::vector<Scalar> k_;
    std::vector<Scalar> omega_;
    std::vector<Scalar> k_old_;
    std::vector<Scalar> omega_old_;
    std::vector<Scalar> nut_;
    std::vector<Scalar> nut_b_;
    std::vector<Scalar> y_;
    std::vector<Scalar> y_plus_;

    std::vector<char> is_wall_face_; // per boundary face
    std::vector<ScalarBoundary> k_bc_;
    std::vector<ScalarBoundary> omega_bc_;
};

} // namespace cfd::fv
