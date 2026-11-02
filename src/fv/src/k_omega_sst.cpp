#include <algorithm>
#include <cfd/fv/k_omega_sst.hpp>
#include <cfd/fv/wall_distance.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cfd::fv {

namespace {

using Mat3 = Eigen::Matrix3d;

constexpr Scalar k_small = 1e-15;

// Solves y+ = ln(E y+) / kappa (intersection of viscous and log law), ~11.53.
Scalar laminar_y_plus(Scalar kappa, Scalar e) {
    Scalar yp = 11.0;
    for (int i = 0; i < 20; ++i) {
        yp = std::log(std::max(e * yp, 1.0)) / kappa;
    }
    return yp;
}

} // namespace

KOmegaSst::KOmegaSst(const FlowSolver& flow, SstSettings settings)
    : geo_(&flow.geometry()), settings_(settings), nu_(flow.settings().nu),
      y_plus_lam_(laminar_y_plus(k_kappa, k_e)) {
    const Mesh& m = geo_->mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nb = static_cast<std::size_t>(m.n_boundary_faces());

    is_wall_face_.assign(nb, 0);
    k_bc_.assign(nb, {});
    omega_bc_.assign(nb, {});
    Scalar k_inlet = 0.0;
    Scalar omega_inlet = 0.0;
    for (std::size_t b = 0; b < nb; ++b) {
        const BoundaryCondition& c = flow.boundary_condition(static_cast<Label>(b));
        if (c.kind == FlowBoundary::Wall) {
            is_wall_face_[b] = 1;
        } else if (c.kind == FlowBoundary::VelocityInlet) {
            const Scalar ti = c.turbulence_intensity * c.velocity.norm();
            const Scalar k_in = std::max(1.5 * ti * ti, k_small);
            const Scalar omega_in = k_in / (nu_ * std::max(c.viscosity_ratio, k_small));
            k_bc_[b] = {ScalarBoundary::Kind::FixedValue, k_in};
            omega_bc_[b] = {ScalarBoundary::Kind::FixedValue, omega_in};
            k_inlet = std::max(k_inlet, k_in);
            omega_inlet = std::max(omega_inlet, omega_in);
        }
    }

    const Scalar k0 = settings_.k_init > 0.0 ? settings_.k_init : (k_inlet > 0.0 ? k_inlet : 1e-4);
    const Scalar omega0 = settings_.omega_init > 0.0
                              ? settings_.omega_init
                              : (omega_inlet > 0.0 ? omega_inlet : k0 / (10.0 * nu_));
    k_.assign(nc, k0);
    omega_.assign(nc, omega0);
    k_old_ = k_;
    omega_old_ = omega_;
    nut_.assign(nc, k0 / omega0);
    nut_b_.assign(nb, 0.0);
    y_plus_.assign(nc, 0.0);
    y_ = fv::wall_distance(*geo_, is_wall_face_);
    for (std::size_t b = 0; b < nb; ++b) {
        if (is_wall_face_[b] == 0) {
            nut_b_[b] = nut_[static_cast<std::size_t>(
                m.owner()[m.n_internal_faces() + static_cast<Label>(b)])];
        }
    }
}

void KOmegaSst::set_fields(std::vector<Scalar> k, std::vector<Scalar> omega) {
    if (k.size() != k_.size() || omega.size() != omega_.size()) {
        throw std::invalid_argument("KOmegaSst::set_fields: size mismatch");
    }
    k_ = std::move(k);
    omega_ = std::move(omega);
    k_old_ = k_;
    omega_old_ = omega_;
    for (std::size_t c = 0; c < nut_.size(); ++c) {
        nut_[c] = k_[c] / std::max(omega_[c], k_small);
    }
    const Mesh& m = geo_->mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    for (std::size_t b = 0; b < nut_b_.size(); ++b) {
        if (is_wall_face_[b] == 0) {
            nut_b_[b] = nut_[static_cast<std::size_t>(m.owner()[nif + b])];
        }
    }
}

void KOmegaSst::set_fields(std::vector<Scalar> k, std::vector<Scalar> omega,
                           std::vector<Scalar> nut, std::vector<Scalar> boundary_nut) {
    if (nut.size() != nut_.size() || boundary_nut.size() != nut_b_.size()) {
        throw std::invalid_argument("KOmegaSst::set_fields: size mismatch");
    }
    set_fields(std::move(k), std::move(omega));
    nut_ = std::move(nut);
    nut_b_ = std::move(boundary_nut);
}

void KOmegaSst::update_nut(const std::vector<Scalar>& s2, const std::vector<Scalar>& f2) {
    const Mesh& m = geo_->mesh();
    for (std::size_t c = 0; c < nut_.size(); ++c) {
        nut_[c] = k_a1 * k_[c] / std::max(k_a1 * omega_[c], k_b1 * f2[c] * std::sqrt(s2[c]));
    }
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    for (std::size_t b = 0; b < nut_b_.size(); ++b) {
        if (is_wall_face_[b] == 0) {
            nut_b_[b] = nut_[static_cast<std::size_t>(m.owner()[nif + b])];
        }
    }
}

std::pair<Scalar, Scalar> KOmegaSst::correct(const FlowSolver& flow, Scalar dt) {
    const Mesh& m = geo_->mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& own = m.owner();
    const auto& nei = m.neighbour();
    const auto& w = geo_->weights();
    const auto& u = flow.velocity();
    const auto& phi = flow.face_flux();

    if (dt > 0.0) {
        k_old_ = k_;
        omega_old_ = omega_;
    }

    const std::vector<Mat3> grad_u = flow.velocity_gradient();
    const std::vector<Vec3> grad_k = scalar_gradient(*geo_, k_, k_bc_);
    const std::vector<Vec3> grad_omega = scalar_gradient(*geo_, omega_, omega_bc_);

    std::vector<Scalar> s2(nc);
    std::vector<Scalar> f1(nc);
    std::vector<Scalar> f2(nc);
    std::vector<Scalar> cd_k_omega(nc);
    std::vector<Scalar> g(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        const Mat3 sym = 0.5 * (grad_u[c] + grad_u[c].transpose());
        s2[c] = 2.0 * sym.squaredNorm();
        const Scalar kc = k_[c];
        const Scalar wc = omega_[c];
        const Scalar y = y_[c];
        cd_k_omega[c] = 2.0 * k_alpha_omega2 * grad_k[c].dot(grad_omega[c]) / wc;
        const Scalar cd_plus = std::max(cd_k_omega[c], 1e-10);
        const Scalar sqrt_k = std::sqrt(kc);
        const Scalar arg_visc = 500.0 * nu_ / (y * y * wc);
        const Scalar arg1 = std::min(std::min(std::max(sqrt_k / (k_beta_star * wc * y), arg_visc),
                                              4.0 * k_alpha_omega2 * kc / (cd_plus * y * y)),
                                     10.0);
        f1[c] = std::tanh(std::pow(arg1, 4));
        const Scalar arg2 =
            std::min(std::max(2.0 * sqrt_k / (k_beta_star * wc * y), arg_visc), 100.0);
        f2[c] = std::tanh(arg2 * arg2);
        g[c] = nut_[c] * s2[c];
    }

    // Wall treatment for wall-adjacent cells.
    const Scalar cmu25 = std::pow(k_beta_star, 0.25);
    std::vector<int> wall_count(nc, 0);
    std::vector<Scalar> omega_wall(nc, 0.0);
    std::vector<Scalar> g_wall(nc, 0.0);
    std::fill(y_plus_.begin(), y_plus_.end(), 0.0);
    for (std::size_t f = nif; f < nf; ++f) {
        const std::size_t b = f - nif;
        if (is_wall_face_[b] == 0) {
            continue;
        }
        const auto p = static_cast<std::size_t>(own[f]);
        const Scalar y = y_[p];
        const Scalar sqrt_k = std::sqrt(k_[p]);
        const Scalar y_plus_k = cmu25 * sqrt_k * y / nu_;

        const Scalar omega_vis = 6.0 * nu_ / (k_beta1 * y * y);
        const Scalar omega_log = sqrt_k / (cmu25 * k_kappa * y);
        omega_wall[p] += std::sqrt(omega_vis * omega_vis + omega_log * omega_log);

        const Vec3& n = geo_->unit_normals()[f];
        Vec3 ut = u[p] - flow.boundary_condition(static_cast<Label>(b)).velocity;
        ut -= ut.dot(n) * n;
        const Scalar mag_grad_uw = ut.norm() / y;

        Scalar nut_w = 0.0;
        if (y_plus_k > y_plus_lam_) {
            nut_w = nu_ * (y_plus_k * k_kappa / std::log(k_e * y_plus_k) - 1.0);
            g_wall[p] += (nu_ + nut_w) * mag_grad_uw * cmu25 * sqrt_k / (k_kappa * y);
        } else {
            g_wall[p] += g[p];
        }
        nut_b_[b] = std::max(nut_w, 0.0);
        const Scalar u_tau = std::sqrt((nu_ + nut_b_[b]) * mag_grad_uw);
        y_plus_[p] = std::max(y_plus_[p], u_tau * y / nu_);
        ++wall_count[p];
    }

    std::vector<char> fixed(nc, 0);
    std::vector<Scalar> fixed_omega(nc, 0.0);
    for (std::size_t c = 0; c < nc; ++c) {
        if (wall_count[c] > 0) {
            fixed[c] = 1;
            fixed_omega[c] = omega_wall[c] / wall_count[c];
            g[c] = g_wall[c] / wall_count[c];
        }
    }

    // Face diffusivities nu + sigma * nu_t.
    auto diffusivity = [&](Scalar alpha1, Scalar alpha2) {
        std::vector<Scalar> cell_sigma_nut(nc);
        for (std::size_t c = 0; c < nc; ++c) {
            cell_sigma_nut[c] = (f1[c] * alpha1 + (1.0 - f1[c]) * alpha2) * nut_[c];
        }
        std::vector<Scalar> gamma(nf);
        for (std::size_t f = 0; f < nif; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const auto n = static_cast<std::size_t>(nei[f]);
            gamma[f] = nu_ + w[f] * cell_sigma_nut[p] + (1.0 - w[f]) * cell_sigma_nut[n];
        }
        for (std::size_t f = nif; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const Scalar sigma = f1[p] * alpha1 + (1.0 - f1[p]) * alpha2;
            gamma[f] = nu_ + sigma * nut_b_[f - nif];
        }
        return gamma;
    };

    const bool steady = dt <= 0.0;

    // omega equation.
    ScalarEquation weq;
    weq.gamma_face = diffusivity(k_alpha_omega1, k_alpha_omega2);
    weq.su.resize(nc);
    weq.sp.resize(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        const Scalar wc = omega_[c];
        const Scalar beta = f1[c] * k_beta1 + (1.0 - f1[c]) * k_beta2;
        const Scalar gamma = f1[c] * k_gamma1 + (1.0 - f1[c]) * k_gamma2;
        const Scalar g_by_nu =
            std::min(s2[c], (k_c1 / k_a1) * k_beta_star * wc *
                                std::max(k_a1 * wc, k_b1 * f2[c] * std::sqrt(s2[c])));
        weq.su[c] = gamma * g_by_nu;
        weq.sp[c] = -beta * wc;
        const Scalar cd = (1.0 - f1[c]) * cd_k_omega[c];
        if (cd > 0.0) {
            weq.su[c] += cd;
        } else {
            weq.sp[c] += cd / wc;
        }
    }
    weq.boundary = omega_bc_;
    weq.fixed_cell = fixed;
    weq.fixed_value = fixed_omega;
    weq.relaxation = steady ? settings_.relax_omega : 1.0;
    weq.dt = dt;
    weq.old_value = &omega_old_;
    const Scalar omega_res =
        solve_scalar(*geo_, phi, weq, omega_, settings_.solver).initial_residual;
    for (auto& v : omega_) {
        v = std::max(v, k_small);
    }

    // k equation.
    ScalarEquation keq;
    keq.gamma_face = diffusivity(k_alpha_k1, k_alpha_k2);
    keq.su.resize(nc);
    keq.sp.resize(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        keq.su[c] = std::min(g[c], k_c1 * k_beta_star * k_[c] * omega_[c]);
        keq.sp[c] = -k_beta_star * omega_[c];
    }
    keq.boundary = k_bc_;
    keq.relaxation = steady ? settings_.relax_k : 1.0;
    keq.dt = dt;
    keq.old_value = &k_old_;
    const Scalar k_res = solve_scalar(*geo_, phi, keq, k_, settings_.solver).initial_residual;
    for (auto& v : k_) {
        v = std::max(v, k_small);
    }

    update_nut(s2, f2);
    return {k_res, omega_res};
}

} // namespace cfd::fv
