// k-omega SST turbulence model for the density-based compressible solver.
#include <algorithm>
#include <cfd/fv/compressible.hpp>
#include <cfd/fv/k_omega_sst.hpp>
#include <cfd/fv/wall_distance.hpp>
#include <cmath>
#include <stdexcept>

namespace cfd::fv {

namespace {

using Mat3 = Eigen::Matrix3d;
using Sst = KOmegaSst; // model constants
constexpr Scalar k_small = 1e-15;

} // namespace

Scalar CompressibleSolver::cell_mu(std::size_t cell) const {
    if (settings_.mu_law.is_constant()) {
        return settings_.mu;
    }
    const GasState st = to_primitive(q_[cell]);
    return settings_.mu_law(st.p / (st.rho * settings_.r_gas));
}

Scalar CompressibleSolver::face_mu(std::size_t face) const {
    const Mesh& m = geo_.mesh();
    const auto p = static_cast<std::size_t>(m.owner()[face]);
    if (settings_.mu_law.is_constant()) {
        return settings_.mu;
    }
    if (face < static_cast<std::size_t>(m.n_internal_faces())) {
        const auto n = static_cast<std::size_t>(m.neighbour()[face]);
        const Scalar w = geo_.weights()[face];
        return w * cell_mu(p) + (1.0 - w) * cell_mu(n);
    }
    return cell_mu(p);
}

Scalar CompressibleSolver::face_mut(std::size_t face) const {
    const Scalar mu = face_mu(face);
    if (mut_.empty()) {
        return mu;
    }
    const Mesh& m = geo_.mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto p = static_cast<std::size_t>(m.owner()[face]);
    if (face < nif) {
        const auto n = static_cast<std::size_t>(m.neighbour()[face]);
        const Scalar w = geo_.weights()[face];
        return mu + w * mut_[p] + (1.0 - w) * mut_[n];
    }
    if (is_wall_face_[face - nif] != 0) {
        return mu;
    }
    return mu + mut_[p];
}

void CompressibleSolver::init_turbulence() {
    if (!(settings_.mu > 0.0)) {
        throw std::invalid_argument("CompressibleSolver: turbulence needs mu > 0");
    }
    const Mesh& m = geo_.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
    is_wall_face_.assign(nb, 0);
    k_bc_.assign(nb, {});
    omega_bc_.assign(nb, {});
    Scalar k_inlet = 0.0;
    Scalar omega_inlet = 0.0;
    for (std::size_t b = 0; b < nb; ++b) {
        const BoundaryCondition& c = bc(b);
        if (c.kind == FlowBoundary::Wall) {
            is_wall_face_[b] = 1;
        } else if (c.kind == FlowBoundary::VelocityInlet) {
            const Scalar rho = c.pressure / (settings_.r_gas * c.temperature);
            const Scalar ti = c.turbulence_intensity * c.velocity.norm();
            const Scalar k_in = std::max(1.5 * ti * ti, k_small);
            const Scalar omega_in =
                rho * k_in / (settings_.mu * std::max(c.viscosity_ratio, k_small));
            k_bc_[b] = {ScalarBoundary::Kind::FixedValue, k_in};
            omega_bc_[b] = {ScalarBoundary::Kind::FixedValue, omega_in};
            k_inlet = std::max(k_inlet, k_in);
            omega_inlet = std::max(omega_inlet, omega_in);
        }
    }
    k_inlet = geo_.comm().max(k_inlet);
    omega_inlet = geo_.comm().max(omega_inlet);
    const Scalar k0 = k_inlet > 0.0 ? k_inlet : 1e-4;
    const Scalar omega0 = omega_inlet > 0.0 ? omega_inlet : k0 / (10.0 * settings_.mu);
    k_.assign(nc, k0);
    omega_.assign(nc, omega0);
    mut_.assign(nc, 0.0);
    y_ = fv::wall_distance(geo_, is_wall_face_);
    set_turbulence(k_, omega_);
}

void CompressibleSolver::set_turbulence(std::vector<Scalar> k, std::vector<Scalar> omega) {
    if (!settings_.turbulence) {
        throw std::logic_error("CompressibleSolver::set_turbulence: turbulence is off");
    }
    if (k.size() != q_.size() || omega.size() != q_.size()) {
        throw std::invalid_argument("CompressibleSolver::set_turbulence: size mismatch");
    }
    k_ = std::move(k);
    omega_ = std::move(omega);
    geo_.exchange(k_);
    geo_.exchange(omega_);
    for (std::size_t c = 0; c < q_.size(); ++c) {
        mut_[c] = q_[c][0] * k_[c] / std::max(omega_[c], k_small);
    }
}

void CompressibleSolver::correct_turbulence(Scalar dt) {
    const Mesh& m = geo_.mesh();
    const auto nc = q_.size();
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& own = m.owner();
    const auto& nei = m.neighbour();
    const auto& w = geo_.weights();
    const auto& sf = m.face_area_vectors();
    const auto& vol = m.cell_volumes();
    std::vector<GasState> s(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        s[c] = to_primitive(q_[c]);
    }

    // Mass flux and Green-Gauss velocity gradient.
    std::vector<Scalar> phi(nf, 0.0);
    std::vector<Mat3> grad_u(nc, Mat3::Zero()); // grad_u(i, j) = du_i / dx_j
    for (std::size_t f = 0; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        Vec3 uf;
        if (f < nif) {
            const auto n = static_cast<std::size_t>(nei[f]);
            uf = w[f] * s[p].u + (1.0 - w[f]) * s[n].u;
            phi[f] = (w[f] * q_[p].segment<3>(1) + (1.0 - w[f]) * q_[n].segment<3>(1)).dot(sf[f]);
            grad_u[n] -= uf * sf[f].transpose();
        } else {
            const BoundaryCondition& c = bc(f - nif);
            if (c.kind == FlowBoundary::Empty) {
                continue;
            }
            const Vec3& nrm = geo_.unit_normals()[f];
            switch (c.kind) {
            case FlowBoundary::Wall:
                uf = c.velocity;
                break;
            case FlowBoundary::Symmetry:
                uf = s[p].u - s[p].u.dot(nrm) * nrm;
                break;
            case FlowBoundary::VelocityInlet: {
                const GasState g = ghost(f, s[p]);
                uf = g.u;
                phi[f] = g.rho * g.u.dot(sf[f]);
                break;
            }
            default:
                uf = s[p].u;
                phi[f] = q_[p].segment<3>(1).dot(sf[f]);
                break;
            }
        }
        grad_u[p] += uf * sf[f].transpose();
    }
    for (std::size_t c = 0; c < nc; ++c) {
        grad_u[c] /= vol[c];
    }
    geo_.exchange(grad_u);

    const std::vector<Vec3> grad_k = scalar_gradient(geo_, k_, k_bc_);
    const std::vector<Vec3> grad_omega = scalar_gradient(geo_, omega_, omega_bc_);

    std::vector<Scalar> rho(nc);
    std::vector<Scalar> s2(nc);
    std::vector<Scalar> f1(nc);
    std::vector<Scalar> f2(nc);
    std::vector<Scalar> cd_k_omega(nc);
    std::vector<Scalar> g(nc); // production rho nu_t S^2
    for (std::size_t c = 0; c < nc; ++c) {
        rho[c] = s[c].rho;
        const Scalar nu = cell_mu(c) / rho[c];
        const Mat3 sym = 0.5 * (grad_u[c] + grad_u[c].transpose());
        s2[c] = 2.0 * sym.squaredNorm();
        const Scalar kc = k_[c];
        const Scalar wc = omega_[c];
        const Scalar y = y_[c];
        cd_k_omega[c] = 2.0 * Sst::k_alpha_omega2 * grad_k[c].dot(grad_omega[c]) / wc;
        const Scalar cd_plus = std::max(rho[c] * cd_k_omega[c], 1e-10);
        const Scalar sqrt_k = std::sqrt(kc);
        const Scalar arg_visc = 500.0 * nu / (y * y * wc);
        const Scalar arg1 =
            std::min(std::min(std::max(sqrt_k / (Sst::k_beta_star * wc * y), arg_visc),
                              4.0 * rho[c] * Sst::k_alpha_omega2 * kc / (cd_plus * y * y)),
                     10.0);
        f1[c] = std::tanh(std::pow(arg1, 4));
        const Scalar arg2 =
            std::min(std::max(2.0 * sqrt_k / (Sst::k_beta_star * wc * y), arg_visc), 100.0);
        f2[c] = std::tanh(arg2 * arg2);
        g[c] = mut_[c] * s2[c];
    }

    // Wall cells: omega from the viscous / log-layer blend.
    const Scalar cmu25 = std::pow(Sst::k_beta_star, 0.25);
    std::vector<int> wall_count(nc, 0);
    std::vector<Scalar> omega_wall(nc, 0.0);
    for (std::size_t f = nif; f < nf; ++f) {
        if (is_wall_face_[f - nif] == 0) {
            continue;
        }
        const auto p = static_cast<std::size_t>(own[f]);
        const Scalar y = y_[p];
        const Scalar nu = cell_mu(p) / rho[p];
        const Scalar omega_vis = 6.0 * nu / (Sst::k_beta1 * y * y);
        const Scalar omega_log = std::sqrt(k_[p]) / (cmu25 * Sst::k_kappa * y);
        omega_wall[p] += std::sqrt(omega_vis * omega_vis + omega_log * omega_log);
        ++wall_count[p];
    }
    std::vector<char> fixed(nc, 0);
    std::vector<Scalar> fixed_omega(nc, 0.0);
    for (std::size_t c = 0; c < nc; ++c) {
        if (wall_count[c] > 0) {
            fixed[c] = 1;
            fixed_omega[c] = omega_wall[c] / wall_count[c];
        }
    }

    auto diffusivity = [&](Scalar alpha1, Scalar alpha2) {
        std::vector<Scalar> sigma_mut(nc);
        for (std::size_t c = 0; c < nc; ++c) {
            sigma_mut[c] = (f1[c] * alpha1 + (1.0 - f1[c]) * alpha2) * mut_[c];
        }
        std::vector<Scalar> gamma(nf);
        for (std::size_t f = 0; f < nif; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const auto n = static_cast<std::size_t>(nei[f]);
            gamma[f] = face_mu(f) + w[f] * sigma_mut[p] + (1.0 - w[f]) * sigma_mut[n];
        }
        for (std::size_t f = nif; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            gamma[f] = face_mu(f) + (is_wall_face_[f - nif] != 0 ? 0.0 : sigma_mut[p]);
        }
        return gamma;
    };

    const bool steady = dt <= 0.0;
    const LinearSolverSettings solver{LinearSolverType::BiCgStab, 1e-10, 0.1, 500};
    // Transient: d(rho phi)/dt with capacity rho^{n+1} and old value rho^n phi^n / rho^{n+1}.
    std::vector<Scalar> k_old;
    std::vector<Scalar> omega_old;
    if (!steady) {
        k_old.resize(nc);
        omega_old.resize(nc);
        for (std::size_t c = 0; c < nc; ++c) {
            const Scalar ratio = rho_old_.empty() ? 1.0 : rho_old_[c] / rho[c];
            k_old[c] = ratio * k_[c];
            omega_old[c] = ratio * omega_[c];
        }
    }

    // omega equation.
    ScalarEquation weq;
    weq.gamma_face = diffusivity(Sst::k_alpha_omega1, Sst::k_alpha_omega2);
    weq.su.resize(nc);
    weq.sp.resize(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        const Scalar wc = omega_[c];
        const Scalar beta = f1[c] * Sst::k_beta1 + (1.0 - f1[c]) * Sst::k_beta2;
        const Scalar gamma = f1[c] * Sst::k_gamma1 + (1.0 - f1[c]) * Sst::k_gamma2;
        const Scalar g_by_nu =
            std::min(s2[c], (Sst::k_c1 / Sst::k_a1) * Sst::k_beta_star * wc *
                                std::max(Sst::k_a1 * wc, Sst::k_b1 * f2[c] * std::sqrt(s2[c])));
        weq.su[c] = rho[c] * gamma * g_by_nu;
        weq.sp[c] = -rho[c] * beta * wc;
        const Scalar cd = rho[c] * (1.0 - f1[c]) * cd_k_omega[c];
        if (cd > 0.0) {
            weq.su[c] += cd;
        } else {
            weq.sp[c] += cd / wc;
        }
    }
    weq.boundary = omega_bc_;
    weq.fixed_cell = fixed;
    weq.fixed_value = fixed_omega;
    weq.relaxation = steady ? 0.7 : 1.0;
    weq.dt = dt;
    weq.old_value = &omega_old;
    weq.capacity = &rho;
    weq.linear_upwind = false; // robust next to the fixed wall-cell omega
    std::vector<Scalar> omega_new = omega_;
    turb_res_.second = solve_scalar(geo_, phi, weq, omega_new, solver).initial_residual;
    for (auto& v : omega_new) {
        v = std::max(v, k_small);
    }

    // k equation.
    ScalarEquation keq;
    keq.gamma_face = diffusivity(Sst::k_alpha_k1, Sst::k_alpha_k2);
    keq.su.resize(nc);
    keq.sp.resize(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        keq.su[c] = std::min(g[c], Sst::k_c1 * Sst::k_beta_star * rho[c] * k_[c] * omega_new[c]);
        keq.sp[c] = -Sst::k_beta_star * rho[c] * omega_new[c];
    }
    keq.boundary = k_bc_;
    keq.relaxation = steady ? 0.7 : 1.0;
    keq.dt = dt;
    keq.old_value = &k_old;
    keq.capacity = &rho;
    keq.linear_upwind = false;
    turb_res_.first = solve_scalar(geo_, phi, keq, k_, solver).initial_residual;
    for (auto& v : k_) {
        v = std::max(v, k_small);
    }
    omega_ = std::move(omega_new);
    geo_.exchange(k_);
    geo_.exchange(omega_);

    for (std::size_t c = 0; c < nc; ++c) {
        mut_[c] = rho[c] * Sst::k_a1 * k_[c] /
                  std::max(Sst::k_a1 * omega_[c], Sst::k_b1 * f2[c] * std::sqrt(s2[c]));
    }
    geo_.exchange(mut_);
}

} // namespace cfd::fv
