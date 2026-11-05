#include <algorithm>
#include <array>
#include <cfd/fv/multiphase.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cfd::fv {

namespace {

using Mat3 = Eigen::Matrix3d;

BoundaryCondition default_condition(const mesh::Patch& patch) {
    switch (patch.type) {
    case mesh::PatchType::Wall:
        return {FlowBoundary::Wall, Vec3::Zero(), 0.0};
    case mesh::PatchType::Symmetry:
        return {FlowBoundary::Symmetry, Vec3::Zero(), 0.0};
    case mesh::PatchType::Outlet:
        return {FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0};
    case mesh::PatchType::Inlet:
    case mesh::PatchType::Patch:
        break;
    }
    throw std::invalid_argument("MultiphaseSolver: no boundary condition for patch '" + patch.name +
                                "'");
}

} // namespace

MultiphaseSolver::MultiphaseSolver(const Mesh& mesh,
                                   std::map<std::string, BoundaryCondition> conditions,
                                   MultiphaseSettings settings, const parallel::Halo* halo)
    : geo_(mesh, halo), settings_(settings) {
    for (std::size_t i = 0; i < mesh.patches().size(); ++i) {
        const auto& patch = mesh.patches()[i];
        if (halo != nullptr && static_cast<int>(i) == halo->halo_patch) {
            patch_bcs_.push_back({FlowBoundary::Empty, Vec3::Zero(), 0.0});
            continue;
        }
        const auto it = conditions.find(patch.name);
        patch_bcs_.push_back(it != conditions.end() ? it->second : default_condition(patch));
        if (patch_bcs_.back().kind == FlowBoundary::PressureOutlet && patch.n_faces > 0) {
            has_fixed_pressure_ = true;
        }
    }
    for (const auto& [name, c] : conditions) {
        if (mesh.find_patch(name) == nullptr) {
            throw std::invalid_argument("MultiphaseSolver: unknown patch '" + name + "'");
        }
    }
    has_fixed_pressure_ = geo_.comm().max(has_fixed_pressure_ ? 1.0 : 0.0) > 0.5;
    const auto nc = static_cast<std::size_t>(mesh.n_cells());
    const auto nf = static_cast<std::size_t>(mesh.n_faces());
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    fixed_velocity_.resize(nb);
    inlet_alpha_.assign(nb, 0.0);
    for (std::size_t b = 0; b < nb; ++b) {
        fixed_velocity_[b] = bc(b).velocity;
    }
    u_.assign(nc, Vec3::Zero());
    p_rgh_.assign(nc, 0.0);
    alpha_.assign(nc, 0.0);
    phi_.assign(nf, 0.0);
    rho_phi_.assign(nf, 0.0);
    gh_.resize(nc);
    ghf_.resize(nf);
    for (std::size_t c = 0; c < nc; ++c) {
        gh_[c] = settings_.gravity.dot(mesh.cell_centres()[c]);
    }
    for (std::size_t f = 0; f < nf; ++f) {
        ghf_[f] = settings_.gravity.dot(mesh.face_centres()[f]);
    }
    set_velocity(u_);
}

const BoundaryCondition& MultiphaseSolver::bc(std::size_t boundary_face) const {
    return patch_bcs_[static_cast<std::size_t>(geo_.boundary_patch()[boundary_face])];
}

Vec3 MultiphaseSolver::boundary_velocity(std::size_t face) const {
    const Mesh& m = geo_.mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const BoundaryCondition& c = bc(face - nif);
    const Vec3& up = u_[static_cast<std::size_t>(m.owner()[face])];
    switch (c.kind) {
    case FlowBoundary::VelocityInlet:
    case FlowBoundary::Wall:
        return fixed_velocity_[face - nif];
    case FlowBoundary::PressureOutlet:
    case FlowBoundary::Empty:
        return up;
    case FlowBoundary::Symmetry:
        break;
    }
    const Vec3& n = geo_.unit_normals()[face];
    return up - up.dot(n) * n;
}

void MultiphaseSolver::set_alpha(std::vector<Scalar> alpha) {
    if (alpha.size() != alpha_.size()) {
        throw std::invalid_argument("MultiphaseSolver::set_alpha: size mismatch");
    }
    alpha_ = std::move(alpha);
    geo_.exchange(alpha_);
}

void MultiphaseSolver::set_inlet_alpha(const std::string& patch, Scalar alpha) {
    const Mesh& m = geo_.mesh();
    const mesh::Patch* p = m.find_patch(patch);
    if (p == nullptr) {
        throw std::invalid_argument("MultiphaseSolver: unknown patch '" + patch + "'");
    }
    const auto start = static_cast<std::size_t>(p->start_face - m.n_internal_faces());
    std::fill_n(inlet_alpha_.begin() + static_cast<std::ptrdiff_t>(start), p->n_faces, alpha);
}

void MultiphaseSolver::set_velocity(std::vector<Vec3> u) {
    if (u.size() != u_.size()) {
        throw std::invalid_argument("MultiphaseSolver::set_velocity: size mismatch");
    }
    u_ = std::move(u);
    geo_.exchange(u_);
    const Mesh& m = geo_.mesh();
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_.weights();
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        phi_[f] = (w[f] * u_[p] + (1.0 - w[f]) * u_[n]).dot(m.face_area_vectors()[f]);
    }
    for (std::size_t f = nif; f < nf; ++f) {
        const FlowBoundary k = bc(f - nif).kind;
        phi_[f] = k == FlowBoundary::Symmetry || k == FlowBoundary::Empty
                      ? 0.0
                      : boundary_velocity(f).dot(m.face_area_vectors()[f]);
    }
}

void MultiphaseSolver::set_state(std::vector<Vec3> u, std::vector<Scalar> p_rgh,
                                 std::vector<Scalar> alpha, std::vector<Scalar> phi) {
    set_alpha(std::move(alpha));
    set_velocity(std::move(u));
    if (p_rgh.size() != p_rgh_.size() || phi.size() != phi_.size()) {
        throw std::invalid_argument("MultiphaseSolver::set_state: size mismatch");
    }
    p_rgh_ = std::move(p_rgh);
    geo_.exchange(p_rgh_);
    phi_ = std::move(phi);
}

std::vector<Scalar> MultiphaseSolver::density() const {
    std::vector<Scalar> rho(alpha_.size());
    for (std::size_t c = 0; c < rho.size(); ++c) {
        const Scalar a = std::clamp(alpha_[c], 0.0, 1.0);
        rho[c] = a * settings_.rho1 + (1.0 - a) * settings_.rho2;
    }
    return rho;
}

std::vector<Scalar> MultiphaseSolver::pressure() const {
    const std::vector<Scalar> rho = density();
    std::vector<Scalar> p(p_rgh_.size());
    for (std::size_t c = 0; c < p.size(); ++c) {
        p[c] = p_rgh_[c] + rho[c] * gh_[c];
    }
    return p;
}

Scalar MultiphaseSolver::phase1_volume() const {
    Scalar v = 0.0;
    for (Label c = 0; c < geo_.n_owned(); ++c) {
        v += alpha_[static_cast<std::size_t>(c)] *
             geo_.mesh().cell_volumes()[static_cast<std::size_t>(c)];
    }
    return geo_.comm().sum(v);
}

Scalar MultiphaseSolver::courant(Scalar dt) const {
    const Mesh& m = geo_.mesh();
    std::vector<Scalar> sum(alpha_.size(), 0.0);
    for (std::size_t f = 0; f < phi_.size(); ++f) {
        sum[static_cast<std::size_t>(m.owner()[f])] += std::abs(phi_[f]);
        if (f < static_cast<std::size_t>(m.n_internal_faces())) {
            sum[static_cast<std::size_t>(m.neighbour()[f])] += std::abs(phi_[f]);
        }
    }
    Scalar co = 0.0;
    for (Label c = 0; c < geo_.n_owned(); ++c) {
        const auto cs = static_cast<std::size_t>(c);
        co = std::max(co, 0.5 * sum[cs] * dt / m.cell_volumes()[cs]);
    }
    return geo_.comm().max(co);
}

std::vector<Vec3> MultiphaseSolver::alpha_gradient() const {
    const Mesh& m = geo_.mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_.weights();
    const auto& sf = m.face_area_vectors();
    std::vector<Vec3> grad(alpha_.size(), Vec3::Zero());
    for (std::size_t f = 0; f < sf.size(); ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        if (f < nif) {
            const auto n = static_cast<std::size_t>(m.neighbour()[f]);
            const Vec3 flux = sf[f] * (w[f] * alpha_[p] + (1.0 - w[f]) * alpha_[n]);
            grad[p] += flux;
            grad[n] -= flux;
        } else {
            const BoundaryCondition& c = bc(f - nif);
            const Scalar ab =
                c.kind == FlowBoundary::VelocityInlet ? inlet_alpha_[f - nif] : alpha_[p];
            grad[p] += sf[f] * ab;
        }
    }
    for (std::size_t c = 0; c < grad.size(); ++c) {
        grad[c] /= m.cell_volumes()[c];
    }
    geo_.exchange(grad);
    return grad;
}

std::vector<Scalar> MultiphaseSolver::curvature_face() const {
    // kappa = -div(n_hat), n_hat = grad(alpha) / |grad(alpha)| interpolated to faces.
    const Mesh& m = geo_.mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_.weights();
    const auto& sf = m.face_area_vectors();
    const std::vector<Vec3> grad = alpha_gradient();
    std::vector<Scalar> kappa(alpha_.size(), 0.0);
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        const Vec3 g = w[f] * grad[p] + (1.0 - w[f]) * grad[n];
        const Scalar delta = 1e-8 / std::cbrt(m.cell_volumes()[p]);
        const Scalar flux = (g / (g.norm() + delta)).dot(sf[f]);
        kappa[p] -= flux;
        kappa[n] += flux;
    }
    for (std::size_t c = 0; c < kappa.size(); ++c) {
        kappa[c] /= m.cell_volumes()[c];
    }
    geo_.exchange(kappa);
    std::vector<Scalar> kf(sf.size(), 0.0);
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        kf[f] = w[f] * kappa[p] + (1.0 - w[f]) * kappa[n];
    }
    return kf;
}

void MultiphaseSolver::solve_alpha(Scalar dt) {
    const Mesh& m = geo_.mesh();
    const auto nc = alpha_.size();
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_.weights();
    const auto& sf = m.face_area_vectors();
    const auto& vol = m.cell_volumes();
    const std::vector<Vec3> grad = alpha_gradient();

    // Low-order (upwind) and high-order (central + interface compression) fluxes.
    std::vector<Scalar> f_low(nf, 0.0);
    std::vector<Scalar> anti(nf, 0.0);
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        const Scalar flux = phi_[f];
        f_low[f] = flux * (flux >= 0.0 ? alpha_[p] : alpha_[n]);
        const Scalar a_central = w[f] * alpha_[p] + (1.0 - w[f]) * alpha_[n];
        const Vec3 g = w[f] * grad[p] + (1.0 - w[f]) * grad[n];
        const Scalar delta = 1e-8 / std::cbrt(vol[p]);
        const Scalar phir = settings_.compression * std::abs(flux) / geo_.face_areas()[f] *
                            (g / (g.norm() + delta)).dot(sf[f]);
        const Scalar a_up = phir >= 0.0 ? alpha_[p] : alpha_[n];
        const Scalar b_up = phir >= 0.0 ? 1.0 - alpha_[n] : 1.0 - alpha_[p];
        const Scalar f_high = flux * a_central + phir * a_up * b_up;
        anti[f] = f_high - f_low[f];
    }
    for (std::size_t f = nif; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const Scalar flux = phi_[f];
        const BoundaryCondition& c = bc(f - nif);
        if (flux >= 0.0) {
            f_low[f] = flux * alpha_[p];
        } else {
            f_low[f] =
                flux * (c.kind == FlowBoundary::VelocityInlet ? inlet_alpha_[f - nif] : alpha_[p]);
        }
    }

    // Zalesak limiter: the corrected update stays within the local bounds (and [0, 1]).
    std::vector<Scalar> a_min(alpha_);
    std::vector<Scalar> a_max(alpha_);
    std::vector<Scalar> low(alpha_);
    std::vector<Scalar> p_in(nc, 0.0);
    std::vector<Scalar> p_out(nc, 0.0);
    for (std::size_t f = 0; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        low[p] -= dt / vol[p] * f_low[f];
        if (f < nif) {
            const auto n = static_cast<std::size_t>(m.neighbour()[f]);
            low[n] += dt / vol[n] * f_low[f];
            a_min[p] = std::min(a_min[p], alpha_[n]);
            a_max[p] = std::max(a_max[p], alpha_[n]);
            a_min[n] = std::min(a_min[n], alpha_[p]);
            a_max[n] = std::max(a_max[n], alpha_[p]);
            p_out[p] += std::max(anti[f], 0.0);
            p_in[p] += std::max(-anti[f], 0.0);
            p_out[n] += std::max(-anti[f], 0.0);
            p_in[n] += std::max(anti[f], 0.0);
        }
    }
    std::vector<Scalar> r_plus(nc, 0.0);
    std::vector<Scalar> r_minus(nc, 0.0);
    for (std::size_t c = 0; c < nc; ++c) {
        const Scalar q_plus = (std::min(a_max[c], 1.0) - low[c]) * vol[c] / dt;
        const Scalar q_minus = (low[c] - std::max(a_min[c], 0.0)) * vol[c] / dt;
        r_plus[c] = p_in[c] > 1e-300 ? std::clamp(q_plus / p_in[c], 0.0, 1.0) : 1.0;
        r_minus[c] = p_out[c] > 1e-300 ? std::clamp(q_minus / p_out[c], 0.0, 1.0) : 1.0;
    }
    geo_.exchange(r_plus);
    geo_.exchange(r_minus);

    std::vector<Scalar> alpha_flux(f_low);
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        const Scalar lambda =
            anti[f] >= 0.0 ? std::min(r_minus[p], r_plus[n]) : std::min(r_plus[p], r_minus[n]);
        alpha_flux[f] += lambda * anti[f];
    }
    for (std::size_t f = 0; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        alpha_[p] -= dt / vol[p] * alpha_flux[f];
        if (f < nif) {
            const auto n = static_cast<std::size_t>(m.neighbour()[f]);
            alpha_[n] += dt / vol[n] * alpha_flux[f];
        }
    }
    geo_.exchange(alpha_);
    for (std::size_t f = 0; f < nf; ++f) {
        rho_phi_[f] += (settings_.rho1 - settings_.rho2) * alpha_flux[f] + settings_.rho2 * phi_[f];
    }
}

void MultiphaseSolver::advance_alpha(Scalar dt) {
    const Scalar co = courant(dt);
    const int n_sub =
        std::max(1, static_cast<int>(std::ceil(co / std::max(settings_.max_alpha_courant, 1e-6))));
    std::fill(rho_phi_.begin(), rho_phi_.end(), 0.0);
    for (int s = 0; s < n_sub; ++s) {
        solve_alpha(dt / n_sub);
    }
    for (Scalar& f : rho_phi_) {
        f /= n_sub;
    }
}

std::vector<Vec3> MultiphaseSolver::reconstruct(const std::vector<Scalar>& face_value) const {
    // Least-squares cell vector g with g . S_f ~ face_value (fvc::reconstruct).
    const Mesh& m = geo_.mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& sf = m.face_area_vectors();
    std::vector<Mat3> mat(u_.size(), Mat3::Zero());
    std::vector<Vec3> rhs(u_.size(), Vec3::Zero());
    for (std::size_t f = 0; f < sf.size(); ++f) {
        const Scalar a = geo_.face_areas()[f];
        const Mat3 ss = sf[f] * sf[f].transpose() / a;
        const Vec3 b = sf[f] * face_value[f] / a;
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        mat[p] += ss;
        rhs[p] += b;
        if (f < nif) {
            const auto n = static_cast<std::size_t>(m.neighbour()[f]);
            mat[n] += ss;
            rhs[n] += b;
        }
    }
    std::vector<Vec3> g(u_.size());
    for (std::size_t c = 0; c < g.size(); ++c) {
        g[c] = mat[c].ldlt().solve(rhs[c]);
    }
    return g;
}

Residuals MultiphaseSolver::step(Scalar dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("MultiphaseSolver::step: dt must be positive");
    }
    const Mesh& m = geo_.mesh();
    const auto nc = u_.size();
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& own = m.owner();
    const auto& nei = m.neighbour();
    const auto& w = geo_.weights();
    const auto& dc = geo_.delta_coeffs();
    const auto& sf = m.face_area_vectors();
    const auto& vol = m.cell_volumes();

    const std::vector<Scalar> rho_old = density();
    const std::vector<Vec3> u_old = u_;
    advance_alpha(dt);
    const std::vector<Scalar> rho = density();
    std::vector<Scalar> mu(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        const Scalar a = std::clamp(alpha_[c], 0.0, 1.0);
        mu[c] = a * settings_.rho1 * settings_.nu1 + (1.0 - a) * settings_.rho2 * settings_.nu2;
    }

    // Momentum matrix (upwind convection with the mass flux, implicit Euler).
    LduMatrix a(m);
    a.pattern = &geo_.csr_pattern();
    a.halo = geo_.halo();
    std::vector<Vec3> source(nc, Vec3::Zero());
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const auto n = static_cast<std::size_t>(nei[f]);
        const Scalar flux = rho_phi_[f];
        const Scalar diff = (w[f] * mu[p] + (1.0 - w[f]) * mu[n]) * dc[f];
        a.upper[f] = std::min(flux, 0.0) - diff;
        a.lower[f] = -std::max(flux, 0.0) - diff;
        a.diag[p] += std::max(flux, 0.0) + diff;
        a.diag[n] += -std::min(flux, 0.0) + diff;
    }
    for (std::size_t f = nif; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const Scalar flux = rho_phi_[f];
        const Scalar diff = mu[p] * dc[f];
        const BoundaryCondition& c = bc(f - nif);
        switch (c.kind) {
        case FlowBoundary::VelocityInlet:
        case FlowBoundary::Wall:
            a.diag[p] += diff + std::max(flux, 0.0);
            source[p] += (diff - std::min(flux, 0.0)) * fixed_velocity_[f - nif];
            break;
        case FlowBoundary::PressureOutlet:
            a.diag[p] += std::max(flux, 0.0);
            break;
        case FlowBoundary::Symmetry:
            a.diag[p] += diff;
            source[p] += diff * boundary_velocity(f);
            break;
        case FlowBoundary::Empty:
            break;
        }
    }
    for (std::size_t c = 0; c < nc; ++c) {
        a.diag[c] += rho[c] * vol[c] / dt;
        source[c] += rho_old[c] * vol[c] / dt * u_old[c];
    }

    // Momentum residual (no momentum predictor).
    Residuals res;
    {
        std::array<std::vector<Scalar>, 3> x;
        Scalar norm = 0.0;
        for (int i = 0; i < 3; ++i) {
            LduMatrix eq = a;
            x[static_cast<std::size_t>(i)].resize(nc);
            for (std::size_t c = 0; c < nc; ++c) {
                eq.source[c] = source[c][i];
                x[static_cast<std::size_t>(i)][c] = u_[c][i];
            }
            const auto [r, n] = eq.residual_norms(x[static_cast<std::size_t>(i)]);
            res.u[i] = r;
            norm = std::max(norm, n);
        }
        res.u /= norm + 1e-300;
    }

    // Face forces: surface tension and gravity (p_rgh formulation), per unit area times |Sf|.
    std::vector<Scalar> phig(nf, 0.0);
    {
        const std::vector<Scalar> kappa =
            settings_.sigma > 0.0 ? curvature_face() : std::vector<Scalar>(nf, 0.0);
        for (std::size_t f = 0; f < nif; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const auto n = static_cast<std::size_t>(nei[f]);
            phig[f] = (settings_.sigma * kappa[f] * (alpha_[n] - alpha_[p]) -
                       ghf_[f] * (rho[n] - rho[p])) *
                      dc[f];
        }
    }

    std::unique_ptr<AmgCache> amg = std::make_unique<AmgCache>();
    for (int corr = 0; corr < settings_.n_piso_correctors; ++corr) {
        std::vector<Vec3> hbya(nc);
        std::vector<Scalar> rau(nc);
        {
            std::vector<Scalar> comp(nc);
            std::vector<Scalar> off;
            for (int i = 0; i < 3; ++i) {
                for (std::size_t c = 0; c < nc; ++c) {
                    comp[c] = u_[c][i];
                }
                a.off_diagonal_product(comp, off);
                for (std::size_t c = 0; c < nc; ++c) {
                    hbya[c][i] = (source[c][i] - off[c]) / a.diag[c];
                }
            }
            for (std::size_t c = 0; c < nc; ++c) {
                rau[c] = vol[c] / a.diag[c];
            }
            geo_.exchange(hbya);
            geo_.exchange(rau);
        }
        std::vector<Scalar> rau_f(nf);
        std::vector<Scalar> phi_hbya(nf, 0.0);
        for (std::size_t f = 0; f < nif; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const auto n = static_cast<std::size_t>(nei[f]);
            rau_f[f] = w[f] * rau[p] + (1.0 - w[f]) * rau[n];
            phi_hbya[f] = (w[f] * hbya[p] + (1.0 - w[f]) * hbya[n]).dot(sf[f]) + rau_f[f] * phig[f];
        }
        for (std::size_t f = nif; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            rau_f[f] = rau[p];
            const BoundaryCondition& c = bc(f - nif);
            switch (c.kind) {
            case FlowBoundary::VelocityInlet:
            case FlowBoundary::Wall:
                phi_hbya[f] = fixed_velocity_[f - nif].dot(sf[f]);
                break;
            case FlowBoundary::PressureOutlet:
                phi_hbya[f] = hbya[p].dot(sf[f]);
                break;
            case FlowBoundary::Symmetry:
            case FlowBoundary::Empty:
                phi_hbya[f] = 0.0;
                break;
            }
        }
        // Outlet: fixed static pressure p -> p_rgh = p - rho g.x.
        auto p_rgh_b = [&](std::size_t f) {
            const auto p = static_cast<std::size_t>(own[f]);
            return bc(f - nif).pressure - rho[p] * ghf_[f];
        };
        LduMatrix eq(m);
        eq.pattern = &geo_.csr_pattern();
        eq.halo = geo_.halo();
        for (std::size_t f = 0; f < nif; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const auto n = static_cast<std::size_t>(nei[f]);
            const Scalar coeff = rau_f[f] * dc[f];
            eq.upper[f] = -coeff;
            eq.lower[f] = -coeff;
            eq.diag[p] += coeff;
            eq.diag[n] += coeff;
            eq.source[p] -= phi_hbya[f];
            eq.source[n] += phi_hbya[f];
        }
        for (std::size_t f = nif; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            eq.source[p] -= phi_hbya[f];
            if (bc(f - nif).kind == FlowBoundary::PressureOutlet) {
                const Scalar coeff = rau_f[f] * dc[f];
                eq.diag[p] += coeff;
                eq.source[p] += coeff * p_rgh_b(f);
            }
        }
        if (!has_fixed_pressure_ && geo_.comm().is_root() && geo_.n_owned() > 0) {
            eq.set_reference(0, 0.0);
        }
        const LinearSolveResult r = eq.solve(p_rgh_, settings_.p_solver, amg.get());
        if (corr == 0) {
            res.p = r.initial_residual;
        }

        // Conservative flux and velocity correction.
        std::vector<Scalar> force(nf, 0.0); // (phig - grad p_rgh) . Sf
        for (std::size_t f = 0; f < nif; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const auto n = static_cast<std::size_t>(nei[f]);
            const Scalar dp = dc[f] * (p_rgh_[n] - p_rgh_[p]);
            phi_[f] = phi_hbya[f] - rau_f[f] * dp;
            force[f] = phig[f] - dp;
        }
        for (std::size_t f = nif; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            phi_[f] = phi_hbya[f];
            if (bc(f - nif).kind == FlowBoundary::PressureOutlet) {
                const Scalar dp = dc[f] * (p_rgh_b(f) - p_rgh_[p]);
                phi_[f] -= rau_f[f] * dp;
                force[f] = -dp;
            }
        }
        const std::vector<Vec3> g = reconstruct(force);
        for (std::size_t c = 0; c < nc; ++c) {
            u_[c] = hbya[c] + rau[c] * g[c];
        }
        geo_.exchange(u_);
    }

    // Continuity and alpha bounds.
    std::vector<Scalar> div(nc, 0.0);
    for (std::size_t f = 0; f < nf; ++f) {
        div[static_cast<std::size_t>(own[f])] += phi_[f];
        if (f < nif) {
            div[static_cast<std::size_t>(nei[f])] -= phi_[f];
        }
    }
    std::array<Scalar, 2> sums{0.0, 0.0};
    Scalar bound = 0.0;
    for (Label c = 0; c < geo_.n_owned(); ++c) {
        const auto cs = static_cast<std::size_t>(c);
        sums[0] += std::abs(div[cs]);
        sums[1] += vol[cs];
        bound = std::max(bound, std::max(alpha_[cs] - 1.0, -alpha_[cs]));
    }
    geo_.comm().sum(sums.data(), sums.size());
    res.continuity = sums[1] > 0.0 ? sums[0] / sums[1] : 0.0;
    res.t = geo_.comm().max(bound);
    return res;
}

} // namespace cfd::fv
