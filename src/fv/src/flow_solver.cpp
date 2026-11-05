#include <algorithm>
#include <array>
#include <cfd/fv/flow_solver.hpp>
#include <cmath>
#include <stdexcept>
#include <tuple>
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
    throw std::invalid_argument("FlowSolver: no boundary condition for patch '" + patch.name + "'");
}

} // namespace

FlowSolver::FlowSolver(const Mesh& mesh, std::map<std::string, BoundaryCondition> conditions,
                       FlowSettings settings, const parallel::Halo* halo)
    : geo_(mesh, halo), settings_(std::move(settings)) {
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
    // Pressure level fixed anywhere in the global domain?
    has_fixed_pressure_ = geo_.comm().max(has_fixed_pressure_ ? 1.0 : 0.0) > 0.5;
    for (const auto& [name, condition] : conditions) {
        if (mesh.find_patch(name) == nullptr) {
            throw std::invalid_argument("FlowSolver: unknown patch '" + name + "'");
        }
    }
    fixed_velocity_.resize(static_cast<std::size_t>(mesh.n_boundary_faces()));
    for (std::size_t b = 0; b < fixed_velocity_.size(); ++b) {
        fixed_velocity_[b] = bc(static_cast<Label>(b)).velocity;
    }
    check_mass_balance();
    const auto nc = static_cast<std::size_t>(mesh.n_cells());
    u_.assign(nc, Vec3::Zero());
    u_old_ = u_;
    p_.assign(nc, 0.0);
    compute_initial_flux();
}

void FlowSolver::check_mass_balance() const {
    if (has_fixed_pressure_) {
        return;
    }
    const Mesh& m = geo_.mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    Scalar net = 0.0;
    Scalar total = 0.0;
    for (std::size_t b = 0; b < fixed_velocity_.size(); ++b) {
        const FlowBoundary kind = bc(static_cast<Label>(b)).kind;
        if ((kind == FlowBoundary::VelocityInlet || kind == FlowBoundary::Wall) &&
            geo_.owned_face(nif + b)) {
            const Scalar flux = fixed_velocity_[b].dot(m.face_area_vectors()[nif + b]);
            net += flux;
            total += std::abs(flux);
        }
    }
    net = geo_.comm().sum(net);
    total = geo_.comm().sum(total);
    if (std::abs(net) > 1e-9 * std::max(total, 1e-300) && std::abs(net) > 1e-300) {
        throw std::invalid_argument(
            "FlowSolver: the prescribed inflow is not balanced by outflow and there is no "
            "pressure outlet (mass cannot be conserved); add a pressure_outlet boundary");
    }
}

const BoundaryCondition& FlowSolver::bc(Label boundary_face_index) const {
    const Label patch = geo_.boundary_patch()[static_cast<std::size_t>(boundary_face_index)];
    return patch_bcs_[static_cast<std::size_t>(patch)];
}

Vec3 FlowSolver::boundary_velocity(Label face) const {
    const Mesh& m = geo_.mesh();
    const Label bi = face - m.n_internal_faces();
    const BoundaryCondition& c = bc(bi);
    const Vec3& up = u_[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(face)])];
    switch (c.kind) {
    case FlowBoundary::VelocityInlet:
    case FlowBoundary::Wall:
        return fixed_velocity_[static_cast<std::size_t>(bi)];
    case FlowBoundary::PressureOutlet:
    case FlowBoundary::Empty:
        return up;
    case FlowBoundary::Symmetry:
        break;
    }
    const Vec3& n = geo_.unit_normals()[static_cast<std::size_t>(face)];
    return up - up.dot(n) * n;
}

Scalar FlowSolver::boundary_pressure(Label face) const {
    const Mesh& m = geo_.mesh();
    const BoundaryCondition& c = bc(face - m.n_internal_faces());
    if (c.kind == FlowBoundary::PressureOutlet) {
        return c.pressure;
    }
    return p_[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(face)])];
}

void FlowSolver::set_cell_acceleration(std::vector<Vec3> acceleration) {
    if (!acceleration.empty() && acceleration.size() != u_.size()) {
        throw std::invalid_argument("FlowSolver::set_cell_acceleration: size mismatch");
    }
    cell_acceleration_ = std::move(acceleration);
}

void FlowSolver::set_uniform_velocity(const Vec3& u) {
    std::fill(u_.begin(), u_.end(), u);
    u_old_ = u_;
    u_old_old_.clear();
    dt_old_ = 0.0;
    compute_initial_flux();
}

void FlowSolver::set_boundary_velocity_profile(const std::string& patch,
                                               std::vector<Vec3> velocities) {
    const Mesh& m = geo_.mesh();
    const mesh::Patch* pt = m.find_patch(patch);
    if (pt == nullptr) {
        throw std::invalid_argument("FlowSolver: unknown patch '" + patch + "'");
    }
    if (velocities.size() != static_cast<std::size_t>(pt->n_faces)) {
        throw std::invalid_argument("FlowSolver: velocity profile size mismatch for '" + patch +
                                    "'");
    }
    const auto start = static_cast<std::size_t>(pt->start_face - m.n_internal_faces());
    // Look the condition up by patch (the patch may have no faces on this process).
    const FlowBoundary kind = patch_bcs_[static_cast<std::size_t>(pt - m.patches().data())].kind;
    if (kind != FlowBoundary::VelocityInlet && kind != FlowBoundary::Wall) {
        throw std::invalid_argument("FlowSolver: velocity profile needs an inlet or wall patch");
    }
    std::copy(velocities.begin(), velocities.end(),
              fixed_velocity_.begin() + static_cast<std::ptrdiff_t>(start));
    check_mass_balance();
    compute_initial_flux();
}

void FlowSolver::set_state(std::vector<Vec3> u, std::vector<Scalar> p) {
    const auto nc = static_cast<std::size_t>(geo_.mesh().n_cells());
    if (u.size() != nc || p.size() != nc) {
        throw std::invalid_argument("FlowSolver::set_state: field size mismatch");
    }
    u_ = std::move(u);
    u_old_ = u_;
    u_old_old_.clear();
    dt_old_ = 0.0;
    p_ = std::move(p);
    compute_initial_flux();
}

void FlowSolver::set_state(std::vector<Vec3> u, std::vector<Scalar> p, std::vector<Scalar> phi) {
    if (phi.size() != static_cast<std::size_t>(geo_.mesh().n_faces())) {
        throw std::invalid_argument("FlowSolver::set_state: face flux size mismatch");
    }
    set_state(std::move(u), std::move(p));
    phi_ = std::move(phi);
}

void FlowSolver::compute_initial_flux() {
    const Mesh& m = geo_.mesh();
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_.weights();
    const auto& sf = m.face_area_vectors();
    phi_.assign(nf, 0.0);
    for (std::size_t f = 0; f < nif; ++f) {
        const Vec3& up = u_[static_cast<std::size_t>(m.owner()[f])];
        const Vec3& un = u_[static_cast<std::size_t>(m.neighbour()[f])];
        phi_[f] = (w[f] * up + (1.0 - w[f]) * un).dot(sf[f]);
    }
    for (std::size_t f = nif; f < nf; ++f) {
        const FlowBoundary kind = bc(static_cast<Label>(f - nif)).kind;
        phi_[f] = kind == FlowBoundary::Symmetry || kind == FlowBoundary::Empty
                      ? 0.0
                      : boundary_velocity(static_cast<Label>(f)).dot(sf[f]);
    }
}

std::vector<Mat3> FlowSolver::velocity_gradient() const {
    const Mesh& m = geo_.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_.weights();
    const auto& sf = m.face_area_vectors();
    std::vector<Mat3> grad(nc, Mat3::Zero());
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        const Mat3 flux = sf[f] * (w[f] * u_[p] + (1.0 - w[f]) * u_[n]).transpose();
        grad[p] += flux;
        grad[n] -= flux;
    }
    for (std::size_t f = nif; f < nf; ++f) {
        grad[static_cast<std::size_t>(m.owner()[f])] +=
            sf[f] * boundary_velocity(static_cast<Label>(f)).transpose();
    }
    const auto& vol = m.cell_volumes();
    for (std::size_t c = 0; c < nc; ++c) {
        grad[c] /= vol[c];
    }
    geo_.exchange(grad);
    return grad;
}

std::vector<Vec3> FlowSolver::pressure_gradient() const {
    const Mesh& m = geo_.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_.weights();
    const auto& sf = m.face_area_vectors();
    const auto& vol = m.cell_volumes();

    std::vector<Vec3> internal(nc, Vec3::Zero());
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        const Vec3 flux = sf[f] * (w[f] * p_[p] + (1.0 - w[f]) * p_[n]);
        internal[p] += flux;
        internal[n] -= flux;
    }

    // Two passes: zero-gradient boundary values are extrapolated tangentially with the
    // first-pass gradient, p_b = p_P + grad(p)_P . (d - (d.n) n), which removes the
    // O(1) gradient error of p_b = p_P on non-orthogonal boundary cells.
    std::vector<Vec3> grad(nc);
    std::vector<Vec3> previous;
    for (int pass = 0; pass < 2; ++pass) {
        grad = internal;
        for (std::size_t f = nif; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(m.owner()[f]);
            Scalar pb = boundary_pressure(static_cast<Label>(f));
            if (pass > 0 && bc(static_cast<Label>(f - nif)).kind != FlowBoundary::PressureOutlet) {
                const Vec3& nrm = geo_.unit_normals()[f];
                const Vec3 d = m.face_centres()[f] - m.cell_centres()[p];
                pb += previous[p].dot(d - d.dot(nrm) * nrm);
            }
            grad[p] += sf[f] * pb;
        }
        for (std::size_t c = 0; c < nc; ++c) {
            grad[c] /= vol[c];
        }
        previous = grad;
    }
    geo_.exchange(grad);
    return grad;
}

PatchForce FlowSolver::patch_force(const std::string& patch) const {
    const Mesh& m = geo_.mesh();
    const mesh::Patch* pp = m.find_patch(patch);
    if (pp == nullptr) {
        throw std::invalid_argument("patch_force: unknown patch '" + patch + "'");
    }
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& sf = m.face_area_vectors();
    const auto& cf = m.face_centres();
    const auto& cc = m.cell_centres();
    const auto& dc = geo_.delta_coeffs();
    const std::vector<Vec3> grad_p = pressure_gradient();
    const std::vector<Scalar>* nut_b =
        turbulence_ != nullptr ? &turbulence_->boundary_nut() : nullptr;
    PatchForce force;
    for (Label i = 0; i < pp->n_faces; ++i) {
        const auto f = static_cast<std::size_t>(pp->start_face + i);
        const auto o = static_cast<std::size_t>(m.owner()[f]);
        if (!geo_.owned_face(f)) {
            continue;
        }
        const BoundaryCondition& c = bc(static_cast<Label>(f - nif));
        const Scalar p_f = c.kind == FlowBoundary::PressureOutlet
                               ? c.pressure
                               : p_[o] + grad_p[o].dot(cf[f] - cc[o]);
        force.pressure += p_f * sf[f];
        if (c.kind == FlowBoundary::Wall || c.kind == FlowBoundary::VelocityInlet) {
            const Scalar nu_f = settings_.nu + (nut_b != nullptr ? (*nut_b)[f - nif] : 0.0);
            force.viscous += nu_f * dc[f] * (u_[o] - fixed_velocity_[f - nif]);
        }
    }
    std::array<Scalar, 6> sums{force.pressure.x(), force.pressure.y(), force.pressure.z(),
                               force.viscous.x(),  force.viscous.y(),  force.viscous.z()};
    geo_.comm().sum(sums.data(), sums.size());
    force.pressure = Vec3(sums[0], sums[1], sums[2]);
    force.viscous = Vec3(sums[3], sums[4], sums[5]);
    return force;
}

Scalar FlowSolver::continuity_error() const {
    const Mesh& m = geo_.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    std::vector<Scalar> div(nc, 0.0);
    for (std::size_t f = 0; f < nf; ++f) {
        div[static_cast<std::size_t>(m.owner()[f])] += phi_[f];
        if (f < nif) {
            div[static_cast<std::size_t>(m.neighbour()[f])] -= phi_[f];
        }
    }
    std::array<Scalar, 2> sums{0.0, 0.0};
    for (std::size_t c = 0; c < static_cast<std::size_t>(geo_.n_owned()); ++c) {
        sums[0] += std::abs(div[c]);
        sums[1] += m.cell_volumes()[c];
    }
    geo_.comm().sum(sums.data(), sums.size());
    return sums[1] > 0.0 ? sums[0] / sums[1] : 0.0;
}

void FlowSolver::assemble_momentum(MomentumSystem& sys, Scalar dt, bool relax) const {
    const Mesh& m = geo_.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& own = m.owner();
    const auto& nei = m.neighbour();
    const auto& w = geo_.weights();
    const auto& dc = geo_.delta_coeffs();
    const auto& k = geo_.corrections();
    const auto& cf = m.face_centres();
    const auto& cc = m.cell_centres();
    const auto& vol = m.cell_volumes();
    const Scalar nu = settings_.nu;
    const bool linear_upwind = settings_.convection != ConvectionScheme::Upwind;
    const bool limited = settings_.convection == ConvectionScheme::LimitedLinearUpwind;
    // Constant-step BDF2 once two old levels exist (Euler otherwise).
    const bool bdf2 = dt > 0.0 && settings_.time_scheme == TimeScheme::Backward &&
                      u_old_old_.size() == u_.size() && std::abs(dt - dt_old_) <= 1e-12 * dt;

    LduMatrix& a = sys.base;
    a.set_zero();
    sys.diag.assign(nc, Vec3::Zero());
    sys.source.assign(nc, Vec3::Zero());

    const std::vector<Mat3> grad = velocity_gradient();
    const auto& sf = m.face_area_vectors();

    // Barth-Jespersen limiter per cell and component: the reconstructed face values must
    // stay within the min/max of the cell and its face neighbours (boundary values included).
    std::vector<Vec3> limiter;
    if (settings_.convection == ConvectionScheme::LimitedLinearUpwind) {
        std::vector<Vec3> u_min(u_);
        std::vector<Vec3> u_max(u_);
        for (std::size_t f = 0; f < nif; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const auto n = static_cast<std::size_t>(nei[f]);
            u_min[p] = u_min[p].cwiseMin(u_[n]);
            u_max[p] = u_max[p].cwiseMax(u_[n]);
            u_min[n] = u_min[n].cwiseMin(u_[p]);
            u_max[n] = u_max[n].cwiseMax(u_[p]);
        }
        for (std::size_t f = nif; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            const Vec3 ub = boundary_velocity(static_cast<Label>(f));
            u_min[p] = u_min[p].cwiseMin(ub);
            u_max[p] = u_max[p].cwiseMax(ub);
        }
        limiter.assign(nc, Vec3::Ones());
        auto limit = [&](std::size_t c, std::size_t f) {
            const Vec3 delta = grad[c].transpose() * Vec3(cf[f] - cc[c]);
            for (int i = 0; i < 3; ++i) {
                const Scalar d = delta[i];
                if (std::abs(d) < 1e-300) {
                    continue;
                }
                const Scalar bound = d > 0.0 ? u_max[c][i] - u_[c][i] : u_min[c][i] - u_[c][i];
                limiter[c][i] = std::min(limiter[c][i], std::max(bound / d, 0.0));
            }
        };
        for (std::size_t f = 0; f < nf; ++f) {
            limit(static_cast<std::size_t>(own[f]), f);
            if (f < nif) {
                limit(static_cast<std::size_t>(nei[f]), f);
            }
        }
        geo_.exchange(limiter);
    }
    const std::vector<Scalar>* nut = turbulence_ != nullptr ? &turbulence_->nut() : nullptr;
    const std::vector<Scalar>* nut_b =
        turbulence_ != nullptr ? &turbulence_->boundary_nut() : nullptr;

    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const auto n = static_cast<std::size_t>(nei[f]);
        const Scalar flux = phi_[f];
        const Scalar nu_f = nut != nullptr ? nu + w[f] * (*nut)[p] + (1.0 - w[f]) * (*nut)[n] : nu;
        const Scalar diff = nu_f * dc[f];

        a.upper[f] = std::min(flux, 0.0) - diff;
        a.lower[f] = -std::max(flux, 0.0) - diff;
        a.diag[p] += std::max(flux, 0.0) + diff;
        a.diag[n] += -std::min(flux, 0.0) + diff;

        Vec3 explicit_flux = Vec3::Zero();
        if (linear_upwind) {
            const std::size_t up = flux >= 0.0 ? p : n;
            const Vec3 r = cf[f] - cc[up];
            const Vec3 delta = grad[up].transpose() * r;
            explicit_flux -= flux * (limited ? Vec3(limiter[up].cwiseProduct(delta)) : delta);
        }
        const Mat3 grad_f = w[f] * grad[p] + (1.0 - w[f]) * grad[n];
        explicit_flux += nu_f * (grad_f.transpose() * k[f]);
        if (nut != nullptr) {
            // div(nu_eff (grad U)^T): vanishes for constant viscosity.
            explicit_flux += nu_f * (grad_f * sf[f]);
        }

        sys.source[p] += explicit_flux;
        sys.source[n] -= explicit_flux;
    }

    for (std::size_t f = nif; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const Scalar flux = phi_[f];
        const Scalar nu_f = nut_b != nullptr ? nu + (*nut_b)[f - nif] : nu;
        const Scalar diff = nu_f * dc[f];
        const BoundaryCondition& c = bc(static_cast<Label>(f - nif));
        if (nut != nullptr && c.kind != FlowBoundary::Empty) {
            sys.source[p] += nu_f * (grad[p] * sf[f]);
        }
        switch (c.kind) {
        case FlowBoundary::VelocityInlet:
        case FlowBoundary::Wall:
            a.diag[p] += diff + std::max(flux, 0.0);
            sys.source[p] += (diff - std::min(flux, 0.0)) * fixed_velocity_[f - nif];
            break;
        case FlowBoundary::PressureOutlet:
            a.diag[p] += flux;
            break;
        case FlowBoundary::Empty:
            break;
        case FlowBoundary::Symmetry:
            // Shared implicit diagonal with the explicit mirrored value U_P - (U_P.n)n keeps
            // the diagonal identical for all components (consistent rAU).
            a.diag[p] += diff;
            sys.source[p] += diff * boundary_velocity(static_cast<Label>(f));
            break;
        }
    }

    for (std::size_t c = 0; c < nc; ++c) {
        sys.source[c] += settings_.body_force * vol[c];
        if (!cell_acceleration_.empty()) {
            sys.source[c] += cell_acceleration_[c] * vol[c];
        }
        if (dt > 0.0) {
            const Scalar ddt = vol[c] / dt;
            if (bdf2) {
                a.diag[c] += 1.5 * ddt;
                sys.source[c] += ddt * (2.0 * u_old_[c] - 0.5 * u_old_old_[c]);
            } else {
                a.diag[c] += ddt;
                sys.source[c] += ddt * u_old_[c];
            }
        }
        sys.diag[c] += Vec3::Constant(a.diag[c]);
        if (relax) {
            const Scalar alpha = settings_.alpha_u;
            for (int i = 0; i < 3; ++i) {
                const Scalar d0 = sys.diag[c][i];
                const Scalar d1 = d0 / alpha;
                sys.diag[c][i] = d1;
                sys.source[c][i] += (d1 - d0) * u_[c][i];
            }
        }
    }
}

Vec3 FlowSolver::solve_momentum(MomentumSystem& sys) {
    const Mesh& m = geo_.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const std::vector<Vec3> grad_p = pressure_gradient();
    const auto& vol = m.cell_volumes();

    // Residuals of all components are normalised with the largest component norm so
    // that a (near-)zero component (e.g. w in a 2D case) does not report noise.
    std::array<LduMatrix, 3> eqs{sys.base, sys.base, sys.base};
    std::array<std::vector<Scalar>, 3> x;
    Vec3 res = Vec3::Zero();
    Scalar norm = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        LduMatrix& eq = eqs[i];
        x[i].resize(nc);
        for (std::size_t c = 0; c < nc; ++c) {
            eq.diag[c] = sys.diag[c][static_cast<Eigen::Index>(i)];
            eq.source[c] = sys.source[c][static_cast<Eigen::Index>(i)] -
                           grad_p[c][static_cast<Eigen::Index>(i)] * vol[c];
            x[i][c] = u_[c][static_cast<Eigen::Index>(i)];
        }
        const auto [r, n] = eq.residual_norms(x[i]);
        res[static_cast<Eigen::Index>(i)] = r;
        norm = std::max(norm, n);
    }
    const Vec3 residual = res / (norm + 1e-20);
    for (std::size_t i = 0; i < 3; ++i) {
        if (residual[static_cast<Eigen::Index>(i)] < settings_.u_solver.tolerance) {
            continue;
        }
        (void)eqs[i].solve(x[i], settings_.u_solver);
        for (std::size_t c = 0; c < nc; ++c) {
            u_[c][static_cast<Eigen::Index>(i)] = x[i][c];
        }
    }
    return residual;
}

Scalar FlowSolver::pressure_correction(const MomentumSystem& sys, bool relax_p) {
    const Mesh& m = geo_.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& own = m.owner();
    const auto& nei = m.neighbour();
    const auto& w = geo_.weights();
    const auto& dc = geo_.delta_coeffs();
    const auto& k = geo_.corrections();
    const auto& sf = m.face_area_vectors();
    const auto& vol = m.cell_volumes();

    // H(U) / A and rAU = V / A.
    std::vector<Vec3> hbya(nc);
    std::vector<Scalar> rau(nc);
    {
        std::vector<Scalar> comp(nc);
        std::vector<Scalar> off;
        for (int i = 0; i < 3; ++i) {
            for (std::size_t c = 0; c < nc; ++c) {
                comp[c] = u_[c][i];
            }
            sys.base.off_diagonal_product(comp, off);
            for (std::size_t c = 0; c < nc; ++c) {
                hbya[c][i] = (sys.source[c][i] - off[c]) / sys.diag[c][i];
            }
        }
        for (std::size_t c = 0; c < nc; ++c) {
            rau[c] = vol[c] / sys.diag[c].mean();
        }
        geo_.exchange(hbya);
        geo_.exchange(rau);
    }

    // Predicted face flux.
    std::vector<Scalar> phi_hbya(nf, 0.0);
    std::vector<Scalar> rau_f(nf, 0.0);
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const auto n = static_cast<std::size_t>(nei[f]);
        phi_hbya[f] = (w[f] * hbya[p] + (1.0 - w[f]) * hbya[n]).dot(sf[f]);
        rau_f[f] = w[f] * rau[p] + (1.0 - w[f]) * rau[n];
    }
    for (std::size_t f = nif; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        rau_f[f] = rau[p];
        const BoundaryCondition& c = bc(static_cast<Label>(f - nif));
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

    // Pressure equation: -div(rAU grad p) = -div(phiHbyA).
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
        const BoundaryCondition& c = bc(static_cast<Label>(f - nif));
        if (c.kind == FlowBoundary::PressureOutlet) {
            const Scalar coeff = rau_f[f] * dc[f];
            eq.diag[p] += coeff;
            eq.source[p] += coeff * c.pressure;
        }
    }
    if (!has_fixed_pressure_ && geo_.comm().is_root() && geo_.n_owned() > 0) {
        eq.set_reference(0, 0.0);
    }

    const std::vector<Scalar> base_source = eq.source;
    const std::vector<Scalar> p_prev = p_;
    std::vector<Scalar> p_new = p_;
    Scalar initial_residual = 0.0;
    std::vector<Scalar> non_orth(nif, 0.0);

    for (int corr = 0; corr <= settings_.n_non_orth_correctors; ++corr) {
        std::fill(non_orth.begin(), non_orth.end(), 0.0);
        eq.source = base_source;
        if (corr > 0 || settings_.n_non_orth_correctors > 0) {
            p_ = p_new;
            const std::vector<Vec3> gp = pressure_gradient();
            for (std::size_t f = 0; f < nif; ++f) {
                const auto p = static_cast<std::size_t>(own[f]);
                const auto n = static_cast<std::size_t>(nei[f]);
                non_orth[f] = rau_f[f] * k[f].dot(w[f] * gp[p] + (1.0 - w[f]) * gp[n]);
                eq.source[p] += non_orth[f];
                eq.source[n] -= non_orth[f];
            }
        }
        const LinearSolveResult r = eq.solve(p_new, settings_.p_solver, p_amg_.get());
        if (corr == 0) {
            initial_residual = r.initial_residual;
        }
    }

    // Conservative face flux from the (unrelaxed) pressure solution.
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const auto n = static_cast<std::size_t>(nei[f]);
        phi_[f] = phi_hbya[f] - rau_f[f] * dc[f] * (p_new[n] - p_new[p]) - non_orth[f];
    }
    for (std::size_t f = nif; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        phi_[f] = phi_hbya[f];
        const BoundaryCondition& c = bc(static_cast<Label>(f - nif));
        if (c.kind == FlowBoundary::PressureOutlet) {
            phi_[f] -= rau_f[f] * dc[f] * (c.pressure - p_new[p]);
        }
    }

    // Pressure relaxation and velocity correction.
    for (std::size_t c = 0; c < nc; ++c) {
        p_[c] = relax_p ? p_prev[c] + settings_.alpha_p * (p_new[c] - p_prev[c]) : p_new[c];
    }
    const std::vector<Vec3> gp = pressure_gradient();
    for (std::size_t c = 0; c < nc; ++c) {
        u_[c] = hbya[c] - rau[c] * gp[c];
    }
    return initial_residual;
}

Residuals FlowSolver::simple_iteration() {
    MomentumSystem sys{LduMatrix(geo_.mesh()), {}, {}};
    sys.base.pattern = &geo_.csr_pattern();
    sys.base.halo = geo_.halo();
    assemble_momentum(sys, 0.0, true);
    Residuals res;
    res.u = solve_momentum(sys);
    res.p = pressure_correction(sys, true);
    if (turbulence_ != nullptr) {
        std::tie(res.k, res.omega) = turbulence_->correct(*this, 0.0);
    }
    res.continuity = continuity_error();
    history_.push_back(res);
    return res;
}

int FlowSolver::solve_steady(Scalar tolerance, int max_iterations) {
    for (int it = 1; it <= max_iterations; ++it) {
        if (simple_iteration().max() < tolerance) {
            return it;
        }
    }
    return max_iterations;
}

Residuals FlowSolver::piso_step(Scalar dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("FlowSolver::piso_step: dt must be positive");
    }
    if (settings_.time_scheme == TimeScheme::Backward && dt_old_ > 0.0) {
        u_old_old_ = u_old_;
    }
    u_old_ = u_;
    MomentumSystem sys{LduMatrix(geo_.mesh()), {}, {}};
    sys.base.pattern = &geo_.csr_pattern();
    sys.base.halo = geo_.halo();
    assemble_momentum(sys, dt, false);
    Residuals res;
    res.u = solve_momentum(sys);
    for (int corr = 0; corr < settings_.n_piso_correctors; ++corr) {
        const Scalar r = pressure_correction(sys, false);
        if (corr == 0) {
            res.p = r;
        }
    }
    if (turbulence_ != nullptr) {
        std::tie(res.k, res.omega) = turbulence_->correct(*this, dt);
    }
    res.continuity = continuity_error();
    dt_old_ = dt;
    history_.push_back(res);
    return res;
}

} // namespace cfd::fv
