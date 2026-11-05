#include <algorithm>
#include <array>
#include <cfd/fv/compressible.hpp>
#include <cmath>
#include <stdexcept>

namespace cfd::fv {

namespace {

using Mat3 = Eigen::Matrix3d;
using Conserved = CompressibleSolver::Conserved;
constexpr int k_vars = 5; // rho, u, v, w, p

std::array<Scalar, k_vars> pack(const GasState& s) {
    return {s.rho, s.u.x(), s.u.y(), s.u.z(), s.p};
}

GasState unpack(const std::array<Scalar, k_vars>& v) {
    return {v[0], Vec3(v[1], v[2], v[3]), v[4]};
}

// HLLC flux through a face with unit normal n (per unit area).
Conserved hllc(const GasState& l, const GasState& r, const Vec3& n, Scalar gamma) {
    auto energy = [&](const GasState& s) {
        return s.p / (gamma - 1.0) + 0.5 * s.rho * s.u.squaredNorm();
    };
    auto flux = [&](const GasState& s, Scalar un) {
        Conserved f;
        const Scalar e = energy(s);
        f[0] = s.rho * un;
        f.segment<3>(1) = s.rho * un * s.u + s.p * n;
        f[4] = (e + s.p) * un;
        return f;
    };
    auto conserved = [&](const GasState& s) {
        Conserved q;
        q[0] = s.rho;
        q.segment<3>(1) = s.rho * s.u;
        q[4] = energy(s);
        return q;
    };
    const Scalar unl = l.u.dot(n);
    const Scalar unr = r.u.dot(n);
    const Scalar cl = std::sqrt(gamma * l.p / l.rho);
    const Scalar cr = std::sqrt(gamma * r.p / r.rho);
    const Scalar sl = std::min(unl - cl, unr - cr);
    const Scalar sr = std::max(unl + cl, unr + cr);
    if (sl >= 0.0) {
        return flux(l, unl);
    }
    if (sr <= 0.0) {
        return flux(r, unr);
    }
    const Scalar s_star = (r.p - l.p + l.rho * unl * (sl - unl) - r.rho * unr * (sr - unr)) /
                          (l.rho * (sl - unl) - r.rho * (sr - unr));
    auto star = [&](const GasState& s, Scalar un, Scalar sk) {
        const Scalar factor = s.rho * (sk - un) / (sk - s_star);
        Conserved q;
        q[0] = factor;
        q.segment<3>(1) = factor * (s.u + (s_star - un) * n);
        q[4] = factor * (energy(s) / s.rho + (s_star - un) * (s_star + s.p / (s.rho * (sk - un))));
        return q;
    };
    if (s_star >= 0.0) {
        return flux(l, unl) + sl * (star(l, unl, sl) - conserved(l));
    }
    return flux(r, unr) + sr * (star(r, unr, sr) - conserved(r));
}

} // namespace

CompressibleSolver::CompressibleSolver(const Mesh& mesh,
                                       std::map<std::string, BoundaryCondition> conditions,
                                       CompressibleSettings settings, const parallel::Halo* halo)
    : geo_(mesh, halo), settings_(settings) {
    for (std::size_t i = 0; i < mesh.patches().size(); ++i) {
        const auto& patch = mesh.patches()[i];
        if (halo != nullptr && static_cast<int>(i) == halo->halo_patch) {
            patch_bcs_.push_back({FlowBoundary::Empty, Vec3::Zero(), 0.0});
            continue;
        }
        const auto it = conditions.find(patch.name);
        if (it != conditions.end()) {
            patch_bcs_.push_back(it->second);
        } else if (patch.type == mesh::PatchType::Wall) {
            patch_bcs_.push_back({FlowBoundary::Wall, Vec3::Zero(), 0.0});
        } else if (patch.type == mesh::PatchType::Symmetry) {
            patch_bcs_.push_back({FlowBoundary::Symmetry, Vec3::Zero(), 0.0});
        } else if (patch.type == mesh::PatchType::Outlet) {
            patch_bcs_.push_back({FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0});
        } else {
            throw std::invalid_argument("CompressibleSolver: no boundary condition for patch '" +
                                        patch.name + "'");
        }
        const BoundaryCondition& c = patch_bcs_.back();
        if (c.kind == FlowBoundary::VelocityInlet && !(c.pressure > 0.0 && c.temperature > 0.0)) {
            throw std::invalid_argument("CompressibleSolver: inlet '" + patch.name +
                                        "' needs pressure and temperature > 0");
        }
    }
    for (const auto& [name, c] : conditions) {
        if (mesh.find_patch(name) == nullptr) {
            throw std::invalid_argument("CompressibleSolver: unknown patch '" + name + "'");
        }
    }
    q_.assign(static_cast<std::size_t>(mesh.n_cells()), to_conserved(GasState{}));
}

Conserved CompressibleSolver::to_conserved(const GasState& s) const {
    Conserved q;
    q[0] = s.rho;
    q.segment<3>(1) = s.rho * s.u;
    q[4] = s.p / (settings_.gamma - 1.0) + 0.5 * s.rho * s.u.squaredNorm();
    return q;
}

GasState CompressibleSolver::to_primitive(const Conserved& q) const {
    GasState s;
    s.rho = q[0];
    s.u = q.segment<3>(1) / q[0];
    s.p = (settings_.gamma - 1.0) * (q[4] - 0.5 * q[0] * s.u.squaredNorm());
    return s;
}

void CompressibleSolver::set_uniform(const GasState& state) {
    std::fill(q_.begin(), q_.end(), to_conserved(state));
}

void CompressibleSolver::set_state(const std::vector<GasState>& states) {
    if (states.size() != q_.size()) {
        throw std::invalid_argument("CompressibleSolver::set_state: size mismatch");
    }
    for (std::size_t c = 0; c < q_.size(); ++c) {
        q_[c] = to_conserved(states[c]);
    }
    geo_.exchange(q_);
}

void CompressibleSolver::set_conserved(std::vector<Conserved> q) {
    if (q.size() != q_.size()) {
        throw std::invalid_argument("CompressibleSolver::set_conserved: size mismatch");
    }
    q_ = std::move(q);
    geo_.exchange(q_);
}

const BoundaryCondition& CompressibleSolver::bc(std::size_t boundary_face) const {
    return patch_bcs_[static_cast<std::size_t>(geo_.boundary_patch()[boundary_face])];
}

GasState CompressibleSolver::ghost(std::size_t face, const GasState& s) const {
    const auto nif = static_cast<std::size_t>(geo_.mesh().n_internal_faces());
    const BoundaryCondition& c = bc(face - nif);
    const Vec3& n = geo_.unit_normals()[face];
    switch (c.kind) {
    case FlowBoundary::VelocityInlet:
        return {c.pressure / (settings_.r_gas * c.temperature), c.velocity, c.pressure};
    case FlowBoundary::PressureOutlet: {
        const Scalar un = s.u.dot(n);
        const Scalar a = std::sqrt(settings_.gamma * s.p / s.rho);
        if (c.pressure <= 0.0 || un >= a) {
            return s; // supersonic outflow: extrapolate everything
        }
        return {s.rho, s.u, c.pressure};
    }
    case FlowBoundary::Wall:
    case FlowBoundary::Symmetry:
    case FlowBoundary::Empty:
        break;
    }
    return {s.rho, s.u - 2.0 * s.u.dot(n) * n, s.p}; // mirror: zero mass flux
}

std::vector<Conserved> CompressibleSolver::residual(const std::vector<Conserved>& q) const {
    const Mesh& m = geo_.mesh();
    const auto nc = q.size();
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& own = m.owner();
    const auto& nei = m.neighbour();
    const auto& w = geo_.weights();
    const auto& sf = m.face_area_vectors();
    const auto& cf = m.face_centres();
    const auto& cc = m.cell_centres();
    const auto& vol = m.cell_volumes();
    const Scalar gamma = settings_.gamma;

    std::vector<std::array<Scalar, k_vars>> prim(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        prim[c] = pack(to_primitive(q[c]));
    }
    std::vector<std::array<Scalar, k_vars>> ghosts(nf - nif);
    for (std::size_t f = nif; f < nf; ++f) {
        ghosts[f - nif] = pack(ghost(f, unpack(prim[static_cast<std::size_t>(own[f])])));
    }

    // Green-Gauss gradients of the primitive variables and Barth-Jespersen limiters.
    using Grad = Eigen::Matrix<Scalar, 3, k_vars>;
    std::vector<Grad> grad(nc, Grad::Zero());
    std::vector<std::array<Scalar, k_vars>> limiter(nc);
    const bool muscl = settings_.second_order;
    const bool viscous = settings_.mu > 0.0;
    if (muscl || viscous) {
        for (std::size_t f = 0; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            if (f >= nif && bc(f - nif).kind == FlowBoundary::Empty) {
                continue; // zero-gradient direction
            }
            for (int v = 0; v < k_vars; ++v) {
                const auto vs = static_cast<std::size_t>(v);
                const Scalar face_value =
                    f < nif ? w[f] * prim[p][vs] +
                                  (1.0 - w[f]) * prim[static_cast<std::size_t>(nei[f])][vs]
                            : 0.5 * (prim[p][vs] + ghosts[f - nif][vs]);
                grad[p].col(v) += sf[f] * face_value;
                if (f < nif) {
                    grad[static_cast<std::size_t>(nei[f])].col(v) -= sf[f] * face_value;
                }
            }
        }
        for (std::size_t c = 0; c < nc; ++c) {
            grad[c] /= vol[c];
        }
        geo_.exchange(grad);
    }
    if (muscl) {
        std::vector<std::array<Scalar, k_vars>> lo(prim);
        std::vector<std::array<Scalar, k_vars>> hi(prim);
        auto extend = [&](std::size_t c, const std::array<Scalar, k_vars>& v) {
            for (std::size_t k = 0; k < k_vars; ++k) {
                lo[c][k] = std::min(lo[c][k], v[k]);
                hi[c][k] = std::max(hi[c][k], v[k]);
            }
        };
        for (std::size_t f = 0; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(own[f]);
            if (f < nif) {
                const auto n = static_cast<std::size_t>(nei[f]);
                extend(p, prim[n]);
                extend(n, prim[p]);
            } else {
                extend(p, ghosts[f - nif]);
            }
        }
        for (std::size_t c = 0; c < nc; ++c) {
            limiter[c].fill(1.0);
        }
        auto limit = [&](std::size_t c, std::size_t f) {
            const Vec3 d = cf[f] - cc[c];
            for (int v = 0; v < k_vars; ++v) {
                const auto vs = static_cast<std::size_t>(v);
                const Scalar delta = grad[c].col(v).dot(d);
                if (std::abs(delta) < 1e-300) {
                    continue;
                }
                const Scalar bound =
                    delta > 0.0 ? hi[c][vs] - prim[c][vs] : lo[c][vs] - prim[c][vs];
                limiter[c][vs] = std::min(limiter[c][vs], std::max(bound / delta, 0.0));
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
    auto reconstruct = [&](std::size_t c, std::size_t f) {
        if (!muscl) {
            return unpack(prim[c]);
        }
        std::array<Scalar, k_vars> v = prim[c];
        const Vec3 d = cf[f] - cc[c];
        for (int k = 0; k < k_vars; ++k) {
            const auto ks = static_cast<std::size_t>(k);
            v[ks] += limiter[c][ks] * grad[c].col(k).dot(d);
        }
        GasState s = unpack(v);
        if (s.rho <= 0.0 || s.p <= 0.0) {
            s = unpack(prim[c]); // positivity fallback
        }
        return s;
    };

    // Temperature gradient for the heat flux.
    const Scalar cp = gamma * settings_.r_gas / (gamma - 1.0);
    const Scalar k_heat = settings_.mu * cp / settings_.prandtl;
    auto temperature = [&](const std::array<Scalar, k_vars>& v) {
        return v[4] / (v[0] * settings_.r_gas);
    };

    std::vector<Conserved> r(nc, Conserved::Zero());
    for (std::size_t f = 0; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const BoundaryCondition* c = f >= nif ? &bc(f - nif) : nullptr;
        if (c != nullptr && c->kind == FlowBoundary::Empty) {
            continue;
        }
        const Vec3 n = geo_.unit_normals()[f];
        const Scalar area = geo_.face_areas()[f];
        const GasState left = reconstruct(p, f);
        const GasState right =
            f < nif ? reconstruct(static_cast<std::size_t>(nei[f]), f) : ghost(f, left);
        Conserved flux = hllc(left, right, n, gamma) * area;

        if (viscous) {
            // Face gradients: average with a correction along the cell-to-cell direction.
            const std::array<Scalar, k_vars>& vp = prim[p];
            std::array<Scalar, k_vars> vn{};
            Vec3 d;
            Grad g;
            bool wall = false;
            if (f < nif) {
                const auto nn = static_cast<std::size_t>(nei[f]);
                vn = prim[nn];
                d = cc[nn] - cc[p];
                g = w[f] * grad[p] + (1.0 - w[f]) * grad[nn];
            } else {
                wall = c->kind == FlowBoundary::Wall;
                if (!wall) {
                    r[p] -= flux;
                    continue; // viscous fluxes only on walls and internal faces
                }
                vn = vp;
                vn[1] = c->velocity.x();
                vn[2] = c->velocity.y();
                vn[3] = c->velocity.z();
                d = cf[f] - cc[p];
                g = grad[p];
            }
            const Scalar d2 = d.squaredNorm();
            Mat3 gu;
            for (int i = 0; i < 3; ++i) {
                const auto is = static_cast<std::size_t>(i + 1);
                const Vec3 gi = g.col(i + 1);
                gu.col(i) = gi + ((vn[is] - vp[is]) - gi.dot(d)) * d / d2; // d(u_i)/dx_j
            }
            const Mat3 grad_u = gu.transpose(); // grad_u(i, j) = du_i / dx_j
            const Mat3 tau = settings_.mu * (grad_u + grad_u.transpose() -
                                             (2.0 / 3.0) * grad_u.trace() * Mat3::Identity());
            const Vec3 u_face =
                f < nif ? Vec3(0.5 * (Vec3(vp[1], vp[2], vp[3]) + Vec3(vn[1], vn[2], vn[3])))
                        : Vec3(vn[1], vn[2], vn[3]);
            Scalar heat = 0.0;
            if (!wall) { // adiabatic walls
                const Scalar tp = temperature(vp);
                const Scalar tn = temperature(vn);
                heat = k_heat * (tn - tp) / d.norm() * area; // along d (approx. normal)
            }
            Conserved fv = Conserved::Zero();
            fv.segment<3>(1) = tau * n * area;
            fv[4] = u_face.dot(tau * n) * area + heat;
            flux -= fv;
        }
        r[p] -= flux;
        if (f < nif) {
            r[static_cast<std::size_t>(nei[f])] += flux;
        }
    }
    return r;
}

std::vector<Scalar> CompressibleSolver::local_dt(const std::vector<Conserved>& q) const {
    const Mesh& m = geo_.mesh();
    const auto nc = q.size();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    std::vector<Scalar> lambda(nc, 0.0);
    std::vector<Scalar> visc(nc, 0.0);
    std::vector<GasState> s(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        s[c] = to_primitive(q[c]);
    }
    for (std::size_t f = 0; f < static_cast<std::size_t>(m.n_faces()); ++f) {
        if (f >= nif && bc(f - nif).kind == FlowBoundary::Empty) {
            continue;
        }
        const Scalar area = geo_.face_areas()[f];
        const Vec3& n = geo_.unit_normals()[f];
        auto add = [&](std::size_t c) {
            const Scalar a = std::sqrt(settings_.gamma * s[c].p / s[c].rho);
            lambda[c] += (std::abs(s[c].u.dot(n)) + a) * area;
            visc[c] += area * area;
        };
        add(static_cast<std::size_t>(m.owner()[f]));
        if (f < nif) {
            add(static_cast<std::size_t>(m.neighbour()[f]));
        }
    }
    std::vector<Scalar> dt(nc);
    const Scalar factor = std::max(4.0 / 3.0, settings_.gamma / settings_.prandtl);
    for (std::size_t c = 0; c < nc; ++c) {
        const Scalar v = m.cell_volumes()[c];
        const Scalar lv = settings_.mu > 0.0 ? factor * settings_.mu / s[c].rho * visc[c] / v : 0.0;
        dt[c] = settings_.cfl * v / (lambda[c] + 4.0 * lv);
    }
    return dt;
}

Scalar CompressibleSolver::stable_dt() const {
    const std::vector<Scalar> dt = local_dt(q_);
    Scalar d = std::numeric_limits<Scalar>::max();
    for (Label c = 0; c < geo_.n_owned(); ++c) {
        d = std::min(d, dt[static_cast<std::size_t>(c)]);
    }
    return geo_.comm().min(d);
}

namespace {

template <typename DtFn>
Scalar rk3(std::vector<Conserved>& q, const std::vector<Scalar>& vol, Label n_owned,
           const parallel::Comm& comm, DtFn&& dt_of, const auto& residual, const auto& exchange) {
    const std::vector<Conserved> q0 = q;
    std::vector<Conserved> r = residual(q);
    // Density residual from the first stage.
    Scalar sum = 0.0;
    for (Label c = 0; c < n_owned; ++c) {
        const auto cs = static_cast<std::size_t>(c);
        sum += std::pow(r[cs][0] / vol[cs], 2);
    }
    std::array<Scalar, 2> sums{sum, static_cast<Scalar>(n_owned)};
    comm.sum(sums.data(), sums.size());
    for (std::size_t c = 0; c < q.size(); ++c) {
        q[c] = q0[c] + dt_of(c) / vol[c] * r[c];
    }
    exchange(q);
    r = residual(q);
    for (std::size_t c = 0; c < q.size(); ++c) {
        q[c] = 0.75 * q0[c] + 0.25 * (q[c] + dt_of(c) / vol[c] * r[c]);
    }
    exchange(q);
    r = residual(q);
    for (std::size_t c = 0; c < q.size(); ++c) {
        q[c] = (1.0 / 3.0) * q0[c] + (2.0 / 3.0) * (q[c] + dt_of(c) / vol[c] * r[c]);
    }
    exchange(q);
    return sums[1] > 0.0 ? std::sqrt(sums[0] / sums[1]) : 0.0;
}

} // namespace

Scalar CompressibleSolver::step(Scalar dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("CompressibleSolver::step: dt must be positive");
    }
    return rk3(
        q_, geo_.mesh().cell_volumes(), geo_.n_owned(), geo_.comm(),
        [dt](std::size_t) { return dt; },
        [this](const std::vector<Conserved>& q) { return residual(q); },
        [this](std::vector<Conserved>& q) { geo_.exchange(q); });
}

Scalar CompressibleSolver::steady_iteration() {
    const std::vector<Scalar> dt = local_dt(q_);
    return rk3(
        q_, geo_.mesh().cell_volumes(), geo_.n_owned(), geo_.comm(),
        [&dt](std::size_t c) { return dt[c]; },
        [this](const std::vector<Conserved>& q) { return residual(q); },
        [this](std::vector<Conserved>& q) { geo_.exchange(q); });
}

GasState CompressibleSolver::state(std::size_t cell) const {
    return to_primitive(q_[cell]);
}

std::vector<Vec3> CompressibleSolver::velocity() const {
    std::vector<Vec3> u(q_.size());
    for (std::size_t c = 0; c < u.size(); ++c) {
        u[c] = state(c).u;
    }
    return u;
}

std::vector<Scalar> CompressibleSolver::pressure() const {
    std::vector<Scalar> p(q_.size());
    for (std::size_t c = 0; c < p.size(); ++c) {
        p[c] = state(c).p;
    }
    return p;
}

std::vector<Scalar> CompressibleSolver::density() const {
    std::vector<Scalar> r(q_.size());
    for (std::size_t c = 0; c < r.size(); ++c) {
        r[c] = q_[c][0];
    }
    return r;
}

std::vector<Scalar> CompressibleSolver::temperature() const {
    std::vector<Scalar> t(q_.size());
    for (std::size_t c = 0; c < t.size(); ++c) {
        const GasState s = state(c);
        t[c] = s.p / (s.rho * settings_.r_gas);
    }
    return t;
}

std::vector<Scalar> CompressibleSolver::mach() const {
    std::vector<Scalar> ma(q_.size());
    for (std::size_t c = 0; c < ma.size(); ++c) {
        const GasState s = state(c);
        ma[c] = s.u.norm() / std::sqrt(settings_.gamma * s.p / s.rho);
    }
    return ma;
}

} // namespace cfd::fv
