#include <algorithm>
#include <cfd/fv/amg.hpp>
#include <cmath>
#include <stdexcept>

namespace cfd::fv {

namespace {
using Triplet = Eigen::Triplet<Scalar, int>;
} // namespace

std::vector<int> amg_aggregate(const CsrMatrix& a, const DenseVector& diag, Scalar theta,
                               int& n_aggregates) {
    const auto n = static_cast<int>(a.rows());
    std::vector<int> agg(static_cast<std::size_t>(n), -1);
    auto strong = [&](Eigen::Index i, Eigen::Index j, Scalar v) {
        return i != j && std::abs(v) > theta * std::sqrt(std::abs(diag[i] * diag[j]));
    };

    n_aggregates = 0;
    // Pass 1: seed aggregates from nodes whose strong neighbourhood is still free.
    for (int i = 0; i < n; ++i) {
        if (agg[static_cast<std::size_t>(i)] >= 0) {
            continue;
        }
        bool free = true;
        bool has_strong = false;
        for (CsrMatrix::InnerIterator it(a, i); it; ++it) {
            if (strong(i, it.col(), it.value())) {
                has_strong = true;
                if (agg[static_cast<std::size_t>(it.col())] >= 0) {
                    free = false;
                    break;
                }
            }
        }
        if (!free || !has_strong) {
            continue;
        }
        agg[static_cast<std::size_t>(i)] = n_aggregates;
        for (CsrMatrix::InnerIterator it(a, i); it; ++it) {
            if (strong(i, it.col(), it.value())) {
                agg[static_cast<std::size_t>(it.col())] = n_aggregates;
            }
        }
        ++n_aggregates;
    }
    // Pass 2: attach remaining nodes to the aggregate of their strongest neighbour.
    std::vector<int> pass1 = agg;
    for (int i = 0; i < n; ++i) {
        if (agg[static_cast<std::size_t>(i)] >= 0) {
            continue;
        }
        Scalar best = 0.0;
        for (CsrMatrix::InnerIterator it(a, i); it; ++it) {
            const auto j = static_cast<int>(it.col());
            if (strong(i, j, it.value()) && pass1[static_cast<std::size_t>(j)] >= 0 &&
                std::abs(it.value()) > best) {
                best = std::abs(it.value());
                agg[static_cast<std::size_t>(i)] = pass1[static_cast<std::size_t>(j)];
            }
        }
    }
    // Pass 3: leftovers (isolated or fully surrounded) form their own aggregates.
    for (int i = 0; i < n; ++i) {
        if (agg[static_cast<std::size_t>(i)] < 0) {
            agg[static_cast<std::size_t>(i)] = n_aggregates;
            for (CsrMatrix::InnerIterator it(a, i); it; ++it) {
                if (strong(i, it.col(), it.value()) &&
                    agg[static_cast<std::size_t>(it.col())] < 0) {
                    agg[static_cast<std::size_t>(it.col())] = n_aggregates;
                }
            }
            ++n_aggregates;
        }
    }
    return agg;
}

namespace {

DenseVector diagonal(const CsrMatrix& a) {
    DenseVector d = DenseVector::Zero(a.rows());
    for (int i = 0; i < a.outerSize(); ++i) {
        for (CsrMatrix::InnerIterator it(a, i); it; ++it) {
            if (it.col() == i) {
                d[i] = it.value();
            }
        }
    }
    return d;
}

void gauss_seidel(const CsrMatrix& a, const DenseVector& inv_diag, const DenseVector& b,
                  DenseVector& x, bool forward) {
    const auto n = static_cast<int>(a.rows());
    auto relax_row = [&](int i) {
        Scalar sum = b[i];
        Scalar d = 0.0;
        for (CsrMatrix::InnerIterator it(a, i); it; ++it) {
            if (it.col() == i) {
                d = it.value();
            } else {
                sum -= it.value() * x[it.col()];
            }
        }
        x[i] = d != 0.0 ? sum * inv_diag[i] : x[i];
    };
    if (forward) {
        for (int i = 0; i < n; ++i) {
            relax_row(i);
        }
    } else {
        for (int i = n - 1; i >= 0; --i) {
            relax_row(i);
        }
    }
}

} // namespace

void AmgPreconditioner::setup(const CsrMatrix& a) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("AmgPreconditioner: matrix must be square");
    }
    levels_.clear();
    levels_.reserve(static_cast<std::size_t>(settings_.max_levels));
    coarse_solver_.reset();

    CsrMatrix current = a;
    for (int l = 0; l < settings_.max_levels; ++l) {
        Level level;
        level.a = current;
        const DenseVector d = diagonal(current);
        level.inv_diag = d.unaryExpr([](Scalar v) { return v != 0.0 ? 1.0 / v : 0.0; });

        const bool last = current.rows() <= settings_.coarse_size || l + 1 == settings_.max_levels;
        if (last) {
            levels_.push_back(std::move(level));
            break;
        }

        int n_agg = 0;
        const std::vector<int> agg = amg_aggregate(current, d, settings_.strength_threshold, n_agg);
        if (n_agg >= current.rows() || n_agg == 0) { // no coarsening possible
            levels_.push_back(std::move(level));
            break;
        }

        // Tentative piecewise-constant prolongation, smoothed with damped Jacobi:
        // P = (I - omega D^-1 A) P_tent, omega = 4 / (3 rho), rho bounded by Gershgorin.
        std::vector<Triplet> t;
        t.reserve(agg.size());
        for (std::size_t i = 0; i < agg.size(); ++i) {
            t.emplace_back(static_cast<int>(i), agg[i], 1.0);
        }
        CsrMatrix p_tent(current.rows(), n_agg);
        p_tent.setFromTriplets(t.begin(), t.end());

        Scalar rho = 0.0;
        for (int i = 0; i < current.outerSize(); ++i) {
            Scalar row = 0.0;
            for (CsrMatrix::InnerIterator it(current, i); it; ++it) {
                row += std::abs(it.value());
            }
            rho = std::max(rho, row * std::abs(level.inv_diag[i]));
        }
        const Scalar omega = rho > 0.0 ? 4.0 / (3.0 * rho) : 0.0;
        const CsrMatrix dinv_a = level.inv_diag.asDiagonal() * current;
        const CsrMatrix smoothed = dinv_a * p_tent;
        CsrMatrix p = p_tent - omega * smoothed;
        p.prune(0.0);

        level.p = p;
        level.r = CsrMatrix(p.transpose());
        const CsrMatrix ap = current * p;
        CsrMatrix coarse = level.r * ap;
        coarse.prune(0.0);
        levels_.push_back(std::move(level));
        current = std::move(coarse);
    }

    // Direct solver on the coarsest level (falls back to Gauss-Seidel if singular).
    const Level& c = levels_.back();
    coarse_solver_ = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>>>();
    coarse_solver_->compute(Eigen::SparseMatrix<Scalar>(c.a));
    if (coarse_solver_->info() != Eigen::Success) {
        coarse_solver_.reset();
    }
}

void AmgPreconditioner::cycle(std::size_t level, const DenseVector& b, DenseVector& x) const {
    const Level& lv = levels_[level];
    if (level + 1 == levels_.size()) {
        if (coarse_solver_) {
            x = coarse_solver_->solve(b);
        } else {
            for (int s = 0; s < 20; ++s) {
                gauss_seidel(lv.a, lv.inv_diag, b, x, true);
                gauss_seidel(lv.a, lv.inv_diag, b, x, false);
            }
        }
        return;
    }
    for (int s = 0; s < settings_.pre_smooth; ++s) {
        gauss_seidel(lv.a, lv.inv_diag, b, x, true);
    }
    const DenseVector residual = b - lv.a * x;
    const DenseVector rc = lv.r * residual;
    DenseVector ec = DenseVector::Zero(rc.size());
    cycle(level + 1, rc, ec);
    x += lv.p * ec;
    for (int s = 0; s < settings_.post_smooth; ++s) {
        gauss_seidel(lv.a, lv.inv_diag, b, x, false);
    }
}

void AmgPreconditioner::apply(const DenseVector& r, DenseVector& z) const {
    if (levels_.empty()) {
        z = r;
        return;
    }
    z = DenseVector::Zero(r.size());
    cycle(0, r, z);
}

Scalar AmgPreconditioner::operator_complexity() const {
    if (levels_.empty()) {
        return 0.0;
    }
    Scalar total = 0.0;
    for (const Level& l : levels_) {
        total += static_cast<Scalar>(l.a.nonZeros());
    }
    return total / static_cast<Scalar>(levels_.front().a.nonZeros());
}

KrylovResult pcg(const CsrMatrix& a, const DenseVector& b, DenseVector& x,
                 const AmgPreconditioner& m, Scalar tolerance, int max_iterations) {
    KrylovResult result;
    DenseVector r = b - a * x;
    result.residual_norm = r.norm();
    if (result.residual_norm <= tolerance) {
        return result;
    }
    DenseVector z;
    m.apply(r, z);
    DenseVector p = z;
    Scalar rz = r.dot(z);
    for (int it = 1; it <= max_iterations; ++it) {
        const DenseVector ap = a * p;
        const Scalar pap = p.dot(ap);
        if (pap <= 0.0) {
            break; // not SPD in this direction
        }
        const Scalar alpha = rz / pap;
        x += alpha * p;
        r -= alpha * ap;
        result.iterations = it;
        result.residual_norm = r.norm();
        if (result.residual_norm <= tolerance) {
            break;
        }
        m.apply(r, z);
        const Scalar rz_new = r.dot(z);
        p = z + (rz_new / rz) * p;
        rz = rz_new;
    }
    return result;
}

} // namespace cfd::fv
