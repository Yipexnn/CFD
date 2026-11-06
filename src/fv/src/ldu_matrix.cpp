#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <cfd/fv/amg.hpp>
#include <cfd/fv/ldu_matrix.hpp>
#include <cfd/fv/parallel_amg.hpp>
#include <cfd/gpu/krylov.hpp>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cfd::fv {

namespace {

using core::Vec3;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, int>;
using EigenVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

constexpr Scalar k_small = 1e-20;

} // namespace

LduMatrix::LduMatrix(const mesh::Mesh& mesh)
    : diag(static_cast<std::size_t>(mesh.n_cells()), 0.0),
      lower(static_cast<std::size_t>(mesh.n_internal_faces()), 0.0),
      upper(static_cast<std::size_t>(mesh.n_internal_faces()), 0.0),
      source(static_cast<std::size_t>(mesh.n_cells()), 0.0), mesh_(&mesh) {}

void LduMatrix::set_zero() {
    std::fill(diag.begin(), diag.end(), 0.0);
    std::fill(lower.begin(), lower.end(), 0.0);
    std::fill(upper.begin(), upper.end(), 0.0);
    std::fill(source.begin(), source.end(), 0.0);
}

void LduMatrix::off_diagonal_product(const std::vector<Scalar>& x, std::vector<Scalar>& y) const {
    const auto& own = mesh_->owner();
    const auto& nei = mesh_->neighbour();
    y.assign(diag.size(), 0.0);
    for (std::size_t f = 0; f < upper.size(); ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const auto n = static_cast<std::size_t>(nei[f]);
        y[p] += upper[f] * x[n];
        y[n] += lower[f] * x[p];
    }
}

void LduMatrix::multiply(const std::vector<Scalar>& x, std::vector<Scalar>& y) const {
    off_diagonal_product(x, y);
    for (std::size_t c = 0; c < diag.size(); ++c) {
        y[c] += diag[c] * x[c];
    }
}

void LduMatrix::residual(const std::vector<Scalar>& x, std::vector<Scalar>& r) const {
    multiply(x, r);
    for (std::size_t c = 0; c < diag.size(); ++c) {
        r[c] = source[c] - r[c];
    }
}

Scalar LduMatrix::normalised_residual(const std::vector<Scalar>& x) const {
    const auto [res, norm] = residual_norms(x);
    return res / (norm + k_small);
}

std::pair<Scalar, Scalar> LduMatrix::residual_norms(const std::vector<Scalar>& x) const {
    const std::size_t n = n_owned();
    const parallel::Comm& comm = halo != nullptr ? *halo->comm : parallel::Comm::self();
    if (comm.sum(static_cast<Label>(n)) == 0) {
        return {0.0, 0.0};
    }
    std::vector<Scalar> ax;
    multiply(x, ax);

    // A * xRef with xRef = mean(x): row sums times the mean.
    std::array<Scalar, 2> mean{
        std::accumulate(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n), 0.0),
        static_cast<Scalar>(n)};
    comm.sum(mean.data(), mean.size());
    const Scalar x_ref = mean[0] / mean[1];
    std::vector<Scalar> row_sum(diag);
    const auto& own = mesh_->owner();
    const auto& nei = mesh_->neighbour();
    for (std::size_t f = 0; f < upper.size(); ++f) {
        row_sum[static_cast<std::size_t>(own[f])] += upper[f];
        row_sum[static_cast<std::size_t>(nei[f])] += lower[f];
    }

    Scalar res = 0.0;
    Scalar norm = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
        const Scalar ax_ref = row_sum[c] * x_ref;
        res += std::abs(source[c] - ax[c]);
        norm += std::abs(ax[c] - ax_ref) + std::abs(source[c] - ax_ref);
    }
    std::array<Scalar, 2> norms{res, norm};
    comm.sum(norms.data(), norms.size());
    return {norms[0], norms[1]};
}

void LduMatrix::set_reference(Label cell, Scalar value) {
    const auto c = static_cast<std::size_t>(cell);
    source[c] += diag[c] * value;
    diag[c] += diag[c];
}

LinearSolveResult LduMatrix::solve(std::vector<Scalar>& x, const LinearSolverSettings& settings,
                                   AmgCache* amg_cache) const {
    if (halo != nullptr && halo->parallel()) {
        return solve_distributed(x, settings, amg_cache);
    }
    LinearSolveResult result;
    const std::size_t n = diag.size();
    if (x.size() != n) {
        throw std::invalid_argument("LduMatrix::solve: size mismatch");
    }
    result.initial_residual = normalised_residual(x);
    result.final_residual = result.initial_residual;
    if (result.initial_residual < settings.tolerance || n == 0) {
        return result;
    }

    // Assemble CSR (fast path with a precomputed pattern).
    SparseMatrix a(static_cast<int>(n), static_cast<int>(n));
    if (pattern != nullptr) {
        a = pattern->structure;
        Scalar* v = a.valuePtr();
        for (std::size_t c = 0; c < n; ++c) {
            v[pattern->diag_pos[c]] = diag[c];
        }
        for (std::size_t f = 0; f < upper.size(); ++f) {
            v[pattern->upper_pos[f]] = upper[f];
            v[pattern->lower_pos[f]] = lower[f];
        }
    } else {
        std::vector<Eigen::Triplet<Scalar, int>> triplets;
        triplets.reserve(n + 2 * upper.size());
        for (std::size_t c = 0; c < n; ++c) {
            triplets.emplace_back(static_cast<int>(c), static_cast<int>(c), diag[c]);
        }
        const auto& own = mesh_->owner();
        const auto& nei = mesh_->neighbour();
        for (std::size_t f = 0; f < upper.size(); ++f) {
            triplets.emplace_back(own[f], nei[f], upper[f]);
            triplets.emplace_back(nei[f], own[f], lower[f]);
        }
        a.setFromTriplets(triplets.begin(), triplets.end());
    }

    const Eigen::Map<const EigenVector> b(source.data(), static_cast<Eigen::Index>(n));
    Eigen::Map<EigenVector> xv(x.data(), static_cast<Eigen::Index>(n));

    // Translate the normalised targets into Eigen's |r|_2 / |b|_2 criterion.
    const EigenVector r0 = b - a * xv;
    const Scalar r0_norm = r0.norm();
    const Scalar b_norm = b.norm();
    const Scalar target =
        std::max(settings.rel_tolerance,
                 settings.tolerance / std::max(result.initial_residual, k_small)) *
        r0_norm;
    const Scalar eigen_tol = b_norm > k_small ? std::clamp(target / b_norm, 1e-15, 1.0) : 1e-15;

    if (settings.type == LinearSolverType::GpuCg ||
        settings.type == LinearSolverType::GpuBiCgStab) {
        a.makeCompressed();
        const gpu::CsrView view{static_cast<int>(n), static_cast<int>(a.nonZeros()),
                                a.outerIndexPtr(), a.innerIndexPtr(), a.valuePtr()};
        const gpu::SolveStats st = gpu::solve(
            view, source.data(), x.data(),
            settings.type == LinearSolverType::GpuCg ? gpu::Method::Cg : gpu::Method::BiCgStab,
            {std::max(target, 1e-300), settings.max_iterations});
        result.iterations = st.iterations;
    } else if (settings.type == LinearSolverType::AmgCg) {
        AmgPreconditioner local;
        AmgPreconditioner* amg = &local;
        bool fresh = true;
        if (amg_cache != nullptr) {
            if (!amg_cache->amg || amg_cache->age >= amg_cache->max_age) {
                amg_cache->amg = std::make_unique<AmgPreconditioner>();
                amg_cache->amg->setup(a);
                amg_cache->age = 0;
            } else {
                fresh = false;
            }
            amg = amg_cache->amg.get();
        } else {
            local.setup(a);
        }
        EigenVector xs = xv;
        KrylovResult kr = pcg(a, b, xs, *amg, eigen_tol * b_norm, settings.max_iterations);
        if (amg_cache != nullptr) {
            if (fresh) {
                amg_cache->setup_iterations = std::max(kr.iterations, 1);
            } else if (kr.iterations > 2 * amg_cache->setup_iterations + 2) {
                // Stale hierarchy: rebuild and solve again from the current iterate.
                amg_cache->amg->setup(a);
                const KrylovResult more =
                    pcg(a, b, xs, *amg_cache->amg, eigen_tol * b_norm, settings.max_iterations);
                kr.iterations += more.iterations;
                amg_cache->age = 0;
                amg_cache->setup_iterations = std::max(more.iterations, 1);
            }
            ++amg_cache->age;
        }
        xv = xs;
        result.iterations = kr.iterations;
    } else if (settings.type == LinearSolverType::ConjugateGradient) {
        Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper,
                                 Eigen::IncompleteCholesky<Scalar>>
            solver;
        solver.setTolerance(eigen_tol);
        solver.setMaxIterations(settings.max_iterations);
        solver.compute(a);
        xv = solver.solveWithGuess(b, EigenVector(xv));
        result.iterations = static_cast<int>(solver.iterations());
    } else {
        Eigen::BiCGSTAB<SparseMatrix, Eigen::DiagonalPreconditioner<Scalar>> solver;
        solver.setTolerance(eigen_tol);
        solver.setMaxIterations(settings.max_iterations);
        solver.compute(a);
        xv = solver.solveWithGuess(b, EigenVector(xv));
        result.iterations = static_cast<int>(solver.iterations());
    }

    result.final_residual = normalised_residual(x);
    return result;
}

LinearSolveResult LduMatrix::solve_distributed(std::vector<Scalar>& x,
                                               const LinearSolverSettings& requested,
                                               AmgCache* amg_cache) const {
    // Accelerator solvers are single-process: use the distributed CPU equivalents.
    LinearSolverSettings settings = requested;
    if (settings.type == LinearSolverType::GpuCg) {
        settings.type = LinearSolverType::AmgCg;
    } else if (settings.type == LinearSolverType::GpuBiCgStab) {
        settings.type = LinearSolverType::BiCgStab;
    }
    const parallel::Comm& comm = *halo->comm;
    const std::size_t n = diag.size();
    const std::size_t no = n_owned();
    if (x.size() != n) {
        throw std::invalid_argument("LduMatrix::solve: size mismatch");
    }
    halo->exchange(x);
    LinearSolveResult result;
    result.initial_residual = normalised_residual(x);
    result.final_residual = result.initial_residual;
    if (result.initial_residual < settings.tolerance) {
        return result;
    }

    // Local CSR (owned and ghost rows); the owned block is used by the preconditioners.
    SparseMatrix a(static_cast<int>(n), static_cast<int>(n));
    if (pattern != nullptr) {
        a = pattern->structure;
        Scalar* v = a.valuePtr();
        for (std::size_t c = 0; c < n; ++c) {
            v[pattern->diag_pos[c]] = diag[c];
        }
        for (std::size_t f = 0; f < upper.size(); ++f) {
            v[pattern->upper_pos[f]] = upper[f];
            v[pattern->lower_pos[f]] = lower[f];
        }
    } else {
        std::vector<Eigen::Triplet<Scalar, int>> triplets;
        triplets.reserve(n + 2 * upper.size());
        for (std::size_t c = 0; c < n; ++c) {
            triplets.emplace_back(static_cast<int>(c), static_cast<int>(c), diag[c]);
        }
        const auto& own = mesh_->owner();
        const auto& nei = mesh_->neighbour();
        for (std::size_t f = 0; f < upper.size(); ++f) {
            triplets.emplace_back(own[f], nei[f], upper[f]);
            triplets.emplace_back(nei[f], own[f], lower[f]);
        }
        a.setFromTriplets(triplets.begin(), triplets.end());
    }
    const auto ino = static_cast<Eigen::Index>(no);
    const bool symmetric = settings.type != LinearSolverType::BiCgStab;
    // Owned rows of the local matrix (columns: owned + ghost cells) for the parallel AMG.
    CsrMatrix a_rows;
    if (symmetric) {
        a_rows = a.topRows(ino);
    }

    std::vector<Scalar> work(x);
    const auto multiply_owned = [&](const EigenVector& v, EigenVector& y) {
        std::copy(v.data(), v.data() + ino, work.begin());
        halo->exchange(work);
        const Eigen::Map<const EigenVector> w(work.data(), static_cast<Eigen::Index>(n));
        y = (a * w).head(ino);
    };
    const auto dot = [&](const EigenVector& u, const EigenVector& v) { return comm.sum(u.dot(v)); };

    const EigenVector b = Eigen::Map<const EigenVector>(source.data(), ino);
    EigenVector xv = Eigen::Map<const EigenVector>(x.data(), ino);
    EigenVector ax(ino);
    multiply_owned(xv, ax);
    EigenVector r = b - ax;
    const Scalar r0_norm = std::sqrt(dot(r, r));
    const Scalar target =
        std::max(settings.rel_tolerance,
                 settings.tolerance / std::max(result.initial_residual, k_small)) *
        r0_norm;

    ParallelAmg local_amg;
    const ParallelAmg* amg = nullptr;
    bool fresh = true;
    if (symmetric) {
        if (settings.type == LinearSolverType::AmgCg && amg_cache != nullptr) {
            if (!amg_cache->parallel_amg || amg_cache->age >= amg_cache->max_age) {
                amg_cache->parallel_amg = std::make_unique<ParallelAmg>();
                amg_cache->parallel_amg->setup(a_rows, *halo);
                amg_cache->age = 0;
            } else {
                fresh = false;
            }
            amg = amg_cache->parallel_amg.get();
        } else {
            local_amg.setup(a_rows, *halo);
            amg = &local_amg;
        }
    }
    EigenVector inv_diag(ino);
    for (Eigen::Index c = 0; c < ino; ++c) {
        const Scalar d = diag[static_cast<std::size_t>(c)];
        inv_diag[c] = std::abs(d) > k_small ? 1.0 / d : 1.0;
    }

    int iterations = 0;
    if (symmetric) {
        auto pcg_run = [&](const ParallelAmg& amg_m) {
            const auto m_apply = [&](const EigenVector& res, EigenVector& z) {
                amg_m.apply(res, z);
            };
            int its = 0;
            EigenVector z(ino);
            EigenVector q(ino);
            m_apply(r, z);
            EigenVector p = z;
            Scalar rz = dot(r, z);
            while (its < settings.max_iterations && std::sqrt(dot(r, r)) > target) {
                multiply_owned(p, q);
                const Scalar pq = dot(p, q);
                if (std::abs(pq) < 1e-300) {
                    break;
                }
                const Scalar alpha = rz / pq;
                xv += alpha * p;
                r -= alpha * q;
                ++its;
                m_apply(r, z);
                const Scalar rz_new = dot(r, z);
                p = z + (rz_new / rz) * p;
                rz = rz_new;
            }
            return its;
        };
        iterations = pcg_run(*amg);
        if (amg_cache != nullptr && settings.type == LinearSolverType::AmgCg) {
            // The decision must be identical on all ranks: use the global maximum.
            const int its = static_cast<int>(comm.max(static_cast<Scalar>(iterations)));
            if (fresh) {
                amg_cache->setup_iterations = std::max(its, 1);
            } else if (its > 2 * amg_cache->setup_iterations + 2) {
                amg_cache->parallel_amg->setup(a_rows, *halo);
                const int more = pcg_run(*amg_cache->parallel_amg);
                iterations += more;
                amg_cache->age = 0;
                // The re-solve starts from a partly converged iterate: keep the reference
                // count (lowering it would trigger a rebuild on almost every solve).
                amg_cache->setup_iterations = std::max(amg_cache->setup_iterations, more);
            }
            ++amg_cache->age;
        }
    } else {
        // Jacobi-preconditioned BiCGStab.
        const EigenVector r_hat = r;
        EigenVector p = EigenVector::Zero(ino);
        EigenVector v = EigenVector::Zero(ino);
        EigenVector p_hat(ino);
        EigenVector s(ino);
        EigenVector s_hat(ino);
        EigenVector t(ino);
        Scalar rho = 1.0;
        Scalar alpha = 1.0;
        Scalar omega = 1.0;
        while (iterations < settings.max_iterations && std::sqrt(dot(r, r)) > target) {
            const Scalar rho_new = dot(r_hat, r);
            if (std::abs(rho_new) < 1e-300) {
                break;
            }
            const Scalar beta = (rho_new / rho) * (alpha / omega);
            p = r + beta * (p - omega * v);
            p_hat = inv_diag.cwiseProduct(p);
            multiply_owned(p_hat, v);
            const Scalar rv = dot(r_hat, v);
            if (std::abs(rv) < 1e-300) {
                break;
            }
            alpha = rho_new / rv;
            s = r - alpha * v;
            ++iterations;
            if (std::sqrt(dot(s, s)) <= target) {
                xv += alpha * p_hat;
                r = s;
                break;
            }
            s_hat = inv_diag.cwiseProduct(s);
            multiply_owned(s_hat, t);
            const Scalar tt = dot(t, t);
            omega = tt > 1e-300 ? dot(t, s) / tt : 0.0;
            xv += alpha * p_hat + omega * s_hat;
            r = s - omega * t;
            rho = rho_new;
            if (omega == 0.0) {
                break;
            }
        }
    }

    std::copy(xv.data(), xv.data() + ino, x.begin());
    halo->exchange(x);
    result.iterations = iterations;
    result.final_residual = normalised_residual(x);
    return result;
}

AmgCache::AmgCache() = default;
AmgCache::~AmgCache() = default;

CsrPattern build_csr_pattern(const mesh::Mesh& mesh) {
    const auto n = static_cast<int>(mesh.n_cells());
    const auto nif = static_cast<std::size_t>(mesh.n_internal_faces());
    const auto& own = mesh.owner();
    const auto& nei = mesh.neighbour();
    std::vector<Eigen::Triplet<Scalar, int>> t;
    t.reserve(static_cast<std::size_t>(n) + 2 * nif);
    for (int c = 0; c < n; ++c) {
        t.emplace_back(c, c, 0.0);
    }
    for (std::size_t f = 0; f < nif; ++f) {
        t.emplace_back(own[f], nei[f], 0.0);
        t.emplace_back(nei[f], own[f], 0.0);
    }
    CsrPattern p;
    p.structure.resize(n, n);
    p.structure.setFromTriplets(t.begin(), t.end());
    p.structure.makeCompressed();

    const int* outer = p.structure.outerIndexPtr();
    const int* inner = p.structure.innerIndexPtr();
    auto find = [&](int row, int col) {
        for (int k = outer[row]; k < outer[row + 1]; ++k) {
            if (inner[k] == col) {
                return k;
            }
        }
        throw std::logic_error("build_csr_pattern: entry not found");
    };
    p.diag_pos.resize(static_cast<std::size_t>(n));
    for (int c = 0; c < n; ++c) {
        p.diag_pos[static_cast<std::size_t>(c)] = find(c, c);
    }
    p.upper_pos.resize(nif);
    p.lower_pos.resize(nif);
    for (std::size_t f = 0; f < nif; ++f) {
        p.upper_pos[f] = find(own[f], nei[f]);
        p.lower_pos[f] = find(nei[f], own[f]);
    }
    return p;
}

} // namespace cfd::fv
