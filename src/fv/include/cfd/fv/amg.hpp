#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <cfd/core/types.hpp>
#include <memory>
#include <vector>

namespace cfd::fv {

using core::Scalar;

using CsrMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, int>;
using DenseVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

struct AmgSettings {
    Scalar strength_threshold = 0.08; // |a_ij| > theta sqrt(a_ii a_jj) is a strong connection
    int max_levels = 25;
    int coarse_size = 400; // direct solve below this many unknowns
    int pre_smooth = 1;    // symmetric Gauss-Seidel sweeps (forward before,
    int post_smooth = 1;   // backward after the coarse correction)
};

// Smoothed-aggregation algebraic multigrid for symmetric positive (semi-)definite
// M-matrix-like systems (pressure equation). Used as a V-cycle preconditioner for CG.
class AmgPreconditioner {
public:
    explicit AmgPreconditioner(AmgSettings settings = {}) : settings_(settings) {}

    void setup(const CsrMatrix& a);

    // z = M^-1 r (one V-cycle with zero initial guess).
    void apply(const DenseVector& r, DenseVector& z) const;

    [[nodiscard]] int levels() const { return static_cast<int>(levels_.size()); }
    [[nodiscard]] Eigen::Index level_size(int level) const {
        return levels_[static_cast<std::size_t>(level)].a.rows();
    }
    // Hierarchy access (e.g. to run the V-cycle on an accelerator): operator, prolongation
    // (to this level from the next coarser one) and restriction of a level, and the dense
    // inverse of the coarsest operator.
    [[nodiscard]] const CsrMatrix& level_matrix(int level) const {
        return levels_[static_cast<std::size_t>(level)].a;
    }
    [[nodiscard]] const CsrMatrix& level_prolongation(int level) const {
        return levels_[static_cast<std::size_t>(level)].p;
    }
    [[nodiscard]] const CsrMatrix& level_restriction(int level) const {
        return levels_[static_cast<std::size_t>(level)].r;
    }
    [[nodiscard]] Eigen::MatrixXd coarse_inverse() const;
    // Sum of non-zeros of all levels / non-zeros of the finest level.
    [[nodiscard]] Scalar operator_complexity() const;

private:
    struct Level {
        CsrMatrix a;
        CsrMatrix p; // prolongation to this level from the next coarser one
        CsrMatrix r; // restriction (p transposed)
        DenseVector inv_diag;
    };

    void cycle(std::size_t level, const DenseVector& b, DenseVector& x) const;

    AmgSettings settings_;
    std::vector<Level> levels_;
    std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>>> coarse_solver_;
};

// Greedy aggregation on the strength graph (Vanek, Mandel & Brezina 1996) of a square
// matrix: aggregate index per row; n_aggregates receives their number.
[[nodiscard]] std::vector<int> amg_aggregate(const CsrMatrix& a, const DenseVector& diag,
                                             Scalar theta, int& n_aggregates);

struct KrylovResult {
    int iterations = 0;
    Scalar residual_norm = 0.0; // |b - Ax|_2
};

// Preconditioned conjugate gradient; stops at |r|_2 <= tolerance or max_iterations.
KrylovResult pcg(const CsrMatrix& a, const DenseVector& b, DenseVector& x,
                 const AmgPreconditioner& m, Scalar tolerance, int max_iterations);

} // namespace cfd::fv
