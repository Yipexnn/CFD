#pragma once

#include <Eigen/SparseCholesky>
#include <cfd/fv/amg.hpp>
#include <cfd/parallel/halo.hpp>
#include <memory>
#include <vector>

namespace cfd::fv {

using core::Label;

// Distributed smoothed-aggregation AMG (V-cycle preconditioner for CG on decomposed meshes).
//
// Every process aggregates its own unknowns; the prolongation is smoothed with damped
// Jacobi over the process-local couplings, and the Galerkin coarse matrices keep the
// couplings between processes (each level has its own halo). Smoother: hybrid symmetric
// Gauss-Seidel (Gauss-Seidel on the owned rows, ghost values from the last exchange). The
// coarsest level is gathered to the root process and solved directly.
class ParallelAmg {
public:
    explicit ParallelAmg(AmgSettings settings = {}) : settings_(settings) {}

    // a: owned rows of the local matrix, columns in local numbering (owned cells first,
    // then the halo's ghost cells). The halo must outlive the preconditioner.
    void setup(const CsrMatrix& a, const parallel::Halo& halo);

    // z = M^-1 r on the owned unknowns (collective).
    void apply(const DenseVector& r, DenseVector& z) const;

    [[nodiscard]] int levels() const { return static_cast<int>(levels_.size()); }
    // Global number of unknowns of a level.
    [[nodiscard]] Label level_size(int level) const {
        return levels_[static_cast<std::size_t>(level)].n_global;
    }

private:
    struct Level {
        CsrMatrix a;          // owned rows x (owned + ghost) columns
        parallel::Halo halo;  // ghost exchange of this level
        DenseVector inv_diag; // owned
        CsrMatrix p;          // owned fine x owned coarse
        CsrMatrix r;          // p transposed
        Label offset = 0;     // global index of the first owned unknown
        Label n_global = 0;
    };

    void cycle(std::size_t level, const DenseVector& b, DenseVector& x) const;
    void smooth(const Level& lv, const DenseVector& b, DenseVector& x, bool forward) const;
    void setup_coarsest();
    void solve_coarsest(const DenseVector& b, DenseVector& x) const;

    AmgSettings settings_;
    const parallel::Comm* comm_ = &parallel::Comm::self();
    std::vector<Level> levels_;
    std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>>> coarse_solver_; // root
};

} // namespace cfd::fv
