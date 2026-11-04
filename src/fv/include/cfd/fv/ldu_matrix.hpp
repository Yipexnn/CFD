#pragma once

#include <Eigen/SparseCore>
#include <cfd/mesh/mesh.hpp>
#include <cfd/parallel/halo.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cfd::fv {

using core::Label;
using core::Scalar;

// ConjugateGradient: incomplete-Cholesky CG; AmgCg: smoothed-aggregation AMG
// preconditioned CG (symmetric matrices, e.g. pressure); BiCgStab: Jacobi-preconditioned.
enum class LinearSolverType { ConjugateGradient, BiCgStab, AmgCg };

struct LinearSolverSettings {
    LinearSolverType type = LinearSolverType::BiCgStab;
    Scalar tolerance = 1e-10;    // absolute, on the normalised residual
    Scalar rel_tolerance = 0.01; // relative to the initial normalised residual
    int max_iterations = 1000;
};

struct LinearSolveResult {
    Scalar initial_residual = 0.0; // normalised (OpenFOAM-style)
    Scalar final_residual = 0.0;
    int iterations = 0;
};

// Sparse matrix in LDU (lower-diagonal-upper) face addressing.
//
// Row P of the system reads: diag[P] x[P] + sum_f upper[f] x[N] (faces owned by P)
//                                         + sum_f lower[f] x[P'] (faces where P is neighbour)
//                          = source[P]
// i.e. upper[f] = a(owner, neighbour) and lower[f] = a(neighbour, owner).
class AmgPreconditioner;

// CSR sparsity pattern of the cell-to-cell matrix of a mesh with the value positions of
// the LDU coefficients, so assembled matrices can be converted to CSR without sorting.
struct CsrPattern {
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor, int> structure;
    std::vector<int> diag_pos;
    std::vector<int> upper_pos; // position of a(owner, neighbour)
    std::vector<int> lower_pos; // position of a(neighbour, owner)
};
[[nodiscard]] CsrPattern build_csr_pattern(const mesh::Mesh& mesh);

// Reusable AMG hierarchy for repeated solves with slowly changing matrices.
struct AmgCache {
    AmgCache();
    ~AmgCache();
    AmgCache(const AmgCache&) = delete;
    AmgCache& operator=(const AmgCache&) = delete;
    AmgCache(AmgCache&&) = delete;
    AmgCache& operator=(AmgCache&&) = delete;

    std::unique_ptr<AmgPreconditioner> amg;
    int age = 0;              // solves since the last setup
    int setup_iterations = 0; // CG iterations of the first solve after setup
    int max_age = 20;         // rebuild at the latest after this many solves
};

class LduMatrix {
public:
    explicit LduMatrix(const mesh::Mesh& mesh);

    [[nodiscard]] Label n_rows() const { return static_cast<Label>(diag.size()); }
    [[nodiscard]] const mesh::Mesh& mesh() const { return *mesh_; }

    void set_zero();

    // y = A x
    void multiply(const std::vector<Scalar>& x, std::vector<Scalar>& y) const;
    // r = source - A x
    void residual(const std::vector<Scalar>& x, std::vector<Scalar>& r) const;
    // sum_nb a_nb x_nb for each row (off-diagonal product)
    void off_diagonal_product(const std::vector<Scalar>& x, std::vector<Scalar>& y) const;
    // OpenFOAM-style normalised residual |b - Ax|_1 / normFactor.
    [[nodiscard]] Scalar normalised_residual(const std::vector<Scalar>& x) const;
    // Returns {|b - Ax|_1, normFactor} used by normalised_residual().
    [[nodiscard]] std::pair<Scalar, Scalar> residual_norms(const std::vector<Scalar>& x) const;

    // Fixes x[cell] = value (pressure reference) while keeping symmetry.
    void set_reference(Label cell, Scalar value);

    // Solves A x = source in place, using x as initial guess.
    // amg_cache (optional, AmgCg only) keeps the AMG hierarchy between calls; it is rebuilt
    // when it gets old or the CG iteration count grows noticeably.
    LinearSolveResult solve(std::vector<Scalar>& x, const LinearSolverSettings& settings,
                            AmgCache* amg_cache = nullptr) const;

    // Optional precomputed pattern (must belong to the same mesh) for fast CSR conversion.
    const CsrPattern* pattern = nullptr;
    // Decomposed mesh (optional): only the owned rows [0, n_owned) are part of the global
    // system; ghost values of x are taken from their owners (halo exchange). Residual norms
    // are global and solve() uses distributed Krylov solvers (CG / BiCGStab with a
    // block-Jacobi AMG or Jacobi preconditioner). Must outlive the matrix.
    const parallel::Halo* halo = nullptr;

    std::vector<Scalar> diag;
    std::vector<Scalar> lower;
    std::vector<Scalar> upper;
    std::vector<Scalar> source;

private:
    [[nodiscard]] std::size_t n_owned() const {
        return halo != nullptr ? static_cast<std::size_t>(halo->n_owned) : diag.size();
    }
    LinearSolveResult solve_distributed(std::vector<Scalar>& x,
                                        const LinearSolverSettings& settings,
                                        AmgCache* amg_cache) const;

    const mesh::Mesh* mesh_;
};

} // namespace cfd::fv
