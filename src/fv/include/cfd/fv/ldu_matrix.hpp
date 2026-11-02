#pragma once

#include <cfd/mesh/mesh.hpp>
#include <string>
#include <utility>
#include <vector>

namespace cfd::fv {

using core::Label;
using core::Scalar;

enum class LinearSolverType { ConjugateGradient, BiCgStab };

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
    LinearSolveResult solve(std::vector<Scalar>& x, const LinearSolverSettings& settings) const;

    std::vector<Scalar> diag;
    std::vector<Scalar> lower;
    std::vector<Scalar> upper;
    std::vector<Scalar> source;

private:
    const mesh::Mesh* mesh_;
};

} // namespace cfd::fv
