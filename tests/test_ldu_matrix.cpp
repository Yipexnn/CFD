#include <cfd/fv/ldu_matrix.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

using cfd::core::Label;
using cfd::core::Scalar;
using cfd::fv::LduMatrix;
using cfd::fv::LinearSolverSettings;
using cfd::fv::LinearSolverType;
using cfd::mesh::BoxSpec;
using cfd::mesh::make_box_mesh;
using cfd::mesh::Mesh;

namespace {

constexpr Label k_n_cells = 10;

std::size_t idx(Label i) {
    return static_cast<std::size_t>(i);
}

Mesh chain_mesh() {
    BoxSpec spec;
    spec.nx = k_n_cells;
    spec.ny = 1;
    spec.nz = 1;
    return make_box_mesh(spec);
}

// Poisson-like chain: upper = lower = -1, diag += 1 at both ends of each internal face.
void assemble_laplacian(LduMatrix& a, const Mesh& m) {
    for (Label f = 0; f < m.n_internal_faces(); ++f) {
        const auto fi = idx(f);
        a.upper[fi] = -1.0;
        a.lower[fi] = -1.0;
        a.diag[idx(m.owner()[fi])] += 1.0;
        a.diag[idx(m.neighbour()[fi])] += 1.0;
    }
}

// Adds 1 to the first and last diagonal entries, making the chain SPD.
void add_dirichlet(LduMatrix& a) {
    a.diag[0] += 1.0;
    a.diag[idx(k_n_cells - 1)] += 1.0;
}

// Convection-like asymmetry: upper reduced by 0.3, owner diagonal increased by 0.3.
void add_asymmetry(LduMatrix& a, const Mesh& m) {
    for (Label f = 0; f < m.n_internal_faces(); ++f) {
        const auto fi = idx(f);
        a.upper[fi] -= 0.3;
        a.diag[idx(m.owner()[fi])] += 0.3;
    }
}

std::vector<Scalar> exact_solution() {
    std::vector<Scalar> xe(idx(k_n_cells));
    for (std::size_t i = 0; i < xe.size(); ++i) {
        xe[i] = std::sin(0.3 * static_cast<Scalar>(i));
    }
    return xe;
}

LinearSolverSettings make_settings(LinearSolverType type) {
    LinearSolverSettings s;
    s.type = type;
    s.tolerance = 1e-12;
    s.rel_tolerance = 0.0;
    s.max_iterations = 1000;
    return s;
}

// Sets source = A x_e, solves from x = 0 and checks the solution against x_e.
void expect_manufactured_solve(LduMatrix& a, LinearSolverType type) {
    const auto xe = exact_solution();
    std::vector<Scalar> b;
    a.multiply(xe, b);
    a.source = b;

    std::vector<Scalar> x(xe.size(), 0.0);
    const auto result = a.solve(x, make_settings(type));

    for (std::size_t i = 0; i < xe.size(); ++i) {
        EXPECT_LT(std::abs(x[i] - xe[i]), 1e-8);
    }
    EXPECT_LT(result.final_residual, result.initial_residual);
}

} // namespace

TEST(LduMatrix, MultiplyMatchesDense) {
    const Mesh m = chain_mesh();
    LduMatrix a(m);
    assemble_laplacian(a, m);
    add_dirichlet(a);

    const auto n = idx(k_n_cells);
    std::vector<std::vector<Scalar>> dense(n, std::vector<Scalar>(n, 0.0));
    for (std::size_t c = 0; c < n; ++c) {
        dense[c][c] = a.diag[c];
    }
    for (std::size_t f = 0; f < a.upper.size(); ++f) {
        const auto p = idx(m.owner()[f]);
        const auto q = idx(m.neighbour()[f]);
        dense[p][q] += a.upper[f];
        dense[q][p] += a.lower[f];
    }

    std::vector<Scalar> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<Scalar>(i + 1);
    }
    std::vector<Scalar> y;
    a.multiply(x, y);

    ASSERT_EQ(y.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        Scalar ref = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            ref += dense[i][j] * x[j];
        }
        EXPECT_NEAR(y[i], ref, 1e-12);
    }
}

TEST(LduMatrix, ResidualIsSourceMinusAx) {
    const Mesh m = chain_mesh();
    LduMatrix a(m);
    assemble_laplacian(a, m);
    add_dirichlet(a);

    const auto n = idx(k_n_cells);
    a.source.assign(n, 0.0);
    std::vector<Scalar> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        a.source[i] = 0.5 + 0.1 * static_cast<Scalar>(i);
        x[i] = std::cos(static_cast<Scalar>(i));
    }

    std::vector<Scalar> ax;
    a.multiply(x, ax);
    std::vector<Scalar> r;
    a.residual(x, r);

    ASSERT_EQ(r.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(r[i], a.source[i] - ax[i], 1e-12);
    }
}

TEST(LduMatrix, SolveConjugateGradient) {
    const Mesh m = chain_mesh();
    LduMatrix a(m);
    assemble_laplacian(a, m);
    add_dirichlet(a);
    expect_manufactured_solve(a, LinearSolverType::ConjugateGradient);
}

TEST(LduMatrix, SolveBiCgStab) {
    const Mesh m = chain_mesh();
    LduMatrix a(m);
    assemble_laplacian(a, m);
    add_dirichlet(a);
    add_asymmetry(a, m);
    expect_manufactured_solve(a, LinearSolverType::BiCgStab);
}

TEST(LduMatrix, NormalisedResidualZeroAtSolution) {
    const Mesh m = chain_mesh();
    LduMatrix a(m);
    assemble_laplacian(a, m);
    add_dirichlet(a);

    const auto xe = exact_solution();
    a.source.clear();
    a.multiply(xe, a.source);

    EXPECT_LT(a.normalised_residual(xe), 1e-12);
}

TEST(LduMatrix, SetReferenceFixesValue) {
    const Mesh m = chain_mesh();
    LduMatrix a(m);
    assemble_laplacian(a, m); // pure Neumann: no Dirichlet term
    a.source.assign(idx(k_n_cells), 0.0);
    a.set_reference(0, 2.0);

    std::vector<Scalar> x(idx(k_n_cells), 0.0);
    a.solve(x, make_settings(LinearSolverType::ConjugateGradient));

    for (const auto v : x) {
        EXPECT_NEAR(v, 2.0, 1e-6);
    }
}
