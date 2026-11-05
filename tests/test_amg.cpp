#include <Eigen/SparseCore>
#include <algorithm>
#include <cfd/fv/amg.hpp>
#include <cfd/fv/ldu_matrix.hpp>
#include <cfd/fv/parallel_amg.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

using cfd::core::Label;
using cfd::core::Scalar;
using cfd::fv::AmgCache;
using cfd::fv::AmgPreconditioner;
using cfd::fv::build_csr_pattern;
using cfd::fv::CsrMatrix;
using cfd::fv::CsrPattern;
using cfd::fv::DenseVector;
using cfd::fv::KrylovResult;
using cfd::fv::LduMatrix;
using cfd::fv::LinearSolverSettings;
using cfd::fv::LinearSolverType;
using cfd::fv::pcg;
using cfd::mesh::BoxSpec;
using cfd::mesh::make_box_mesh;
using cfd::mesh::Mesh;
using cfd::mesh::Patch;

namespace {

std::size_t idx(Label i) {
    return static_cast<std::size_t>(i);
}

Mesh box(Label nx, Label ny, Label nz) {
    BoxSpec spec;
    spec.nx = nx;
    spec.ny = ny;
    spec.nz = nz;
    return make_box_mesh(spec);
}

// Unit-coefficient Poisson operator on a box mesh. Internal faces give c = A / d; when
// dirichlet_xmin is set, the xmin patch contributes A / |Cf - C_P| to the owner diagonal.
LduMatrix build_poisson(const Mesh& m, bool dirichlet_xmin) {
    LduMatrix a(m);
    const auto& cc = m.cell_centres();
    const auto& sa = m.face_area_vectors();
    const auto& own = m.owner();
    const auto& nei = m.neighbour();
    for (Label f = 0; f < m.n_internal_faces(); ++f) {
        const auto fi = idx(f);
        const Scalar area = sa[fi].norm();
        const Scalar d = (cc[idx(nei[fi])] - cc[idx(own[fi])]).norm();
        const Scalar c = area / d;
        a.upper[fi] = -c;
        a.lower[fi] = -c;
        a.diag[idx(own[fi])] += c;
        a.diag[idx(nei[fi])] += c;
    }
    if (dirichlet_xmin) {
        const Patch* p = m.find_patch("xmin");
        for (Label k = 0; k < p->n_faces; ++k) {
            const auto fi = idx(p->start_face + k);
            const Scalar area = sa[fi].norm();
            const Scalar d = (m.face_centres()[fi] - cc[idx(own[fi])]).norm();
            a.diag[idx(own[fi])] += area / d;
        }
    }
    return a;
}

// Converts the LDU coefficients to a CSR matrix via triplets.
CsrMatrix to_csr(const LduMatrix& a) {
    const Mesh& m = a.mesh();
    const auto n = static_cast<int>(a.n_rows());
    std::vector<Eigen::Triplet<Scalar, int>> t;
    t.reserve(idx(n) + 2 * idx(m.n_internal_faces()));
    for (int c = 0; c < n; ++c) {
        t.emplace_back(c, c, a.diag[idx(c)]);
    }
    for (Label f = 0; f < m.n_internal_faces(); ++f) {
        const auto fi = idx(f);
        t.emplace_back(static_cast<int>(m.owner()[fi]), static_cast<int>(m.neighbour()[fi]),
                       a.upper[fi]);
        t.emplace_back(static_cast<int>(m.neighbour()[fi]), static_cast<int>(m.owner()[fi]),
                       a.lower[fi]);
    }
    CsrMatrix csr(n, n);
    csr.setFromTriplets(t.begin(), t.end());
    csr.makeCompressed();
    return csr;
}

// Smooth-ish manufactured solution x_i = sin(s i) + 1 over all cells.
std::vector<Scalar> manufactured(std::size_t n, Scalar s) {
    std::vector<Scalar> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::sin(s * static_cast<Scalar>(i)) + 1.0;
    }
    return x;
}

DenseVector to_dense(const std::vector<Scalar>& v) {
    DenseVector d(static_cast<Eigen::Index>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) {
        d[static_cast<Eigen::Index>(i)] = v[i];
    }
    return d;
}

// Max-norm of (x - xe) for std::vector arguments.
Scalar max_abs_diff(const std::vector<Scalar>& x, const std::vector<Scalar>& xe) {
    Scalar m = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        m = std::max(m, std::abs(x[i] - xe[i]));
    }
    return m;
}

LinearSolverSettings amg_settings() {
    LinearSolverSettings s;
    s.type = LinearSolverType::AmgCg;
    s.tolerance = 1e-12;
    s.rel_tolerance = 0.0;
    s.max_iterations = 200;
    return s;
}

// Solves the Dirichlet-xmin Poisson problem with a manufactured right-hand side through
// pcg, checks convergence and accuracy, and returns the iteration count.
int run_pcg_poisson(const Mesh& m) {
    const LduMatrix ldu = build_poisson(m, true);
    const CsrMatrix a = to_csr(ldu);
    const auto n = idx(ldu.n_rows());
    const std::vector<Scalar> xe = manufactured(n, 0.01);
    const DenseVector x_exact = to_dense(xe);
    const DenseVector b = a * x_exact;
    const Scalar tol = 1e-8 * b.norm();

    AmgPreconditioner amg;
    amg.setup(a);
    DenseVector x = DenseVector::Zero(static_cast<Eigen::Index>(n));
    const KrylovResult r = pcg(a, b, x, amg, tol, 200);

    EXPECT_LE(r.residual_norm, tol);
    EXPECT_LT(r.iterations, 30);
    EXPECT_LT((x - x_exact).cwiseAbs().maxCoeff(), 1e-5);
    return r.iterations;
}

} // namespace

TEST(Amg, HierarchyIsBuilt) {
    const Mesh m = box(64, 64, 1);
    const CsrMatrix a = to_csr(build_poisson(m, true));

    AmgPreconditioner amg;
    amg.setup(a);

    const int levels = amg.levels();
    EXPECT_GE(levels, 3);
    for (int l = 1; l < levels; ++l) {
        EXPECT_LT(amg.level_size(l), amg.level_size(l - 1));
    }
    EXPECT_LE(amg.level_size(levels - 1), 400);
    EXPECT_GE(amg.operator_complexity(), 1.0);
    EXPECT_LE(amg.operator_complexity(), 3.0);
}

TEST(Amg, PcgConvergesFastAndIndependentOfSize) {
    const Label sizes[] = {32, 64, 128};
    int iters[3] = {0, 0, 0};
    for (std::size_t k = 0; k < 3; ++k) {
        const Mesh m = box(sizes[k], sizes[k], 1);
        const LduMatrix ldu = build_poisson(m, true);
        const CsrMatrix a = to_csr(ldu);
        const auto n = idx(ldu.n_rows());
        const std::vector<Scalar> xe = manufactured(n, 0.01);
        const DenseVector x_exact = to_dense(xe);
        const DenseVector b = a * x_exact;
        const Scalar tol = 1e-8 * b.norm();

        AmgPreconditioner amg;
        amg.setup(a);
        DenseVector x = DenseVector::Zero(static_cast<Eigen::Index>(n));
        const KrylovResult r = pcg(a, b, x, amg, tol, 200);
        iters[k] = r.iterations;

        EXPECT_LE(r.residual_norm, tol);
        EXPECT_LT(r.iterations, 30);
        EXPECT_LT((x - x_exact).cwiseAbs().maxCoeff(), 1e-5);
    }
    EXPECT_LE(iters[2], iters[0] + 10);
}

TEST(Amg, Poisson3D) {
    const Mesh m = box(16, 16, 16);
    const int iterations = run_pcg_poisson(m);
    EXPECT_LT(iterations, 30);
}

TEST(Amg, LduSolveAmgCg) {
    const Mesh m = box(40, 40, 1);
    const LduMatrix proto = build_poisson(m, true);
    const auto n = idx(proto.n_rows());
    const std::vector<Scalar> xe = manufactured(n, 0.01);

    std::vector<Scalar> b;
    proto.multiply(xe, b);

    // Without a precomputed pattern.
    LduMatrix eq = proto;
    eq.source = b;
    std::vector<Scalar> x(n, 0.0);
    (void)eq.solve(x, amg_settings());
    EXPECT_LT(max_abs_diff(x, xe), 1e-6);

    // With a precomputed CSR pattern.
    const CsrPattern pattern = build_csr_pattern(m);
    LduMatrix eq_pat = proto;
    eq_pat.source = b;
    eq_pat.pattern = &pattern;
    std::vector<Scalar> xp(n, 0.0);
    (void)eq_pat.solve(xp, amg_settings());
    EXPECT_LT(max_abs_diff(xp, xe), 1e-6);

    EXPECT_LT(max_abs_diff(x, xp), 1e-10);
}

TEST(Amg, AmgCacheReuse) {
    const Mesh m = box(40, 40, 1);
    const LduMatrix proto = build_poisson(m, true);
    const auto n = idx(proto.n_rows());

    AmgCache cache;
    const LinearSolverSettings settings = amg_settings();
    for (int k = 0; k < 5; ++k) {
        const std::vector<Scalar> xe = manufactured(n, 0.01 * static_cast<Scalar>(k + 1));
        std::vector<Scalar> b;
        proto.multiply(xe, b);

        LduMatrix eq = proto;
        eq.source = b;
        std::vector<Scalar> x(n, 0.0);
        (void)eq.solve(x, settings, &cache);
        EXPECT_LT(max_abs_diff(x, xe), 1e-6);
    }
    EXPECT_NE(cache.amg, nullptr);
    EXPECT_EQ(cache.age, 5);
}

TEST(Amg, SingularNeumannWithReference) {
    const Mesh m = box(40, 40, 1);
    LduMatrix a = build_poisson(m, false);
    a.source.assign(idx(a.n_rows()), 0.0);
    a.set_reference(0, 3.0);

    std::vector<Scalar> x(idx(a.n_rows()), 0.0);
    (void)a.solve(x, amg_settings());
    for (const Scalar v : x) {
        EXPECT_NEAR(v, 3.0, 1e-6);
    }
}

// The distributed AMG on a single process (no ghosts) is a working V-cycle preconditioner.
TEST(Amg, ParallelAmgSingleProcess) {
    const Mesh m = box(96, 96, 1);
    const CsrMatrix a = to_csr(build_poisson(m, true));
    cfd::parallel::Halo halo;
    halo.n_owned = static_cast<Label>(a.rows());
    cfd::fv::ParallelAmg amg;
    amg.setup(a, halo);
    EXPECT_GE(amg.levels(), 3);
    for (int l = 1; l < amg.levels(); ++l) {
        EXPECT_LT(amg.level_size(l), amg.level_size(l - 1));
    }
    // Preconditioned CG.
    const DenseVector b = DenseVector::Ones(a.rows());
    DenseVector x = DenseVector::Zero(a.rows());
    DenseVector r = b;
    DenseVector z;
    amg.apply(r, z);
    DenseVector p = z;
    Scalar rz = r.dot(z);
    int it = 0;
    while (it < 100 && r.norm() > 1e-8 * b.norm()) {
        const DenseVector q = a * p;
        const Scalar alpha = rz / p.dot(q);
        x += alpha * p;
        r -= alpha * q;
        amg.apply(r, z);
        const Scalar rz_new = r.dot(z);
        p = z + (rz_new / rz) * p;
        rz = rz_new;
        ++it;
    }
    EXPECT_LT(it, 25);
    EXPECT_LT((b - a * x).norm(), 1e-7 * b.norm());
}
