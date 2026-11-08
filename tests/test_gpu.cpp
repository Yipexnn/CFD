#include <cfd/case/case_config.hpp>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/gpu/krylov.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using cfd::core::Scalar;
using cfd::core::Vec3;

namespace {

// 2D five-point operator on an m x m grid: -laplace + c d/dx (c = 0: symmetric).
struct Csr {
    std::vector<int> rp{0};
    std::vector<int> ci;
    std::vector<double> va;
    int n = 0;

    explicit Csr(int m, double c) : n(m * m) {
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < m; ++i) {
                auto add = [&](int ii, int jj, double v) {
                    if (ii >= 0 && ii < m && jj >= 0 && jj < m) {
                        ci.push_back(ii + jj * m);
                        va.push_back(v);
                    }
                };
                add(i, j, 4.0);
                add(i - 1, j, -1.0 - c);
                add(i + 1, j, -1.0 + c);
                add(i, j - 1, -1.0);
                add(i, j + 1, -1.0);
                rp.push_back(static_cast<int>(ci.size()));
            }
        }
    }

    [[nodiscard]] cfd::gpu::CsrView view() const {
        return {n, static_cast<int>(va.size()), rp.data(), ci.data(), va.data()};
    }

    [[nodiscard]] double residual(const std::vector<double>& b,
                                  const std::vector<double>& x) const {
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            double ax = 0.0;
            for (int k = rp[static_cast<std::size_t>(i)]; k < rp[static_cast<std::size_t>(i) + 1];
                 ++k) {
                ax += va[static_cast<std::size_t>(k)] *
                      x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            }
            s += std::pow(b[static_cast<std::size_t>(i)] - ax, 2);
        }
        return std::sqrt(s);
    }
};

} // namespace

TEST(Gpu, BackendReported) {
    const std::string name = cfd::gpu::backend_name();
    EXPECT_TRUE(name == "host" || name == "cuda" || name == "openmp-offload") << name;
    EXPECT_FALSE(cfd::gpu::device_description().empty());
}

TEST(Gpu, CgSolvesPoisson) {
    const Csr a(40, 0.0);
    std::vector<double> b(static_cast<std::size_t>(a.n), 1.0);
    std::vector<double> x(b.size(), 0.0);
    const auto st =
        cfd::gpu::solve(a.view(), b.data(), x.data(), cfd::gpu::Method::Cg, {1e-10, 2000});
    EXPECT_TRUE(st.converged);
    EXPECT_GT(st.iterations, 10);
    EXPECT_LT(st.residual, 1e-10);
    EXPECT_LT(a.residual(b, x), 1e-9);
}

TEST(Gpu, BiCgStabSolvesNonSymmetric) {
    const Csr a(40, 0.4);
    std::vector<double> b(static_cast<std::size_t>(a.n));
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = std::sin(0.1 * static_cast<double>(i));
    }
    std::vector<double> x(b.size(), 0.0);
    const auto st =
        cfd::gpu::solve(a.view(), b.data(), x.data(), cfd::gpu::Method::BiCgStab, {1e-10, 2000});
    EXPECT_TRUE(st.converged);
    EXPECT_LT(a.residual(b, x), 1e-9);
}

TEST(Gpu, ZeroIterationsWhenConverged) {
    const Csr a(10, 0.0);
    std::vector<double> x(static_cast<std::size_t>(a.n), 1.0);
    std::vector<double> b(x.size());
    for (int i = 0; i < a.n; ++i) {
        double s = 0.0;
        for (int k = a.rp[static_cast<std::size_t>(i)]; k < a.rp[static_cast<std::size_t>(i) + 1];
             ++k) {
            s += a.va[static_cast<std::size_t>(k)];
        }
        b[static_cast<std::size_t>(i)] = s;
    }
    const auto st =
        cfd::gpu::solve(a.view(), b.data(), x.data(), cfd::gpu::Method::Cg, {1e-12, 10});
    EXPECT_TRUE(st.converged);
    EXPECT_EQ(st.iterations, 0);
}

// Lid-driven cavity with the accelerator solvers (pressure CG, momentum BiCGStab) converges
// to the same solution as the CPU AMG solvers.
TEST(Gpu, CavityMatchesCpuSolvers) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 24;
    spec.ny = 24;
    spec.nz = 1;
    spec.max = Vec3(1.0, 1.0, 1.0 / 24);
    spec.patch_types[4] = cfd::mesh::PatchType::Symmetry;
    spec.patch_types[5] = cfd::mesh::PatchType::Symmetry;
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    auto run = [&](bool gpu) {
        cfd::fv::FlowSettings s;
        s.nu = 0.01;
        if (gpu) {
            s.p_solver.type = cfd::fv::LinearSolverType::GpuCg;
            s.u_solver.type = cfd::fv::LinearSolverType::GpuBiCgStab;
        }
        cfd::fv::FlowSolver solver(
            mesh, {{"ymax", {cfd::fv::FlowBoundary::Wall, Vec3(1.0, 0.0, 0.0), 0.0}}}, s);
        EXPECT_LT(solver.solve_steady(1e-8, 4000), 4000);
        return solver.velocity();
    };
    const auto u_cpu = run(false);
    const auto u_gpu = run(true);
    for (std::size_t c = 0; c < u_cpu.size(); ++c) {
        EXPECT_NEAR((u_gpu[c] - u_cpu[c]).norm(), 0.0, 1e-5);
    }
}

TEST(Gpu, CaseFileDevice) {
    const std::string json = R"({"name": "c", "mesh": {"type": "box", "cells": [4, 4, 1]},
                                 "solver": {"device": "gpu"}})";
    const auto c = cfd::cases::parse_case(json);
    EXPECT_EQ(c.solver.flow.p_solver.type, cfd::fv::LinearSolverType::GpuCg);
    EXPECT_EQ(c.solver.flow.u_solver.type, cfd::fv::LinearSolverType::GpuBiCgStab);
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c));
    EXPECT_EQ(b.solver.flow.p_solver.type, cfd::fv::LinearSolverType::GpuCg);
    const std::string bad = R"({"name": "c", "mesh": {"type": "box", "cells": [4, 4, 1]},
                                "solver": {"device": "tpu"}})";
    EXPECT_THROW((void)cfd::cases::parse_case(bad), std::runtime_error);
}

#include <cfd/fv/amg.hpp>

// AMG-preconditioned CG with the V-cycle on the accelerator path: converges to the same
// solution with a small, mesh-independent iteration count (Jacobi-PCG needs many more).
TEST(Gpu, AmgCgSolvesPoissonFast) {
    for (const int m : {40, 80}) {
        const Csr a(m, 0.0);
        std::vector<double> b(static_cast<std::size_t>(a.n), 1.0);
        cfd::fv::CsrMatrix e(a.n, a.n);
        std::vector<Eigen::Triplet<double, int>> trip;
        for (int i = 0; i < a.n; ++i) {
            for (int k = a.rp[static_cast<std::size_t>(i)];
                 k < a.rp[static_cast<std::size_t>(i) + 1]; ++k) {
                trip.emplace_back(i, a.ci[static_cast<std::size_t>(k)],
                                  a.va[static_cast<std::size_t>(k)]);
            }
        }
        e.setFromTriplets(trip.begin(), trip.end());
        e.makeCompressed();
        cfd::fv::AmgPreconditioner amg;
        amg.setup(e);
        ASSERT_GT(amg.levels(), 1);
        std::vector<cfd::fv::CsrMatrix> keep;
        keep.reserve(3 * static_cast<std::size_t>(amg.levels()));
        auto view = [](const cfd::fv::CsrMatrix& c) {
            return cfd::gpu::CsrView{static_cast<int>(c.rows()), static_cast<int>(c.nonZeros()),
                                     c.outerIndexPtr(), c.innerIndexPtr(), c.valuePtr()};
        };
        cfd::gpu::AmgView h;
        for (int l = 0; l < amg.levels(); ++l) {
            cfd::gpu::AmgLevelView lv;
            keep.push_back(amg.level_matrix(l));
            keep.back().makeCompressed();
            lv.a = view(keep.back());
            if (l + 1 < amg.levels()) {
                keep.push_back(amg.level_prolongation(l));
                keep.back().makeCompressed();
                lv.p = view(keep.back());
                keep.push_back(amg.level_restriction(l));
                keep.back().makeCompressed();
                lv.r = view(keep.back());
            }
            h.levels.push_back(lv);
        }
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inv =
            amg.coarse_inverse();
        h.coarse_inverse = inv.data();
        std::vector<double> x(b.size(), 0.0);
        const auto st = cfd::gpu::solve_amg_cg(h, b.data(), x.data(), {1e-10, 500});
        EXPECT_TRUE(st.converged);
        EXPECT_LT(a.residual(b, x), 1e-9);
        std::vector<double> xj(b.size(), 0.0);
        const auto sj =
            cfd::gpu::solve(a.view(), b.data(), xj.data(), cfd::gpu::Method::Cg, {1e-10, 5000});
        EXPECT_LT(3 * st.iterations, sj.iterations) << "m = " << m;
        EXPECT_LT(st.iterations, 40) << "m = " << m;
    }
}

#include <cfd/fv/ldu_matrix.hpp>

// Persistent device system: repeated solves with changing LDU coefficients on one pattern
// (assembly from LDU, cached AMG levels) agree with the one-shot CPU solvers.
TEST(Gpu, PersistentSystemRepeatedSolves) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 16;
    spec.ny = 16;
    spec.nz = 1;
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    const cfd::fv::CsrPattern pattern = cfd::fv::build_csr_pattern(mesh);
    cfd::fv::AmgCache cache;
    for (int round = 0; round < 3; ++round) {
        // Diffusion-like matrix with a small diagonal shift that changes per round.
        cfd::fv::LduMatrix a(mesh);
        a.pattern = &pattern;
        const auto nc = static_cast<std::size_t>(mesh.n_cells());
        for (std::size_t f = 0; f < a.upper.size(); ++f) {
            const auto o = static_cast<std::size_t>(mesh.owner()[f]);
            const auto nb = static_cast<std::size_t>(mesh.neighbour()[f]);
            a.upper[f] = -1.0;
            a.lower[f] = -1.0;
            a.diag[o] += 1.0;
            a.diag[nb] += 1.0;
        }
        for (std::size_t c = 0; c < nc; ++c) {
            a.diag[c] += 0.01 * (round + 1);
            a.source[c] = std::sin(0.3 * static_cast<double>(c));
        }
        for (const auto type :
             {cfd::fv::LinearSolverType::GpuCg, cfd::fv::LinearSolverType::GpuBiCgStab}) {
            std::vector<double> x(nc, 0.0);
            std::vector<double> ref(nc, 0.0);
            (void)a.solve(x, {type, 1e-14, 1e-10, 1000}, &cache);
            (void)a.solve(ref, {cfd::fv::LinearSolverType::AmgCg, 1e-14, 1e-10, 1000});
            for (std::size_t c = 0; c < nc; ++c) {
                EXPECT_NEAR(x[c], ref[c], 1e-7 * (1.0 + std::abs(ref[c])));
            }
        }
    }
    ASSERT_NE(cache.device, nullptr);
    EXPECT_EQ(cache.device->on_device(),
              cfd::gpu::device_available() && cfd::gpu::backend_name() == "cuda");
}
