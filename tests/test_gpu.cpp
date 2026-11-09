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
    // Default "auto": GPU solvers only if the build finds an accelerator; round trip keeps it.
    const auto a =
        cfd::cases::parse_case(R"({"name": "c", "mesh": {"type": "box", "cells": [4, 4, 1]}})");
    EXPECT_EQ(a.solver.device, cfd::cases::ComputeDevice::Auto);
    EXPECT_EQ(a.solver.flow.p_solver.type == cfd::fv::LinearSolverType::GpuCg,
              cfd::gpu::device_available());
    EXPECT_EQ(cfd::cases::parse_case(cfd::cases::case_to_json(a)).solver.device,
              cfd::cases::ComputeDevice::Auto);
    const auto cpu = cfd::cases::parse_case(
        R"({"name": "c", "mesh": {"type": "box", "cells": [4, 4, 1]}, "solver": {"device": "cpu"}})");
    EXPECT_EQ(cpu.solver.flow.p_solver.type, cfd::fv::LinearSolverType::AmgCg);
    EXPECT_EQ(cpu.solver.flow.u_solver.type, cfd::fv::LinearSolverType::BiCgStab);
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

#include <cfd/fv/scalar_transport.hpp>

// FV assembly kernel = CPU assembly; a scalar transport solve with the accelerator solvers
// (assembly + solve) gives the CPU result.
TEST(Gpu, AssemblyMatchesCpu) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 7;
    spec.ny = 5;
    spec.nz = 3;
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    const auto nif = static_cast<std::size_t>(mesh.n_internal_faces());
    const auto nc = static_cast<std::size_t>(mesh.n_cells());
    std::vector<double> flux(nif);
    std::vector<double> gamma(nif);
    std::vector<double> dc(nif);
    for (std::size_t f = 0; f < nif; ++f) {
        flux[f] = std::sin(1.3 * static_cast<double>(f));
        gamma[f] = 0.1 + 0.01 * static_cast<double>(f % 7);
        dc[f] = 2.0 + 0.1 * static_cast<double>(f % 3);
    }
    std::vector<double> up(nif), lo(nif), diag(nc, 1.0), div(nc, 0.0);
    cfd::gpu::assemble_convection_diffusion(
        static_cast<int>(nc), static_cast<int>(nif), mesh.owner().data(), mesh.neighbour().data(),
        flux.data(), gamma.data(), dc.data(), 1.0, up.data(), lo.data(), diag.data(), div.data());
    std::vector<double> diag_ref(nc, 1.0), div_ref(nc, 0.0);
    for (std::size_t f = 0; f < nif; ++f) {
        const double g = gamma[f] * dc[f];
        EXPECT_DOUBLE_EQ(up[f], std::min(flux[f], 0.0) - g);
        EXPECT_DOUBLE_EQ(lo[f], -std::max(flux[f], 0.0) - g);
        const auto p = static_cast<std::size_t>(mesh.owner()[f]);
        const auto n = static_cast<std::size_t>(mesh.neighbour()[f]);
        diag_ref[p] += std::max(flux[f], 0.0) + g;
        diag_ref[n] += -std::min(flux[f], 0.0) + g;
        div_ref[p] += flux[f];
        div_ref[n] -= flux[f];
    }
    for (std::size_t c = 0; c < nc; ++c) {
        EXPECT_NEAR(diag[c], diag_ref[c], 1e-12);
        EXPECT_NEAR(div[c], div_ref[c], 1e-12);
    }

    // Diffusion + source solve through solve_scalar with CPU and accelerator settings.
    const cfd::fv::FvGeometry geo(mesh);
    cfd::fv::ScalarEquation eq;
    eq.gamma_face.assign(static_cast<std::size_t>(mesh.n_faces()), 0.5);
    eq.su.assign(nc, 1.0);
    eq.sp.assign(nc, -0.2);
    eq.boundary.assign(static_cast<std::size_t>(mesh.n_boundary_faces()),
                       {cfd::fv::ScalarBoundary::Kind::FixedValue, 0.0});
    const std::vector<double> zero_flux(static_cast<std::size_t>(mesh.n_faces()), 0.0);
    std::vector<double> t_cpu(nc, 0.0);
    std::vector<double> t_gpu(nc, 0.0);
    (void)cfd::fv::solve_scalar(geo, zero_flux, eq, t_cpu,
                                {cfd::fv::LinearSolverType::BiCgStab, 1e-14, 1e-12, 1000});
    (void)cfd::fv::solve_scalar(geo, zero_flux, eq, t_gpu,
                                {cfd::fv::LinearSolverType::GpuBiCgStab, 1e-14, 1e-12, 1000});
    for (std::size_t c = 0; c < nc; ++c) {
        EXPECT_NEAR(t_gpu[c], t_cpu[c], 1e-9);
    }
}

#include <cfd/fv/scalar_transport.hpp>
#include <cfd/gpu/fv_device.hpp>

namespace {

// Box mesh with the interior points moved in x / y (skewed, non-orthogonal cells; planar faces).
cfd::mesh::Mesh skewed_box(int nx, int ny, int nz, double lz) {
    cfd::mesh::BoxSpec spec;
    spec.nx = nx;
    spec.ny = ny;
    spec.nz = nz;
    spec.max = Vec3(1.0, 1.0, lz);
    if (nz == 1) {
        spec.patch_types[4] = cfd::mesh::PatchType::Symmetry;
        spec.patch_types[5] = cfd::mesh::PatchType::Symmetry;
    }
    const auto box = cfd::mesh::make_box_mesh(spec);
    std::vector<Vec3> pts = box.points();
    const double hx = 1.0 / nx;
    const double hy = 1.0 / ny;
    for (auto& p : pts) {
        if (p.x() > 1e-9 && p.x() < 1.0 - 1e-9 && p.y() > 1e-9 && p.y() < 1.0 - 1e-9) {
            const double s = 17.0 * p.x() + 31.0 * p.y();
            p.x() += 0.25 * hx * std::sin(s);
            p.y() += 0.25 * hy * std::cos(1.7 * s);
        }
    }
    std::vector<cfd::core::Label> offsets{0};
    std::vector<cfd::core::Label> fp;
    for (cfd::core::Label f = 0; f < box.n_faces(); ++f) {
        const auto face = box.face(f);
        fp.insert(fp.end(), face.begin(), face.end());
        offsets.push_back(static_cast<cfd::core::Label>(fp.size()));
    }
    return {pts, offsets, fp, box.owner(), box.neighbour(), box.patches(), box.n_cells()};
}

} // namespace

// Device gradients (Green-Gauss, least squares) and scalar face corrections equal the CPU loops.
TEST(Gpu, FvKernelsMatchCpu) {
    const auto mesh = skewed_box(6, 5, 4, 1.0);
    const cfd::fv::FvGeometry geo(mesh);
    const auto nc = static_cast<std::size_t>(mesh.n_cells());
    const auto nf = static_cast<std::size_t>(mesh.n_faces());
    const auto nif = static_cast<std::size_t>(mesh.n_internal_faces());
    auto field = [](const Vec3& x) {
        return std::sin(2.0 * x.x()) + x.y() * x.y() + x.x() * x.z();
    };
    std::vector<Scalar> phi(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        phi[c] = field(mesh.cell_centres()[c]);
    }
    std::vector<Scalar> bv(nf - nif);
    for (std::size_t f = nif; f < nf; ++f) {
        bv[f - nif] = field(mesh.face_centres()[f]);
    }
    for (const auto scheme :
         {cfd::fv::GradientScheme::Gauss, cfd::fv::GradientScheme::LeastSquares}) {
        const auto cpu = geo.gradient(phi, bv, scheme, false);
        const auto dev = geo.gradient(phi, bv, scheme, true);
        for (std::size_t c = 0; c < nc; ++c) {
            EXPECT_NEAR((dev[c] - cpu[c]).norm(), 0.0, 1e-12);
        }
    }
    EXPECT_EQ(geo.device().on_device(), cfd::gpu::device_available());

    // Skewness-corrected Gauss: exact for a linear field on the skewed mesh (plain Gauss is not).
    {
        const Vec3 a(0.7, -1.3, 2.1);
        std::vector<Scalar> lin(nc);
        std::vector<Scalar> lin_b(nf - nif);
        for (std::size_t c = 0; c < nc; ++c) {
            lin[c] = a.dot(mesh.cell_centres()[c]) + 0.4;
        }
        for (std::size_t f = nif; f < nf; ++f) {
            lin_b[f - nif] = a.dot(mesh.face_centres()[f]) + 0.4;
        }
        const auto gc = geo.gradient(lin, lin_b, cfd::fv::GradientScheme::GaussCorrected);
        const auto gg = geo.gradient(lin, lin_b, cfd::fv::GradientScheme::Gauss);
        double err_c = 0.0;
        double err_g = 0.0;
        for (std::size_t c = 0; c < nc; ++c) {
            err_c = std::max(err_c, (gc[c] - a).norm());
            err_g = std::max(err_g, (gg[c] - a).norm());
        }
        EXPECT_LT(err_c, 1e-10);
        EXPECT_GT(err_g, 1e-3);
    }

    // Explicit linear-upwind + non-orthogonal corrections against the CPU face loop.
    std::vector<Scalar> flux(nf);
    std::vector<Scalar> gamma(nf);
    for (std::size_t f = 0; f < nf; ++f) {
        flux[f] = std::sin(0.7 * static_cast<double>(f));
        gamma[f] = 0.05 + 0.01 * static_cast<double>(f % 5);
    }
    const auto g = geo.gradient(phi, bv, cfd::fv::GradientScheme::LeastSquares, false);
    std::vector<Scalar> ref(nc, 0.5);
    std::vector<Scalar> src(nc, 0.5);
    const auto& cf = mesh.face_centres();
    const auto& cc = mesh.cell_centres();
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(mesh.owner()[f]);
        const auto n = static_cast<std::size_t>(mesh.neighbour()[f]);
        const std::size_t up = flux[f] >= 0.0 ? p : n;
        const double w = geo.weights()[f];
        const double e = -flux[f] * g[up].dot(cf[f] - cc[up]) +
                         gamma[f] * geo.corrections()[f].dot(w * g[p] + (1.0 - w) * g[n]);
        ref[p] += e;
        ref[n] -= e;
    }
    geo.device().scalar_corrections(phi.data(), bv.data(), true, flux.data(), gamma.data(), true,
                                    src.data());
    for (std::size_t c = 0; c < nc; ++c) {
        EXPECT_NEAR(src[c], ref[c], 1e-12);
    }
}

// Lid-driven cavity on a skewed mesh with limited linear upwind and least-squares gradients:
// the accelerator path (gradients, limiter, momentum and scalar corrections, assembly,
// AMG-PCG) converges to the CPU solution; the variable viscosity enables the transpose term.
TEST(Gpu, SkewedCavityMatchesCpu) {
    const auto mesh = skewed_box(16, 16, 1, 1.0 / 16);
    auto run = [&](bool gpu, cfd::fv::Limiter limiter, cfd::fv::ConvectionScheme scheme) {
        cfd::fv::FlowSettings s;
        s.nu = 0.01;
        s.convection = scheme;
        s.limiter = limiter;
        s.gradient = cfd::fv::GradientScheme::LeastSquares;
        s.n_non_orth_correctors = 1;
        if (gpu) {
            s.p_solver.type = cfd::fv::LinearSolverType::GpuCg;
            s.u_solver.type = cfd::fv::LinearSolverType::GpuBiCgStab;
        }
        cfd::fv::FlowSolver solver(
            mesh, {{"ymax", {cfd::fv::FlowBoundary::Wall, Vec3(1.0, 0.0, 0.0), 0.0}}}, s);
        std::vector<Scalar> nu(static_cast<std::size_t>(mesh.n_cells()));
        for (std::size_t c = 0; c < nu.size(); ++c) {
            nu[c] = 0.01 * (1.0 + 0.5 * mesh.cell_centres()[c].x());
        }
        solver.set_cell_viscosity(nu);
        EXPECT_LT(solver.solve_steady(1e-8, 4000), 4000);
        return solver.velocity();
    };
    for (const auto limiter :
         {cfd::fv::Limiter::BarthJespersen, cfd::fv::Limiter::Venkatakrishnan}) {
        const auto u_cpu = run(false, limiter, cfd::fv::ConvectionScheme::LimitedLinearUpwind);
        const auto u_gpu = run(true, limiter, cfd::fv::ConvectionScheme::LimitedLinearUpwind);
        for (std::size_t c = 0; c < u_cpu.size(); ++c) {
            EXPECT_NEAR((u_gpu[c] - u_cpu[c]).norm(), 0.0, 1e-5);
        }
    }
    {
        const auto u_cpu =
            run(false, cfd::fv::Limiter::BarthJespersen, cfd::fv::ConvectionScheme::Lust);
        const auto u_gpu =
            run(true, cfd::fv::Limiter::BarthJespersen, cfd::fv::ConvectionScheme::Lust);
        for (std::size_t c = 0; c < u_cpu.size(); ++c) {
            EXPECT_NEAR((u_gpu[c] - u_cpu[c]).norm(), 0.0, 1e-5);
        }
    }
    const std::string json = R"({"name": "c", "mesh": {"type": "box", "cells": [4, 4, 1]},
        "solver": {"convection": "limited_linear_upwind", "limiter": "venkatakrishnan",
                   "venkat_k": 0.2}})";
    const auto c = cfd::cases::parse_case(cfd::cases::case_to_json(cfd::cases::parse_case(json)));
    EXPECT_EQ(c.solver.flow.limiter, cfd::fv::Limiter::Venkatakrishnan);
    EXPECT_DOUBLE_EQ(c.solver.flow.venkat_k, 0.2);
}
