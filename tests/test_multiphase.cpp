#include <cfd/fv/multiphase.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <gtest/gtest.h>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

cfd::mesh::Mesh box2d(int n, Scalar lx, Scalar ly) {
    cfd::mesh::BoxSpec spec;
    spec.nx = n;
    spec.ny = static_cast<int>(std::lround(n * ly / lx));
    spec.nz = 1;
    spec.max = Vec3(lx, ly, lx / n);
    return cfd::mesh::make_box_mesh(spec);
}

const std::map<std::string, BoundaryCondition>& empty_z() {
    static const std::map<std::string, BoundaryCondition> bcs{
        {"zmin", {FlowBoundary::Empty, Vec3::Zero(), 0.0}},
        {"zmax", {FlowBoundary::Empty, Vec3::Zero(), 0.0}}};
    return bcs;
}

// Water below air at rest: stays at rest, hydrostatic pressure.
TEST(Multiphase, HydrostaticTankStaysAtRest) {
    const auto mesh = box2d(10, 1.0, 1.0);
    MultiphaseSettings s;
    s.gravity = Vec3(0, -9.81, 0);
    MultiphaseSolver solver(mesh, empty_z(), s);
    std::vector<Scalar> alpha(100);
    for (std::size_t c = 0; c < alpha.size(); ++c) {
        alpha[c] = mesh.cell_centres()[c].y() < 0.5 ? 1.0 : 0.0;
    }
    solver.set_alpha(alpha);
    for (int n = 0; n < 20; ++n) {
        (void)solver.step(1e-3);
    }
    Scalar u_max = 0.0;
    for (const auto& u : solver.velocity()) {
        u_max = std::max(u_max, u.norm());
    }
    EXPECT_LT(u_max, 1e-6);
    // Pressure difference between bottom and top water cells = rho g dy.
    const auto p = solver.pressure();
    EXPECT_NEAR(p[0] - p[40], 1000.0 * 9.81 * 0.4, 1.0);
    EXPECT_NEAR(solver.phase1_volume(), 0.5 * 0.1, 1e-12);
}

// Circular drop (2D) at rest: Laplace pressure jump sigma / R.
TEST(Multiphase, StaticDropLaplacePressure) {
    const auto mesh = box2d(40, 1.0, 1.0);
    MultiphaseSettings s;
    s.rho1 = 1.0;
    s.rho2 = 1.0;
    s.nu1 = 0.1;
    s.nu2 = 0.1;
    s.sigma = 1.0;
    MultiphaseSolver solver(mesh, empty_z(), s);
    const Scalar r = 0.25;
    std::vector<Scalar> alpha(static_cast<std::size_t>(mesh.n_cells()));
    for (std::size_t c = 0; c < alpha.size(); ++c) {
        const Vec3 d = mesh.cell_centres()[c] - Vec3(0.5, 0.5, 0.0);
        // Smooth initial profile over about one cell.
        alpha[c] = 0.5 * (1.0 - std::tanh((std::hypot(d.x(), d.y()) - r) / 0.025));
    }
    solver.set_alpha(alpha);
    for (int n = 0; n < 50; ++n) {
        (void)solver.step(1e-3);
    }
    const auto p = solver.pressure();
    Scalar inside = 0.0;
    Scalar outside = 0.0;
    int n_in = 0;
    int n_out = 0;
    for (std::size_t c = 0; c < p.size(); ++c) {
        const Vec3 d = mesh.cell_centres()[c] - Vec3(0.5, 0.5, 0.0);
        const Scalar dist = std::hypot(d.x(), d.y());
        if (dist < 0.15) {
            inside += p[c];
            ++n_in;
        } else if (dist > 0.4) {
            outside += p[c];
            ++n_out;
        }
    }
    const Scalar jump = inside / n_in - outside / n_out;
    EXPECT_NEAR(jump / (s.sigma / r), 1.0, 0.1);
}

// Dam break: mass conservation, boundedness, the front moves.
TEST(Multiphase, DamBreakConservesMassAndStaysBounded) {
    const Scalar a = 0.05715;
    const auto mesh = box2d(40, 5 * a, 2.5 * a);
    MultiphaseSettings s;
    s.gravity = Vec3(0, -9.81, 0);
    s.sigma = 0.07;
    MultiphaseSolver solver(mesh, empty_z(), s);
    std::vector<Scalar> alpha(static_cast<std::size_t>(mesh.n_cells()));
    for (std::size_t c = 0; c < alpha.size(); ++c) {
        const Vec3& x = mesh.cell_centres()[c];
        alpha[c] = x.x() < a && x.y() < 2 * a ? 1.0 : 0.0;
    }
    solver.set_alpha(alpha);
    const Scalar v0 = solver.phase1_volume();
    Scalar bound = 0.0;
    for (int n = 0; n < 100; ++n) {
        bound = std::max(bound, solver.step(5e-4).t);
    }
    EXPECT_NEAR(solver.phase1_volume() / v0, 1.0, 1e-10);
    EXPECT_LT(bound, 1e-6);
    Scalar front = 0.0;
    for (std::size_t c = 0; c < alpha.size(); ++c) {
        if (mesh.cell_centres()[c].y() < 0.0036 && solver.alpha()[c] > 0.5) {
            front = std::max(front, mesh.cell_centres()[c].x());
        }
    }
    EXPECT_GT(front, 1.15 * a); // t = 0.05 s: T ~ 0.93, Z ~ 1.3
    EXPECT_LT(front, 1.8 * a);
    EXPECT_LT(solver.courant(5e-4), 1.0);
}

} // namespace

#include <cfd/case/case_config.hpp>
#include <cfd/case/case_runner.hpp>
#include <filesystem>

namespace {

constexpr auto kDamCase = R"({
    "name": "dam",
    "mesh": {"type": "box", "cells": [24, 12, 1], "max": [0.28, 0.14, 0.01],
             "patch_types": {"zmin": "patch", "zmax": "patch"}},
    "physics": {"multiphase": {"rho1": 1000, "nu1": 1e-6, "rho2": 1, "nu2": 1.5e-5,
                               "sigma": 0.07, "gravity": [0, -9.81, 0]}},
    "boundaries": {"zmin": {"type": "empty"}, "zmax": {"type": "empty"}},
    "initial": {"alpha_regions": [{"min": [0, 0, -1], "max": [0.06, 0.12, 1]}]},
    "solver": {"mode": "transient", "dt": 1e-3, "end_time": 0.04, "max_courant": 0.5}
})";

TEST(Multiphase, CaseFileRunAndRestart) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_multiphase_case";
    std::filesystem::remove_all(dir);
    cfd::cases::CaseConfig c = cfd::cases::parse_case(kDamCase, dir);
    ASSERT_TRUE(c.multiphase.has_value());
    EXPECT_DOUBLE_EQ(c.multiphase->rho1, 1000.0);
    EXPECT_EQ(c.initial.alpha_regions.size(), 1U);
    EXPECT_DOUBLE_EQ(c.solver.max_courant, 0.5);
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c), dir);
    ASSERT_TRUE(b.multiphase.has_value());
    EXPECT_DOUBLE_EQ(b.multiphase->sigma, 0.07);
    EXPECT_TRUE(b.initial.alpha_regions[0].second.isApprox(Vec3(0.06, 0.12, 1)));

    cfd::cases::CaseRunner runner(c);
    ASSERT_NE(runner.multiphase(), nullptr);
    const Scalar v0 = runner.multiphase()->phase1_volume();
    EXPECT_NEAR(v0 / (0.06 * 0.12 * 0.01), 1.0, 0.1); // cell-centre selection
    const auto result = runner.run();
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(runner.time(), 0.04, 1e-12);
    EXPECT_NEAR(runner.multiphase()->phase1_volume() / v0, 1.0, 1e-10);
    EXPECT_TRUE(std::filesystem::exists(dir / "results" / "dam.vtu"));

    c.solver.end_time = 0.05;
    c.restart = "results/checkpoint.bin";
    cfd::cases::CaseRunner second(c);
    EXPECT_NEAR(second.time(), 0.04, 1e-12);
    for (std::size_t i = 0; i < second.multiphase()->alpha().size(); ++i) {
        EXPECT_DOUBLE_EQ(second.multiphase()->alpha()[i], runner.multiphase()->alpha()[i]);
    }
    (void)second.run();
    EXPECT_NEAR(second.time(), 0.05, 1e-12);

    auto bad = std::string(kDamCase);
    bad.replace(bad.find("\"transient\""), 11, "\"steady\"   ");
    EXPECT_THROW((void)cfd::cases::parse_case(bad), std::runtime_error);
}

} // namespace
