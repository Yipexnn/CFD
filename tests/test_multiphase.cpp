#include <cfd/fv/k_omega_sst.hpp>
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

// A semicircular droplet on a wall relaxes towards a circular cap with the prescribed static
// contact angle (wetting and non-wetting).
TEST(Multiphase, DropletSpreadsToContactAngle) {
    for (const Scalar theta_deg : {60.0, 120.0}) {
        cfd::mesh::BoxSpec spec;
        spec.nx = 48;
        spec.ny = 20;
        spec.nz = 1;
        const Scalar dx = 6e-3 / spec.nx;
        spec.min = Vec3(-3e-3, 0.0, 0.0);
        spec.max = Vec3(3e-3, 2.5e-3, dx);
        const auto mesh = cfd::mesh::make_box_mesh(spec);
        MultiphaseSettings s;
        s.nu1 = 1e-4;
        s.sigma = 0.07;
        BoundaryCondition wall{FlowBoundary::Wall, Vec3::Zero(), 0.0};
        wall.contact_angle = theta_deg;
        const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
        const BoundaryCondition slip{FlowBoundary::Symmetry, Vec3::Zero(), 0.0};
        MultiphaseSolver solver(mesh,
                                {{"ymin", wall},
                                 {"xmin", slip},
                                 {"xmax", slip},
                                 {"ymax", {FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0}},
                                 {"zmin", empty},
                                 {"zmax", empty}},
                                s);
        const Scalar r0 = 1e-3;
        std::vector<Scalar> alpha(static_cast<std::size_t>(mesh.n_cells()));
        for (std::size_t c = 0; c < alpha.size(); ++c) {
            alpha[c] = mesh.cell_centres()[c].norm() < r0 ? 1.0 : 0.0;
        }
        solver.set_alpha(alpha);
        for (int n = 0; n < 1200; ++n) {
            (void)solver.step(2.4e-5);
        }
        Scalar area = 0.0;
        Scalar h = 0.0;
        for (std::size_t c = 0; c < alpha.size(); ++c) {
            const Scalar a = solver.alpha()[c] * dx * 1.25e-4;
            area += a;
            const Scalar x = mesh.cell_centres()[c].x();
            h += x > 0.0 && x < dx ? a / dx : 0.0;
        }
        auto cap_height = [&](Scalar th) {
            return std::sqrt(area / (th - std::sin(th) * std::cos(th))) * (1.0 - std::cos(th));
        };
        Scalar lo = 0.05;
        Scalar hi = 3.0;
        for (int k = 0; k < 100; ++k) {
            const Scalar mid = 0.5 * (lo + hi);
            (cap_height(mid) < h ? lo : hi) = mid;
        }
        EXPECT_NEAR(0.5 * (lo + hi) * 180.0 / 3.141592653589793, theta_deg, 7.0);
    }
}

// Mixture k-omega SST: a VOF channel filled with one phase reproduces the single-phase
// SIMPLE + SST solution (developing turbulent half channel with wall functions).
TEST(Multiphase, TurbulentSinglePhaseChannelMatchesFlowSolver) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 40;
    spec.ny = 16;
    spec.nz = 1;
    spec.max = Vec3(10.0, 1.0, 0.1);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const std::map<std::string, BoundaryCondition> bcs{
        {"xmin", {FlowBoundary::VelocityInlet, Vec3(1, 0, 0), 0.0}},
        {"xmax", {FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0}},
        {"ymin", {FlowBoundary::Wall, Vec3::Zero(), 0.0}},
        {"ymax", {FlowBoundary::Symmetry, Vec3::Zero(), 0.0}},
        {"zmin", empty},
        {"zmax", empty}};
    const Scalar nu = 1e-5;

    FlowSettings fs;
    fs.nu = nu;
    fs.convection = ConvectionScheme::Upwind; // as the VOF momentum equation
    FlowSolver flow(mesh, bcs, fs);
    KOmegaSst sst(flow);
    flow.set_turbulence_model(&sst);
    flow.set_uniform_velocity(Vec3(1, 0, 0));
    int it = 0;
    for (; it < 5000 && flow.simple_iteration().max() > 1e-8; ++it) {
    }
    EXPECT_LT(it, 5000);

    MultiphaseSettings ms;
    ms.rho1 = 1.0;
    ms.nu1 = nu;
    ms.rho2 = 1.0;
    ms.nu2 = nu;
    ms.turbulence = true;
    MultiphaseSolver vof(mesh, bcs, ms);
    vof.set_alpha(std::vector<Scalar>(static_cast<std::size_t>(mesh.n_cells()), 1.0));
    vof.set_inlet_alpha("xmin", 1.0);
    vof.set_velocity(std::vector<Vec3>(static_cast<std::size_t>(mesh.n_cells()), Vec3(1, 0, 0)));
    ASSERT_NE(vof.sst(), nullptr);
    Residuals r;
    for (int n = 0; n < 400; ++n) {
        r = vof.step(0.2);
    }
    EXPECT_GT(r.k, 0.0);
    Scalar du = 0.0;
    Scalar dnut = 0.0;
    Scalar nut_max = 0.0;
    for (std::size_t c = 0; c < flow.velocity().size(); ++c) {
        du = std::max(du, (vof.velocity()[c] - flow.velocity()[c]).norm());
        dnut = std::max(dnut, std::abs(vof.sst()->nut()[c] - sst.nut()[c]));
        nut_max = std::max(nut_max, sst.nut()[c]);
    }
    EXPECT_GT(nut_max, 10.0 * nu); // turbulent
    EXPECT_LT(du, 0.02);
    EXPECT_LT(dnut, 0.05 * nut_max);
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

// Multiphase case with "turbulence": "kOmegaSST": mixture model active, k in the results
// and restored from the checkpoint.
TEST(Multiphase, TurbulentCaseFileRestart) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_multiphase_sst_case";
    std::filesystem::remove_all(dir);
    std::string json = kDamCase;
    const auto pos = json.find("\"physics\": {");
    ASSERT_NE(pos, std::string::npos);
    json.insert(pos + 12, "\"turbulence\": \"kOmegaSST\", ");
    cfd::cases::CaseConfig c = cfd::cases::parse_case(json, dir);
    ASSERT_TRUE(c.multiphase.has_value());
    EXPECT_TRUE(c.multiphase->turbulence);
    cfd::cases::CaseRunner runner(c);
    ASSERT_NE(runner.sst(), nullptr);
    const auto result = runner.run();
    EXPECT_GT(result.final_residuals.k, 0.0);
    c.restart = "results/checkpoint.bin";
    cfd::cases::CaseRunner second(c);
    for (std::size_t i = 0; i < runner.sst()->k().size(); ++i) {
        EXPECT_DOUBLE_EQ(second.sst()->k()[i], runner.sst()->k()[i]);
    }
}

} // namespace
