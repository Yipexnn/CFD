#include <algorithm>
#include <cfd/fv/compressible.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <string>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

cfd::mesh::Mesh tube(int n) {
    cfd::mesh::BoxSpec spec;
    spec.nx = n;
    spec.ny = 1;
    spec.nz = 1;
    spec.max = Vec3(1.0, 0.01, 0.01);
    return cfd::mesh::make_box_mesh(spec);
}

std::map<std::string, BoundaryCondition> tube_bcs(FlowBoundary ends) {
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    return {{"ymin", empty},
            {"ymax", empty},
            {"zmin", empty},
            {"zmax", empty},
            {"xmin", {ends, Vec3::Zero(), 0.0}},
            {"xmax", {ends, Vec3::Zero(), 0.0}}};
}

void sod_init(CompressibleSolver& solver, const cfd::mesh::Mesh& mesh) {
    std::vector<GasState> init(static_cast<std::size_t>(mesh.n_cells()));
    for (std::size_t c = 0; c < init.size(); ++c) {
        init[c] = mesh.cell_centres()[c].x() < 0.5 ? GasState{1.0, Vec3::Zero(), 1.0}
                                                   : GasState{0.125, Vec3::Zero(), 0.1};
    }
    solver.set_state(init);
}

// Sod at t = 0.2: plateau values between the contact and the shock (exact: p* = 0.30313,
// u* = 0.92745, rho right of the contact 0.26557) and shock position x = 0.8504.
TEST(Compressible, SodShockTube) {
    const auto mesh = tube(200);
    CompressibleSolver solver(mesh, tube_bcs(FlowBoundary::PressureOutlet), CompressibleSettings{});
    sod_init(solver, mesh);
    Scalar t = 0.0;
    while (t < 0.2 - 1e-12) {
        const Scalar dt = std::min(solver.stable_dt(), 0.2 - t);
        (void)solver.step(dt);
        t += dt;
    }
    const GasState plateau = solver.state(155); // x = 0.7775
    EXPECT_NEAR(plateau.p, 0.30313, 0.003);
    EXPECT_NEAR(plateau.u.x(), 0.92745, 0.01);
    EXPECT_NEAR(plateau.rho, 0.26557, 0.003);
    EXPECT_NEAR(solver.state(175).rho, 0.125, 1e-6); // x = 0.8775: ahead of the shock
    EXPECT_GT(solver.state(165).rho, 0.2);           // x = 0.8275: behind the shock
}

// Closed tube (walls): mass and total energy are conserved exactly.
TEST(Compressible, ClosedTubeConservesMassAndEnergy) {
    const auto mesh = tube(64);
    CompressibleSolver solver(mesh, tube_bcs(FlowBoundary::Wall), CompressibleSettings{});
    sod_init(solver, mesh);
    auto totals = [&] {
        Scalar mass = 0.0;
        Scalar energy = 0.0;
        for (std::size_t c = 0; c < solver.conserved().size(); ++c) {
            mass += solver.conserved()[c][0] * mesh.cell_volumes()[c];
            energy += solver.conserved()[c][4] * mesh.cell_volumes()[c];
        }
        return std::make_pair(mass, energy);
    };
    const auto [m0, e0] = totals();
    for (int n = 0; n < 200; ++n) {
        (void)solver.step(solver.stable_dt());
    }
    const auto [m1, e1] = totals();
    EXPECT_NEAR(m1 / m0, 1.0, 1e-13);
    EXPECT_NEAR(e1 / e0, 1.0, 1e-13);
}

// Uniform supersonic flow through a box (inlet / outlet) stays uniform (also viscous).
TEST(Compressible, FreeStreamPreserved) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 8;
    spec.ny = 4;
    spec.nz = 1;
    spec.grading = Vec3(2.0, 1.0, 1.0);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    CompressibleSettings s;
    s.mu = 1e-5;
    BoundaryCondition inlet{FlowBoundary::VelocityInlet, Vec3(700, 0, 0), 1e5};
    inlet.temperature = 300.0;
    const BoundaryCondition slip{FlowBoundary::Symmetry, Vec3::Zero(), 0.0};
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    CompressibleSolver solver(mesh,
                              {{"xmin", inlet},
                               {"xmax", {FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0}},
                               {"ymin", slip},
                               {"ymax", slip},
                               {"zmin", empty},
                               {"zmax", empty}},
                              s);
    const GasState free{1e5 / (287.0 * 300.0), Vec3(700, 0, 0), 1e5};
    solver.set_uniform(free);
    for (int n = 0; n < 20; ++n) {
        (void)solver.steady_iteration();
    }
    for (std::size_t c = 0; c < 32; ++c) {
        EXPECT_NEAR(solver.state(c).p / free.p, 1.0, 1e-10);
        EXPECT_NEAR(solver.state(c).u.x() / 700.0, 1.0, 1e-10);
    }
    EXPECT_NEAR(solver.mach()[0], 700.0 / std::sqrt(1.4 * 287.0 * 300.0), 1e-10);
}

// Implicit LU-SGS (with and without Krylov acceleration) converges to the explicit RK3
// steady solution of an oblique shock reflecting from a slip wall, in far fewer iterations.
TEST(Compressible, ImplicitMatchesExplicitSteadyShock) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 30;
    spec.ny = 15;
    spec.nz = 1;
    spec.max = Vec3(2.0, 1.0, 0.05);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    const Scalar t = 300.0;
    const Scalar p = 1e5;
    const Vec3 u = 2.5 * std::sqrt(1.4 * 287.0 * t) * Vec3(std::cos(0.2), -std::sin(0.2), 0.0);
    BoundaryCondition inlet{FlowBoundary::VelocityInlet, u, p};
    inlet.temperature = t;
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const std::map<std::string, BoundaryCondition> bcs{
        {"xmin", inlet},
        {"ymax", inlet},
        {"xmax", {FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0}},
        {"ymin", {FlowBoundary::Symmetry, Vec3::Zero(), 0.0}},
        {"zmin", empty},
        {"zmax", empty}};
    auto run = [&](bool implicit, int krylov, Scalar cfl, int& iterations) {
        CompressibleSettings s;
        s.limiter = Limiter::Venkatakrishnan;
        s.implicit = implicit;
        s.krylov_dim = krylov;
        s.cfl = cfl;
        CompressibleSolver solver(mesh, bcs, s);
        solver.set_uniform({p / (287.0 * t), u, p});
        Scalar r0 = 0.0;
        for (iterations = 0; iterations < 20000; ++iterations) {
            const Scalar r = solver.steady_iteration();
            r0 = iterations < 5 ? std::max(r0, r) : r0;
            if (r < 1e-10 * r0) {
                break;
            }
        }
        return solver.pressure();
    };
    int n_explicit = 0;
    int n_lusgs = 0;
    int n_krylov = 0;
    const auto p_explicit = run(false, 0, 0.8, n_explicit);
    const auto p_lusgs = run(true, 0, 50.0, n_lusgs);
    const auto p_krylov = run(true, 10, 1000.0, n_krylov);
    EXPECT_LT(n_explicit, 20000);
    EXPECT_LT(3 * n_lusgs, n_explicit);
    EXPECT_LT(5 * n_krylov, n_explicit);
    Scalar p_max = 0.0;
    for (std::size_t c = 0; c < p_explicit.size(); ++c) {
        EXPECT_NEAR(p_lusgs[c] / p_explicit[c], 1.0, 1e-6);
        EXPECT_NEAR(p_krylov[c] / p_explicit[c], 1.0, 1e-6);
        p_max = std::max(p_max, p_explicit[c]);
    }
    EXPECT_GT(p_max, 1.5 * p); // reflected shock present
}

// Compressible Couette flow at low Mach number: the implicit Newton-Krylov solver gives the
// linear velocity profile on a strongly stretched wall grid (explicit stepping would need
// ~1e7 iterations; low-Mach stiffness still costs some hundred implicit iterations).
TEST(Compressible, KrylovCouetteOnStretchedGrid) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 1;
    spec.ny = 24;
    spec.nz = 1;
    spec.max = Vec3(0.01, 0.01, 0.01);
    spec.grading = Vec3(1.0, 200.0, 1.0);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    CompressibleSettings s;
    s.mu = 1.8e-5;
    s.implicit = true;
    s.krylov_dim = 20;
    s.cfl = 1e5;
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const Scalar u_wall = 10.0;
    CompressibleSolver solver(mesh,
                              {{"xmin", empty},
                               {"xmax", empty},
                               {"zmin", empty},
                               {"zmax", empty},
                               {"ymin", {FlowBoundary::Wall, Vec3::Zero(), 0.0}},
                               {"ymax", {FlowBoundary::Wall, Vec3(u_wall, 0, 0), 0.0}}},
                              s);
    solver.set_uniform({1.16, Vec3::Zero(), 1e5});
    auto error = [&] {
        Scalar e = 0.0;
        for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_cells()); ++c) {
            const Scalar y = mesh.cell_centres()[c].y();
            e = std::max(e, std::abs(solver.state(c).u.x() - u_wall * y / 0.01));
        }
        return e / u_wall;
    };
    int it = 0;
    for (; it < 1500 && error() > 5e-4; ++it) {
        (void)solver.steady_iteration();
    }
    EXPECT_LT(it, 1500);
    for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_cells()); ++c) {
        const Scalar y = mesh.cell_centres()[c].y();
        EXPECT_NEAR(solver.state(c).u.x(), u_wall * y / 0.01, 1e-3 * u_wall) << "y = " << y;
    }
}

// Turbulent flat plate (Mach 0.2, Re 5e6 per metre) with compressible k-omega SST on a
// coarse wall-resolved grid: skin friction near the 1/5-power-law value.
TEST(Compressible, TurbulentFlatPlateSkinFriction) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 24;
    spec.ny = 32;
    spec.nz = 1;
    spec.max = Vec3(2.0, 1.0, 0.01);
    spec.grading = Vec3(8.0, 2.0e4, 1.0);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    const Scalar t = 300.0;
    const Scalar p = 1e5;
    const Scalar rho = p / (287.0 * t);
    const Scalar u = 0.2 * std::sqrt(1.4 * 287.0 * t);
    CompressibleSettings s;
    s.mu = rho * u / 5e6;
    s.implicit = true;
    s.krylov_dim = 30;
    s.cfl = 1000.0;
    s.turbulence = true;
    s.limiter = Limiter::Venkatakrishnan;
    BoundaryCondition inlet{FlowBoundary::VelocityInlet, Vec3(u, 0, 0), p};
    inlet.temperature = t;
    inlet.turbulence_intensity = 0.01; // enough free-stream turbulence for a turbulent
    inlet.viscosity_ratio = 1.0;       // solution from the leading edge (SST is bistable)
    const BoundaryCondition outlet{FlowBoundary::PressureOutlet, Vec3::Zero(), p};
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    CompressibleSolver solver(mesh,
                              {{"xmin", inlet},
                               {"xmax", outlet},
                               {"ymax", outlet},
                               {"ymin", {FlowBoundary::Wall, Vec3::Zero(), 0.0}},
                               {"zmin", empty},
                               {"zmax", empty}},
                              s);
    solver.set_uniform({rho, Vec3(u, 0, 0), p});
    Scalar r0 = 0.0;
    int it = 0;
    for (; it < 3000; ++it) {
        const Scalar r = solver.steady_iteration();
        r0 = it < 5 ? std::max(r0, r) : r0;
        if (r < 1e-6 * r0) {
            break;
        }
    }
    EXPECT_LT(it, 3000);
    // Wall cell closest to x = 1.
    Scalar best = 1e9;
    Scalar cf = 0.0;
    for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_cells()); ++c) {
        const Vec3& x = mesh.cell_centres()[c];
        if (x.y() < 1e-4 && std::abs(x.x() - 1.0) < best) {
            best = std::abs(x.x() - 1.0);
            cf = s.mu * solver.state(c).u.x() / x.y() / (0.5 * rho * u * u);
        }
    }
    const Scalar re_x = 5e6 * 1.0;
    EXPECT_NEAR(cf / (0.0592 * std::pow(re_x, -0.2)), 1.0, 0.15) << "Cf = " << cf;
    for (const Scalar m : solver.eddy_viscosity()) {
        EXPECT_GE(m, 0.0);
    }
}

// Characteristic far field: an acoustic pulse leaves a subsonic tube through both ends
// almost without reflection, while fixed-state / fixed-pressure ends reflect it.
TEST(Compressible, FarFieldIsNonReflecting) {
    const auto mesh = tube(200);
    const Scalar gamma = 1.4;
    const Scalar p0 = 1.0;
    const Scalar rho0 = 1.0;
    const Scalar u0 = 0.3;
    const Scalar t0 = p0 / rho0; // r_gas = 1
    const Scalar amp = 1e-3;
    auto run = [&](bool far_field) {
        CompressibleSettings s;
        s.r_gas = 1.0;
        s.gamma = gamma;
        BoundaryCondition in{FlowBoundary::VelocityInlet, Vec3(u0, 0, 0), p0};
        in.temperature = t0;
        in.far_field = far_field;
        BoundaryCondition out = in;
        if (!far_field) {
            out = {FlowBoundary::PressureOutlet, Vec3::Zero(), p0};
        }
        auto bcs = tube_bcs(FlowBoundary::Empty);
        bcs["xmin"] = in;
        bcs["xmax"] = out;
        CompressibleSolver solver(mesh, bcs, s);
        std::vector<GasState> init(static_cast<std::size_t>(mesh.n_cells()));
        const Scalar c0 = std::sqrt(gamma * p0 / rho0);
        for (std::size_t c = 0; c < init.size(); ++c) {
            const Scalar x = mesh.cell_centres()[c].x();
            const Scalar dp = amp * std::exp(-std::pow((x - 0.5) / 0.05, 2));
            init[c] = {rho0 + dp / (c0 * c0), Vec3(u0, 0, 0), p0 + dp};
        }
        solver.set_state(init);
        Scalar t = 0.0;
        while (t < 1.2) {
            const Scalar dt = std::min(solver.stable_dt(), 1.2 - t);
            (void)solver.step(dt);
            t += dt;
        }
        Scalar dev = 0.0;
        for (const Scalar p : solver.pressure()) {
            dev = std::max(dev, std::abs(p - p0));
        }
        return dev / amp;
    };
    const Scalar far = run(true);
    const Scalar fixed = run(false);
    RecordProperty("far_field_residual", std::to_string(far));
    RecordProperty("fixed_residual", std::to_string(fixed));
    EXPECT_LT(far, 0.02) << "far field residual pulse " << far;
    EXPECT_GT(fixed, 5.0 * far) << "fixed boundaries " << fixed;
}

// Uniform subsonic flow with far-field conditions on all sides stays exactly uniform.
TEST(Compressible, FarFieldPreservesFreeStream) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 8;
    spec.ny = 4;
    spec.nz = 1;
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    CompressibleSettings s;
    BoundaryCondition ff{FlowBoundary::VelocityInlet, Vec3(150, 40, 0), 1e5};
    ff.temperature = 300.0;
    ff.far_field = true;
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    CompressibleSolver solver(
        mesh,
        {{"xmin", ff}, {"xmax", ff}, {"ymin", ff}, {"ymax", ff}, {"zmin", empty}, {"zmax", empty}},
        s);
    const GasState free{1e5 / (287.0 * 300.0), Vec3(150, 40, 0), 1e5};
    solver.set_uniform(free);
    for (int n = 0; n < 50; ++n) {
        (void)solver.steady_iteration();
    }
    for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_cells()); ++c) {
        EXPECT_NEAR(solver.state(c).p / free.p, 1.0, 1e-10);
        EXPECT_NEAR(solver.state(c).u.y() / 40.0, 1.0, 1e-9);
    }
}

} // namespace

#include <cfd/case/case_config.hpp>
#include <cfd/case/case_runner.hpp>
#include <filesystem>

namespace {

TEST(Compressible, CaseFileRunAndRestart) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_compressible_case";
    std::filesystem::remove_all(dir);
    const std::string json = R"({
        "name": "sod",
        "mesh": {"type": "box", "cells": [100, 1, 1], "max": [1, 0.01, 0.01],
                 "patch_types": {"xmin": "outlet", "xmax": "outlet", "ymin": "patch",
                                 "ymax": "patch", "zmin": "patch", "zmax": "patch"}},
        "physics": {"compressible": {"r_gas": 1.0}},
        "boundaries": {"ymin": {"type": "empty"}, "ymax": {"type": "empty"},
                       "zmin": {"type": "empty"}, "zmax": {"type": "empty"}},
        "initial": {"pressure": 0.1, "temperature": 0.8,
                    "regions": [{"min": [-1, -1, -1], "max": [0.5, 1, 1],
                                 "pressure": 1.0, "temperature": 1.0}]},
        "solver": {"mode": "transient", "dt": 1.0, "end_time": 0.1}
    })";
    cfd::cases::CaseConfig c = cfd::cases::parse_case(json, dir);
    ASSERT_TRUE(c.compressible.has_value());
    EXPECT_FALSE(c.initial.temperature.has_value()); // gas temperature, not energy
    ASSERT_EQ(c.initial.regions.size(), 1U);
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c), dir);
    ASSERT_TRUE(b.compressible.has_value());
    EXPECT_DOUBLE_EQ(b.compressible->r_gas, 1.0);
    EXPECT_DOUBLE_EQ(b.initial.temperature_gas, 0.8);
    ASSERT_EQ(b.initial.regions.size(), 1U);
    EXPECT_DOUBLE_EQ(b.initial.regions[0].pressure, 1.0);

    cfd::cases::CaseRunner runner(c);
    ASSERT_NE(runner.compressible(), nullptr);
    EXPECT_NEAR(runner.compressible()->state(10).rho, 1.0, 1e-12);
    EXPECT_NEAR(runner.compressible()->state(90).rho, 0.125, 1e-12);
    const auto result = runner.run();
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(runner.time(), 0.1, 1e-12);
    EXPECT_TRUE(std::filesystem::exists(dir / "results" / "sod.vtu"));

    c.solver.end_time = 0.12;
    c.restart = "results/checkpoint.bin";
    cfd::cases::CaseRunner second(c);
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(second.compressible()->state(i).p, runner.compressible()->state(i).p);
    }
    (void)second.run();
    EXPECT_NEAR(second.time(), 0.12, 1e-12);
}

// Turbulent, implicit compressible case from a case file: options round-trip, residuals of
// k / omega, k in the results and an exact restart of the turbulence fields.
TEST(Compressible, TurbulentImplicitCaseFile) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_compressible_sst_case";
    std::filesystem::remove_all(dir);
    const std::string json = R"({
        "name": "plate",
        "mesh": {"type": "box", "cells": [12, 16, 1], "max": [2, 1, 0.01],
                 "grading": [4, 2000, 1],
                 "patch_types": {"xmin": "inlet", "xmax": "outlet", "ymin": "wall",
                                 "ymax": "outlet", "zmin": "patch", "zmax": "patch"}},
        "physics": {"turbulence": "kOmegaSST",
                    "compressible": {"mu": 1.6e-5, "implicit": true, "krylov_dim": 10,
                                     "cfl": 500, "limiter": "venkatakrishnan",
                                     "prandtl_t": 0.85}},
        "boundaries": {"xmin": {"type": "velocity_inlet", "velocity": [69.4, 0, 0],
                                "pressure": 1e5, "temperature": 300,
                                "turbulence_intensity": 0.01},
                       "xmax": {"type": "pressure_outlet", "pressure": 1e5},
                       "ymax": {"type": "pressure_outlet", "pressure": 1e5},
                       "zmin": {"type": "empty"}, "zmax": {"type": "empty"}},
        "initial": {"velocity": [69.4, 0, 0], "pressure": 1e5, "temperature": 300},
        "solver": {"mode": "steady", "max_iterations": 20, "tolerance": 1e-12},
        "output": {"checkpoint_interval": 20}
    })";
    cfd::cases::CaseConfig c = cfd::cases::parse_case(json, dir);
    ASSERT_TRUE(c.compressible.has_value());
    EXPECT_TRUE(c.compressible->turbulence);
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c), dir);
    ASSERT_TRUE(b.compressible.has_value());
    EXPECT_TRUE(b.compressible->implicit);
    EXPECT_EQ(b.compressible->krylov_dim, 10);
    EXPECT_EQ(b.compressible->limiter, Limiter::Venkatakrishnan);
    EXPECT_DOUBLE_EQ(b.compressible->prandtl_t, 0.85);
    EXPECT_TRUE(b.compressible->turbulence);

    cfd::cases::CaseRunner runner(c);
    const auto result = runner.run();
    EXPECT_GT(result.final_residuals.k, 0.0);
    EXPECT_GT(result.final_residuals.omega, 0.0);
    const auto* cs = runner.compressible();
    ASSERT_NE(cs, nullptr);
    Scalar mut_max = 0.0;
    for (const Scalar m : cs->eddy_viscosity()) {
        mut_max = std::max(mut_max, m);
    }
    EXPECT_GT(mut_max, 0.0);
    EXPECT_TRUE(std::filesystem::exists(dir / "results" / "plate.vtu"));

    c.restart = "results/checkpoint.bin";
    cfd::cases::CaseRunner second(c);
    for (std::size_t i = 0; i < cs->k().size(); ++i) {
        EXPECT_DOUBLE_EQ(second.compressible()->k()[i], cs->k()[i]);
        EXPECT_DOUBLE_EQ(second.compressible()->omega()[i], cs->omega()[i]);
    }
}

TEST(Compressible, FarFieldCaseSyntax) {
    const std::string json = R"({"name": "ff", "mesh": {"type": "box", "cells": [4, 4, 1]},
        "physics": {"compressible": {}},
        "boundaries": {"xmin": {"type": "far_field", "velocity": [100, 0, 0], "pressure": 1e5,
                                "temperature": 300}}})";
    const auto c = cfd::cases::parse_case(json);
    EXPECT_TRUE(c.boundaries.at("xmin").far_field);
    EXPECT_EQ(c.boundaries.at("xmin").kind, FlowBoundary::VelocityInlet);
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c));
    EXPECT_TRUE(b.boundaries.at("xmin").far_field);
    std::string incompressible = json;
    incompressible.replace(incompressible.find("\"compressible\": {}"), 18, "\"nu\": 1e-3");
    EXPECT_THROW((void)cfd::cases::parse_case(incompressible), std::runtime_error);
}

} // namespace
