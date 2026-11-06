#include <cfd/fv/compressible.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <gtest/gtest.h>

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

} // namespace
