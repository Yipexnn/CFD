#include <cfd/case/case_config.hpp>
#include <cfd/case/case_runner.hpp>
#include <cfd/fv/energy.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/parallel/decomposition.hpp>
#include <filesystem>
#include <gtest/gtest.h>

using cfd::core::Label;
using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;
namespace par = cfd::parallel;

namespace {

cfd::mesh::Mesh slab() {
    cfd::mesh::BoxSpec spec;
    spec.nx = 20;
    spec.ny = 2;
    spec.nz = 1;
    spec.max = Vec3(2.0, 1.0, 0.5);
    return cfd::mesh::make_box_mesh(spec);
}

std::vector<int> zones_by_x(const cfd::mesh::Mesh& m, Scalar x_split) {
    std::vector<int> zones(static_cast<std::size_t>(m.n_cells()));
    for (std::size_t c = 0; c < zones.size(); ++c) {
        zones[c] = m.cell_centres()[c].x() > x_split ? 1 : 0;
    }
    return zones;
}

TEST(Conjugate, ExtractZone) {
    const auto mesh = slab();
    const auto zones = zones_by_x(mesh, 1.0);
    const par::SubMesh fluid = par::extract_zone(mesh, zones, {"fluid", "plate"}, 0, nullptr);
    EXPECT_EQ(fluid.mesh.n_cells(), 20);
    const auto* interface = fluid.mesh.find_patch("plate");
    ASSERT_NE(interface, nullptr);
    EXPECT_EQ(interface->type, cfd::mesh::PatchType::Wall);
    EXPECT_EQ(interface->n_faces, 2);
    EXPECT_EQ(fluid.mesh.find_patch("xmax")->n_faces, 0);
    EXPECT_EQ(fluid.mesh.find_patch("xmin")->n_faces, 2);
    for (Label c = 0; c < fluid.mesh.n_cells(); ++c) {
        const auto p = static_cast<std::size_t>(fluid.parent_cell[static_cast<std::size_t>(c)]);
        EXPECT_DOUBLE_EQ(fluid.mesh.cell_volumes()[static_cast<std::size_t>(c)],
                         mesh.cell_volumes()[p]);
    }
    // Interface faces point out of the fluid (+x).
    for (Label i = 0; i < interface->n_faces; ++i) {
        EXPECT_GT(
            fluid.mesh.face_area_vectors()[static_cast<std::size_t>(interface->start_face + i)].x(),
            0.0);
    }
}

// Fluid at rest (x < 1, K = alpha = 1) and a solid (x > 1, k = 4): series resistance,
// q = dT / (L1 / K1 + L2 / K2) = 0.8, interface temperature 0.2.
TEST(Conjugate, CompositeWallConduction) {
    const auto mesh = slab();
    const std::vector<std::string> names{"fluid", "plate"};
    ConjugateMesh cm;
    cm.zone_of_cell = zones_by_x(mesh, 1.0);
    cm.zone_names = names;
    cm.solids = {{"plate", 4.0, 3.0}};
    const par::SubMesh fluid = par::extract_zone(mesh, cm.zone_of_cell, names, 0, nullptr);
    const FvGeometry geo(mesh);
    cm.geometry = &geo;
    cm.fluid = &fluid;

    FlowSolver flow(fluid.mesh,
                    {{"zmin", {FlowBoundary::Empty, Vec3::Zero(), 0.0}},
                     {"zmax", {FlowBoundary::Empty, Vec3::Zero(), 0.0}}},
                    FlowSettings{});
    EnergySettings es;
    es.alpha = 1.0;
    es.relaxation = 1.0;
    EnergyEquation energy(flow, cm, es,
                          {{"xmin", {ThermalBoundary::Kind::FixedTemperature, 1.0}},
                           {"xmax", {ThermalBoundary::Kind::FixedTemperature, 0.0}}});
    for (int i = 0; i < 5; ++i) {
        (void)energy.correct(flow, 0.0);
    }
    for (std::size_t c = 0; c < energy.temperature().size(); ++c) {
        const Scalar x = mesh.cell_centres()[c].x();
        const Scalar exact = x < 1.0 ? 1.0 - 0.8 * x : 0.2 - 0.2 * (x - 1.0);
        EXPECT_NEAR(energy.temperature()[c], exact, 1e-8) << x;
    }
    const Scalar area = 1.0 * 0.5;
    EXPECT_NEAR(energy.patch_heat_flow(flow, "xmin"), 0.8 * area, 1e-8);
    EXPECT_NEAR(energy.patch_heat_flow(flow, "xmax"), -0.8 * area, 1e-8);
    EXPECT_NEAR(energy.patch_heat_flow(flow, "plate"), -0.8 * area, 1e-8); // solid -> fluid
    EXPECT_EQ(energy.buoyancy().size(), static_cast<std::size_t>(fluid.mesh.n_cells()));

    // Transient: the solid heat capacity enters the time term (C = 3).
    energy.set_uniform_temperature(0.0);
    const Scalar r = energy.correct(flow, 1e-3);
    EXPECT_GT(r, 0.0);
}

} // namespace

namespace {

constexpr auto kPlateCase = R"({
    "name": "plate",
    "mesh": {"type": "box", "cells": [30, 12, 1], "max": [3, 1.2, 0.1],
             "patch_types": {"xmin": "inlet", "xmax": "outlet", "zmin": "patch", "zmax": "patch"}},
    "physics": {"nu": 0.02, "energy": {"prandtl": 0.71, "rho_cp": 2}},
    "solids": [{"name": "plate", "min": [-1, -1, -1], "max": [10, 0.2, 1],
                "conductivity": 0.5, "rho_cp": 3}],
    "boundaries": {"xmin": {"type": "velocity_inlet", "velocity": [1, 0, 0]},
                   "zmin": {"type": "empty"}, "zmax": {"type": "empty"},
                   "ymin": {"thermal": {"type": "fixed", "value": 1}}},
    "solver": {"max_iterations": 2000, "tolerance": 1e-9},
    "heat": {"patches": ["ymin", "plate"]}
})";

TEST(Conjugate, CaseFileRunAndRestart) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_conjugate_case";
    std::filesystem::remove_all(dir);
    cfd::cases::CaseConfig c = cfd::cases::parse_case(kPlateCase, dir);
    ASSERT_EQ(c.solids.size(), 1U);
    EXPECT_DOUBLE_EQ(c.solids[0].conductivity, 0.5);
    EXPECT_DOUBLE_EQ(c.energy->rho_cp, 2.0);
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c), dir);
    ASSERT_EQ(b.solids.size(), 1U);
    EXPECT_TRUE(b.solids[0].max->isApprox(Vec3(10, 0.2, 1)));
    EXPECT_DOUBLE_EQ(b.solids[0].rho_cp, 3.0);

    cfd::cases::CaseRunner runner(c);
    EXPECT_EQ(runner.full_mesh().n_cells(), 360);
    EXPECT_EQ(runner.mesh().n_cells(), 360 - 60); // 2 of 12 rows are solid
    EXPECT_NE(runner.mesh().find_patch("plate"), nullptr);
    const auto result = runner.run();
    EXPECT_TRUE(result.converged);
    // Steady state: all heat entering through the base flows from the plate into the fluid.
    const Scalar q_in = runner.heat_flow("ymin");
    EXPECT_GT(q_in, 0.0);
    EXPECT_NEAR(runner.heat_flow("plate"), q_in, 1e-6 * q_in);
    // Solid cells have no velocity in the output fields.
    const auto u = runner.to_full_cells(runner.flow().velocity().data()->data(), 3);
    EXPECT_EQ(u[0], 0.0);

    c.solver.max_iterations = result.steps + 1;
    c.restart = "results/checkpoint.bin";
    cfd::cases::CaseRunner second(c);
    EXPECT_NEAR(second.heat_flow("plate"), runner.heat_flow("plate"), 1e-9);
    for (std::size_t i = 0; i < runner.flow().velocity().size(); i += 37) {
        EXPECT_NEAR((second.flow().velocity()[i] - runner.flow().velocity()[i]).norm(), 0.0, 1e-12);
    }

    auto bad = c;
    bad.solids[0].min = Vec3(20, 20, 20);
    bad.solids[0].max = Vec3(21, 21, 21);
    bad.restart.reset();
    EXPECT_THROW(cfd::cases::CaseRunner{bad}, std::runtime_error);
}

} // namespace
