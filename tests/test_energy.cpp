#include <cfd/case/case_config.hpp>
#include <cfd/case/case_runner.hpp>
#include <cfd/fv/energy.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <filesystem>
#include <gtest/gtest.h>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

cfd::mesh::Mesh slab(int n) {
    cfd::mesh::BoxSpec spec;
    spec.nx = n;
    spec.ny = 2;
    spec.nz = 1;
    spec.max = Vec3(2.0, 1.0, 0.5);
    return cfd::mesh::make_box_mesh(spec);
}

const std::map<std::string, BoundaryCondition>& empty_z() {
    static const std::map<std::string, BoundaryCondition> bcs{
        {"zmin", {FlowBoundary::Empty, Vec3::Zero(), 0.0}},
        {"zmax", {FlowBoundary::Empty, Vec3::Zero(), 0.0}}};
    return bcs;
}

// Pure conduction between two fixed temperatures: linear profile, Q = alpha A dT / L.
TEST(Energy, ConductionSlab) {
    const auto mesh = slab(10);
    FlowSolver flow(mesh, empty_z(), FlowSettings{});
    EnergySettings es;
    es.alpha = 0.3;
    es.relaxation = 1.0;
    EnergyEquation energy(flow, es,
                          {{"xmin", {ThermalBoundary::Kind::FixedTemperature, 1.0}},
                           {"xmax", {ThermalBoundary::Kind::FixedTemperature, 0.0}}});
    for (int i = 0; i < 5; ++i) {
        (void)energy.correct(flow, 0.0);
    }
    for (std::size_t c = 0; c < energy.temperature().size(); ++c) {
        EXPECT_NEAR(energy.temperature()[c], 1.0 - mesh.cell_centres()[c].x() / 2.0, 1e-8);
    }
    const Scalar q = 0.3 * 0.5 * 1.0 / 2.0; // alpha * area * dT / L
    EXPECT_NEAR(energy.patch_heat_flow(flow, "xmin"), q, 1e-8);
    EXPECT_NEAR(energy.patch_heat_flow(flow, "xmax"), -q, 1e-8);
    EXPECT_FALSE(energy.buoyant());
}

// Prescribed heat flux in at xmin, fixed T = 0 at xmax: T(xmin) = q L / alpha.
TEST(Energy, HeatFluxBoundary) {
    const auto mesh = slab(16);
    FlowSolver flow(mesh, empty_z(), FlowSettings{});
    EnergySettings es;
    es.alpha = 0.5;
    es.relaxation = 1.0;
    EnergyEquation energy(flow, es,
                          {{"xmin", {ThermalBoundary::Kind::HeatFlux, 2.0}},
                           {"xmax", {ThermalBoundary::Kind::FixedTemperature, 0.0}}});
    for (int i = 0; i < 5; ++i) {
        (void)energy.correct(flow, 0.0);
    }
    for (std::size_t c = 0; c < energy.temperature().size(); ++c) {
        EXPECT_NEAR(energy.temperature()[c], 2.0 / 0.5 * (2.0 - mesh.cell_centres()[c].x()), 1e-7);
    }
    const auto tb = energy.boundary_temperature();
    const auto* xmin = mesh.find_patch("xmin");
    EXPECT_NEAR(tb[static_cast<std::size_t>(xmin->start_face - mesh.n_internal_faces())], 8.0,
                1e-7);
    EXPECT_NEAR(energy.patch_heat_flow(flow, "xmin"), 2.0 * 0.5, 1e-10);
    EXPECT_NEAR(energy.patch_heat_flow(flow, "xmax"), -2.0 * 0.5, 1e-7);
}

// De Vahl Davis cavity, Ra = 1e3 on 24 x 24 cells: mean Nu = 1.118.
TEST(Energy, NaturalConvectionCavityRa1e3) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 24;
    spec.ny = 24;
    spec.nz = 1;
    spec.max = Vec3(1.0, 1.0, 1.0 / 24.0);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    FlowSettings settings;
    settings.nu = 0.71;
    settings.convection = ConvectionScheme::LimitedLinearUpwind;
    FlowSolver flow(mesh, empty_z(), settings);
    EnergySettings es;
    es.alpha = 1.0;
    es.beta = 1.0;
    es.t_ref = 0.5;
    es.gravity = Vec3(0.0, -1e3 * 0.71, 0.0);
    EnergyEquation energy(flow, es,
                          {{"xmin", {ThermalBoundary::Kind::FixedTemperature, 1.0}},
                           {"xmax", {ThermalBoundary::Kind::FixedTemperature, 0.0}}});
    int it = 0;
    for (; it < 3000; ++it) {
        flow.set_cell_acceleration(energy.buoyancy());
        Residuals r = flow.simple_iteration();
        r.t = energy.correct(flow, 0.0);
        if (r.max() < 1e-8) {
            break;
        }
    }
    EXPECT_LT(it, 3000);
    const Scalar nu = energy.patch_heat_flow(flow, "xmin") * 24.0;
    EXPECT_NEAR(nu / 1.118, 1.0, 0.01);
    EXPECT_NEAR(-energy.patch_heat_flow(flow, "xmax") * 24.0, nu, 1e-4);
}

TEST(Energy, CaseFileRoundTripAndRestart) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_energy_case";
    std::filesystem::remove_all(dir);
    const std::string json = R"({
        "name": "hot",
        "mesh": {"type": "box", "cells": [12, 12, 1], "max": [1, 1, 0.1],
                 "patch_types": {"zmin": "patch", "zmax": "patch"}},
        "physics": {"nu": 0.71, "energy": {"prandtl": 0.71, "beta": 1, "t_ref": 0.5,
                                           "gravity": [0, -710, 0]}},
        "boundaries": {"zmin": {"type": "empty"}, "zmax": {"type": "empty"},
                       "xmin": {"thermal": {"type": "fixed", "value": 1}},
                       "xmax": {"type": "wall", "thermal": {"type": "heat_flux", "value": -0.5}}},
        "initial": {"temperature": 0.5},
        "solver": {"max_iterations": 20, "tolerance": 1e-12},
        "heat": {"patches": ["xmin", "xmax"], "rho_cp": 2, "reference_area": 0.1}
    })";
    cfd::cases::CaseConfig c = cfd::cases::parse_case(json, dir);
    ASSERT_TRUE(c.energy.has_value());
    EXPECT_DOUBLE_EQ(c.energy->alpha, 1.0); // nu / Pr
    EXPECT_EQ(c.thermal.at("xmin").kind, ThermalBoundary::Kind::FixedTemperature);
    EXPECT_EQ(c.thermal.at("xmax").kind, ThermalBoundary::Kind::HeatFlux);
    EXPECT_EQ(c.boundaries.count("xmin"), 0U); // thermal only
    EXPECT_EQ(c.boundaries.count("xmax"), 1U);
    EXPECT_DOUBLE_EQ(*c.initial.temperature, 0.5);

    const cfd::cases::CaseConfig b = cfd::cases::parse_case(cfd::cases::case_to_json(c), dir);
    ASSERT_TRUE(b.energy.has_value());
    EXPECT_DOUBLE_EQ(b.energy->gravity.y(), -710.0);
    EXPECT_DOUBLE_EQ(b.thermal.at("xmax").value, -0.5);
    EXPECT_EQ(b.heat.patches, c.heat.patches);
    EXPECT_DOUBLE_EQ(b.heat.rho_cp, 2.0);

    cfd::cases::CaseRunner first(c);
    (void)first.run();
    ASSERT_NE(first.energy(), nullptr);
    EXPECT_TRUE(std::filesystem::exists(dir / "results" / "heat.csv"));
    EXPECT_NEAR(first.heat_flow("xmax"), 2.0 * -0.5 * 0.1, 1e-9);

    c.solver.max_iterations = 25;
    c.restart = "results/checkpoint.bin";
    cfd::cases::CaseRunner second(c);
    for (std::size_t i = 0; i < first.energy()->temperature().size(); ++i) {
        EXPECT_DOUBLE_EQ(second.energy()->temperature()[i], first.energy()->temperature()[i]);
    }

    auto bad = json;
    bad.replace(bad.find("\"energy\""), 8, "\"energyx\"");
    EXPECT_THROW((void)cfd::cases::parse_case(bad), std::runtime_error);
}

} // namespace
