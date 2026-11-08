#include <cfd/fv/energy.hpp>
#include <cfd/fv/radiation.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <gtest/gtest.h>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

// Radiative equilibrium of a grey slab of optical thickness tau = a L between two grey walls
// at T1 / T2 (negligible conduction). P1 with Marshak conditions gives the exact P1 value
//   q / (sigma (T1^4 - T2^4)) = 1 / (3 tau / 4 + 1 / eps1 + 1 / eps2 - 1).
Scalar equilibrium_flux(Scalar tau, Scalar eps, Scalar& q_expected) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 60;
    spec.ny = 1;
    spec.nz = 1;
    spec.max = Vec3(1.0, 0.1, 0.1);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    const BoundaryCondition slip{FlowBoundary::Symmetry, Vec3::Zero(), 0.0};
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    FlowSolver flow(mesh, {{"ymin", slip}, {"ymax", slip}, {"zmin", empty}, {"zmax", empty}},
                    FlowSettings{});
    EnergySettings es;
    es.alpha = 1e-9; // radiation-dominated
    es.relaxation = 1.0;
    const Scalar t1 = 1000.0;
    const Scalar t2 = 500.0;
    EnergyEquation energy(flow, es,
                          {{"xmin", {ThermalBoundary::Kind::FixedTemperature, t1}},
                           {"xmax", {ThermalBoundary::Kind::FixedTemperature, t2}}});
    energy.set_uniform_temperature(750.0);
    RadiationSettings rs;
    rs.absorption = tau; // L = 1
    rs.emissivity = eps;
    P1Radiation rad(flow, rs);
    std::vector<Scalar> su;
    std::vector<Scalar> sp;
    for (int it = 0; it < 200; ++it) {
        (void)rad.correct(energy.temperature(), energy.boundary_temperature());
        rad.energy_source(energy.temperature(), su, sp);
        energy.set_source(su, sp); // rho c_p = 1
        (void)energy.correct(flow, 0.0);
    }
    const Scalar sigma = P1Radiation::k_sigma;
    q_expected = sigma * (std::pow(t1, 4) - std::pow(t2, 4)) / (0.75 * tau + 2.0 / eps - 1.0);
    // Radiative flux from the gradient of G in the middle of the slab.
    const auto& g = rad.incident_radiation();
    const std::size_t i = 30;
    const Scalar dx = mesh.cell_centres()[i].x() - mesh.cell_centres()[i - 1].x();
    const Scalar q_mid = -rad.gamma() * (g[i] - g[i - 1]) / dx;
    // Wall fluxes balance (equilibrium) and match the interior flux.
    const Scalar area = 0.01;
    EXPECT_NEAR(rad.patch_heat_flow("xmin") / area, q_mid, 2e-3 * q_mid);
    EXPECT_NEAR(rad.patch_heat_flow("xmax") / area, -q_mid, 2e-3 * q_mid);
    return q_mid;
}

} // namespace

TEST(Radiation, P1RadiativeEquilibriumBlackWalls) {
    for (const Scalar tau : {0.5, 1.0, 5.0}) {
        Scalar expected = 0.0;
        const Scalar q = equilibrium_flux(tau, 1.0, expected);
        EXPECT_NEAR(q / expected, 1.0, 5e-3) << "tau = " << tau;
    }
}

TEST(Radiation, P1RadiativeEquilibriumGreyWalls) {
    Scalar expected = 0.0;
    const Scalar q = equilibrium_flux(1.0, 0.5, expected);
    EXPECT_NEAR(q / expected, 1.0, 5e-3);
}

TEST(Radiation, InvalidSettingsThrow) {
    cfd::mesh::BoxSpec spec;
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    FlowSolver flow(mesh, {}, FlowSettings{});
    RadiationSettings rs;
    rs.absorption = 0.0;
    EXPECT_THROW(P1Radiation(flow, rs), std::invalid_argument);
    rs.absorption = 1.0;
    rs.patch_emissivity["nope"] = 0.5;
    EXPECT_THROW(P1Radiation(flow, rs), std::invalid_argument);
}

#include <cfd/case/case_config.hpp>
#include <cfd/case/case_runner.hpp>
#include <filesystem>

// Case file: energy with P1 radiation (radiative equilibrium slab, tau = 1, black walls);
// heat_flow includes the radiative part, G is written, settings round-trip.
TEST(Radiation, CaseFile) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_radiation_case";
    std::filesystem::remove_all(dir);
    const std::string json = R"({
        "name": "slab",
        "mesh": {"type": "box", "cells": [60, 1, 1], "max": [1, 0.1, 0.1],
                 "patch_types": {"ymin": "symmetry", "ymax": "symmetry", "zmin": "patch",
                                 "zmax": "patch"}},
        "physics": {"nu": 1, "energy": {"alpha": 1e-9, "t_ref": 750,
                    "radiation": {"absorption": 1, "emissivity": 1,
                                  "patch_emissivity": {"xmax": 1}}}},
        "boundaries": {"zmin": {"type": "empty"}, "zmax": {"type": "empty"},
                       "xmin": {"thermal": {"type": "fixed", "value": 1000}},
                       "xmax": {"thermal": {"type": "fixed", "value": 500}}},
        "solver": {"max_iterations": 300, "tolerance": 1e-14},
        "heat": {"patches": ["xmin", "xmax"]}
    })";
    const auto c = cfd::cases::parse_case(json, dir);
    ASSERT_TRUE(c.radiation.has_value());
    EXPECT_DOUBLE_EQ(c.radiation->patch_emissivity.at("xmax"), 1.0);
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c), dir);
    ASSERT_TRUE(b.radiation.has_value());
    EXPECT_DOUBLE_EQ(b.radiation->absorption, 1.0);
    cfd::cases::CaseRunner runner(c);
    ASSERT_NE(runner.radiation(), nullptr);
    (void)runner.run();
    const Scalar expected =
        P1Radiation::k_sigma * (std::pow(1000.0, 4) - std::pow(500.0, 4)) / 1.75;
    EXPECT_NEAR(runner.heat_flow("xmin") / 0.01, expected, 5e-3 * expected);
    EXPECT_TRUE(std::filesystem::exists(dir / "results" / "slab.vtu"));
    std::string bad = json;
    bad.replace(bad.find("\"absorption\": 1"), 15, "\"absorption\": 0");
    EXPECT_THROW((void)cfd::cases::parse_case(bad, dir), std::runtime_error);
}

// Conjugate: a conducting plate (k = 50 W/(m K), x in [0, 1], T(0) = 1000 K) radiates across
// an almost transparent gas gap (x in [1, 2]) to a black wall at 500 K. The interface
// temperature satisfies k (1000 - Ti) = sigma (Ti^4 - 500^4) (black surfaces, tau -> 0).
TEST(Radiation, ConjugatePlateRadiatesAcrossGap) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_radiation_conjugate";
    std::filesystem::remove_all(dir);
    const std::string json = R"({
        "name": "gap",
        "mesh": {"type": "box", "cells": [80, 1, 1], "max": [2, 0.1, 0.1],
                 "patch_types": {"ymin": "symmetry", "ymax": "symmetry", "zmin": "patch",
                                 "zmax": "patch"}},
        "physics": {"nu": 1e-5, "energy": {"alpha": 1e-12, "t_ref": 750, "rho_cp": 1000,
                    "radiation": {"absorption": 1e-3}}},
        "solids": [{"name": "plate", "min": [-1, -1, -1], "max": [1, 1, 1],
                    "conductivity": 50, "rho_cp": 1000}],
        "boundaries": {"zmin": {"type": "empty"}, "zmax": {"type": "empty"},
                       "xmin": {"thermal": {"type": "fixed", "value": 1000}},
                       "xmax": {"type": "wall", "thermal": {"type": "fixed", "value": 500}}},
        "initial": {"temperature": 750},
        "solver": {"max_iterations": 400, "tolerance": 1e-14},
        "heat": {"patches": ["xmin", "xmax"], "rho_cp": 1000}
    })";
    const auto c = cfd::cases::parse_case(json, dir);
    cfd::cases::CaseRunner runner(c);
    ASSERT_NE(runner.radiation(), nullptr);
    (void)runner.run();
    // Analytic interface temperature.
    const Scalar sigma = P1Radiation::k_sigma;
    Scalar lo = 500.0;
    Scalar hi = 1000.0;
    for (int k = 0; k < 100; ++k) {
        const Scalar ti = 0.5 * (lo + hi);
        (50.0 * (1000.0 - ti) > sigma * (std::pow(ti, 4) - std::pow(500.0, 4)) ? lo : hi) = ti;
    }
    const Scalar ti = 0.5 * (lo + hi);
    const Scalar q = 50.0 * (1000.0 - ti); // W/m^2
    // Solid temperature next to the interface (cell centre at x = 1 - dx/2).
    const auto& t = runner.energy()->temperature();
    const auto& mesh = runner.full_mesh();
    Scalar t_near = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (std::abs(mesh.cell_centres()[i].x() - (1.0 - 0.0125)) < 1e-6) {
            t_near = t[i];
        }
    }
    EXPECT_NEAR(t_near, ti + q / 50.0 * 0.0125, 0.01 * (1000.0 - ti));
    // Heat balance: conduction in at xmin = radiation out at xmax (W).
    EXPECT_NEAR(runner.heat_flow("xmin") / 0.01, q, 0.02 * q);
    EXPECT_NEAR(-runner.heat_flow("xmax") / 0.01, q, 0.02 * q);
}
