#include <cfd/case/case_config.hpp>
#include <cfd/case/case_runner.hpp>
#include <cfd/fv/compressible.hpp>
#include <cfd/fv/properties.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

using cfd::core::Scalar;
using cfd::core::Vec3;
using cfd::fv::PropertyLaw;

TEST(Properties, Laws) {
    PropertyLaw sutherland;
    sutherland.kind = PropertyLaw::Kind::Sutherland;
    sutherland.reference = 1.716e-5;
    EXPECT_NEAR(sutherland(273.15), 1.716e-5, 1e-12);
    EXPECT_NEAR(sutherland(300.0), 1.846e-5, 1e-8); // air
    PropertyLaw poly;
    poly.kind = PropertyLaw::Kind::Polynomial;
    poly.coefficients = {1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(poly(2.0), 1.0 + 4.0 + 12.0);
    PropertyLaw power;
    power.kind = PropertyLaw::Kind::PowerLaw;
    power.reference = 2.0;
    power.t_ref = 100.0;
    power.exponent = 0.5;
    EXPECT_DOUBLE_EQ(power(400.0), 4.0);
    PropertyLaw constant;
    constant.reference = 3.0;
    EXPECT_DOUBLE_EQ(constant(1e6), 3.0);
}

namespace {

// 1D slab in y (one cell in x and z, Empty sides), walls at y = 0 / 1.
std::string slab_case(const std::string& physics, const std::string& top) {
    return R"({"name": "slab",
        "mesh": {"type": "box", "cells": [1, 40, 1], "max": [0.1, 1, 0.1],
                 "patch_types": {"xmin": "patch", "xmax": "patch", "zmin": "patch",
                                 "zmax": "patch"}},
        "physics": )" +
           physics + R"(,
        "boundaries": {"xmin": {"type": "empty"}, "xmax": {"type": "empty"},
                       "zmin": {"type": "empty"}, "zmax": {"type": "empty"},
                       "ymin": {"type": "wall", "thermal": {"type": "fixed", "value": 0}},
                       "ymax": )" +
           top + R"(},
        "initial": {"temperature": 0.5},
        "solver": {"max_iterations": 3000, "tolerance": 1e-10}})";
}

} // namespace

// Conduction with alpha(T) = a0 (1 + b T): the Kirchhoff transform a0 (T + b T^2 / 2) is
// linear in y.
TEST(Properties, VariableConductivitySlab) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_prop_slab";
    std::filesystem::remove_all(dir);
    const std::string json = slab_case(
        R"({"nu": 1, "energy": {"alpha": 1, "properties": {"alpha": {"law": "polynomial", "coefficients": [1, 2]}}}})",
        R"({"type": "wall", "thermal": {"type": "fixed", "value": 1}})");
    const auto c = cfd::cases::parse_case(json, dir);
    ASSERT_TRUE(c.energy.has_value());
    EXPECT_FALSE(c.energy->alpha_law.is_constant());
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c), dir);
    EXPECT_EQ(b.energy->alpha_law.coefficients, c.energy->alpha_law.coefficients);
    cfd::cases::CaseRunner runner(c);
    EXPECT_TRUE(runner.run().converged);
    const auto& t = runner.energy()->temperature();
    const auto& mesh = runner.mesh();
    for (std::size_t i = 0; i < t.size(); ++i) {
        const Scalar y = mesh.cell_centres()[i].y();
        // T + T^2 = 2 y (T(0) = 0, T(1) = 1) -> T = (-1 + sqrt(1 + 8 y)) / 2
        EXPECT_NEAR(t[i], 0.5 * (-1.0 + std::sqrt(1.0 + 8.0 * y)), 2e-3) << "y = " << y;
    }
}

// Couette flow with nu(T) linear in y (fixed wall temperatures, pure conduction):
// u(y) = U ln(nu(y) / nu_0) / ln(nu_1 / nu_0).
TEST(Properties, VariableViscosityCouette) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_prop_couette";
    std::filesystem::remove_all(dir);
    const std::string json = slab_case(
        R"({"nu": 0.1, "energy": {"alpha": 1, "properties": {"nu": {"law": "polynomial", "coefficients": [0.1, 0.9]}}}})",
        R"({"type": "wall", "velocity": [1, 0, 0], "thermal": {"type": "fixed", "value": 1}})");
    const auto c = cfd::cases::parse_case(json, dir);
    ASSERT_TRUE(c.energy.has_value());
    EXPECT_FALSE(c.energy->nu_law.is_constant());
    cfd::cases::CaseRunner runner(c);
    EXPECT_TRUE(runner.run().converged);
    const auto& u = runner.velocity();
    const auto& mesh = runner.mesh();
    for (std::size_t i = 0; i < u.size(); ++i) {
        const Scalar y = mesh.cell_centres()[i].y();
        const Scalar nu = 0.1 + 0.9 * y; // T = y
        EXPECT_NEAR(u[i].x(), std::log(nu / 0.1) / std::log(10.0), 5e-3) << "y = " << y;
    }
}

// Compressible Sutherland viscosity: a low-speed Couette at 300 K gives the same profile
// as the constant viscosity mu(300 K); the case-file syntax round-trips.
TEST(Properties, CompressibleSutherland) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 1;
    spec.ny = 16;
    spec.nz = 1;
    spec.max = Vec3(0.01, 0.01, 0.01);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    using namespace cfd::fv;
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const std::map<std::string, BoundaryCondition> bcs{
        {"xmin", empty},
        {"xmax", empty},
        {"zmin", empty},
        {"zmax", empty},
        {"ymin", {FlowBoundary::Wall, Vec3::Zero(), 0.0}},
        {"ymax", {FlowBoundary::Wall, Vec3(5.0, 0, 0), 0.0}}};
    auto run = [&](bool sutherland) {
        CompressibleSettings s;
        s.mu = 1.846e-5;
        if (sutherland) {
            s.mu_law.kind = PropertyLaw::Kind::Sutherland;
            s.mu_law.reference = 1.716e-5;
        }
        s.implicit = true;
        s.krylov_dim = 20;
        s.cfl = 1e5;
        CompressibleSolver solver(mesh, bcs, s);
        solver.set_uniform({1e5 / (287.0 * 300.0), Vec3::Zero(), 1e5});
        for (int n = 0; n < 300; ++n) {
            (void)solver.steady_iteration();
        }
        return solver.velocity();
    };
    const auto a = run(false);
    const auto b = run(true);
    for (std::size_t c = 0; c < a.size(); ++c) {
        EXPECT_NEAR(b[c].x(), a[c].x(), 2e-3);
    }

    const std::string json = R"({"name": "c", "mesh": {"type": "box", "cells": [4, 4, 1]},
        "physics": {"compressible": {"viscosity": {"law": "sutherland", "reference": 1.716e-5}}},
        "boundaries": {}})";
    const auto cfg = cfd::cases::parse_case(json);
    ASSERT_TRUE(cfg.compressible.has_value());
    EXPECT_EQ(cfg.compressible->mu_law.kind, PropertyLaw::Kind::Sutherland);
    EXPECT_GT(cfg.compressible->mu, 0.0); // viscous terms enabled
    const auto rt = cfd::cases::parse_case(cfd::cases::case_to_json(cfg));
    EXPECT_EQ(rt.compressible->mu_law.kind, PropertyLaw::Kind::Sutherland);
    EXPECT_DOUBLE_EQ(rt.compressible->mu_law.reference, 1.716e-5);
    EXPECT_THROW((void)cfd::cases::parse_case(
                     R"({"name": "c", "mesh": {"type": "box", "cells": [4, 4, 1]},
                "physics": {"compressible": {"viscosity": {"law": "magic", "reference": 1}}}})"),
                 std::runtime_error);
}
