#include <cfd/case/case_config.hpp>
#include <cfd/case/case_runner.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace cc = cfd::cases;
using cfd::core::Vec3;
using cfd::geometry::Body;

namespace {

constexpr auto kCylinderCase = R"({
    "mesh": {
        "type": "geometry",
        "min": [0, 0, 0],
        "max": [2.2, 0.41, 0.01],
        "two_d": true,
        "bodies": [
            {"type": "cylinder", "base": [0.2, 0.2, 0], "radius": 0.05, "patch": "cylinder"},
            {"type": "file", "file": "part.stl"}
        ],
        "size": 0.02,
        "size_near": 0.002,
        "refine_distance": 0.15,
        "boundary_layer": {"layers": 8, "first": 2.5e-4, "ratio": 1.15},
        "quads": false,
        "domain_patches": {"xmin": "inlet", "xmax": "outlet", "zmin": "front"},
        "patches": [{"name": "top", "min": [0, 0.4, 0], "max": [2.2, 0.42, 0.01]}]
    },
    "boundaries": {
        "inlet": {"type": "velocity_inlet", "velocity": [0.3, 0, 0],
                  "profile": "parabolic", "profile_axes": "y"}
    },
    "forces": {"patches": ["cylinder"], "density": 2, "reference_velocity": 0.2,
               "reference_area": 0.001}
})";

TEST(CaseGeometry, ParseGeometryMesh) {
    const cc::CaseConfig c = cc::parse_case(kCylinderCase);
    ASSERT_EQ(c.mesh.source, cc::MeshSource::Geometry);
    const auto& g = c.mesh.geometry;
    EXPECT_TRUE(g.two_d);
    EXPECT_FALSE(g.internal);
    EXPECT_FALSE(g.quads);
    EXPECT_DOUBLE_EQ(g.domain_max.y(), 0.41);
    ASSERT_EQ(g.bodies.size(), 2U);
    EXPECT_EQ(g.bodies[0].kind, Body::Kind::Cylinder);
    EXPECT_DOUBLE_EQ(g.bodies[0].radius, 0.05);
    EXPECT_DOUBLE_EQ(g.bodies[0].b.z(), 1.0); // default axis
    EXPECT_EQ(g.bodies[0].patch, "cylinder");
    EXPECT_EQ(g.bodies[1].kind, Body::Kind::File);
    EXPECT_EQ(g.bodies[1].file, std::filesystem::path("part.stl"));
    EXPECT_EQ(g.bodies[1].patch, "obstacle");
    EXPECT_DOUBLE_EQ(g.size_near, 0.002);
    EXPECT_EQ(g.bl_layers, 8);
    EXPECT_DOUBLE_EQ(g.bl_ratio, 1.15);
    EXPECT_EQ(g.domain_patch_names[0], "inlet");
    EXPECT_EQ(g.domain_patch_names[3], "ymax"); // default kept
    ASSERT_EQ(g.patches.size(), 1U);
    EXPECT_EQ(g.patches[0].name, "top");

    ASSERT_EQ(c.profiles.count("inlet"), 1U);
    EXPECT_EQ(c.profiles.at("inlet").axes, "y");
    ASSERT_EQ(c.forces.patches.size(), 1U);
    EXPECT_DOUBLE_EQ(c.forces.density, 2.0);
    EXPECT_DOUBLE_EQ(c.forces.reference_area, 0.001);
}

TEST(CaseGeometry, RoundTrip) {
    const cc::CaseConfig a = cc::parse_case(kCylinderCase);
    const cc::CaseConfig b = cc::parse_case(cc::case_to_json(a));
    ASSERT_EQ(b.mesh.source, cc::MeshSource::Geometry);
    const auto& ga = a.mesh.geometry;
    const auto& gb = b.mesh.geometry;
    ASSERT_EQ(gb.bodies.size(), ga.bodies.size());
    for (std::size_t i = 0; i < ga.bodies.size(); ++i) {
        EXPECT_EQ(gb.bodies[i].kind, ga.bodies[i].kind);
        EXPECT_EQ(gb.bodies[i].patch, ga.bodies[i].patch);
        EXPECT_EQ(gb.bodies[i].file, ga.bodies[i].file);
        EXPECT_DOUBLE_EQ(gb.bodies[i].radius, ga.bodies[i].radius);
        EXPECT_TRUE(gb.bodies[i].a.isApprox(ga.bodies[i].a));
    }
    EXPECT_EQ(gb.two_d, ga.two_d);
    EXPECT_EQ(gb.quads, ga.quads);
    EXPECT_DOUBLE_EQ(gb.size, ga.size);
    EXPECT_DOUBLE_EQ(gb.refine_distance, ga.refine_distance);
    EXPECT_EQ(gb.bl_layers, ga.bl_layers);
    EXPECT_DOUBLE_EQ(gb.bl_first, ga.bl_first);
    EXPECT_EQ(gb.domain_patch_names, ga.domain_patch_names);
    ASSERT_EQ(gb.patches.size(), 1U);
    EXPECT_TRUE(gb.patches[0].max.isApprox(ga.patches[0].max));
    EXPECT_EQ(b.profiles.at("inlet").axes, "y");
    EXPECT_EQ(b.forces.patches, a.forces.patches);
    EXPECT_DOUBLE_EQ(b.forces.reference_velocity, 0.2);
}

TEST(CaseGeometry, Errors) {
    auto parse = [](const std::string& mesh, const std::string& rest = "") {
        return cc::parse_case(R"({"mesh": )" + mesh + rest + "}");
    };
    EXPECT_THROW((void)parse(R"({"type": "geometry", "bodies": [{"type": "cone"}]})"),
                 std::runtime_error);
    EXPECT_THROW((void)parse(R"({"type": "geometry", "bodies": [{"type": "sphere"}]})"),
                 std::runtime_error); // radius missing
    EXPECT_THROW((void)parse(R"({"type": "geometry", "bodies": [{"type": "file"}]})"),
                 std::runtime_error);
    EXPECT_THROW((void)parse(R"({"type": "geometry", "size": 0})"), std::runtime_error);
    EXPECT_THROW((void)parse(R"({"type": "geometry", "domain_patches": {"left": "a"}})"),
                 std::runtime_error);
    EXPECT_THROW((void)parse(R"({"type": "box"})", R"(, "boundaries": {"xmin":
                     {"type": "wall", "profile": "parabolic"}})"),
                 std::runtime_error); // profile only for inlets
    EXPECT_THROW((void)parse(R"({"type": "box"})", R"(, "boundaries": {"xmin":
                     {"type": "velocity_inlet", "velocity": [1, 0, 0], "profile": "parabolic",
                      "profile_axes": "w"}})"),
                 std::runtime_error);
    EXPECT_THROW((void)parse(R"({"type": "box"})", R"(, "forces": {"patches": "xmin"})"),
                 std::runtime_error);
}

TEST(CaseGeometry, ParabolicInletProfileAndForces) {
    const auto dir = std::filesystem::temp_directory_path() / "cfd_case_geometry_profile";
    std::filesystem::remove_all(dir);
    const cc::CaseConfig c = cc::parse_case(R"({
        "mesh": {"type": "box", "cells": [10, 8, 1], "max": [2, 1, 0.1],
                 "patch_types": {"xmin": "inlet", "xmax": "outlet",
                                 "zmin": "symmetry", "zmax": "symmetry"}},
        "physics": {"nu": 0.1},
        "boundaries": {"xmin": {"type": "velocity_inlet", "velocity": [1.5, 0, 0],
                                "profile": "parabolic", "profile_axes": "y"}},
        "solver": {"max_iterations": 5},
        "forces": {"patches": ["ymin", "ymax"]},
        "output": {"log_interval": 2}
    })",
                                            dir);
    cc::CaseRunner runner(c);
    const auto& mesh = runner.mesh();
    const auto* inlet = mesh.find_patch("xmin");
    for (int q = 0; q < inlet->n_faces; ++q) {
        const auto f = inlet->start_face + q;
        const double y = mesh.face_centres()[static_cast<std::size_t>(f)].y();
        EXPECT_NEAR(runner.flow().boundary_velocity(f).x(), 1.5 * 4.0 * y * (1.0 - y), 1e-12);
    }
    (void)runner.run();
    EXPECT_TRUE(std::filesystem::exists(dir / "results" / "forces.csv"));
    EXPECT_GT(runner.force("ymin").x(), 0.0); // flow drags the walls downstream

    cc::CaseConfig bad = c;
    bad.forces.patches = {"nope"};
    EXPECT_THROW(cc::CaseRunner{bad}, std::runtime_error);
}

TEST(CaseGeometry, RunGeometryCase) {
    if (cfd::geometry::find_gmsh().empty()) {
        GTEST_SKIP() << "Gmsh not installed";
    }
    const auto dir = std::filesystem::temp_directory_path() / "cfd_case_geometry_run";
    std::filesystem::remove_all(dir);
    cc::CaseConfig c = cc::parse_case(kCylinderCase, dir);
    c.mesh.geometry.bodies.pop_back(); // no STL file
    c.mesh.geometry.patches.clear();
    c.mesh.geometry.size = 0.05;
    c.mesh.geometry.size_near = 0.01;
    c.mesh.geometry.domain_patch_names = {"inlet", "outlet", "walls", "walls", "front", "back"};
    c.boundaries["outlet"].kind = cfd::fv::FlowBoundary::PressureOutlet;
    c.boundaries["front"].kind = cfd::fv::FlowBoundary::Empty;
    c.boundaries["back"].kind = cfd::fv::FlowBoundary::Empty;
    c.solver.max_iterations = 3;
    cc::CaseRunner runner(c);
    EXPECT_NE(runner.mesh().find_patch("cylinder"), nullptr);
    EXPECT_TRUE(std::filesystem::exists(dir / "results" / "mesh" / "geometry.msh"));
    (void)runner.run();
    EXPECT_TRUE(std::filesystem::exists(dir / "results" / "forces.csv"));
}

} // namespace

namespace {

// Solid geometry body: meshed as a zone, conjugate heat transfer through the case file.
TEST(CaseGeometry, SolidBodyConjugate) {
    const std::string json = R"({
        "mesh": {"type": "geometry", "min": [0, 0, 0], "max": [2, 0.5, 0.05], "two_d": true,
                 "bodies": [{"type": "box", "min": [0.6, 0, 0], "max": [1.0, 0.15, 0],
                             "patch": "block", "solid": {"conductivity": 0.4, "rho_cp": 2}}],
                 "size": 0.05,
                 "domain_patches": {"xmin": "inlet", "xmax": "outlet", "ymin": "bottom",
                                    "ymax": "top", "zmin": "front", "zmax": "back"}},
        "physics": {"nu": 0.01, "energy": {"prandtl": 0.71}},
        "boundaries": {"inlet": {"type": "velocity_inlet", "velocity": [1, 0, 0]},
                       "outlet": {"type": "pressure_outlet"},
                       "bottom": {"type": "wall", "thermal": {"type": "fixed", "value": 1}},
                       "top": {"type": "wall"},
                       "front": {"type": "empty"}, "back": {"type": "empty"}},
        "solver": {"max_iterations": 1500, "tolerance": 1e-8},
        "heat": {"patches": ["block"]}
    })";
    cc::CaseConfig c = cc::parse_case(json);
    ASSERT_EQ(c.solids.size(), 1U);
    EXPECT_EQ(c.solids[0].name, "block");
    EXPECT_FALSE(c.solids[0].min.has_value());
    EXPECT_TRUE(c.mesh.geometry.bodies[0].solid);
    // Round trip: the solid stays with its body, not in the top-level list.
    const std::string text = cc::case_to_json(c);
    EXPECT_EQ(text.find("\"solids\""), std::string::npos);
    const cc::CaseConfig b = cc::parse_case(text);
    ASSERT_EQ(b.solids.size(), 1U);
    EXPECT_DOUBLE_EQ(b.solids[0].rho_cp, 2.0);

    if (cfd::geometry::find_gmsh().empty()) {
        GTEST_SKIP() << "Gmsh not installed";
    }
    const auto dir = std::filesystem::temp_directory_path() / "cfd_case_solid_body";
    std::filesystem::remove_all(dir);
    c.base_directory = dir;
    cc::CaseRunner runner(c);
    EXPECT_EQ(runner.full_mesh().find_patch("block"), nullptr); // interface is internal
    ASSERT_NE(runner.mesh().find_patch("block"), nullptr);      // wall of the fluid mesh
    EXPECT_LT(runner.mesh().n_cells(), runner.full_mesh().n_cells());
    const auto result = runner.run();
    EXPECT_TRUE(result.converged);
    EXPECT_GT(runner.heat_flow("block"), 0.0); // the heated block warms the fluid
}

} // namespace
