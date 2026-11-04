#include <cfd/geometry/geometry.hpp>
#include <cfd/mesh/quality.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <numbers>
#include <stdexcept>
#include <string>

using cfd::core::Vec3;
using cfd::geometry::Body;
using cfd::geometry::GeometrySpec;

namespace {

GeometrySpec cylinder_2d() {
    GeometrySpec s;
    s.two_d = true;
    s.domain_max = Vec3(2.2, 0.41, 0.01);
    Body c;
    c.kind = Body::Kind::Cylinder;
    c.a = Vec3(0.2, 0.2, 0.0);
    c.radius = 0.05;
    c.patch = "cylinder";
    s.bodies.push_back(c);
    s.size = 0.04;
    s.size_near = 0.008;
    s.refine_distance = 0.1;
    s.bl_layers = 4;
    s.bl_first = 1e-3;
    s.domain_patch_names = {"inlet", "outlet", "walls", "walls", "front", "back"};
    return s;
}

bool contains(const std::string& text, const std::string& what) {
    return text.find(what) != std::string::npos;
}

TEST(Geometry, WriteGeo2dCylinder) {
    const std::string geo = cfd::geometry::write_geo(cylinder_2d());
    EXPECT_TRUE(contains(geo, "SetFactory(\"OpenCASCADE\")"));
    EXPECT_TRUE(contains(geo, "Disk"));
    EXPECT_TRUE(contains(geo, "BoundaryLayer"));
    EXPECT_TRUE(contains(geo, "Extrude"));
    EXPECT_TRUE(contains(geo, "Physical Surface(\"cylinder\")"));
    EXPECT_TRUE(contains(geo, "Physical Surface(\"inlet\")"));
    EXPECT_TRUE(contains(geo, "Physical Volume(\"fluid\")"));
    EXPECT_TRUE(contains(geo, "Mesh.MshFileVersion = 4.1"));
    // "walls" is used by two box faces but must be defined once.
    const auto first = geo.find("Physical Surface(\"walls\")");
    ASSERT_NE(first, std::string::npos);
    EXPECT_EQ(geo.find("Physical Surface(\"walls\")", first + 1), std::string::npos);
}

TEST(Geometry, WriteGeo3dPrimitives) {
    GeometrySpec s;
    s.domain_max = Vec3(2, 1, 1);
    Body sphere;
    sphere.kind = Body::Kind::Sphere;
    sphere.a = Vec3(0.5, 0.5, 0.5);
    sphere.radius = 0.1;
    Body box;
    box.kind = Body::Kind::Box;
    box.a = Vec3(1.2, 0.4, 0.4);
    box.b = Vec3(1.4, 0.6, 0.6);
    box.patch = "block";
    s.bodies = {sphere, box};
    const std::string geo = cfd::geometry::write_geo(s);
    EXPECT_TRUE(contains(geo, "Sphere("));
    EXPECT_TRUE(contains(geo, "BooleanDifference"));
    EXPECT_TRUE(contains(geo, "Physical Surface(\"obstacle\")"));
    EXPECT_TRUE(contains(geo, "Physical Surface(\"block\")"));
}

TEST(Geometry, InvalidSpecsThrow) {
    GeometrySpec internal;
    internal.internal = true; // needs exactly one STEP body
    EXPECT_THROW((void)cfd::geometry::write_geo(internal), std::invalid_argument);

    GeometrySpec bl3d;
    bl3d.bl_layers = 3; // boundary layers only in 2D
    bl3d.bl_first = 1e-3;
    EXPECT_THROW((void)cfd::geometry::write_geo(bl3d), std::invalid_argument);

    GeometrySpec sphere2d = cylinder_2d();
    sphere2d.bodies[0].kind = Body::Kind::Sphere;
    EXPECT_THROW((void)cfd::geometry::write_geo(sphere2d), std::invalid_argument);

    GeometrySpec bad_domain;
    bad_domain.domain_max = Vec3(1, 0, 1);
    EXPECT_THROW((void)cfd::geometry::write_geo(bad_domain), std::invalid_argument);
}

TEST(Geometry, GenerateMesh2dCylinder) {
    const std::string gmsh = cfd::geometry::find_gmsh();
    if (gmsh.empty()) {
        GTEST_SKIP() << "Gmsh not installed";
    }
    const auto dir = std::filesystem::temp_directory_path() / "cfd_test_geometry_2d";
    const auto mesh = cfd::geometry::generate_mesh(cylinder_2d(), dir, gmsh);
    for (const char* name : {"inlet", "outlet", "walls", "front", "back", "cylinder"}) {
        EXPECT_NE(mesh.find_patch(name), nullptr) << name;
    }
    EXPECT_EQ(mesh.find_patch("cylinder")->type, cfd::mesh::PatchType::Wall);
    const auto q = cfd::mesh::compute_quality(mesh);
    const double expected = (2.2 * 0.41 - std::numbers::pi * 0.05 * 0.05) * 0.01;
    EXPECT_NEAR(q.total_volume, expected, 2e-3 * expected);
    EXPECT_EQ(q.n_negative_volumes, 0);
    // One cell layer: front and back have one face per cell.
    EXPECT_EQ(mesh.find_patch("front")->n_faces, mesh.n_cells());
}

} // namespace
