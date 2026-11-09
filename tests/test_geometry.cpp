#include <cfd/case/case_config.hpp>
#include <cfd/geometry/geometry.hpp>
#include <cfd/mesh/gmsh_reader.hpp>
#include <cfd/mesh/quality.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    bl3d.bl_layers = 3; // 3D boundary layers need bodies
    bl3d.bl_first = 1e-3;
    EXPECT_THROW((void)cfd::geometry::write_geo(bl3d), std::invalid_argument);

    GeometrySpec sphere2d = cylinder_2d();
    sphere2d.bodies[0].kind = Body::Kind::Sphere;
    EXPECT_THROW((void)cfd::geometry::write_geo(sphere2d), std::invalid_argument);

    GeometrySpec bad_domain;
    bad_domain.domain_max = Vec3(1, 0, 1);
    EXPECT_THROW((void)cfd::geometry::write_geo(bad_domain), std::invalid_argument);
}

// Refinement boxes: Box size fields combined with the body distance field; smaller cells
// inside the box; case-file round trip.
TEST(Geometry, RefineBoxes) {
    GeometrySpec s;
    s.domain_max = Vec3(4, 2, 2);
    Body sphere;
    sphere.kind = Body::Kind::Sphere;
    sphere.a = Vec3(1.0, 1.0, 1.0);
    sphere.radius = 0.2;
    s.bodies = {sphere};
    s.size = 0.5;
    s.size_near = 0.1;
    s.refine_distance = 0.5;
    s.refine_boxes.push_back({Vec3(1.5, 0.5, 0.5), Vec3(3.5, 1.5, 1.5), 0.15});
    const std::string geo = cfd::geometry::write_geo(s);
    EXPECT_TRUE(contains(geo, "Field[10] = Box;"));
    EXPECT_TRUE(contains(geo, "Field[9].FieldsList = {2, 10}"));
    EXPECT_TRUE(contains(geo, "Background Field = 9;"));

    const std::string json = R"({"name": "rb", "mesh": {"type": "geometry", "min": [0, 0, 0],
        "max": [4, 2, 2], "size": 0.5,
        "refine_boxes": [{"min": [1.5, 0.5, 0.5], "max": [3.5, 1.5, 1.5], "size": 0.15}]}})";
    const auto c = cfd::cases::parse_case(json);
    ASSERT_EQ(c.mesh.geometry.refine_boxes.size(), 1U);
    const auto b = cfd::cases::parse_case(cfd::cases::case_to_json(c));
    ASSERT_EQ(b.mesh.geometry.refine_boxes.size(), 1U);
    EXPECT_DOUBLE_EQ(b.mesh.geometry.refine_boxes[0].size, 0.15);
    EXPECT_DOUBLE_EQ(b.mesh.geometry.refine_boxes[0].max.x(), 3.5);

    const std::string gmsh = cfd::geometry::find_gmsh();
    if (gmsh.empty()) {
        GTEST_SKIP() << "Gmsh not installed";
    }
    const auto dir = std::filesystem::temp_directory_path() / "cfd_test_refine_box";
    const auto mesh = cfd::geometry::generate_mesh(s, dir, gmsh);
    double v_in = 0.0;
    double v_out = 0.0;
    int n_in = 0;
    int n_out = 0;
    for (std::size_t i = 0; i < mesh.cell_volumes().size(); ++i) {
        const Vec3& x = mesh.cell_centres()[i];
        if (x.x() > 2.0 && x.x() < 3.3 && std::abs(x.y() - 1.0) < 0.4 &&
            std::abs(x.z() - 1.0) < 0.4) {
            v_in += mesh.cell_volumes()[i];
            ++n_in;
        } else if (x.x() > 3.6 || x.y() < 0.3 || x.y() > 1.7) {
            v_out += mesh.cell_volumes()[i];
            ++n_out;
        }
    }
    ASSERT_GT(n_in, 0);
    ASSERT_GT(n_out, 0);
    EXPECT_LT(v_in / n_in, 0.2 * v_out / n_out);
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

// 3D prism boundary layers around a sphere: prisms on the body, tets elsewhere, exact
// volume, valid cells; the case-file syntax enables them for 3D geometries.
TEST(Geometry, GenerateMesh3dPrismLayers) {
    const std::string gmsh = cfd::geometry::find_gmsh();
    if (gmsh.empty()) {
        GTEST_SKIP() << "Gmsh not installed";
    }
    GeometrySpec s;
    s.domain_min = Vec3(-1.5, -1, -1);
    s.domain_max = Vec3(2.5, 1, 1);
    Body sphere;
    sphere.kind = Body::Kind::Sphere;
    sphere.radius = 0.25;
    sphere.patch = "sphere";
    s.bodies.push_back(sphere);
    s.size = 0.3;
    s.size_near = 0.06;
    s.refine_distance = 0.5;
    s.bl_layers = 5;
    s.bl_first = 0.005;
    s.bl_ratio = 1.3;
    s.domain_patch_names = {"inlet", "outlet", "side", "side", "side", "side"};
    EXPECT_NE(cfd::geometry::write_geo(s).find("Recombine"), std::string::npos);
    const auto dir = std::filesystem::temp_directory_path() / "cfd_test_geometry_bl3d";
    const auto mesh = cfd::geometry::generate_mesh(s, dir, gmsh);
    ASSERT_NE(mesh.find_patch("sphere"), nullptr);
    EXPECT_EQ(mesh.find_patch("sphere")->type, cfd::mesh::PatchType::Wall);
    EXPECT_NE(mesh.find_patch("inlet"), nullptr);
    EXPECT_NE(mesh.find_patch("side"), nullptr);
    const auto q = cfd::mesh::compute_quality(mesh);
    const double expected = 4.0 * 2.0 * 2.0 - 4.0 / 3.0 * std::numbers::pi * std::pow(0.25, 3);
    EXPECT_NEAR(q.total_volume, expected, 2e-3 * expected);
    EXPECT_EQ(q.n_negative_volumes, 0);
    // Prism cells (5 faces) form the layers: 5 per sphere surface triangle.
    int prisms = 0;
    for (cfd::core::Label c = 0; c < mesh.n_cells(); ++c) {
        prisms += mesh.cell_face_offsets()[static_cast<std::size_t>(c) + 1] -
                              mesh.cell_face_offsets()[static_cast<std::size_t>(c)] ==
                          5
                      ? 1
                      : 0;
    }
    EXPECT_EQ(prisms, 5 * mesh.find_patch("sphere")->n_faces);
    // First layer thickness: wall cell centres at about bl_first / 2 from their wall face.
    const auto* wall = mesh.find_patch("sphere");
    for (cfd::core::Label i = 0; i < wall->n_faces; ++i) {
        const auto f = static_cast<std::size_t>(wall->start_face + i);
        const auto c = static_cast<std::size_t>(mesh.owner()[f]);
        const Vec3 n = mesh.face_area_vectors()[f].normalized();
        const double dn = std::abs((mesh.cell_centres()[c] - mesh.face_centres()[f]).dot(n));
        EXPECT_NEAR(dn, 0.0025, 0.0008);
    }
}

} // namespace

namespace {

// Gmsh physical volumes become cell zones of the mesh.
TEST(Geometry, GmshPhysicalVolumesAsZones) {
    const std::string gmsh = cfd::geometry::find_gmsh();
    if (gmsh.empty()) {
        GTEST_SKIP() << "Gmsh not installed";
    }
    const auto dir = std::filesystem::temp_directory_path() / "cfd_test_zones";
    std::filesystem::create_directories(dir);
    {
        std::ofstream geo(dir / "zones.geo");
        geo << "SetFactory(\"OpenCASCADE\");\n"
               "Box(1) = {0, 0, 0, 1, 1, 1};\nBox(2) = {1, 0, 0, 1, 1, 1};\n"
               "v() = BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; };\n"
               "Mesh.MeshSizeMax = 0.25;\n"
               "Physical Volume(\"fluid\") = {1};\nPhysical Volume(\"solid\") = {2};\n"
               "Physical Surface(\"walls\") = {Surface{:}};\n"
               "Mesh.MshFileVersion = 4.1;\n";
    }
    std::string cmd = "\"" + gmsh + "\" \"" + (dir / "zones.geo").string() + "\" -3 -o \"" +
                      (dir / "zones.msh").string() + "\" > \"" + (dir / "log.txt").string() +
                      "\" 2>&1";
#ifdef _WIN32
    cmd = "\"" + cmd + "\""; // cmd.exe strips one level of quotes
#endif
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    const auto mesh = cfd::mesh::read_gmsh(dir / "zones.msh");
    ASSERT_EQ(mesh.cell_zone_names().size(), 2U);
    std::vector<double> volume(2, 0.0);
    for (std::size_t c = 0; c < mesh.cell_zones().size(); ++c) {
        volume[static_cast<std::size_t>(mesh.cell_zones()[c])] += mesh.cell_volumes()[c];
    }
    EXPECT_NEAR(volume[0], 1.0, 1e-9);
    EXPECT_NEAR(volume[1], 1.0, 1e-9);
    for (std::size_t c = 0; c < mesh.cell_zones().size(); ++c) {
        const bool right = mesh.cell_centres()[c].x() > 1.0;
        EXPECT_EQ(mesh.cell_zone_names()[static_cast<std::size_t>(mesh.cell_zones()[c])],
                  right ? "solid" : "fluid");
    }
}

} // namespace

#include <cfd/fv/fv_geometry.hpp>

namespace {

// Least-squares gradients are exact for linear fields on a skewed tetrahedral mesh; the
// Green-Gauss gradient with linear face interpolation is not.
TEST(Geometry, LeastSquaresGradientExactOnTets) {
    const std::string gmsh = cfd::geometry::find_gmsh();
    if (gmsh.empty()) {
        GTEST_SKIP() << "Gmsh not installed";
    }
    GeometrySpec s;
    s.domain_min = Vec3(0, 0, 0);
    s.domain_max = Vec3(1, 1, 1);
    Body sphere;
    sphere.kind = Body::Kind::Sphere;
    sphere.a = Vec3(0.5, 0.5, 0.5);
    sphere.radius = 0.2;
    s.bodies.push_back(sphere);
    s.size = 0.15;
    const auto dir = std::filesystem::temp_directory_path() / "cfd_test_lsq_gradient";
    const auto mesh = cfd::geometry::generate_mesh(s, dir, gmsh);
    const cfd::fv::FvGeometry geo(mesh);
    const Vec3 g(1.5, -2.0, 0.75);
    auto phi = [&](const Vec3& x) { return 3.0 + g.dot(x); };
    std::vector<double> cells(static_cast<std::size_t>(mesh.n_cells()));
    for (std::size_t c = 0; c < cells.size(); ++c) {
        cells[c] = phi(mesh.cell_centres()[c]);
    }
    const auto nif = static_cast<std::size_t>(mesh.n_internal_faces());
    std::vector<double> bnd(static_cast<std::size_t>(mesh.n_boundary_faces()));
    for (std::size_t b = 0; b < bnd.size(); ++b) {
        bnd[b] = phi(mesh.face_centres()[nif + b]);
    }
    const auto lsq = geo.least_squares_gradient(cells, bnd);
    double err_lsq = 0.0;
    double err_gauss = 0.0;
    for (std::size_t c = 0; c < cells.size(); ++c) {
        err_lsq = std::max(err_lsq, (lsq[c] - g).norm());
        Vec3 gauss = Vec3::Zero();
        for (auto k = static_cast<std::size_t>(mesh.cell_face_offsets()[c]);
             k < static_cast<std::size_t>(mesh.cell_face_offsets()[c + 1]); ++k) {
            const auto f = static_cast<std::size_t>(mesh.cell_faces()[k]);
            double value = 0.0;
            if (f < nif) {
                const double w = geo.weights()[f];
                value = w * cells[static_cast<std::size_t>(mesh.owner()[f])] +
                        (1.0 - w) * cells[static_cast<std::size_t>(mesh.neighbour()[f])];
            } else {
                value = bnd[f - nif];
            }
            const double sign = static_cast<std::size_t>(mesh.owner()[f]) == c ? 1.0 : -1.0;
            gauss += sign * value * mesh.face_area_vectors()[f];
        }
        gauss /= mesh.cell_volumes()[c];
        err_gauss = std::max(err_gauss, (gauss - g).norm());
    }
    EXPECT_LT(err_lsq, 1e-9 * g.norm());
    EXPECT_GT(err_gauss, 1e-3 * g.norm()); // inconsistent on skewed cells
}

} // namespace
