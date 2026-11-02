#include <cfd/mesh/gmsh_reader.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>

using cfd::core::Vec3;
using cfd::mesh::Mesh;
using cfd::mesh::PatchType;

namespace {

// Unit cube, one hex, three physical surface groups.
const char* const k_hex_cube = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
4
2 1 "inlet"
2 2 "outlet"
2 3 "walls"
3 4 "fluid"
$EndPhysicalNames
$Entities
0 0 3 1
1 0 0 0 1 1 0 1 1 0
2 0 0 1 1 1 1 1 2 0
3 0 0 0 1 1 1 1 3 0
1 0 0 0 1 1 1 1 4 3 1 2 3
$EndEntities
$Nodes
1 8 1 8
3 1 0 8
1
2
3
4
5
6
7
8
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
$EndNodes
$Elements
4 7 1 7
2 1 3 1
1 1 2 3 4
2 2 3 1
2 5 6 7 8
2 3 3 4
3 1 2 6 5
4 2 3 7 6
5 3 4 8 7
6 4 1 5 8
3 1 5 1
7 1 2 3 4 5 6 7 8
$EndElements
)";

// Prism with a tet on top (sharing the top triangle), no physical groups,
// non-contiguous node tags and an unknown section.
const char* const k_prism_tet = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$Comments
anything here is ignored
$EndComments
$Entities
0 0 0 1
1 0 0 0 1 1 2 0 0
$EndEntities
$Nodes
1 7 10 70
3 1 0 7
10
20
30
40
50
60
70
0 0 0
1 0 0
0 1 0
0 0 1
1 0 1
0 1 1
0 0 2
$EndNodes
$Elements
2 2 1 2
3 1 6 1
1 10 20 30 40 50 60
3 1 4 1
2 40 50 60 70
$EndElements
)";

std::string replace_once(std::string text, const std::string& from, const std::string& to) {
    const auto pos = text.find(from);
    if (pos != std::string::npos) {
        text.replace(pos, from.size(), to);
    }
    return text;
}

Mesh read_string(const std::string& text) {
    std::istringstream in(text);
    return cfd::mesh::read_gmsh(in);
}

} // namespace

TEST(GmshReader, HexCubeWithPhysicalGroups) {
    const Mesh mesh = read_string(k_hex_cube);
    ASSERT_EQ(mesh.n_cells(), 1);
    EXPECT_EQ(mesh.n_points(), 8);
    EXPECT_EQ(mesh.n_internal_faces(), 0);
    EXPECT_EQ(mesh.n_boundary_faces(), 6);
    EXPECT_NEAR(mesh.cell_volumes()[0], 1.0, 1e-12);

    const auto* inlet = mesh.find_patch("inlet");
    const auto* outlet = mesh.find_patch("outlet");
    const auto* walls = mesh.find_patch("walls");
    ASSERT_NE(inlet, nullptr);
    ASSERT_NE(outlet, nullptr);
    ASSERT_NE(walls, nullptr);
    EXPECT_EQ(mesh.find_patch("defaultFaces"), nullptr);
    EXPECT_EQ(mesh.find_patch("fluid"), nullptr);

    EXPECT_EQ(inlet->type, PatchType::Inlet);
    EXPECT_EQ(outlet->type, PatchType::Outlet);
    EXPECT_EQ(walls->type, PatchType::Wall);
    EXPECT_EQ(inlet->n_faces, 1);
    EXPECT_EQ(outlet->n_faces, 1);
    EXPECT_EQ(walls->n_faces, 4);

    const auto& sf = mesh.face_area_vectors();
    EXPECT_TRUE(sf[static_cast<std::size_t>(inlet->start_face)].isApprox(Vec3(0, 0, -1)));
    EXPECT_TRUE(sf[static_cast<std::size_t>(outlet->start_face)].isApprox(Vec3(0, 0, 1)));
}

TEST(GmshReader, PrismAndTetWithoutPhysicalGroups) {
    const Mesh mesh = read_string(k_prism_tet);
    ASSERT_EQ(mesh.n_cells(), 2);
    EXPECT_EQ(mesh.n_points(), 7);
    EXPECT_EQ(mesh.n_internal_faces(), 1);
    EXPECT_EQ(mesh.n_boundary_faces(), 7);
    EXPECT_NEAR(mesh.cell_volumes()[0], 0.5, 1e-12);
    EXPECT_NEAR(mesh.cell_volumes()[1], 1.0 / 6.0, 1e-12);

    // Shared triangle at z = 1, normal from prism (owner 0) to tet (neighbour 1).
    EXPECT_EQ(mesh.owner()[0], 0);
    EXPECT_EQ(mesh.neighbour()[0], 1);
    EXPECT_TRUE(mesh.face_area_vectors()[0].isApprox(Vec3(0, 0, 0.5)));

    ASSERT_EQ(mesh.patches().size(), 1U);
    EXPECT_EQ(mesh.patches()[0].name, "defaultFaces");
    EXPECT_EQ(mesh.patches()[0].type, PatchType::Wall);
    EXPECT_EQ(mesh.patches()[0].n_faces, 7);
}

TEST(GmshReader, RejectsOldVersion) {
    EXPECT_THROW((void)read_string(replace_once(k_hex_cube, "4.1 0 8", "2.2 0 8")),
                 std::runtime_error);
}

TEST(GmshReader, RejectsBinary) {
    EXPECT_THROW((void)read_string(replace_once(k_hex_cube, "4.1 0 8", "4.1 1 8")),
                 std::runtime_error);
}

TEST(GmshReader, RejectsSecondOrderElements) {
    EXPECT_THROW((void)read_string(replace_once(k_prism_tet, "3 1 4 1", "3 1 11 1")),
                 std::runtime_error);
}

TEST(GmshReader, RejectsUnknownNode) {
    EXPECT_THROW((void)read_string(replace_once(k_prism_tet, "2 40 50 60 70", "2 40 50 60 99")),
                 std::runtime_error);
}

TEST(GmshReader, RejectsTruncatedInput) {
    const std::string text(k_hex_cube);
    EXPECT_THROW((void)read_string(text.substr(0, text.find("$Elements") + 20)),
                 std::runtime_error);
}

TEST(GmshReader, MissingFileThrows) {
    EXPECT_THROW((void)cfd::mesh::read_gmsh(std::filesystem::path("does/not/exist.msh")),
                 std::runtime_error);
}

TEST(GmshReader, InferPatchType) {
    EXPECT_EQ(cfd::mesh::infer_patch_type("Inlet_1"), PatchType::Inlet);
    EXPECT_EQ(cfd::mesh::infer_patch_type("outletTop"), PatchType::Outlet);
    EXPECT_EQ(cfd::mesh::infer_patch_type("WALLS"), PatchType::Wall);
    EXPECT_EQ(cfd::mesh::infer_patch_type("symmetryPlane"), PatchType::Symmetry);
    EXPECT_EQ(cfd::mesh::infer_patch_type("farfield"), PatchType::Patch);
}
