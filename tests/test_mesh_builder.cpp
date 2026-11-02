#include <cfd/core/types.hpp>
#include <cfd/mesh/cell_shapes.hpp>
#include <cfd/mesh/mesh.hpp>
#include <cfd/mesh/mesh_builder.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace cfd::mesh;
using cfd::core::Scalar;
using cfd::core::Vec3;

namespace {

constexpr Scalar k_tol = 1e-12;

template <typename T> const T& at(const std::vector<T>& v, Label i) {
    return v[static_cast<std::size_t>(i)];
}

// Single-cell mesh whose cell uses all given points in order.
Mesh single_cell_mesh(CellShape shape, const std::vector<Vec3>& pts) {
    MeshBuilder builder(pts);
    std::vector<Label> nodes(pts.size());
    std::iota(nodes.begin(), nodes.end(), Label{0});
    builder.add_cell(shape, nodes);
    return builder.build();
}

// Norm of the sum of outward face area vectors of cell c (zero for a closed cell).
Scalar closure_norm(const Mesh& mesh, Label c) {
    Vec3 sum = Vec3::Zero();
    for (Label f = 0; f < mesh.n_faces(); ++f) {
        if (at(mesh.owner(), f) == c) {
            sum += at(mesh.face_area_vectors(), f);
        }
        if (f < mesh.n_internal_faces() && at(mesh.neighbour(), f) == c) {
            sum -= at(mesh.face_area_vectors(), f);
        }
    }
    return sum.norm();
}

void expect_vec_near(const Vec3& actual, const Vec3& expected) {
    EXPECT_NEAR(actual.x(), expected.x(), k_tol);
    EXPECT_NEAR(actual.y(), expected.y(), k_tol);
    EXPECT_NEAR(actual.z(), expected.z(), k_tol);
}

std::vector<Vec3> tet_points() {
    return {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
}

std::vector<Vec3> pyramid_points() {
    return {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0), Vec3(0.5, 0.5, 1)};
}

std::vector<Vec3> prism_points() {
    return {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0),
            Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(0, 1, 1)};
}

std::vector<Vec3> hex_points() {
    return {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
            Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1), Vec3(0, 1, 1)};
}

void check_single_shape(CellShape shape, const std::vector<Vec3>& pts, Scalar volume,
                        const Vec3& centre) {
    const Mesh mesh = single_cell_mesh(shape, pts);

    EXPECT_EQ(mesh.n_cells(), 1);
    EXPECT_EQ(mesh.n_internal_faces(), 0);
    EXPECT_EQ(mesh.n_boundary_faces(), static_cast<Label>(local_faces(shape).size()));

    EXPECT_NEAR(at(mesh.cell_volumes(), 0), volume, k_tol);
    expect_vec_near(at(mesh.cell_centres(), 0), centre);
    EXPECT_NEAR(closure_norm(mesh, 0), 0.0, k_tol);

    const Vec3& cc = at(mesh.cell_centres(), 0);
    for (Label f = 0; f < mesh.n_faces(); ++f) {
        const Vec3 d = at(mesh.face_centres(), f) - cc;
        EXPECT_GT(at(mesh.face_area_vectors(), f).dot(d), 0.0) << "face " << f << " not outward";
    }

    const Patch* patch = mesh.find_patch("defaultFaces");
    ASSERT_NE(patch, nullptr);
    EXPECT_EQ(patch->n_faces, mesh.n_faces());
}

} // namespace

TEST(MeshBuilder, SingleTet) {
    check_single_shape(CellShape::Tet, tet_points(), 1.0 / 6.0, Vec3(0.25, 0.25, 0.25));
}

TEST(MeshBuilder, SinglePyramid) {
    check_single_shape(CellShape::Pyramid, pyramid_points(), 1.0 / 3.0, Vec3(0.5, 0.5, 0.25));
}

TEST(MeshBuilder, SinglePrism) {
    check_single_shape(CellShape::Prism, prism_points(), 0.5, Vec3(1.0 / 3.0, 1.0 / 3.0, 0.5));
}

TEST(MeshBuilder, SingleHex) {
    check_single_shape(CellShape::Hex, hex_points(), 1.0, Vec3(0.5, 0.5, 0.5));
}

TEST(MeshBuilder, ShearedHexHasUnitVolume) {
    std::vector<Vec3> pts = hex_points();
    for (Vec3& p : pts) {
        p.x() += 0.5 * p.z();
    }
    const Mesh mesh = single_cell_mesh(CellShape::Hex, pts);

    ASSERT_EQ(mesh.n_cells(), 1);
    EXPECT_NEAR(at(mesh.cell_volumes(), 0), 1.0, k_tol);
    EXPECT_NEAR(closure_norm(mesh, 0), 0.0, k_tol);
}

TEST(MeshBuilder, TwoHexesShareAFace) {
    // Hex A is the unit cube; hex B is shifted by +1 in x and shares four points with A.
    std::vector<Vec3> pts = hex_points();
    pts.push_back(Vec3(2, 0, 0)); // 8
    pts.push_back(Vec3(2, 1, 0)); // 9
    pts.push_back(Vec3(2, 0, 1)); // 10
    pts.push_back(Vec3(2, 1, 1)); // 11

    MeshBuilder builder(pts);
    const std::vector<Label> hex_a{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<Label> hex_b{1, 8, 9, 2, 5, 10, 11, 6};
    builder.add_cell(CellShape::Hex, hex_a);
    builder.add_cell(CellShape::Hex, hex_b);
    const Mesh mesh = builder.build();

    EXPECT_EQ(mesh.n_cells(), 2);
    EXPECT_EQ(mesh.n_points(), 12);
    EXPECT_EQ(mesh.n_internal_faces(), 1);
    EXPECT_EQ(mesh.n_boundary_faces(), 10);
    EXPECT_EQ(at(mesh.owner(), 0), 0);
    EXPECT_EQ(at(mesh.neighbour(), 0), 1);
    expect_vec_near(at(mesh.face_area_vectors(), 0), Vec3(1, 0, 0));

    EXPECT_NEAR(at(mesh.cell_volumes(), 0) + at(mesh.cell_volumes(), 1), 2.0, k_tol);
    EXPECT_NEAR(closure_norm(mesh, 0), 0.0, k_tol);
    EXPECT_NEAR(closure_norm(mesh, 1), 0.0, k_tol);
}

TEST(MeshBuilder, PatchTaggingAndContiguity) {
    MeshBuilder builder(hex_points());
    const std::vector<Label> nodes{0, 1, 2, 3, 4, 5, 6, 7};
    builder.add_cell(CellShape::Hex, nodes);
    builder.add_patch("bottom", PatchType::Wall);
    const std::vector<Label> bottom{3, 1, 0, 2}; // any order
    builder.add_boundary_face(bottom, "bottom");
    const Mesh mesh = builder.build();

    ASSERT_EQ(mesh.patches().size(), 2U);
    const Patch& bottom_patch = mesh.patches()[0];
    const Patch& default_patch = mesh.patches()[1];

    EXPECT_EQ(bottom_patch.name, "bottom");
    EXPECT_EQ(bottom_patch.type, PatchType::Wall);
    EXPECT_EQ(bottom_patch.n_faces, 1);

    const Patch* found_bottom = mesh.find_patch("bottom");
    ASSERT_NE(found_bottom, nullptr);
    EXPECT_EQ(found_bottom->n_faces, 1);
    expect_vec_near(at(mesh.face_area_vectors(), bottom_patch.start_face), Vec3(0, 0, -1));

    const Patch* found_default = mesh.find_patch("defaultFaces");
    ASSERT_NE(found_default, nullptr);
    EXPECT_EQ(found_default->n_faces, 5);

    EXPECT_EQ(default_patch.start_face, bottom_patch.start_face + bottom_patch.n_faces);
    EXPECT_EQ(default_patch.n_faces, 5);
}

TEST(MeshBuilder, AddCellWrongNodeCountThrows) {
    MeshBuilder builder(tet_points());
    const std::vector<Label> too_few{0, 1, 2};
    EXPECT_THROW(builder.add_cell(CellShape::Tet, too_few), std::invalid_argument);
}

TEST(MeshBuilder, BoundaryFaceNotOnAnyCellThrowsOnBuild) {
    // Point 4 is not used by the tet, so the tagged triangle {0, 1, 4} is not a mesh face.
    std::vector<Vec3> pts = tet_points();
    pts.push_back(Vec3(5, 5, 5));

    MeshBuilder builder(pts);
    const std::vector<Label> tet{0, 1, 2, 3};
    builder.add_cell(CellShape::Tet, tet);
    const std::vector<Label> stray{0, 1, 4};
    builder.add_boundary_face(stray, "inlet");

    EXPECT_THROW(static_cast<void>(builder.build()), std::runtime_error);
}
