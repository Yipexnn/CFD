#include "vtk_convert.hpp"

#include <array>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/mesh/cell_shapes.hpp>
#include <cfd/mesh/mesh_builder.hpp>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkDataArray.h>
#include <vtkUnstructuredGrid.h>

namespace {

using cfd::core::Scalar;
using cfd::core::Vec3;
using cfd::gui::boundary_to_vtk;
using cfd::gui::FieldSet;
using cfd::gui::mesh_to_vtk;
using cfd::gui::scalar_field_names;
using cfd::gui::set_fields;
using cfd::mesh::BoxSpec;
using cfd::mesh::CellShape;
using cfd::mesh::Label;
using cfd::mesh::MeshBuilder;

constexpr double k_tol = 1e-12;

using Vec3d = std::array<double, 3>;

Vec3d point(vtkUnstructuredGrid& grid, vtkIdType id) {
    const double* p = grid.GetPoint(id);
    return {p[0], p[1], p[2]};
}

Vec3d sub(const Vec3d& a, const Vec3d& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3d cross(const Vec3d& a, const Vec3d& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double dot(const Vec3d& a, const Vec3d& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

vtkDataArray* array_by_name(vtkCellData* cd, const char* name) {
    return vtkDataArray::SafeDownCast(cd->GetArray(name));
}

// Builds a grid from a mesh containing exactly one cell of the given shape.
vtkSmartPointer<vtkUnstructuredGrid> single_cell_grid(CellShape shape, const std::vector<Vec3>& pts,
                                                      const std::vector<Label>& nodes) {
    MeshBuilder builder(pts);
    builder.add_cell(shape, nodes);
    return mesh_to_vtk(builder.build());
}

} // namespace

TEST(VtkConvert, BoxMeshToGrid) {
    BoxSpec spec;
    spec.nx = 3;
    spec.ny = 4;
    spec.nz = 5;
    spec.min = Vec3(-1.0, 0.5, 2.0);
    spec.max = Vec3(2.0, 4.5, 5.0);
    const auto grid = mesh_to_vtk(cfd::mesh::make_box_mesh(spec));

    EXPECT_EQ(grid->GetNumberOfCells(), 60);
    EXPECT_EQ(grid->GetNumberOfPoints(), 120);
    for (vtkIdType i = 0; i < grid->GetNumberOfCells(); ++i) {
        EXPECT_EQ(grid->GetCellType(i), VTK_POLYHEDRON) << "cell " << i;
    }

    const double* b = grid->GetBounds();
    EXPECT_NEAR(b[0], -1.0, k_tol);
    EXPECT_NEAR(b[1], 2.0, k_tol);
    EXPECT_NEAR(b[2], 0.5, k_tol);
    EXPECT_NEAR(b[3], 4.5, k_tol);
    EXPECT_NEAR(b[4], 2.0, k_tol);
    EXPECT_NEAR(b[5], 5.0, k_tol);
}

TEST(VtkConvert, PolyhedronFacesOutward) {
    const BoxSpec spec{};
    const auto grid = mesh_to_vtk(cfd::mesh::make_box_mesh(spec));
    ASSERT_EQ(grid->GetNumberOfCells(), 1);

    vtkCell* cell = grid->GetCell(vtkIdType{0});
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ(cell->GetNumberOfFaces(), 6);
    EXPECT_EQ(cell->GetNumberOfPoints(), 8);

    // Cell centroid from its unique points.
    Vec3d cell_centre{0.0, 0.0, 0.0};
    for (int p = 0; p < cell->GetNumberOfPoints(); ++p) {
        const Vec3d x = point(*grid, cell->GetPointId(p));
        for (std::size_t d = 0; d < 3; ++d) {
            cell_centre[d] += x[d] / 8.0;
        }
    }

    // Each face normal (from its first three points) must point away from the cell centre.
    for (int f = 0; f < cell->GetNumberOfFaces(); ++f) {
        vtkCell* face = cell->GetFace(f);
        ASSERT_NE(face, nullptr);
        ASSERT_EQ(face->GetNumberOfPoints(), 4);
        const Vec3d a = point(*grid, face->GetPointId(0));
        const Vec3d b = point(*grid, face->GetPointId(1));
        const Vec3d c = point(*grid, face->GetPointId(2));
        Vec3d face_centre{0.0, 0.0, 0.0};
        for (int p = 0; p < 4; ++p) {
            const Vec3d x = point(*grid, face->GetPointId(p));
            for (std::size_t d = 0; d < 3; ++d) {
                face_centre[d] += x[d] / 4.0;
            }
        }
        const Vec3d normal = cross(sub(b, a), sub(c, a));
        EXPECT_GT(dot(normal, sub(face_centre, cell_centre)), 0.0) << "face " << f;
    }
}

TEST(VtkConvert, MixedShapes) {
    const auto tet = single_cell_grid(
        CellShape::Tet,
        {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)},
        {0, 1, 2, 3});
    ASSERT_EQ(tet->GetNumberOfCells(), 1);
    EXPECT_EQ(tet->GetCellType(0), VTK_POLYHEDRON);
    EXPECT_EQ(tet->GetCell(vtkIdType{0})->GetNumberOfFaces(), 4);

    const auto prism =
        single_cell_grid(CellShape::Prism,
                         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                          Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 1.0), Vec3(0.0, 1.0, 1.0)},
                         {0, 1, 2, 3, 4, 5});
    ASSERT_EQ(prism->GetNumberOfCells(), 1);
    EXPECT_EQ(prism->GetCellType(0), VTK_POLYHEDRON);
    EXPECT_EQ(prism->GetCell(vtkIdType{0})->GetNumberOfFaces(), 5);
}

TEST(VtkConvert, BoundaryGrid) {
    BoxSpec spec;
    spec.nx = 2;
    spec.ny = 2;
    spec.nz = 2;
    const auto grid = boundary_to_vtk(cfd::mesh::make_box_mesh(spec));

    EXPECT_EQ(grid->GetNumberOfCells(), 24);
    vtkDataArray* patch = array_by_name(grid->GetCellData(), "patch");
    ASSERT_NE(patch, nullptr);

    std::array<int, 6> counts{};
    for (vtkIdType i = 0; i < grid->GetNumberOfCells(); ++i) {
        const auto value = static_cast<int>(patch->GetTuple1(i));
        ASSERT_GE(value, 0);
        ASSERT_LT(value, 6);
        ++counts[static_cast<std::size_t>(value)];
    }
    for (int count : counts) {
        EXPECT_EQ(count, 4);
    }
}

TEST(VtkConvert, SetFields) {
    BoxSpec spec;
    spec.nx = 2;
    spec.ny = 2;
    spec.nz = 1;
    const auto grid = mesh_to_vtk(cfd::mesh::make_box_mesh(spec));
    const auto n_cells = static_cast<std::size_t>(grid->GetNumberOfCells());
    ASSERT_EQ(n_cells, 4u);

    FieldSet fields;
    for (std::size_t i = 0; i < n_cells; ++i) {
        const auto x = static_cast<Scalar>(i);
        fields.u.emplace_back(x, 2.0 * x, 3.0 * x);
        fields.p.push_back(x);
    }
    set_fields(*grid, fields);

    vtkCellData* cd = grid->GetCellData();
    EXPECT_EQ(cd->GetNumberOfArrays(), 6);
    EXPECT_EQ(cd->GetArray("k"), nullptr);
    ASSERT_NE(cd->GetVectors(), nullptr);
    EXPECT_EQ(std::string(cd->GetVectors()->GetName()), "U");

    vtkDataArray* U = array_by_name(cd, "U");
    vtkDataArray* mag = array_by_name(cd, "|U|");
    vtkDataArray* ux = array_by_name(cd, "Ux");
    vtkDataArray* uy = array_by_name(cd, "Uy");
    vtkDataArray* uz = array_by_name(cd, "Uz");
    vtkDataArray* p = array_by_name(cd, "p");
    ASSERT_NE(U, nullptr);
    ASSERT_NE(mag, nullptr);
    ASSERT_NE(ux, nullptr);
    ASSERT_NE(uy, nullptr);
    ASSERT_NE(uz, nullptr);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(U->GetNumberOfComponents(), 3);

    for (std::size_t i = 0; i < n_cells; ++i) {
        const auto id = static_cast<vtkIdType>(i);
        const auto x = static_cast<double>(i);
        EXPECT_NEAR(U->GetComponent(id, 0), x, k_tol) << "cell " << i;
        EXPECT_NEAR(U->GetComponent(id, 1), 2.0 * x, k_tol) << "cell " << i;
        EXPECT_NEAR(U->GetComponent(id, 2), 3.0 * x, k_tol) << "cell " << i;
        EXPECT_NEAR(mag->GetTuple1(id), std::sqrt(14.0) * x, k_tol) << "cell " << i;
        EXPECT_NEAR(ux->GetTuple1(id), x, k_tol) << "cell " << i;
        EXPECT_NEAR(uy->GetTuple1(id), 2.0 * x, k_tol) << "cell " << i;
        EXPECT_NEAR(uz->GetTuple1(id), 3.0 * x, k_tol) << "cell " << i;
        EXPECT_NEAR(p->GetTuple1(id), x, k_tol) << "cell " << i;
    }

    EXPECT_EQ(scalar_field_names(fields), (std::vector<std::string>{"|U|", "Ux", "Uy", "Uz", "p"}));
}

TEST(VtkConvert, SetFieldsReplacesArrays) {
    BoxSpec spec;
    spec.nx = 2;
    spec.ny = 2;
    spec.nz = 1;
    const auto grid = mesh_to_vtk(cfd::mesh::make_box_mesh(spec));

    FieldSet both;
    both.u.assign(4, Vec3(1.0, 0.0, 0.0));
    both.p.assign(4, 2.0);
    set_fields(*grid, both);
    ASSERT_NE(grid->GetCellData()->GetArray("U"), nullptr);

    FieldSet only_p;
    only_p.p.assign(4, 5.0);
    set_fields(*grid, only_p);

    EXPECT_EQ(grid->GetCellData()->GetArray("U"), nullptr);
    EXPECT_EQ(grid->GetCellData()->GetArray("|U|"), nullptr);
    ASSERT_NE(grid->GetCellData()->GetArray("p"), nullptr);
    EXPECT_EQ(grid->GetCellData()->GetNumberOfArrays(), 1);
}

TEST(VtkConvert, SizeMismatchThrows) {
    BoxSpec spec;
    spec.nx = 2;
    spec.ny = 2;
    spec.nz = 1;
    const auto grid = mesh_to_vtk(cfd::mesh::make_box_mesh(spec));

    FieldSet bad;
    bad.p.assign(3, 1.0);
    EXPECT_THROW(set_fields(*grid, bad), std::invalid_argument);
}
