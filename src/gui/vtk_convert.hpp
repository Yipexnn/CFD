#pragma once

#include <cfd/core/types.hpp>
#include <cfd/mesh/mesh.hpp>
#include <string>
#include <vector>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

namespace cfd::gui {

using core::Scalar;
using core::Vec3;

// Cell fields of a solution snapshot.
struct FieldSet {
    std::vector<Vec3> u;
    std::vector<Scalar> p;
    std::vector<Scalar> k;
    std::vector<Scalar> omega;
    std::vector<Scalar> nut;
    std::vector<Scalar> y_plus;
    std::vector<Scalar> t; // temperature
};

// Builds an unstructured grid with one VTK_POLYHEDRON per mesh cell (works for any
// cell shape). Patch ids are stored as cell data "patch" on a separate boundary grid.
[[nodiscard]] vtkSmartPointer<vtkUnstructuredGrid> mesh_to_vtk(const mesh::Mesh& mesh);

// Replaces the cell data arrays of grid with the given fields ("U", "|U|", "p", ...).
// Empty vectors are skipped; sizes must match the number of cells.
void set_fields(vtkUnstructuredGrid& grid, const FieldSet& fields);

// Boundary faces as polygons (cell data "patch" = patch index) for surface display and
// picking of patches in the case editor.
[[nodiscard]] vtkSmartPointer<vtkUnstructuredGrid> boundary_to_vtk(const mesh::Mesh& mesh);

// Names of the scalar arrays set_fields() creates for the given field set.
[[nodiscard]] std::vector<std::string> scalar_field_names(const FieldSet& fields);

} // namespace cfd::gui
