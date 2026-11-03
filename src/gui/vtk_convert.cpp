#include "vtk_convert.hpp"

#include <algorithm>
#include <stdexcept>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkDoubleArray.h>
#include <vtkIdList.h>
#include <vtkIntArray.h>
#include <vtkPoints.h>

namespace cfd::gui {

using core::Label;

namespace {

vtkSmartPointer<vtkPoints> make_points(const mesh::Mesh& mesh) {
    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataTypeToDouble();
    points->SetNumberOfPoints(mesh.n_points());
    for (Label i = 0; i < mesh.n_points(); ++i) {
        const Vec3& p = mesh.points()[static_cast<std::size_t>(i)];
        points->SetPoint(i, p.x(), p.y(), p.z());
    }
    return points;
}

void add_scalar(vtkUnstructuredGrid& grid, const char* name, const std::vector<Scalar>& values) {
    if (values.empty()) {
        return;
    }
    if (static_cast<vtkIdType>(values.size()) != grid.GetNumberOfCells()) {
        throw std::invalid_argument(std::string("set_fields: size mismatch for ") + name);
    }
    auto array = vtkSmartPointer<vtkDoubleArray>::New();
    array->SetName(name);
    array->SetNumberOfTuples(static_cast<vtkIdType>(values.size()));
    for (std::size_t c = 0; c < values.size(); ++c) {
        array->SetValue(static_cast<vtkIdType>(c), values[c]);
    }
    grid.GetCellData()->AddArray(array);
}

} // namespace

vtkSmartPointer<vtkUnstructuredGrid> mesh_to_vtk(const mesh::Mesh& mesh) {
    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(make_points(mesh));
    grid->Allocate(mesh.n_cells());

    const auto& offsets = mesh.cell_face_offsets();
    const auto& cell_faces = mesh.cell_faces();
    const auto& owner = mesh.owner();
    std::vector<vtkIdType> point_ids;
    std::vector<vtkIdType> face_stream;

    for (Label c = 0; c < mesh.n_cells(); ++c) {
        point_ids.clear();
        face_stream.clear();
        const auto begin = static_cast<std::size_t>(offsets[static_cast<std::size_t>(c)]);
        const auto end = static_cast<std::size_t>(offsets[static_cast<std::size_t>(c) + 1]);
        for (std::size_t k = begin; k < end; ++k) {
            const Label f = cell_faces[k];
            const auto pts = mesh.face(f);
            face_stream.push_back(static_cast<vtkIdType>(pts.size()));
            // Outward orientation for this cell: reverse faces it does not own.
            const bool owned = owner[static_cast<std::size_t>(f)] == c;
            for (std::size_t i = 0; i < pts.size(); ++i) {
                const Label pid = owned ? pts[i] : pts[pts.size() - 1 - i];
                face_stream.push_back(pid);
                point_ids.push_back(pid);
            }
        }
        std::sort(point_ids.begin(), point_ids.end());
        point_ids.erase(std::unique(point_ids.begin(), point_ids.end()), point_ids.end());
        // The face-stream overload is deprecated in VTK >= 9.4 but available in all 9.x.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        grid->InsertNextCell(VTK_POLYHEDRON, static_cast<vtkIdType>(point_ids.size()),
                             point_ids.data(), static_cast<vtkIdType>(end - begin),
                             face_stream.data());
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    }
    return grid;
}

vtkSmartPointer<vtkUnstructuredGrid> boundary_to_vtk(const mesh::Mesh& mesh) {
    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(make_points(mesh));
    grid->Allocate(mesh.n_boundary_faces());
    auto patch_ids = vtkSmartPointer<vtkIntArray>::New();
    patch_ids->SetName("patch");
    std::vector<vtkIdType> ids;
    const auto& patches = mesh.patches();
    for (std::size_t p = 0; p < patches.size(); ++p) {
        for (Label i = 0; i < patches[p].n_faces; ++i) {
            const auto pts = mesh.face(patches[p].start_face + i);
            ids.assign(pts.begin(), pts.end());
            grid->InsertNextCell(VTK_POLYGON, static_cast<vtkIdType>(ids.size()), ids.data());
            patch_ids->InsertNextValue(static_cast<int>(p));
        }
    }
    grid->GetCellData()->AddArray(patch_ids);
    grid->GetCellData()->SetActiveScalars("patch");
    return grid;
}

void set_fields(vtkUnstructuredGrid& grid, const FieldSet& fields) {
    vtkCellData* cd = grid.GetCellData();
    while (cd->GetNumberOfArrays() > 0) {
        cd->RemoveArray(0);
    }
    if (!fields.u.empty()) {
        if (static_cast<vtkIdType>(fields.u.size()) != grid.GetNumberOfCells()) {
            throw std::invalid_argument("set_fields: size mismatch for U");
        }
        auto u = vtkSmartPointer<vtkDoubleArray>::New();
        u->SetName("U");
        u->SetNumberOfComponents(3);
        u->SetNumberOfTuples(static_cast<vtkIdType>(fields.u.size()));
        std::vector<Scalar> mag(fields.u.size());
        std::vector<Scalar> comp[3];
        for (auto& v : comp) {
            v.resize(fields.u.size());
        }
        for (std::size_t c = 0; c < fields.u.size(); ++c) {
            const Vec3& v = fields.u[c];
            u->SetTuple3(static_cast<vtkIdType>(c), v.x(), v.y(), v.z());
            mag[c] = v.norm();
            comp[0][c] = v.x();
            comp[1][c] = v.y();
            comp[2][c] = v.z();
        }
        cd->AddArray(u);
        cd->SetActiveVectors("U");
        add_scalar(grid, "|U|", mag);
        add_scalar(grid, "Ux", comp[0]);
        add_scalar(grid, "Uy", comp[1]);
        add_scalar(grid, "Uz", comp[2]);
    }
    add_scalar(grid, "p", fields.p);
    add_scalar(grid, "k", fields.k);
    add_scalar(grid, "omega", fields.omega);
    add_scalar(grid, "nut", fields.nut);
    add_scalar(grid, "yPlus", fields.y_plus);
    grid.Modified();
}

std::vector<std::string> scalar_field_names(const FieldSet& fields) {
    std::vector<std::string> names;
    if (!fields.u.empty()) {
        names.insert(names.end(), {"|U|", "Ux", "Uy", "Uz"});
    }
    if (!fields.p.empty()) {
        names.emplace_back("p");
    }
    if (!fields.k.empty()) {
        names.emplace_back("k");
    }
    if (!fields.omega.empty()) {
        names.emplace_back("omega");
    }
    if (!fields.nut.empty()) {
        names.emplace_back("nut");
    }
    if (!fields.y_plus.empty()) {
        names.emplace_back("yPlus");
    }
    return names;
}

} // namespace cfd::gui
