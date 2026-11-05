#pragma once

#include <cfd/core/types.hpp>
#include <span>
#include <string>
#include <vector>

namespace cfd::mesh {

using core::Label;
using core::Scalar;
using core::Vec3;

enum class PatchType { Wall, Inlet, Outlet, Symmetry, Patch };

// Contiguous range of boundary faces sharing a name and type.
struct Patch {
    std::string name;
    PatchType type = PatchType::Patch;
    Label start_face = 0;
    Label n_faces = 0;
};

// Face-based unstructured polyhedral finite-volume mesh.
//
// Faces [0, n_internal_faces) are internal and sorted by (owner, neighbour) with
// owner < neighbour. Boundary faces follow, grouped by patch. Face points are ordered
// so that the right-hand normal points from owner to neighbour (outward on boundaries).
class Mesh {
public:
    Mesh() = default;
    Mesh(std::vector<Vec3> points, std::vector<Label> face_point_offsets,
         std::vector<Label> face_points, std::vector<Label> owner, std::vector<Label> neighbour,
         std::vector<Patch> patches, Label n_cells);

    [[nodiscard]] Label n_points() const { return static_cast<Label>(points_.size()); }
    [[nodiscard]] Label n_faces() const { return static_cast<Label>(owner_.size()); }
    [[nodiscard]] Label n_internal_faces() const { return static_cast<Label>(neighbour_.size()); }
    [[nodiscard]] Label n_boundary_faces() const { return n_faces() - n_internal_faces(); }
    [[nodiscard]] Label n_cells() const { return n_cells_; }

    [[nodiscard]] const std::vector<Vec3>& points() const { return points_; }
    [[nodiscard]] std::span<const Label> face(Label f) const;
    [[nodiscard]] const std::vector<Label>& owner() const { return owner_; }
    [[nodiscard]] const std::vector<Label>& neighbour() const { return neighbour_; }
    [[nodiscard]] const std::vector<Patch>& patches() const { return patches_; }
    [[nodiscard]] const Patch* find_patch(const std::string& name) const;
    // Changes the type of a patch (geometry is unaffected). Throws if the name is unknown.
    void set_patch_type(const std::string& name, PatchType type);

    // Optional cell zones (e.g. Gmsh physical volumes): zone index per cell and zone names.
    // Empty if the mesh has no zones.
    void set_cell_zones(std::vector<std::string> names, std::vector<int> zone_of_cell);
    [[nodiscard]] const std::vector<std::string>& cell_zone_names() const { return zone_names_; }
    [[nodiscard]] const std::vector<int>& cell_zones() const { return cell_zones_; }

    // Geometry (computed in the constructor).
    [[nodiscard]] const std::vector<Vec3>& face_centres() const { return face_centres_; }
    [[nodiscard]] const std::vector<Vec3>& face_area_vectors() const { return face_area_vectors_; }
    [[nodiscard]] const std::vector<Vec3>& cell_centres() const { return cell_centres_; }
    [[nodiscard]] const std::vector<Scalar>& cell_volumes() const { return cell_volumes_; }

    // Faces of each cell (CSR), built on demand.
    [[nodiscard]] const std::vector<Label>& cell_face_offsets() const;
    [[nodiscard]] const std::vector<Label>& cell_faces() const;

private:
    void validate() const;
    void compute_geometry();
    void build_cell_faces() const;

    std::vector<Vec3> points_;
    std::vector<Label> face_point_offsets_{0};
    std::vector<Label> face_points_;
    std::vector<Label> owner_;
    std::vector<Label> neighbour_;
    std::vector<Patch> patches_;
    Label n_cells_ = 0;
    std::vector<std::string> zone_names_;
    std::vector<int> cell_zones_;

    std::vector<Vec3> face_centres_;
    std::vector<Vec3> face_area_vectors_;
    std::vector<Vec3> cell_centres_;
    std::vector<Scalar> cell_volumes_;

    mutable std::vector<Label> cell_face_offsets_;
    mutable std::vector<Label> cell_faces_;
};

[[nodiscard]] std::string to_string(PatchType type);
[[nodiscard]] PatchType patch_type_from_string(const std::string& name);

} // namespace cfd::mesh
