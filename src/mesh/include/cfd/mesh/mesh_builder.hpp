#pragma once

#include <array>
#include <cfd/mesh/cell_shapes.hpp>
#include <cfd/mesh/mesh.hpp>
#include <span>
#include <string>
#include <vector>

namespace cfd::mesh {

// Builds a face-based Mesh from cell-to-node connectivity.
//
// Shared faces are matched automatically. Boundary faces are assigned to patches via
// add_boundary_face(); unassigned boundary faces go to the default patch.
class MeshBuilder {
public:
    explicit MeshBuilder(std::vector<Vec3> points);

    // Adds a cell; returns its index. nodes.size() must equal n_nodes(shape).
    Label add_cell(CellShape shape, std::span<const Label> nodes);

    // Declares a patch (creation order defines patch order). Returns its index.
    Label add_patch(const std::string& name, PatchType type = PatchType::Patch);

    // Tags a boundary face (3 or 4 nodes, any orientation) with a patch (created if needed).
    void add_boundary_face(std::span<const Label> nodes, const std::string& patch);

    [[nodiscard]] Label n_cells() const { return static_cast<Label>(cell_shapes_.size()); }

    // Assembles the mesh. Throws std::runtime_error on non-manifold faces or on tagged
    // faces that are not part of the mesh.
    [[nodiscard]] Mesh build(const std::string& default_patch = "defaultFaces") const;

private:
    struct PatchDef {
        std::string name;
        PatchType type = PatchType::Patch;
    };
    struct TaggedFace {
        std::array<Label, 4> nodes{};
        int n_nodes = 0;
        Label patch = 0;
    };

    std::vector<Vec3> points_;
    std::vector<CellShape> cell_shapes_;
    std::vector<Label> cell_node_offsets_{0};
    std::vector<Label> cell_nodes_;
    std::vector<PatchDef> patches_;
    std::vector<TaggedFace> tagged_faces_;
};

} // namespace cfd::mesh
