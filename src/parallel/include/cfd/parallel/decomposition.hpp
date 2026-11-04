#pragma once

#include <cfd/mesh/mesh.hpp>
#include <cfd/parallel/halo.hpp>
#include <string>
#include <vector>

namespace cfd::parallel {

// Name of the patch formed by the outer faces of the ghost-cell layer.
inline const std::string k_halo_patch = "_halo";

// Cell -> part assignment by recursive coordinate bisection of the cell centres
// (balanced cell counts, deterministic).
[[nodiscard]] std::vector<int> partition_rcb(const mesh::Mesh& mesh, int n_parts);

// Local mesh of one part: owned cells (ascending global index) followed by one layer of
// ghost cells (face neighbours owned by other parts). All global patches are kept in order
// (possibly empty); the halo patch is appended last.
struct LocalMesh {
    mesh::Mesh mesh;
    Halo halo;
    std::vector<Label> global_cell; // local cell -> global cell
    std::vector<Label> global_face; // local face -> global face (-1: halo face)
};

// Builds the local mesh of `part`. comm is stored in the halo (its rank need not equal
// part, which allows building all parts in one process for testing).
[[nodiscard]] LocalMesh decompose(const mesh::Mesh& global, const std::vector<int>& parts, int part,
                                  const Comm& comm = Comm::world());

} // namespace cfd::parallel
