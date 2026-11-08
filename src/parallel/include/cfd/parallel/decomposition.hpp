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
// Same for a point set.
[[nodiscard]] std::vector<int> partition_points(const std::vector<core::Vec3>& points, int n_parts);

// Local mesh of one part: owned cells (ascending global index) followed by one layer of
// ghost cells (face neighbours owned by other parts). All global patches are kept in order
// (possibly empty); the halo patch is appended last.
struct LocalMesh {
    mesh::Mesh mesh;
    Halo halo;
    std::vector<Label> global_cell; // local cell -> global cell
    std::vector<Label> global_face; // local face -> global face (-1: halo face)
    std::vector<char> face_flipped; // 1: local orientation opposite to the global face
};

// Builds the local mesh of `part`. comm is stored in the halo (its rank need not equal
// part, which allows building all parts in one process for testing).
[[nodiscard]] LocalMesh decompose(const mesh::Mesh& global, const std::vector<int>& parts, int part,
                                  const Comm& comm = Comm::world());

// Serialised local mesh (for sending to another process) and its inverse; the halo uses comm.
[[nodiscard]] std::vector<char> pack_local_mesh(const LocalMesh& lm);
[[nodiscard]] LocalMesh unpack_local_mesh(const std::vector<char>& bytes, const Comm& comm);

// Parallel mesh distribution: only the root process holds (reads / generates) the global
// mesh; it partitions it (RCB), builds every local mesh and sends it to its process. Other
// processes pass nullptr and receive only their part. Collective.
[[nodiscard]] LocalMesh distribute(const mesh::Mesh* global, const Comm& comm = Comm::world());

// Cells of one zone of a (possibly decomposed) mesh as a mesh of their own, e.g. the fluid
// region of a conjugate heat transfer case. Faces towards cells of another zone form one
// patch per zone, named after it (type wall), appended after the parent patches. Cell and
// face order follow the parent, so owned cells stay first.
struct SubMesh {
    mesh::Mesh mesh;
    std::vector<Label> parent_cell; // sub cell -> parent cell
    std::vector<Label> parent_face; // sub face -> parent face
    std::vector<char> face_flipped; // 1: orientation opposite to the parent face
    Halo halo;                      // parent halo restricted to the zone (if given)
};

// zone_of_cell / zone_names describe the parent cells; parent_halo may be nullptr.
[[nodiscard]] SubMesh extract_zone(const mesh::Mesh& parent, const std::vector<int>& zone_of_cell,
                                   const std::vector<std::string>& zone_names, int zone,
                                   const Halo* parent_halo);

} // namespace cfd::parallel
