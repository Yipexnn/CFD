# Phase 1 – Mesh core

Goal: unstructured 3D finite-volume mesh with geometry, a built-in box mesher, Gmsh `.msh` import and quality metrics.

## Design

New module `cfd::mesh` (`src/mesh`, static library `cfd_mesh`, alias `cfd::mesh`, depends on `cfd::core`).

**Face-based polyhedral mesh** (OpenFOAM-style):
- `points`: `std::vector<Vec3>`
- faces: CSR storage (`face_point_offsets`, `face_points`); points ordered so that the right-hand normal points from owner to neighbour (boundary faces: outward)
- `owner[f]` for all faces, `neighbour[f]` for internal faces only
- internal faces first, sorted by (owner, neighbour) → upper-triangular order
- boundary faces grouped in contiguous `Patch { name, type, start_face, n_faces }`
- geometry (computed): `face_centres`, `face_area_vectors` (Sf, |Sf| = area), `cell_centres`, `cell_volumes`

**Geometry algorithms**
- Face: triangle fan around the point average → area-weighted centre, sum of triangle area vectors.
- Cell: pyramids from the estimated centre (average of face centres) to each face → volume-weighted centre, sum of volumes.

**Building** (`MeshBuilder`): input = points + cells as shape type + node lists (tet, pyramid, prism, hex) + boundary faces (node lists) tagged with patch names. Steps: generate cell faces with outward orientation, match shared faces through a sorted-node-key hash map, orient each internal face owner→neighbour (owner = lower cell index), sort, and attach boundary faces to patches. Unmatched faces without a patch go into a `defaultFaces` patch.

**Box mesher**: `make_box_mesh(BoxSpec)` with `nx, ny, nz`, `min`, `max` and patch names `xmin..zmax` (default names).

**Gmsh reader**: `.msh` 4.1 ASCII (`$MeshFormat`, `$PhysicalNames`, `$Entities`, `$Nodes`, `$Elements`). 3D elements (4 = tet, 5 = hex, 6 = prism, 7 = pyramid) become cells. 2D elements (2 = tri, 3 = quad) on surfaces that belong to a physical group become boundary patches. Clear error on unsupported format/version.

**Quality**: `MeshQuality` report: cell count, min/max/total volume, max/average non-orthogonality (degrees, angle between d = C_N − C_P and Sf), max skewness, max aspect ratio, number of negative-volume cells; `bool ok()`.

**CLI**: `cfd mesh-box nx ny nz` and `cfd check-mesh <file.msh>` print counts, patches and quality.

## Tasks
| # | File(s) | Who |
|---|---|---|
| 1 | `src/mesh/include/cfd/mesh/mesh.hpp`, `src/mesh/src/mesh.cpp` (data, accessors, geometry) | O |
| 2 | `cell_shapes.hpp` (local face definitions of tet/pyr/prism/hex, outward orientation) | O |
| 3 | `mesh_builder.hpp/.cpp` | O |
| 4 | `box_mesher.hpp/.cpp` | H |
| 5 | `gmsh_reader.hpp/.cpp` | O |
| 6 | `quality.hpp/.cpp` | H |
| 7 | `src/mesh/CMakeLists.txt`, root `add_subdirectory` | O |
| 8 | tests: geometry/builder (single tet, pyramid, prism, hex: volume, closure Σ Sf = 0, centres) | H |
| 9 | tests: box mesher (counts, total volume, patch sizes, orthogonality = 0) | H |
| 10 | tests: gmsh reader (embedded small `.msh` strings: hex cube, mixed tets) | O |
| 11 | CLI subcommands in `src/app/main.cpp` | H |

## Definition of Done
- [x] CI green (Windows, Linux, format)
- [x] All shapes: volume exact, every cell closed (|Σ ±Sf| < 1e-12·scale)
- [x] Box mesh: cell/face/patch counts exact, volume = box volume, non-orthogonality 0
- [x] Gmsh 4.1 files read with correct patches
- [x] `cfd check-mesh` works
