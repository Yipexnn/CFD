#pragma once

#include <cfd/mesh/mesh.hpp>
#include <filesystem>
#include <istream>
#include <string>

namespace cfd::mesh {

// Reads a Gmsh MSH 4.1 ASCII mesh.
//
// 3D elements (tet, hex, prism, pyramid; first order) become cells. 2D elements on
// surfaces belonging to a physical group become boundary patches named after the group;
// the patch type is inferred from the name prefix (inlet, outlet, wall, sym*). Remaining
// boundary faces go to the "defaultFaces" wall patch.
// Throws std::runtime_error on unsupported or malformed input.
[[nodiscard]] Mesh read_gmsh(std::istream& in);
[[nodiscard]] Mesh read_gmsh(const std::filesystem::path& file);

// Patch type from a patch name prefix (case-insensitive).
[[nodiscard]] PatchType infer_patch_type(const std::string& name);

} // namespace cfd::mesh
