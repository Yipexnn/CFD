#pragma once

#include <array>
#include <cfd/mesh/mesh.hpp>
#include <string>

namespace cfd::mesh {

struct BoxSpec {
    Label nx = 1;
    Label ny = 1;
    Label nz = 1;
    Vec3 min = Vec3::Zero();
    Vec3 max = Vec3::Ones();
    // Geometric grading per direction: size of the last cell / size of the first cell
    // (1 = uniform, > 1 = cells grow towards max).
    Vec3 grading = Vec3::Ones();
    // Patch order: xmin, xmax, ymin, ymax, zmin, zmax.
    std::array<std::string, 6> patch_names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    std::array<PatchType, 6> patch_types{PatchType::Wall, PatchType::Wall, PatchType::Wall,
                                         PatchType::Wall, PatchType::Wall, PatchType::Wall};
};

// Structured hex mesh of an axis-aligned box. Throws std::invalid_argument if any n < 1 or
// max <= min in any direction.
[[nodiscard]] Mesh make_box_mesh(const BoxSpec& spec);

} // namespace cfd::mesh
