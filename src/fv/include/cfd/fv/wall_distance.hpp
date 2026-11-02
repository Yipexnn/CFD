#pragma once

#include <cfd/fv/fv_geometry.hpp>
#include <vector>

namespace cfd::fv {

// Distance from each cell centre to the nearest wall face.
//
// Uses the Poisson method (solve lap(phi) = -1 with phi = 0 on walls,
// y = sqrt(|grad phi|^2 + 2 phi) - |grad phi|) and the exact normal distance for cells
// adjacent to a wall. is_wall_face is indexed by boundary face (f - n_internal_faces).
// Returns a large value everywhere if there are no walls.
[[nodiscard]] std::vector<Scalar> wall_distance(const FvGeometry& geo,
                                                const std::vector<char>& is_wall_face);

} // namespace cfd::fv
