#pragma once

#include <array>
#include <cfd/core/types.hpp>
#include <span>
#include <string_view>

namespace cfd::mesh {

// First-order cell shapes. Node ordering follows Gmsh/VTK conventions.
enum class CellShape { Tet, Pyramid, Prism, Hex };

// Local face of a cell shape: node indices into the cell's node list, ordered so
// that the right-hand normal points out of the cell. n_nodes is 3 or 4.
struct LocalFace {
    std::array<int, 4> nodes{};
    int n_nodes = 0;
};

namespace detail {

inline constexpr std::array<LocalFace, 4> k_tet_faces{{
    {{0, 2, 1, 0}, 3},
    {{0, 1, 3, 0}, 3},
    {{0, 3, 2, 0}, 3},
    {{1, 2, 3, 0}, 3},
}};

inline constexpr std::array<LocalFace, 5> k_pyramid_faces{{
    {{0, 3, 2, 1}, 4},
    {{0, 1, 4, 0}, 3},
    {{1, 2, 4, 0}, 3},
    {{2, 3, 4, 0}, 3},
    {{3, 0, 4, 0}, 3},
}};

inline constexpr std::array<LocalFace, 5> k_prism_faces{{
    {{0, 2, 1, 0}, 3},
    {{3, 4, 5, 0}, 3},
    {{0, 1, 4, 3}, 4},
    {{1, 2, 5, 4}, 4},
    {{0, 3, 5, 2}, 4},
}};

inline constexpr std::array<LocalFace, 6> k_hex_faces{{
    {{0, 3, 2, 1}, 4},
    {{4, 5, 6, 7}, 4},
    {{0, 1, 5, 4}, 4},
    {{1, 2, 6, 5}, 4},
    {{2, 3, 7, 6}, 4},
    {{3, 0, 4, 7}, 4},
}};

} // namespace detail

[[nodiscard]] constexpr int n_nodes(CellShape shape) {
    switch (shape) {
    case CellShape::Tet:
        return 4;
    case CellShape::Pyramid:
        return 5;
    case CellShape::Prism:
        return 6;
    case CellShape::Hex:
        break;
    }
    return 8;
}

[[nodiscard]] constexpr std::span<const LocalFace> local_faces(CellShape shape) {
    switch (shape) {
    case CellShape::Tet:
        return detail::k_tet_faces;
    case CellShape::Pyramid:
        return detail::k_pyramid_faces;
    case CellShape::Prism:
        return detail::k_prism_faces;
    case CellShape::Hex:
        break;
    }
    return detail::k_hex_faces;
}

[[nodiscard]] constexpr std::string_view to_string(CellShape shape) {
    switch (shape) {
    case CellShape::Tet:
        return "tet";
    case CellShape::Pyramid:
        return "pyramid";
    case CellShape::Prism:
        return "prism";
    case CellShape::Hex:
        break;
    }
    return "hex";
}

} // namespace cfd::mesh
