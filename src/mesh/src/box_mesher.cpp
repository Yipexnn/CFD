#include <array>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/mesh/cell_shapes.hpp>
#include <cfd/mesh/mesh_builder.hpp>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace cfd::mesh {

namespace {

void validate(const BoxSpec& spec) {
    if (spec.nx < 1 || spec.ny < 1 || spec.nz < 1) {
        throw std::invalid_argument("make_box_mesh: cell counts nx, ny, nz must be >= 1");
    }
    const Vec3 extent = spec.max - spec.min;
    // Written as "not > 0" so that NaN extents are rejected as well.
    if (!(extent.x() > 0.0 && extent.y() > 0.0 && extent.z() > 0.0)) {
        throw std::invalid_argument(
            "make_box_mesh: max must be greater than min in every direction");
    }
}

// Point index for lattice position (i, j, k), i fastest.
Label point_index(const BoxSpec& spec, Label i, Label j, Label k) {
    return i + (spec.nx + 1) * (j + (spec.ny + 1) * k);
}

std::vector<Vec3> make_points(const BoxSpec& spec) {
    const Vec3 extent = spec.max - spec.min;
    const auto n_points = static_cast<std::size_t>(spec.nx + 1) *
                          static_cast<std::size_t>(spec.ny + 1) *
                          static_cast<std::size_t>(spec.nz + 1);

    std::vector<Vec3> points;
    points.reserve(n_points);
    for (Label k = 0; k <= spec.nz; ++k) {
        for (Label j = 0; j <= spec.ny; ++j) {
            for (Label i = 0; i <= spec.nx; ++i) {
                const Vec3 frac(static_cast<Scalar>(i) / static_cast<Scalar>(spec.nx),
                                static_cast<Scalar>(j) / static_cast<Scalar>(spec.ny),
                                static_cast<Scalar>(k) / static_cast<Scalar>(spec.nz));
                points.push_back(spec.min + extent.cwiseProduct(frac));
            }
        }
    }
    return points;
}

} // namespace

Mesh make_box_mesh(const BoxSpec& spec) {
    validate(spec);

    MeshBuilder builder(make_points(spec));

    constexpr std::size_t n_patches = 6;
    for (std::size_t p = 0; p < n_patches; ++p) {
        builder.add_patch(spec.patch_names[p], spec.patch_types[p]);
    }

    const auto pid = [&spec](Label i, Label j, Label k) { return point_index(spec, i, j, k); };

    // Cells, k outermost and i innermost.
    for (Label k = 0; k < spec.nz; ++k) {
        for (Label j = 0; j < spec.ny; ++j) {
            for (Label i = 0; i < spec.nx; ++i) {
                const std::array<Label, 8> nodes{
                    pid(i, j, k),
                    pid(i + 1, j, k),
                    pid(i + 1, j + 1, k),
                    pid(i, j + 1, k),
                    pid(i, j, k + 1),
                    pid(i + 1, j, k + 1),
                    pid(i + 1, j + 1, k + 1),
                    pid(i, j + 1, k + 1),
                };
                builder.add_cell(CellShape::Hex, std::span<const Label>(nodes));
            }
        }
    }

    // Boundary faces. Patch index is 2 * direction + (0 for min side, 1 for max side).
    const auto tag = [&builder, &spec](const std::array<Label, 4>& quad, std::size_t patch) {
        builder.add_boundary_face(std::span<const Label>(quad), spec.patch_names[patch]);
    };

    // x = min (i = 0) and x = max (i = nx).
    for (const Label i : {Label{0}, spec.nx}) {
        const std::size_t patch = (i == 0) ? 0 : 1;
        for (Label k = 0; k < spec.nz; ++k) {
            for (Label j = 0; j < spec.ny; ++j) {
                tag({pid(i, j, k), pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i, j, k + 1)},
                    patch);
            }
        }
    }

    // y = min (j = 0) and y = max (j = ny).
    for (const Label j : {Label{0}, spec.ny}) {
        const std::size_t patch = (j == 0) ? 2 : 3;
        for (Label k = 0; k < spec.nz; ++k) {
            for (Label i = 0; i < spec.nx; ++i) {
                tag({pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j, k + 1), pid(i, j, k + 1)},
                    patch);
            }
        }
    }

    // z = min (k = 0) and z = max (k = nz).
    for (const Label k : {Label{0}, spec.nz}) {
        const std::size_t patch = (k == 0) ? 4 : 5;
        for (Label j = 0; j < spec.ny; ++j) {
            for (Label i = 0; i < spec.nx; ++i) {
                tag({pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)},
                    patch);
            }
        }
    }

    return builder.build();
}

} // namespace cfd::mesh
