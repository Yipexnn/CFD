#include <array>
#include <cfd/core/types.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/mesh/mesh.hpp>
#include <cfd/mesh/quality.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cfd::mesh;
using cfd::core::Scalar;
using cfd::core::Vec3;

namespace {

constexpr Scalar k_tol = 1e-12;

template <typename T> const T& at(const std::vector<T>& v, Label i) {
    return v[static_cast<std::size_t>(i)];
}

// 3 x 4 x 5 cells over the box [0,1] x [0,2] x [0,3]: dx = 1/3, dy = 1/2, dz = 3/5.
BoxSpec unit_spec() {
    BoxSpec spec;
    spec.nx = 3;
    spec.ny = 4;
    spec.nz = 5;
    spec.min = Vec3(0.0, 0.0, 0.0);
    spec.max = Vec3(1.0, 2.0, 3.0);
    return spec;
}

// Checks that every face of the named patch has an area vector pointing along `sign * e_axis`
// with the given magnitude and no off-axis component.
void expect_outward(const Mesh& mesh, const std::string& name, int axis, Scalar sign, Scalar area) {
    const Patch* patch = mesh.find_patch(name);
    ASSERT_NE(patch, nullptr) << "missing patch " << name;
    const Label end = patch->start_face + patch->n_faces;
    for (Label f = patch->start_face; f < end; ++f) {
        const Vec3& sf = at(mesh.face_area_vectors(), f);
        for (int d = 0; d < 3; ++d) {
            if (d == axis) {
                EXPECT_GT(sign * sf[d], 0.0) << "patch " << name << " face " << f;
            } else {
                EXPECT_NEAR(sf[d], 0.0, k_tol) << "patch " << name << " face " << f;
            }
        }
        EXPECT_NEAR(sf.norm(), area, k_tol) << "patch " << name << " face " << f;
    }
}

} // namespace

TEST(BoxMesher, Counts) {
    const Mesh mesh = make_box_mesh(unit_spec());

    EXPECT_EQ(mesh.n_cells(), 60);
    EXPECT_EQ(mesh.n_internal_faces(), 133);
    EXPECT_EQ(mesh.n_boundary_faces(), 94);
    EXPECT_EQ(mesh.n_points(), 4 * 5 * 6);

    const std::array<std::string, 6> names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    const std::array<Label, 6> counts{20, 20, 15, 15, 12, 12};
    ASSERT_EQ(mesh.patches().size(), names.size());

    Label expected_start = mesh.n_internal_faces();
    for (std::size_t i = 0; i < names.size(); ++i) {
        const Patch& patch = mesh.patches()[i];
        EXPECT_EQ(patch.name, names[i]);
        EXPECT_EQ(patch.n_faces, counts[i]) << "patch " << names[i];
        EXPECT_EQ(patch.type, PatchType::Wall) << "patch " << names[i];
        EXPECT_EQ(patch.start_face, expected_start) << "patch " << names[i];
        expected_start += patch.n_faces;
    }
}

TEST(BoxMesher, Geometry) {
    const Mesh mesh = make_box_mesh(unit_spec());

    constexpr Scalar cell_volume = 1.0 / 3.0 * 0.5 * 0.6;
    Scalar total = 0.0;
    for (const Scalar v : mesh.cell_volumes()) {
        EXPECT_NEAR(v, 0.1, k_tol);
        EXPECT_NEAR(v, cell_volume, k_tol);
        total += v;
    }
    EXPECT_NEAR(total, 6.0, k_tol);

    for (Label f = 0; f < mesh.n_internal_faces(); ++f) {
        EXPECT_LT(at(mesh.owner(), f), at(mesh.neighbour(), f)) << "internal face " << f;
    }

    // Boundary area vectors point out of the box. Face areas: dy*dz, dx*dz, dx*dy.
    const Scalar dx = 1.0 / 3.0;
    const Scalar dy = 2.0 / 4.0;
    const Scalar dz = 3.0 / 5.0;
    expect_outward(mesh, "xmin", 0, -1.0, dy * dz);
    expect_outward(mesh, "xmax", 0, +1.0, dy * dz);
    expect_outward(mesh, "ymin", 1, -1.0, dx * dz);
    expect_outward(mesh, "ymax", 1, +1.0, dx * dz);
    expect_outward(mesh, "zmin", 2, -1.0, dx * dy);
    expect_outward(mesh, "zmax", 2, +1.0, dx * dy);
}

TEST(BoxMesher, CellClosure) {
    const Mesh mesh = make_box_mesh(unit_spec());
    const std::vector<Label>& offsets = mesh.cell_face_offsets();
    const std::vector<Label>& faces = mesh.cell_faces();

    for (Label c = 0; c < mesh.n_cells(); ++c) {
        Vec3 sum = Vec3::Zero();
        const Label begin = at(offsets, c);
        const Label end = at(offsets, c + 1);
        for (Label k = begin; k < end; ++k) {
            const Label f = at(faces, k);
            const Scalar sign = (at(mesh.owner(), f) == c) ? 1.0 : -1.0;
            sum += sign * at(mesh.face_area_vectors(), f);
        }
        EXPECT_LT(sum.norm(), k_tol) << "cell " << c;
    }
}

TEST(BoxMesher, Quality) {
    const Mesh mesh = make_box_mesh(unit_spec());
    const MeshQuality q = compute_quality(mesh);

    EXPECT_LT(q.max_non_orthogonality, 1e-6);
    EXPECT_LT(q.max_skewness, 1e-9);
    EXPECT_NEAR(q.max_aspect_ratio, 0.6 / (1.0 / 3.0), 1e-9);
    EXPECT_EQ(q.n_negative_volumes, 0);
    EXPECT_TRUE(q.ok());
    EXPECT_NEAR(q.total_volume, 6.0, k_tol);
}

TEST(BoxMesher, CustomPatchNameAndType) {
    BoxSpec spec = unit_spec();
    spec.patch_names[0] = "inlet";
    spec.patch_types[0] = PatchType::Inlet;
    const Mesh mesh = make_box_mesh(spec);

    const Patch* inlet = mesh.find_patch("inlet");
    ASSERT_NE(inlet, nullptr);
    EXPECT_EQ(inlet->type, PatchType::Inlet);
    EXPECT_EQ(inlet->n_faces, 20);
    EXPECT_EQ(mesh.find_patch("xmin"), nullptr);
}

TEST(BoxMesher, InvalidSpecThrows) {
    BoxSpec zero_nx = unit_spec();
    zero_nx.nx = 0;
    EXPECT_THROW(static_cast<void>(make_box_mesh(zero_nx)), std::invalid_argument);

    BoxSpec flat = unit_spec();
    flat.max.x() = flat.min.x();
    EXPECT_THROW(static_cast<void>(make_box_mesh(flat)), std::invalid_argument);
}
