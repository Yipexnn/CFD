#include <algorithm>
#include <cfd/core/types.hpp>
#include <cfd/fv/fv_geometry.hpp>
#include <cfd/fv/scalar_transport.hpp>
#include <cfd/fv/wall_distance.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/mesh/mesh.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using cfd::core::Label;
using cfd::core::Scalar;
using cfd::core::Vec3;
using cfd::fv::FvGeometry;
using cfd::fv::LinearSolverSettings;
using cfd::fv::LinearSolverType;
using cfd::fv::ScalarBoundary;
using cfd::fv::ScalarEquation;
using cfd::fv::solve_scalar;
using cfd::fv::wall_distance;
using cfd::mesh::BoxSpec;
using cfd::mesh::make_box_mesh;
using cfd::mesh::Mesh;

namespace {

constexpr Label k_ny = 40;

// Unit box with a single cell in x and z, ny cells in y.
BoxSpec slab_spec(Label ny) {
    BoxSpec spec;
    spec.nx = 1;
    spec.ny = ny;
    spec.nz = 1;
    spec.min = Vec3(0.0, 0.0, 0.0);
    spec.max = Vec3(1.0, 1.0, 1.0);
    return spec;
}

// Name of the patch owning boundary face b (b = f - n_internal_faces).
std::string patch_name(const Mesh& msh, const FvGeometry& geo, std::size_t b) {
    const auto p = static_cast<std::size_t>(geo.boundary_patch()[b]);
    return msh.patches()[p].name;
}

// Wall flags: 1 on the ymin and ymax patches, 0 elsewhere.
std::vector<char> y_wall_flags(const Mesh& msh, const FvGeometry& geo) {
    std::vector<char> flags(static_cast<std::size_t>(msh.n_boundary_faces()), 0);
    for (std::size_t b = 0; b < flags.size(); ++b) {
        const std::string name = patch_name(msh, geo, b);
        flags[b] = (name == "ymin" || name == "ymax") ? 1 : 0;
    }
    return flags;
}

ScalarBoundary fixed(Scalar value) {
    return ScalarBoundary{ScalarBoundary::Kind::FixedValue, value};
}

ScalarBoundary zero_gradient() {
    return ScalarBoundary{ScalarBoundary::Kind::ZeroGradient, 0.0};
}

// Per-boundary-face BCs: ymin_bc on ymin faces, ymax_bc on ymax faces, zero gradient elsewhere.
std::vector<ScalarBoundary> y_boundaries(const Mesh& msh, const FvGeometry& geo,
                                         ScalarBoundary ymin_bc, ScalarBoundary ymax_bc) {
    std::vector<ScalarBoundary> bcs(static_cast<std::size_t>(msh.n_boundary_faces()),
                                    zero_gradient());
    for (std::size_t b = 0; b < bcs.size(); ++b) {
        const std::string name = patch_name(msh, geo, b);
        if (name == "ymin") {
            bcs[b] = ymin_bc;
        } else if (name == "ymax") {
            bcs[b] = ymax_bc;
        }
    }
    return bcs;
}

LinearSolverSettings solver_settings() {
    return LinearSolverSettings{LinearSolverType::BiCgStab, 1e-14, 0.0, 2000};
}

} // namespace

TEST(BoxMesher, BoxGrading) {
    BoxSpec spec;
    spec.nx = 1;
    spec.ny = 10;
    spec.nz = 1;
    spec.grading = Vec3(1.0, 4.0, 1.0);
    const Mesh msh = make_box_mesh(spec);
    const auto n = static_cast<std::size_t>(msh.n_cells());
    ASSERT_EQ(n, 10U);

    // Cell heights: volume / (dx * dz), with dx = dz = 1 for the unit box.
    std::vector<Scalar> h(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        h[j] = msh.cell_volumes()[j];
    }
    for (std::size_t j = 1; j < n; ++j) {
        EXPECT_GT(h[j], h[j - 1]);
    }
    EXPECT_NEAR(h.back() / h.front(), 4.0, 1e-9);
    const Scalar total = std::accumulate(h.begin(), h.end(), 0.0);
    EXPECT_NEAR(total, 1.0, 1e-9);

    BoxSpec bad = spec;
    bad.grading = Vec3(1.0, 0.0, 1.0);
    EXPECT_THROW(static_cast<void>(make_box_mesh(bad)), std::invalid_argument);
}

TEST(WallDistance, WallDistance1D) {
    const Mesh msh = make_box_mesh(slab_spec(k_ny));
    const FvGeometry geo(msh);
    const auto n = static_cast<std::size_t>(msh.n_cells());
    const std::vector<char> flags = y_wall_flags(msh, geo);
    const std::vector<Scalar> y = wall_distance(geo, flags);
    ASSERT_EQ(y.size(), n);

    for (std::size_t c = 0; c < n; ++c) {
        const Scalar cy = msh.cell_centres()[c].y();
        const Scalar exact = std::min(cy, 1.0 - cy);
        if (c == 0 || c == n - 1) {
            EXPECT_NEAR(y[c], exact, 1e-12);
        }
        EXPECT_NEAR(y[c], exact, 0.02 * exact);
    }
}

TEST(WallDistance, WallDistanceNoWalls) {
    const Mesh msh = make_box_mesh(slab_spec(k_ny));
    const FvGeometry geo(msh);
    const auto n = static_cast<std::size_t>(msh.n_cells());
    const std::vector<char> flags(static_cast<std::size_t>(msh.n_boundary_faces()), 0);
    const std::vector<Scalar> y = wall_distance(geo, flags);
    ASSERT_EQ(y.size(), n);
    for (std::size_t c = 0; c < n; ++c) {
        EXPECT_GT(y[c], 1e10);
    }
}

TEST(ScalarTransport, PureDiffusion1D) {
    const Mesh msh = make_box_mesh(slab_spec(k_ny));
    const FvGeometry geo(msh);
    const auto n_cells = static_cast<std::size_t>(msh.n_cells());
    const auto n_faces = static_cast<std::size_t>(msh.n_faces());

    ScalarEquation eq;
    eq.gamma_face.assign(n_faces, 1.0);
    eq.su.assign(n_cells, 0.0);
    eq.sp.assign(n_cells, 0.0);
    eq.boundary = y_boundaries(msh, geo, fixed(0.0), fixed(1.0));
    eq.linear_upwind = false;

    const std::vector<Scalar> face_flux(n_faces, 0.0);
    std::vector<Scalar> phi(n_cells, 0.0);
    static_cast<void>(solve_scalar(geo, face_flux, eq, phi, solver_settings()));

    for (std::size_t c = 0; c < n_cells; ++c) {
        EXPECT_NEAR(phi[c], msh.cell_centres()[c].y(), 1e-8);
    }
}

TEST(ScalarTransport, SourceTerm1D) {
    const Mesh msh = make_box_mesh(slab_spec(k_ny));
    const FvGeometry geo(msh);
    const auto n_cells = static_cast<std::size_t>(msh.n_cells());
    const auto n_faces = static_cast<std::size_t>(msh.n_faces());

    ScalarEquation eq;
    eq.gamma_face.assign(n_faces, 1.0);
    eq.su.assign(n_cells, 2.0);
    eq.sp.assign(n_cells, 0.0);
    eq.boundary = y_boundaries(msh, geo, fixed(0.0), fixed(0.0));
    eq.linear_upwind = false;

    const std::vector<Scalar> face_flux(n_faces, 0.0);
    std::vector<Scalar> phi(n_cells, 0.0);
    static_cast<void>(solve_scalar(geo, face_flux, eq, phi, solver_settings()));

    // Exact solution of -phi'' = 2 with phi(0) = phi(1) = 0 is y (1 - y).
    for (std::size_t c = 0; c < n_cells; ++c) {
        const Scalar cy = msh.cell_centres()[c].y();
        EXPECT_NEAR(phi[c], cy * (1.0 - cy), 1e-3);
    }
}

TEST(ScalarTransport, ConvectionUpwindBounded) {
    const Mesh msh = make_box_mesh(slab_spec(k_ny));
    const FvGeometry geo(msh);
    const auto n_cells = static_cast<std::size_t>(msh.n_cells());
    const auto n_faces = static_cast<std::size_t>(msh.n_faces());

    // Uniform velocity (0, 0.5, 0): flux through each face is 0.5 * Sf_y.
    std::vector<Scalar> face_flux(n_faces, 0.0);
    for (std::size_t f = 0; f < n_faces; ++f) {
        face_flux[f] = 0.5 * msh.face_area_vectors()[f].y();
    }

    ScalarEquation eq;
    eq.gamma_face.assign(n_faces, 1e-3);
    eq.su.assign(n_cells, 0.0);
    eq.sp.assign(n_cells, 0.0);
    eq.boundary = y_boundaries(msh, geo, fixed(1.0), zero_gradient());
    eq.linear_upwind = true;

    std::vector<Scalar> phi(n_cells, 0.0);
    // Linear-upwind is a deferred (explicit) correction: iterate to convergence.
    for (int outer = 0; outer < 50; ++outer) {
        static_cast<void>(solve_scalar(geo, face_flux, eq, phi, solver_settings()));
    }

    for (std::size_t c = 0; c < n_cells; ++c) {
        EXPECT_GE(phi[c], 0.0);
        EXPECT_LE(phi[c], 1.0 + 1e-9);
        EXPECT_GT(phi[c], 0.99);
    }
}
