#include <algorithm>
#include <array>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/mesh/mesh_builder.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;
using cfd::mesh::BoxSpec;
using cfd::mesh::PatchType;

namespace {

// Ghia, Ghia & Shin (1982), Re = 100: u along the vertical centreline x = 0.5.
constexpr std::array<Scalar, 11> k_ghia_y{0.9766, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
                                          0.4531, 0.2813, 0.1719, 0.1016, 0.0547};
constexpr std::array<Scalar, 11> k_ghia_u{0.84123,  0.68717,  0.23151,  0.00332,
                                          -0.13641, -0.20581, -0.21090, -0.15662,
                                          -0.10150, -0.06434, -0.03717};

// 5 x 1 channel between plates (y = 0, 1), uniform inlet U = 1, quasi-2D (z symmetry).
constexpr int k_nx = 50;
constexpr int k_ny = 20;
constexpr Scalar k_channel_nu = 0.1;

cfd::mesh::Mesh channel_mesh() {
    BoxSpec spec;
    spec.nx = k_nx;
    spec.ny = k_ny;
    spec.nz = 1;
    spec.max = Vec3(5.0, 1.0, 0.05);
    spec.patch_types = {PatchType::Inlet, PatchType::Outlet,   PatchType::Wall,
                        PatchType::Wall,  PatchType::Symmetry, PatchType::Symmetry};
    return cfd::mesh::make_box_mesh(spec);
}

FlowSolver channel_solver(const cfd::mesh::Mesh& mesh) {
    FlowSettings settings;
    settings.nu = k_channel_nu;
    return FlowSolver(mesh, {{"xmin", {FlowBoundary::VelocityInlet, Vec3(1.0, 0.0, 0.0), 0.0}}},
                      settings);
}

// Checks the developed parabolic profile u = 6 U y (1 - y) and dp/dx = -12 nu U / h^2.
void expect_poiseuille(const FlowSolver& solver) {
    const auto& u = solver.velocity();
    const auto& p = solver.pressure();
    const int i = 45; // x = 4.55, fully developed
    Scalar max_err = 0.0;
    for (int j = 0; j < k_ny; ++j) {
        const Scalar y = (j + 0.5) / k_ny;
        const Scalar exact = 6.0 * y * (1.0 - y);
        max_err =
            std::max(max_err, std::abs(u[static_cast<std::size_t>(i + j * k_nx)].x() - exact));
    }
    EXPECT_LT(max_err / 1.5, 0.01);

    auto column_mean = [&](int col) {
        Scalar sum = 0.0;
        for (int j = 0; j < k_ny; ++j) {
            sum += p[static_cast<std::size_t>(col + j * k_nx)];
        }
        return sum / k_ny;
    };
    const Scalar dpdx = (column_mean(45) - column_mean(30)) / 1.5;
    const Scalar exact_dpdx = -12.0 * k_channel_nu;
    EXPECT_NEAR(dpdx / exact_dpdx, 1.0, 0.02);
}

} // namespace

TEST(FlowValidation, LidDrivenCavityRe100) {
    const int n = 32;
    BoxSpec spec;
    spec.nx = n;
    spec.ny = n;
    spec.nz = 1;
    spec.max = Vec3(1.0, 1.0, 1.0 / n);
    spec.patch_types[4] = PatchType::Symmetry;
    spec.patch_types[5] = PatchType::Symmetry;
    const auto mesh = cfd::mesh::make_box_mesh(spec);

    FlowSettings settings;
    settings.nu = 0.01;
    FlowSolver solver(mesh, {{"ymax", {FlowBoundary::Wall, Vec3(1.0, 0.0, 0.0), 0.0}}}, settings);
    const int iterations = solver.solve_steady(1e-6, 4000);
    EXPECT_LT(iterations, 4000);
    EXPECT_LT(solver.continuity_error(), 1e-6);

    // Centreline x = 0.5 lies between columns n/2 - 1 and n/2.
    const auto& u = solver.velocity();
    auto centre_u = [&](int row) {
        return 0.5 * (u[static_cast<std::size_t>(n / 2 - 1 + row * n)].x() +
                      u[static_cast<std::size_t>(n / 2 + row * n)].x());
    };
    Scalar max_err = 0.0;
    for (std::size_t k = 0; k < k_ghia_y.size(); ++k) {
        const Scalar pos = k_ghia_y[k] * n - 0.5;
        const int row = std::clamp(static_cast<int>(std::floor(pos)), 0, n - 2);
        const Scalar t = std::clamp(pos - row, 0.0, 1.0);
        const Scalar value = (1.0 - t) * centre_u(row) + t * centre_u(row + 1);
        max_err = std::max(max_err, std::abs(value - k_ghia_u[k]));
    }
    EXPECT_LT(max_err, 0.01);
}

TEST(FlowValidation, PoiseuilleChannelSimple) {
    const auto mesh = channel_mesh();
    FlowSolver solver = channel_solver(mesh);
    EXPECT_LT(solver.solve_steady(1e-7, 3000), 3000);
    EXPECT_LT(solver.continuity_error(), 1e-6);
    expect_poiseuille(solver);
}

TEST(FlowValidation, PoiseuilleChannelPisoReachesSteadyState) {
    const auto mesh = channel_mesh();
    FlowSolver solver = channel_solver(mesh);
    for (int step = 0; step < 400; ++step) {
        (void)solver.piso_step(0.05);
    }
    EXPECT_LT(solver.history().back().max(), 1e-6);
    expect_poiseuille(solver);
}

TEST(FlowValidation, PoiseuilleOnNonOrthogonalMesh) {
    // Channel mesh sheared by x += 0.5 y sin(pi x / L): inlet/outlet stay vertical, interior
    // cells are up to ~27 deg non-orthogonal.
    const int nx = k_nx;
    const int ny = k_ny;
    std::vector<Vec3> points;
    for (int k = 0; k <= 1; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const Scalar x = 5.0 * i / nx;
                const Scalar y = static_cast<Scalar>(j) / ny;
                points.emplace_back(x + 0.5 * y * std::sin(std::numbers::pi * x / 5.0), y,
                                    0.05 * k);
            }
        }
    }
    auto id = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    cfd::mesh::MeshBuilder builder(points);
    builder.add_patch("inlet", PatchType::Inlet);
    builder.add_patch("outlet", PatchType::Outlet);
    builder.add_patch("walls", PatchType::Wall);
    builder.add_patch("sides", PatchType::Symmetry);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const std::array<int, 8> cell{id(i, j, 0),         id(i + 1, j, 0), id(i + 1, j + 1, 0),
                                          id(i, j + 1, 0),     id(i, j, 1),     id(i + 1, j, 1),
                                          id(i + 1, j + 1, 1), id(i, j + 1, 1)};
            builder.add_cell(cfd::mesh::CellShape::Hex, cell);
        }
    }
    // Inlet/outlet/walls are tagged; the remaining z faces default to "sides" below.
    for (int j = 0; j < ny; ++j) {
        const std::array<int, 4> in{id(0, j, 0), id(0, j + 1, 0), id(0, j + 1, 1), id(0, j, 1)};
        const std::array<int, 4> out{id(nx, j, 0), id(nx, j + 1, 0), id(nx, j + 1, 1),
                                     id(nx, j, 1)};
        builder.add_boundary_face(in, "inlet");
        builder.add_boundary_face(out, "outlet");
    }
    for (int i = 0; i < nx; ++i) {
        for (const int j : {0, ny}) {
            const std::array<int, 4> wall{id(i, j, 0), id(i + 1, j, 0), id(i + 1, j, 1),
                                          id(i, j, 1)};
            builder.add_boundary_face(wall, "walls");
        }
    }
    const auto mesh = builder.build("sides");
    ASSERT_NE(mesh.find_patch("sides"), nullptr);

    FlowSettings settings;
    settings.nu = k_channel_nu;
    settings.n_non_orth_correctors = 1;
    BoundaryCondition sides{FlowBoundary::Symmetry, Vec3::Zero(), 0.0};
    FlowSolver solver(
        mesh,
        {{"inlet", {FlowBoundary::VelocityInlet, Vec3(1.0, 0.0, 0.0), 0.0}}, {"sides", sides}},
        settings);
    EXPECT_LT(solver.solve_steady(1e-7, 3000), 3000);

    // Mid-channel column (maximum shear), fully developed.
    Scalar max_err = 0.0;
    for (int j = 0; j < ny; ++j) {
        const auto c = static_cast<std::size_t>(25 + j * nx);
        const Scalar y = mesh.cell_centres()[c].y();
        max_err = std::max(max_err, std::abs(solver.velocity()[c].x() - 6.0 * y * (1.0 - y)));
    }
    EXPECT_LT(max_err / 1.5, 0.01);
}

TEST(FlowValidation, MassConservationPerCell) {
    const auto mesh = channel_mesh();
    FlowSolver solver = channel_solver(mesh);
    (void)solver.solve_steady(1e-7, 3000);

    // Net outflow equals inflow (1 m/s through a 1 x 0.05 inlet).
    Scalar inflow = 0.0;
    Scalar outflow = 0.0;
    const auto& phi = solver.face_flux();
    for (const auto& patch : mesh.patches()) {
        for (cfd::core::Label i = 0; i < patch.n_faces; ++i) {
            const Scalar f = phi[static_cast<std::size_t>(patch.start_face + i)];
            (patch.name == "xmin" ? inflow : outflow) += f;
        }
    }
    EXPECT_NEAR(inflow, -0.05, 1e-12);
    EXPECT_NEAR(outflow, 0.05, 1e-8);
}

TEST(FlowValidation, MissingInletConditionThrows) {
    const auto mesh = channel_mesh();
    EXPECT_THROW((void)FlowSolver(mesh, {}, FlowSettings{}), std::invalid_argument);
}

TEST(FlowValidation, UnknownPatchThrows) {
    const auto mesh = channel_mesh();
    EXPECT_THROW((void)FlowSolver(mesh,
                                  {{"xmin", {FlowBoundary::VelocityInlet, Vec3(1, 0, 0), 0.0}},
                                   {"nope", {FlowBoundary::Wall, Vec3::Zero(), 0.0}}},
                                  FlowSettings{}),
                 std::invalid_argument);
}
