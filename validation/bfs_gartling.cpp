// Laminar backward-facing step, Re = 800, expansion ratio 2 (Gartling 1990,
// Int. J. Numer. Meth. Fluids 11:953-967). Domain 0 <= x <= 30, -0.5 <= y <= 0.5 (H = 1),
// parabolic inflow u = 24 y (0.5 - y) on 0 <= y <= 0.5, step wall below.
// Reference: lower-wall reattachment x = 6.10, upper-wall eddy 4.85 .. 10.48.
//
// Usage: bfs_gartling [nx=300] [ny=40] [max_iterations=6000]
#include <array>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/mesh/mesh_builder.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using cfd::core::Vec3;

int main(int argc, char** argv) {
    const int nx = argc > 1 ? std::atoi(argv[1]) : 300;
    const int ny = argc > 2 ? std::atoi(argv[2]) : 40;
    const int max_iterations = argc > 3 ? std::atoi(argv[3]) : 6000;
    const double length = 30.0;

    std::vector<Vec3> points;
    for (int k = 0; k <= 1; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                points.emplace_back(length * i / nx, -0.5 + static_cast<double>(j) / ny, 0.05 * k);
            }
        }
    }
    auto id = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    cfd::mesh::MeshBuilder builder(points);
    builder.add_patch("inlet", cfd::mesh::PatchType::Inlet);
    builder.add_patch("outlet", cfd::mesh::PatchType::Outlet);
    builder.add_patch("walls", cfd::mesh::PatchType::Wall);
    builder.add_patch("sides", cfd::mesh::PatchType::Patch);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const std::array<int, 8> cell{id(i, j, 0),         id(i + 1, j, 0), id(i + 1, j + 1, 0),
                                          id(i, j + 1, 0),     id(i, j, 1),     id(i + 1, j, 1),
                                          id(i + 1, j + 1, 1), id(i, j + 1, 1)};
            builder.add_cell(cfd::mesh::CellShape::Hex, cell);
        }
    }
    for (int j = 0; j < ny; ++j) {
        const std::array<int, 4> left{id(0, j, 0), id(0, j + 1, 0), id(0, j + 1, 1), id(0, j, 1)};
        builder.add_boundary_face(left, j >= ny / 2 ? "inlet" : "walls");
        const std::array<int, 4> right{id(nx, j, 0), id(nx, j + 1, 0), id(nx, j + 1, 1),
                                       id(nx, j, 1)};
        builder.add_boundary_face(right, "outlet");
    }
    for (int i = 0; i < nx; ++i) {
        for (const int j : {0, ny}) {
            const std::array<int, 4> wall{id(i, j, 0), id(i + 1, j, 0), id(i + 1, j, 1),
                                          id(i, j, 1)};
            builder.add_boundary_face(wall, "walls");
        }
    }
    const auto mesh = builder.build("sides");

    cfd::fv::FlowSettings settings;
    settings.nu = 1.0 / 800.0;
    settings.convection = cfd::fv::ConvectionScheme::LimitedLinearUpwind;
    settings.alpha_u = 0.5;
    settings.alpha_p = 0.2;
    cfd::fv::FlowSolver flow(mesh,
                             {{"inlet", {cfd::fv::FlowBoundary::VelocityInlet, Vec3(1, 0, 0), 0.0}},
                              {"sides", {cfd::fv::FlowBoundary::Empty, Vec3::Zero(), 0.0}}},
                             settings);
    const auto* inlet = mesh.find_patch("inlet");
    std::vector<Vec3> profile;
    for (int q = 0; q < inlet->n_faces; ++q) {
        const double y = mesh.face_centres()[static_cast<std::size_t>(inlet->start_face + q)].y();
        profile.emplace_back(24.0 * y * (0.5 - y), 0.0, 0.0);
    }
    flow.set_boundary_velocity_profile("inlet", profile);

    const auto t0 = std::chrono::steady_clock::now();
    int it = 0;
    for (; it < max_iterations; ++it) {
        const auto r = flow.simple_iteration();
        if (it % 500 == 0) {
            std::printf("iteration %d  residual %.2e\n", it, r.max());
        }
        if (r.max() < 1e-6) {
            break;
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Zero crossings of u in the wall-adjacent cell rows (linear interpolation).
    auto crossings = [&](int row) {
        std::vector<double> x;
        for (int i = 0; i + 1 < nx; ++i) {
            const double a = flow.velocity()[static_cast<std::size_t>(i + row * nx)].x();
            const double b = flow.velocity()[static_cast<std::size_t>(i + 1 + row * nx)].x();
            if ((a < 0.0) != (b < 0.0)) {
                x.push_back(length * (i + 0.5 + a / (a - b)) / nx);
            }
        }
        return x;
    };
    std::printf("cells %d, iterations %d, %.1f s\n", mesh.n_cells(), it, seconds);
    std::printf("lower wall:");
    for (const double x : crossings(0)) {
        std::printf(" %.3f", x);
    }
    std::printf("   (Gartling: 6.10)\nupper wall:");
    for (const double x : crossings(ny - 1)) {
        std::printf(" %.3f", x);
    }
    std::printf("   (Gartling: 4.85, 10.48)\n");
    return 0;
}
