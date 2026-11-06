// Supersonic flow over a compression ramp (inviscid): Mach 2.5 onto a 15 degree ramp.
// The oblique shock angle beta and the pressure ratio p2/p1 behind it are compared with
// theta-beta-M theory (weak shock).
//
// Usage: supersonic_ramp [nx=120] [max_iterations=4000]
#include <array>
#include <cfd/fv/compressible.hpp>
#include <cfd/mesh/mesh_builder.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

int main(int argc, char** argv) {
    const int nx = argc > 1 ? std::atoi(argv[1]) : 120;
    const int max_iterations = argc > 2 ? std::atoi(argv[2]) : 4000;
    const int ny = nx / 2;
    const Scalar length = 2.0;
    const Scalar height = 1.0;
    const Scalar x0 = 0.5;
    const Scalar theta = 15.0 * std::numbers::pi / 180.0;
    const Scalar mach = 2.5;
    const Scalar gamma = 1.4;

    // Structured mesh following the ramp, one cell layer in z.
    std::vector<Vec3> points;
    for (int k = 0; k <= 1; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const Scalar x = length * i / nx;
                const Scalar yb = x > x0 ? (x - x0) * std::tan(theta) : 0.0;
                points.emplace_back(x, yb + (height - yb) * j / ny, 0.02 * k);
            }
        }
    }
    auto id = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    cfd::mesh::MeshBuilder b(points);
    b.add_patch("inlet", cfd::mesh::PatchType::Patch);
    b.add_patch("outlet", cfd::mesh::PatchType::Patch);
    b.add_patch("ramp", cfd::mesh::PatchType::Wall);
    b.add_patch("top", cfd::mesh::PatchType::Symmetry);
    b.add_patch("sides", cfd::mesh::PatchType::Patch);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const std::array<int, 8> cell{id(i, j, 0),         id(i + 1, j, 0), id(i + 1, j + 1, 0),
                                          id(i, j + 1, 0),     id(i, j, 1),     id(i + 1, j, 1),
                                          id(i + 1, j + 1, 1), id(i, j + 1, 1)};
            b.add_cell(cfd::mesh::CellShape::Hex, cell);
        }
    }
    for (int j = 0; j < ny; ++j) {
        b.add_boundary_face(
            std::array<int, 4>{id(0, j, 0), id(0, j + 1, 0), id(0, j + 1, 1), id(0, j, 1)},
            "inlet");
        b.add_boundary_face(
            std::array<int, 4>{id(nx, j, 0), id(nx, j + 1, 0), id(nx, j + 1, 1), id(nx, j, 1)},
            "outlet");
    }
    for (int i = 0; i < nx; ++i) {
        b.add_boundary_face(
            std::array<int, 4>{id(i, 0, 0), id(i + 1, 0, 0), id(i + 1, 0, 1), id(i, 0, 1)}, "ramp");
        b.add_boundary_face(
            std::array<int, 4>{id(i, ny, 0), id(i + 1, ny, 0), id(i + 1, ny, 1), id(i, ny, 1)},
            "top");
    }
    const auto mesh = b.build("sides");

    CompressibleSettings s;
    s.gamma = gamma;
    s.cfl = 0.8;
    const Scalar p1 = 1e5;
    const Scalar t1 = 300.0;
    const Scalar u1 = mach * std::sqrt(gamma * s.r_gas * t1);
    BoundaryCondition inlet{FlowBoundary::VelocityInlet, Vec3(u1, 0, 0), p1};
    inlet.temperature = t1;
    CompressibleSolver solver(mesh,
                              {{"inlet", inlet},
                               {"outlet", {FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0}},
                               {"sides", {FlowBoundary::Empty, Vec3::Zero(), 0.0}}},
                              s);
    solver.set_uniform({p1 / (s.r_gas * t1), Vec3(u1, 0, 0), p1});
    Scalar r0 = 0.0;
    int it = 0;
    for (; it < max_iterations; ++it) {
        const Scalar r = solver.steady_iteration();
        r0 = it == 0 ? r : r0;
        if (it % 500 == 0) {
            std::printf("iteration %d  density residual %.3e\n", it, r / r0);
        }
        if (r < 1e-7 * r0) {
            break;
        }
    }

    // Theory: weak oblique shock from tan(theta) = 2 cot(b) (M^2 sin^2 b - 1) / (M^2 (g + cos 2b) +
    // 2).
    auto deflection = [&](Scalar beta) {
        const Scalar m2 = mach * mach * std::sin(beta) * std::sin(beta);
        return std::atan(2.0 / std::tan(beta) * (m2 - 1.0) /
                         (mach * mach * (gamma + std::cos(2.0 * beta)) + 2.0));
    };
    Scalar lo = std::asin(1.0 / mach);
    Scalar hi = 64.0 * std::numbers::pi / 180.0; // below the maximum-deflection angle
    for (int k = 0; k < 100; ++k) {
        const Scalar mid = 0.5 * (lo + hi);
        (deflection(mid) < theta ? lo : hi) = mid;
    }
    const Scalar beta = 0.5 * (lo + hi);
    const Scalar ratio =
        1.0 + 2.0 * gamma / (gamma + 1.0) * (mach * mach * std::pow(std::sin(beta), 2) - 1.0);

    // Simulation: mean wall pressure on the ramp between x = 0.8 and 1.3 (behind the shock,
    // ahead of reflections), and the shock angle from the 50 % pressure rise at y = 0.4.
    Scalar sum = 0.0;
    int count = 0;
    const auto p = solver.pressure();
    for (int i = 0; i < nx; ++i) {
        const auto c = static_cast<std::size_t>(i); // first cell row
        const Scalar x = mesh.cell_centres()[c].x();
        if (x > 0.8 && x < 1.3) {
            sum += p[c];
            ++count;
        }
    }
    const Scalar sim_ratio = sum / count / p1;
    const int j = static_cast<int>(0.4 * ny);
    Scalar x_shock = 0.0;
    Scalar y_shock = 0.0;
    for (int i = 0; i + 1 < nx; ++i) {
        const auto c0 = static_cast<std::size_t>(i + j * nx);
        const auto c1 = static_cast<std::size_t>(i + 1 + j * nx);
        const Scalar mid = 0.5 * (1.0 + ratio) * p1;
        if (p[c0] < mid && p[c1] >= mid) {
            const Scalar f = (mid - p[c0]) / (p[c1] - p[c0]);
            const Vec3 xs =
                mesh.cell_centres()[c0] + f * (mesh.cell_centres()[c1] - mesh.cell_centres()[c0]);
            x_shock = xs.x();
            y_shock = xs.y();
            break;
        }
    }
    const Scalar sim_beta = std::atan2(y_shock, x_shock - x0);
    std::printf("cells %d, %d iterations\n", mesh.n_cells(), it);
    std::printf("p2/p1: simulation %.4f, theory %.4f\n", sim_ratio, ratio);
    std::printf("shock angle: simulation %.2f deg, theory %.2f deg\n",
                sim_beta * 180.0 / std::numbers::pi, beta * 180.0 / std::numbers::pi);
    return 0;
}
