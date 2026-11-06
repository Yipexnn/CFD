// Turbulent flat plate, zero pressure gradient (NASA TMR "2DZP" setup): Mach 0.2,
// Re = 5e6 per unit length, plate 0 <= x <= 2 behind a symmetry run-up from x = -0.33.
// Compressible k-omega SST with implicit LU-SGS; the skin friction is compared with the
// TMR reference value at x = 0.97 (Cf ~ 0.00271) and with turbulent-plate correlations.
//
// Usage: flat_plate_turbulent [nx_plate=64] [ny=64] [max_iterations=6000] [cfl=1000]
//        [turbulent|laminar] [krylov_dim=30 (0: plain LU-SGS)]
#include <algorithm>
#include <array>
#include <cfd/fv/compressible.hpp>
#include <cfd/mesh/mesh_builder.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

// Geometric distribution of n cells over [0, length] with first cell size first.
std::vector<Scalar> graded(int n, Scalar length, Scalar first) {
    Scalar lo = 1.0 + 1e-9;
    Scalar hi = 2.0;
    for (int k = 0; k < 200; ++k) {
        const Scalar r = 0.5 * (lo + hi);
        const Scalar total = first * (std::pow(r, n) - 1.0) / (r - 1.0);
        (total > length ? hi : lo) = r;
    }
    const Scalar r = 0.5 * (lo + hi);
    std::vector<Scalar> x(static_cast<std::size_t>(n) + 1, 0.0);
    Scalar d = first;
    for (std::size_t i = 1; i < x.size(); ++i) {
        x[i] = x[i - 1] + d;
        d *= r;
    }
    const Scalar scale = length / x.back();
    for (auto& v : x) {
        v *= scale;
    }
    return x;
}

} // namespace

int main(int argc, char** argv) {
    const int nx_plate = argc > 1 ? std::atoi(argv[1]) : 64;
    const int ny = argc > 2 ? std::atoi(argv[2]) : 64;
    const int max_iterations = argc > 3 ? std::atoi(argv[3]) : 6000;
    const Scalar cfl = argc > 4 ? std::atof(argv[4]) : 1000.0;
    const bool turbulent = !(argc > 5 && std::string(argv[5]) == "laminar");
    const int krylov = argc > 6 ? std::atoi(argv[6]) : 30;
    const int nx_up = nx_plate / 4;

    const Scalar gamma = 1.4;
    const Scalar r_gas = 287.0;
    const Scalar t_inf = 300.0;
    const Scalar p_inf = 1e5;
    const Scalar rho_inf = p_inf / (r_gas * t_inf);
    const Scalar u_inf = 0.2 * std::sqrt(gamma * r_gas * t_inf);
    const Scalar mu = rho_inf * u_inf / 5e6;
    const Scalar dx_le = 2.0 / (16.0 * nx_plate); // streamwise size at the leading edge

    // x: run-up (graded towards the leading edge) + plate; y: graded towards the wall.
    std::vector<Scalar> xs;
    const std::vector<Scalar> up = graded(nx_up, 0.33, dx_le);
    for (int i = nx_up; i > 0; --i) {
        xs.push_back(-up[static_cast<std::size_t>(i)]);
    }
    for (const Scalar x : graded(nx_plate, 2.0, dx_le)) {
        xs.push_back(x);
    }
    const std::vector<Scalar> ys = graded(ny, 1.0, 4e-6);
    const int nx = static_cast<int>(xs.size()) - 1;

    std::vector<Vec3> points;
    for (int k = 0; k <= 1; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                points.emplace_back(xs[static_cast<std::size_t>(i)],
                                    ys[static_cast<std::size_t>(j)], 0.01 * k);
            }
        }
    }
    auto id = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    cfd::mesh::MeshBuilder b(points);
    b.add_patch("inlet", cfd::mesh::PatchType::Patch);
    b.add_patch("outlet", cfd::mesh::PatchType::Outlet);
    b.add_patch("top", cfd::mesh::PatchType::Outlet);
    b.add_patch("runup", cfd::mesh::PatchType::Symmetry);
    b.add_patch("plate", cfd::mesh::PatchType::Wall);
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
            std::array<int, 4>{id(i, 0, 0), id(i + 1, 0, 0), id(i + 1, 0, 1), id(i, 0, 1)},
            i < nx_up ? "runup" : "plate");
        b.add_boundary_face(
            std::array<int, 4>{id(i, ny, 0), id(i + 1, ny, 0), id(i + 1, ny, 1), id(i, ny, 1)},
            "top");
    }
    const auto mesh = b.build("sides");

    CompressibleSettings s;
    s.gamma = gamma;
    s.r_gas = r_gas;
    s.mu = mu;
    s.cfl = cfl;
    s.implicit = true;
    s.krylov_dim = krylov;
    s.turbulence = turbulent;
    s.limiter = Limiter::Venkatakrishnan;
    BoundaryCondition inlet{FlowBoundary::VelocityInlet, Vec3(u_inf, 0, 0), p_inf};
    inlet.temperature = t_inf;
    inlet.turbulence_intensity = 0.01; // turbulent from the leading edge (SST is bistable)
    inlet.viscosity_ratio = 1.0;
    BoundaryCondition outlet{FlowBoundary::PressureOutlet, Vec3::Zero(), p_inf};
    CompressibleSolver solver(mesh,
                              {{"inlet", inlet},
                               {"outlet", outlet},
                               {"top", outlet},
                               {"sides", {FlowBoundary::Empty, Vec3::Zero(), 0.0}}},
                              s);
    solver.set_uniform({rho_inf, Vec3(u_inf, 0, 0), p_inf});
    // Skin friction from the wall-cell velocity (first cell row, y+ ~ 1).
    const Scalar q_inf = 0.5 * rho_inf * u_inf * u_inf;
    auto cf_at = [&](Scalar x_target) {
        for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_cells()); ++c) {
            const Vec3& cc = mesh.cell_centres()[c];
            if (cc.y() > ys[1]) {
                continue;
            }
            for (int i = nx_up; i < nx; ++i) {
                const Scalar x0 = xs[static_cast<std::size_t>(i)];
                const Scalar x1 = xs[static_cast<std::size_t>(i) + 1];
                if (x_target >= x0 && x_target < x1 && cc.x() > x0 && cc.x() < x1) {
                    return mu * solver.state(c).u.x() / cc.y() / q_inf;
                }
            }
        }
        return 0.0;
    };
    Scalar r0 = 0.0;
    int it = 0;
    for (; it < max_iterations; ++it) {
        const Scalar r = solver.steady_iteration();
        r0 = it < 20 ? std::max(r0, r) : r0;
        if (it % 200 == 0) {
            const auto [rk, rw] =
                turbulent ? solver.turbulence_residuals() : std::pair<Scalar, Scalar>{0.0, 0.0};
            std::printf("iteration %d  density %.3e  k %.2e  omega %.2e  Cf(0.97) %.5f\n", it,
                        r / r0, rk, rw, cf_at(0.97));
        }
        if (r < 1e-8 * r0) {
            break;
        }
    }

    std::printf("cells %d, %d iterations\n", mesh.n_cells(), it);
    std::printf("   x      Re_x      Cf      Cf(0.0592 Re^-0.2)  Cf(White)\n");
    for (const Scalar x : {0.25, 0.5, 0.97, 1.5}) {
        const Scalar re = 5e6 * x;
        const Scalar l = std::log(0.06 * re);
        std::printf("%5.2f  %.3e  %.5f  %.5f  %.5f\n", x, re, cf_at(x), 0.0592 * std::pow(re, -0.2),
                    0.455 / (l * l));
    }
    std::printf("Cf(x = 0.97): simulation %.5f, TMR SST reference 0.00271\n", cf_at(0.97));
    return 0;
}
