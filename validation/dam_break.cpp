// Dam break (Martin & Moyce 1952, n^2 = 2): water column a x 2a collapses in a closed tank.
// Leading-edge position Z = x / a versus T = t sqrt(2 g / a), compared with the experiment.
//
// Usage: dam_break [n_x=100] [dt=5e-4] [T_end=3]
#include <cfd/fv/multiphase.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

int main(int argc, char** argv) {
    const int nx = argc > 1 ? std::atoi(argv[1]) : 100;
    const Scalar dt = argc > 2 ? std::atof(argv[2]) : 5e-4;
    const Scalar t_end_nd = argc > 3 ? std::atof(argv[3]) : 3.0;
    const Scalar a = 0.05715;
    const Scalar g = 9.81;
    const Scalar length = 5.0 * a;
    const Scalar height = 2.5 * a;

    cfd::mesh::BoxSpec spec;
    spec.nx = nx;
    spec.ny = nx / 2;
    spec.nz = 1;
    spec.max = Vec3(length, height, length / nx);
    const auto mesh = cfd::mesh::make_box_mesh(spec);

    MultiphaseSettings s;
    s.rho1 = 1000.0;
    s.nu1 = 1e-6;
    s.rho2 = 1.0;
    s.nu2 = 1.48e-5;
    s.sigma = 0.07;
    s.gravity = Vec3(0.0, -g, 0.0);
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    MultiphaseSolver solver(mesh, {{"zmin", empty}, {"zmax", empty}}, s);
    std::vector<Scalar> alpha(static_cast<std::size_t>(mesh.n_cells()), 0.0);
    for (std::size_t c = 0; c < alpha.size(); ++c) {
        const Vec3& x = mesh.cell_centres()[c];
        alpha[c] = x.x() < a && x.y() < 2.0 * a ? 1.0 : 0.0;
    }
    solver.set_alpha(alpha);
    const Scalar v0 = solver.phase1_volume();

    const Scalar time_scale = std::sqrt(2.0 * g / a);
    const int steps = static_cast<int>(t_end_nd / time_scale / dt + 0.5);
    const Scalar dx = length / nx;
    std::printf("    T      Z (sim)   Z (Martin & Moyce)\n");
    const Scalar exp_t[] = {0.41, 0.84, 1.19, 1.43, 1.63, 1.83, 1.98,
                            2.20, 2.32, 2.51, 2.65, 2.83, 2.97};
    const Scalar exp_z[] = {1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33,
                            2.56, 2.78, 3.00, 3.22, 3.44, 3.67};
    std::size_t next = 0;
    Scalar max_bound = 0.0;
    Scalar max_error = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int n = 1; n <= steps; ++n) {
        const Residuals r = solver.step(dt);
        max_bound = std::max(max_bound, r.t);
        const Scalar t_nd = n * dt * time_scale;
        if (next < std::size(exp_t) && t_nd >= exp_t[next]) {
            // Leading edge: furthest cell of the bottom row with alpha > 0.5 (+ fraction).
            Scalar front = 0.0;
            for (std::size_t c = 0; c < alpha.size(); ++c) {
                const Vec3& x = mesh.cell_centres()[c];
                if (x.y() < height / spec.ny && solver.alpha()[c] > 0.5) {
                    front = std::max(front, x.x() + 0.5 * dx);
                }
            }
            const Scalar z = front / a;
            max_error = std::max(max_error, std::abs(z - exp_z[next]) / exp_z[next]);
            std::printf("  %5.2f    %6.3f    %6.2f\n", t_nd, z, exp_z[next]);
            ++next;
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("cells %d, %d steps, %.1f s; volume change %.2e, max alpha bound error %.1e\n",
                mesh.n_cells(), steps, seconds, solver.phase1_volume() / v0 - 1.0, max_bound);
    std::printf("max relative deviation of the front from the experiment: %.1f %%\n",
                100.0 * max_error);
    return 0;
}
