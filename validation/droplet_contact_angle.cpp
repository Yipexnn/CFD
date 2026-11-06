// Droplet spreading on a wall with a static contact angle (2D, no gravity): a semicircular
// droplet (90 degrees) relaxes to a circular cap with the prescribed angle theta. The final
// height h and the droplet area A give the measured angle from h = R (1 - cos t),
// A = R^2 (t - sin t cos t), compared with theta.
//
// Usage: droplet_contact_angle [theta_deg=60] [n_x=96] [t_end=0.02]
#include <cfd/fv/multiphase.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

int main(int argc, char** argv) {
    const Scalar theta_deg = argc > 1 ? std::atof(argv[1]) : 60.0;
    const int nx = argc > 2 ? std::atoi(argv[2]) : 96;
    const Scalar t_end = argc > 3 ? std::atof(argv[3]) : 0.02;
    const Scalar r0 = 1e-3;
    const Scalar half_width = 3e-3;
    const Scalar height = 2.5e-3;

    cfd::mesh::BoxSpec spec;
    spec.nx = nx;
    spec.ny = static_cast<int>(std::lround(nx * height / (2.0 * half_width)));
    spec.nz = 1;
    const Scalar dx = 2.0 * half_width / nx;
    spec.min = Vec3(-half_width, 0.0, 0.0);
    spec.max = Vec3(half_width, height, dx);
    const auto mesh = cfd::mesh::make_box_mesh(spec);

    MultiphaseSettings s;
    s.rho1 = 1000.0;
    s.nu1 = 1e-4;
    s.rho2 = 1.0;
    s.nu2 = 1.5e-5;
    s.sigma = 0.07;
    BoundaryCondition wall{FlowBoundary::Wall, Vec3::Zero(), 0.0};
    wall.contact_angle = theta_deg;
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const BoundaryCondition slip{FlowBoundary::Symmetry, Vec3::Zero(), 0.0};
    MultiphaseSolver solver(mesh,
                            {{"ymin", wall},
                             {"xmin", slip},
                             {"xmax", slip},
                             {"ymax", {FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0}},
                             {"zmin", empty},
                             {"zmax", empty}},
                            s);
    // Semicircle of radius r0 on the wall (area fractions by sub-sampling).
    std::vector<Scalar> alpha(static_cast<std::size_t>(mesh.n_cells()), 0.0);
    const Scalar dy = height / spec.ny;
    for (std::size_t c = 0; c < alpha.size(); ++c) {
        const Vec3& x = mesh.cell_centres()[c];
        int inside = 0;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                const Scalar px = x.x() + dx * ((i + 0.5) / 8.0 - 0.5);
                const Scalar py = x.y() + dy * ((j + 0.5) / 8.0 - 0.5);
                inside += px * px + py * py < r0 * r0 ? 1 : 0;
            }
        }
        alpha[c] = inside / 64.0;
    }
    solver.set_alpha(alpha);

    // Capillary time-step limit sqrt(rho_mean dx^3 / (2 pi sigma)).
    const Scalar dt = 0.5 * std::sqrt(0.5 * (s.rho1 + s.rho2) * dx * dx * dx /
                                      (2.0 * std::numbers::pi * s.sigma));
    Scalar t = 0.0;
    int steps = 0;
    while (t < t_end) {
        (void)solver.step(dt);
        t += dt;
        ++steps;
    }

    // Droplet area and height (alpha integrated over the column at x = 0).
    Scalar area = 0.0;
    Scalar h = 0.0;
    for (std::size_t c = 0; c < alpha.size(); ++c) {
        const Scalar a = solver.alpha()[c];
        area += a * dx * dy;
        if (std::abs(mesh.cell_centres()[c].x()) < 0.5 * dx + 1e-12 &&
            mesh.cell_centres()[c].x() > 0.0) {
            h += a * dy;
        }
    }
    // Angle with R(t) (1 - cos t) = h for the measured area.
    auto height_of = [&](Scalar th) {
        const Scalar r = std::sqrt(area / (th - std::sin(th) * std::cos(th)));
        return r * (1.0 - std::cos(th));
    };
    Scalar lo = 0.05;
    Scalar hi = std::numbers::pi - 0.05;
    for (int k = 0; k < 100; ++k) {
        const Scalar mid = 0.5 * (lo + hi);
        (height_of(mid) < h ? lo : hi) = mid;
    }
    const Scalar th = theta_deg * std::numbers::pi / 180.0;
    std::printf("cells %d, %d steps (dt %.2e), area change %.2f %%\n", mesh.n_cells(), steps, dt,
                100.0 * (area / (0.5 * std::numbers::pi * r0 * r0) - 1.0));
    std::printf("height: simulation %.4e, equilibrium cap %.4e\n", h, height_of(th));
    std::printf("contact angle: simulation %.1f deg, prescribed %.1f deg\n",
                0.5 * (lo + hi) * 180.0 / std::numbers::pi, theta_deg);
    return 0;
}
