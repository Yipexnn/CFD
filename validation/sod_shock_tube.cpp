// Sod shock tube: left (rho, p) = (1, 1), right (0.125, 0.1), gamma = 1.4, t = 0.2.
// Compared with the exact Riemann solution (L1 errors of density, velocity, pressure).
//
// Usage: sod_shock_tube [cells=400] [first_order=0]
#include <cfd/fv/compressible.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

struct Exact {
    Scalar rho, u, p;
};

// Exact solution of the Riemann problem (Toro, chapter 4) at x / t = s.
Exact riemann(Scalar s, Scalar gamma) {
    const Scalar rl = 1.0, pl = 1.0, rr = 0.125, pr = 0.1;
    const Scalar cl = std::sqrt(gamma * pl / rl), cr = std::sqrt(gamma * pr / rr);
    auto f = [&](Scalar p, Scalar rk, Scalar pk, Scalar ck, Scalar& df) {
        if (p > pk) { // shock
            const Scalar a = 2.0 / ((gamma + 1.0) * rk);
            const Scalar b = (gamma - 1.0) / (gamma + 1.0) * pk;
            const Scalar q = std::sqrt(a / (p + b));
            df = q * (1.0 - 0.5 * (p - pk) / (p + b));
            return (p - pk) * q;
        }
        const Scalar e = (gamma - 1.0) / (2.0 * gamma);
        df = std::pow(p / pk, -(gamma + 1.0) / (2.0 * gamma)) / (rk * ck);
        return 2.0 * ck / (gamma - 1.0) * (std::pow(p / pk, e) - 1.0);
    };
    Scalar p = 0.5 * (pl + pr);
    for (int it = 0; it < 50; ++it) {
        Scalar dl = 0.0, dr = 0.0;
        const Scalar g = f(p, rl, pl, cl, dl) + f(p, rr, pr, cr, dr);
        p = std::max(p - g / (dl + dr), 1e-8);
    }
    Scalar dl = 0.0, dr = 0.0;
    const Scalar u = 0.5 * (f(p, rr, pr, cr, dr) - f(p, rl, pl, cl, dl));
    const Scalar g1 = (gamma - 1.0) / (gamma + 1.0);
    if (s < u) { // left of contact: rarefaction (pl > p)
        const Scalar c_star = cl * std::pow(p / pl, (gamma - 1.0) / (2.0 * gamma));
        if (s < -cl) {
            return {rl, 0.0, pl};
        }
        if (s > u - c_star) {
            return {rl * std::pow(p / pl, 1.0 / gamma), u, p};
        }
        const Scalar c = 2.0 / (gamma + 1.0) * cl + g1 * (-s);
        const Scalar uf = 2.0 / (gamma + 1.0) * (cl + 0.5 * (gamma - 1.0) * 0.0 + s);
        return {rl * std::pow(c / cl, 2.0 / (gamma - 1.0)), uf,
                pl * std::pow(c / cl, 2.0 * gamma / (gamma - 1.0))};
    }
    // right of contact: shock (p > pr)
    const Scalar ratio = p / pr;
    const Scalar shock =
        cr * std::sqrt((gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma));
    if (s > shock) {
        return {rr, 0.0, pr};
    }
    return {rr * (ratio + g1) / (g1 * ratio + 1.0), u, p};
}

} // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 400;
    const bool first_order = argc > 2 && std::atoi(argv[2]) != 0;
    cfd::mesh::BoxSpec spec;
    spec.nx = n;
    spec.ny = 1;
    spec.nz = 1;
    spec.max = Vec3(1.0, 0.01, 0.01);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    CompressibleSettings s;
    s.second_order = !first_order;
    s.cfl = 0.5;
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const BoundaryCondition out{FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0};
    CompressibleSolver solver(mesh,
                              {{"ymin", empty},
                               {"ymax", empty},
                               {"zmin", empty},
                               {"zmax", empty},
                               {"xmin", out},
                               {"xmax", out}},
                              s);
    std::vector<GasState> init(static_cast<std::size_t>(n));
    for (std::size_t c = 0; c < init.size(); ++c) {
        init[c] = mesh.cell_centres()[c].x() < 0.5 ? GasState{1.0, Vec3::Zero(), 1.0}
                                                   : GasState{0.125, Vec3::Zero(), 0.1};
    }
    solver.set_state(init);
    Scalar t = 0.0;
    int steps = 0;
    while (t < 0.2 - 1e-12) {
        const Scalar dt = std::min(solver.stable_dt(), 0.2 - t);
        (void)solver.step(dt);
        t += dt;
        ++steps;
    }
    Scalar e_rho = 0.0, e_u = 0.0, e_p = 0.0;
    for (std::size_t c = 0; c < init.size(); ++c) {
        const Scalar x = mesh.cell_centres()[c].x();
        const Exact ex = riemann((x - 0.5) / 0.2, s.gamma);
        const GasState g = solver.state(c);
        e_rho += std::abs(g.rho - ex.rho) / n;
        e_u += std::abs(g.u.x() - ex.u) / n;
        e_p += std::abs(g.p - ex.p) / n;
    }
    std::printf("Sod, %d cells, %s, %d steps: L1 error rho %.5f  u %.5f  p %.5f\n", n,
                first_order ? "1st order" : "MUSCL", steps, e_rho, e_u, e_p);
    return 0;
}
