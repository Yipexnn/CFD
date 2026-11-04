// Natural convection in a differentially heated square cavity (de Vahl Davis 1983):
// hot wall x = 0 (T = 1), cold wall x = 1 (T = 0), adiabatic top/bottom, Pr = 0.71.
// Non-dimensional: L = 1, alpha = 1, nu = Pr, g beta dT = Ra Pr (velocity scale alpha / L).
// Reference mean Nusselt numbers: Ra 1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800.
//
// Usage: cavity_de_vahl_davis [Ra=1e4] [n=64] [max_iterations=20000]
#include <cfd/fv/energy.hpp>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

int main(int argc, char** argv) {
    const Scalar ra = argc > 1 ? std::atof(argv[1]) : 1e4;
    const int n = argc > 2 ? std::atoi(argv[2]) : 64;
    const int max_iterations = argc > 3 ? std::atoi(argv[3]) : 20000;
    const Scalar pr = 0.71;
    const Scalar depth = 1.0 / n;

    cfd::mesh::BoxSpec spec;
    spec.nx = n;
    spec.ny = n;
    spec.nz = 1;
    spec.max = Vec3(1.0, 1.0, depth);
    const auto mesh = cfd::mesh::make_box_mesh(spec);

    FlowSettings settings;
    settings.nu = pr;
    settings.convection = ConvectionScheme::LimitedLinearUpwind;
    settings.alpha_u = 0.7;
    settings.alpha_p = 0.3;
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    FlowSolver flow(mesh, {{"zmin", empty}, {"zmax", empty}}, settings);

    EnergySettings es;
    es.alpha = 1.0;
    es.beta = 1.0;
    es.t_ref = 0.5;
    es.gravity = Vec3(0.0, -ra * pr, 0.0);
    es.relaxation = 0.9;
    EnergyEquation energy(flow, es,
                          {{"xmin", {ThermalBoundary::Kind::FixedTemperature, 1.0}},
                           {"xmax", {ThermalBoundary::Kind::FixedTemperature, 0.0}}});

    const auto t0 = std::chrono::steady_clock::now();
    int it = 0;
    for (; it < max_iterations; ++it) {
        flow.set_cell_acceleration(energy.buoyancy());
        Residuals r = flow.simple_iteration();
        r.t = energy.correct(flow, 0.0);
        if (it % 1000 == 0) {
            std::printf("iteration %d  residual %.2e  (T %.2e)\n", it, r.max(), r.t);
        }
        if (r.max() < 1e-7) {
            break;
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Mean Nusselt number = heat flow / (alpha dT L / L * depth).
    const Scalar nu_hot = energy.patch_heat_flow(flow, "xmin") / depth;
    const Scalar nu_cold = -energy.patch_heat_flow(flow, "xmax") / depth;
    Scalar u_max = 0.0;
    for (const auto& u : flow.velocity()) {
        u_max = std::max(u_max, u.norm());
    }
    std::printf("Ra %.0e, %dx%d cells, %d iterations, %.1f s\n", ra, n, n, it, seconds);
    std::printf("Nu hot %.4f  cold %.4f  |U|max %.3f\n", nu_hot, nu_cold, u_max);
    return 0;
}
