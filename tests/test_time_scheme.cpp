#include <cfd/fv/flow_solver.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

// Start-up of a body-force driven channel (1D in y): velocity at t = 0.2 for a time step,
// on a fixed mesh, so only the temporal error differs between runs.
std::vector<Vec3> startup(TimeScheme scheme, Scalar dt) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 1;
    spec.ny = 16;
    spec.nz = 1;
    spec.max = Vec3(0.1, 1.0, 0.1);
    static const auto mesh = cfd::mesh::make_box_mesh(spec);
    FlowSettings settings;
    settings.nu = 0.05;
    settings.body_force = Vec3(1.0, 0.0, 0.0);
    settings.time_scheme = scheme;
    settings.n_piso_correctors = 3;
    // Tight linear solves: only the temporal discretisation error may remain.
    settings.u_solver = {LinearSolverType::BiCgStab, 1e-14, 1e-10, 1000};
    settings.p_solver = {LinearSolverType::AmgCg, 1e-14, 1e-10, 500};
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    FlowSolver flow(mesh, {{"xmin", empty}, {"xmax", empty}, {"zmin", empty}, {"zmax", empty}},
                    settings);
    const int steps = static_cast<int>(std::lround(0.2 / dt));
    for (int n = 0; n < steps; ++n) {
        (void)flow.piso_step(dt);
    }
    return flow.velocity();
}

Scalar error(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    Scalar e = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) {
        e = std::max(e, (a[c] - b[c]).norm());
    }
    return e;
}

TEST(TimeScheme, BackwardIsSecondOrderEulerFirstOrder) {
    const auto reference = startup(TimeScheme::Backward, 0.2 / 3200.0);
    const Scalar euler_ratio = error(startup(TimeScheme::Euler, 0.01), reference) /
                               error(startup(TimeScheme::Euler, 0.005), reference);
    const Scalar bdf2_ratio = error(startup(TimeScheme::Backward, 0.01), reference) /
                              error(startup(TimeScheme::Backward, 0.005), reference);
    std::printf("convergence ratios: Euler %.3f, BDF2 %.3f\n", euler_ratio, bdf2_ratio);
    EXPECT_NEAR(euler_ratio, 2.0, 0.3);
    EXPECT_GT(bdf2_ratio, 3.3); // ~4 (first step Euler)
    EXPECT_LT(error(startup(TimeScheme::Backward, 0.01), reference),
              0.2 * error(startup(TimeScheme::Euler, 0.01), reference));
}

} // namespace
