#include <cfd/fv/flow_solver.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <gtest/gtest.h>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

// Half channel 0 <= y <= 1 driven by a body force: wall at ymin, symmetry at ymax, one
// cell in x and z (empty). In steady state the wall force balances the body force.
TEST(PatchForce, WallShearBalancesBodyForce) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 1;
    spec.ny = 20;
    spec.nz = 1;
    spec.max = Vec3(0.1, 1.0, 0.1);
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    FlowSettings settings;
    settings.nu = 0.1;
    settings.body_force = Vec3(2.0, 0.0, 0.0);
    settings.alpha_u = 0.9;
    settings.alpha_p = 0.5;
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    FlowSolver flow(mesh,
                    {{"xmin", empty},
                     {"xmax", empty},
                     {"zmin", empty},
                     {"zmax", empty},
                     {"ymax", {FlowBoundary::Symmetry, Vec3::Zero(), 0.0}}},
                    settings);
    for (int it = 0; it < 2000 && flow.simple_iteration().max() > 1e-10; ++it) {
    }
    const PatchForce wall = flow.patch_force("ymin");
    const Scalar volume = 0.1 * 1.0 * 0.1;
    EXPECT_NEAR(wall.viscous.x(), 2.0 * volume, 1e-6);
    EXPECT_NEAR(wall.viscous.y(), 0.0, 1e-9);
    // Pressure acts normal to the wall (-y); the x component vanishes.
    EXPECT_NEAR(wall.pressure.x(), 0.0, 1e-12);
    EXPECT_NEAR(wall.total().x(), 2.0 * volume, 1e-6);

    const PatchForce side = flow.patch_force("xmin");
    EXPECT_NEAR(side.viscous.norm(), 0.0, 1e-12);
    EXPECT_THROW((void)flow.patch_force("nope"), std::invalid_argument);
}

// Uniform pressure on a closed body gives no net force: check the pressure part on the
// two opposite walls of a channel with p = const.
TEST(PatchForce, PressureForceOnWalls) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 4;
    spec.ny = 4;
    spec.nz = 1;
    const auto mesh = cfd::mesh::make_box_mesh(spec);
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    FlowSolver flow(mesh, {{"zmin", empty}, {"zmax", empty}}, FlowSettings{});
    flow.set_state(std::vector<Vec3>(16, Vec3::Zero()), std::vector<Scalar>(16, 3.0));
    // Wall area 1 x 1, outward normal -y at ymin: F = p A n.
    EXPECT_NEAR(flow.patch_force("ymin").pressure.y(), -3.0, 1e-12);
    EXPECT_NEAR(flow.patch_force("ymax").pressure.y(), 3.0, 1e-12);
    EXPECT_NEAR(flow.patch_force("ymin").viscous.norm(), 0.0, 1e-12);
}

} // namespace
