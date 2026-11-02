#include <algorithm>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/k_omega_sst.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cmath>
#include <gtest/gtest.h>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;

namespace {

// Fully developed half channel (wall at y = 0, centreline symmetry at y = h = 1) as a
// 1D problem: one cell in x and z with Empty sides, driven by the body force u_tau^2 / h
// with u_tau = 1, so Re_tau = 1 / nu and all velocities are in wall units.
struct Channel {
    cfd::mesh::Mesh mesh;
    FlowSolver flow;
    KOmegaSst sst;

    Channel(int ny, Scalar grading, Scalar re_tau)
        : mesh(make_mesh(ny, grading)), flow(mesh, conditions(), settings(re_tau)),
          sst(flow, SstSettings{1.0, 10.0}) {
        flow.set_uniform_velocity(Vec3(15.0, 0.0, 0.0));
        flow.set_turbulence_model(&sst);
    }

    static cfd::mesh::Mesh make_mesh(int ny, Scalar grading) {
        cfd::mesh::BoxSpec spec;
        spec.nx = 1;
        spec.ny = ny;
        spec.nz = 1;
        spec.max = Vec3(0.1, 1.0, 0.1);
        spec.grading.y() = grading;
        return cfd::mesh::make_box_mesh(spec);
    }

    static std::map<std::string, BoundaryCondition> conditions() {
        const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
        return {{"xmin", empty},
                {"xmax", empty},
                {"zmin", empty},
                {"zmax", empty},
                {"ymax", {FlowBoundary::Symmetry, Vec3::Zero(), 0.0}}};
    }

    static FlowSettings settings(Scalar re_tau) {
        FlowSettings s;
        s.nu = 1.0 / re_tau;
        s.body_force = Vec3(1.0, 0.0, 0.0);
        s.alpha_u = 0.95;
        s.alpha_p = 0.5;
        return s;
    }

    int solve(int max_iterations) {
        for (int it = 1; it <= max_iterations; ++it) {
            if (flow.simple_iteration().max() < 1e-7) {
                return it;
            }
        }
        return max_iterations;
    }

    // Volume-averaged (bulk) velocity.
    [[nodiscard]] Scalar bulk_velocity() const {
        Scalar sum = 0.0;
        Scalar volume = 0.0;
        for (std::size_t c = 0; c < flow.velocity().size(); ++c) {
            sum += flow.velocity()[c].x() * mesh.cell_volumes()[c];
            volume += mesh.cell_volumes()[c];
        }
        return sum / volume;
    }

    [[nodiscard]] Scalar centre_velocity() const { return flow.velocity().back().x(); }

    // Velocity at wall distance y (linear interpolation between cell centres).
    [[nodiscard]] Scalar velocity_at(Scalar y) const {
        const auto& cc = mesh.cell_centres();
        std::size_t j = 0;
        while (j + 2 < cc.size() && cc[j + 1].y() < y) {
            ++j;
        }
        const Scalar t = std::clamp((y - cc[j].y()) / (cc[j + 1].y() - cc[j].y()), 0.0, 1.0);
        return (1.0 - t) * flow.velocity()[j].x() + t * flow.velocity()[j + 1].x();
    }

    // Total wall shear stress (nu + nu_t,wall) dU/dy from the wall face.
    [[nodiscard]] Scalar wall_shear() const {
        const auto* wall = mesh.find_patch("ymin");
        const auto b = static_cast<std::size_t>(wall->start_face - mesh.n_internal_faces());
        const Scalar nu_eff = flow.settings().nu + sst.boundary_nut()[b];
        return nu_eff * flow.velocity()[0].x() / mesh.cell_centres()[0].y();
    }
};

} // namespace

TEST(Turbulence, SstChannelRe395WallResolved) {
    Channel ch(60, 20.0, 395.0);
    EXPECT_LT(ch.solve(20000), 20000);

    EXPECT_LT(ch.sst.y_plus()[0], 1.0);
    EXPECT_NEAR(ch.wall_shear(), 1.0, 0.01); // force balance
    // DNS (Moser, Kim & Mansour 1999): U_b+ = 17.5, U_c+ = 20.1.
    EXPECT_NEAR(ch.bulk_velocity() / 17.5, 1.0, 0.05);
    EXPECT_NEAR(ch.centre_velocity() / 20.1, 1.0, 0.05);

    // Log-law slope 1/kappa between y+ = 100 and 300.
    const Scalar nu = ch.flow.settings().nu;
    const Scalar slope = (ch.velocity_at(300.0 * nu) - ch.velocity_at(100.0 * nu)) / std::log(3.0);
    EXPECT_NEAR(slope * 0.41, 1.0, 0.05);

    // Viscous sublayer U+ = y+.
    EXPECT_NEAR(ch.velocity_at(1.0 * nu), 1.0, 0.05);

    for (std::size_t c = 0; c < ch.sst.k().size(); ++c) {
        ASSERT_GT(ch.sst.k()[c], 0.0);
        ASSERT_GT(ch.sst.omega()[c], 0.0);
        ASSERT_GE(ch.sst.nut()[c], 0.0);
    }
}

TEST(Turbulence, SstChannelRe2000WallFunction) {
    Channel ch(20, 1.0, 2000.0);
    EXPECT_LT(ch.solve(20000), 20000);

    EXPECT_GT(ch.sst.y_plus()[0], 40.0); // log-layer first cell
    EXPECT_LT(ch.sst.y_plus()[0], 60.0);
    EXPECT_NEAR(ch.wall_shear(), 1.0, 0.03);
    // Dean's correlation Cf = 0.073 Re_b^-0.25 gives U_b+ = 21.7 at Re_tau = 2000.
    EXPECT_NEAR(ch.bulk_velocity() / 21.7, 1.0, 0.06);
}

TEST(Turbulence, PisoKeepsSteadyTurbulentChannel) {
    Channel ch(60, 20.0, 395.0);
    (void)ch.solve(20000);
    const Scalar ub = ch.bulk_velocity();
    for (int step = 0; step < 20; ++step) {
        (void)ch.flow.piso_step(0.01);
    }
    EXPECT_NEAR(ch.bulk_velocity() / ub, 1.0, 0.005);
}

TEST(Turbulence, LaminarTestsUnaffectedWithoutModel) {
    Channel ch(20, 1.0, 395.0);
    EXPECT_EQ(ch.flow.turbulence_model(), &ch.sst);
    ch.flow.set_turbulence_model(nullptr);
    EXPECT_EQ(ch.flow.turbulence_model(), nullptr);
    EXPECT_EQ(ch.flow.simple_iteration().k, 0.0);
}
