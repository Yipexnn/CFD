// Laminar flow around a cylinder, Schäfer & Turek benchmark 2D-1 (steady, Re = 20).
// Channel 2.2 x 0.41, cylinder D = 0.1 at (0.2, 0.2), parabolic inflow U_max = 0.3,
// nu = 1e-3. Mesh: Gmsh (quad boundary layer, one extruded cell layer, empty sides).
// Reference: Cd = 5.57953, Cl = 0.010618, dp = p(0.15, 0.2) - p(0.25, 0.2) = 0.11752.
//
// Usage: cylinder_schaefer_turek [size=0.01] [size_near=0.001] [max_iterations=4000]
#include <cfd/fv/flow_solver.hpp>
#include <cfd/geometry/geometry.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <vector>

using cfd::core::Scalar;
using cfd::core::Vec3;

int main(int argc, char** argv) {
    namespace geo = cfd::geometry;
    const double height = 0.41;
    const double diameter = 0.1;
    const double depth = 0.01;
    const double u_max = 0.3;
    const double u_mean = 2.0 * u_max / 3.0;

    geo::GeometrySpec spec;
    spec.two_d = true;
    spec.domain_min = Vec3(0, 0, 0);
    spec.domain_max = Vec3(2.2, height, depth);
    geo::Body cylinder;
    cylinder.kind = geo::Body::Kind::Cylinder;
    cylinder.a = Vec3(0.2, 0.2, 0.0);
    cylinder.radius = 0.5 * diameter;
    cylinder.patch = "cylinder";
    spec.bodies.push_back(cylinder);
    spec.size = argc > 1 ? std::atof(argv[1]) : 0.01;
    spec.size_near = argc > 2 ? std::atof(argv[2]) : 0.001;
    spec.refine_distance = 0.15;
    spec.bl_layers = 8;
    spec.bl_first = 2.5e-4;
    spec.bl_ratio = 1.2;
    spec.domain_patch_names = {"inlet", "outlet", "walls", "walls", "front", "back"};
    const int max_iterations = argc > 3 ? std::atoi(argv[3]) : 4000;

    const auto work = std::filesystem::temp_directory_path() / "cfd_cylinder_schaefer_turek";
    const auto mesh = geo::generate_mesh(spec, work);

    cfd::fv::FlowSettings settings;
    settings.nu = 1e-3;
    settings.convection = cfd::fv::ConvectionScheme::LimitedLinearUpwind;
    settings.alpha_u = 0.7;
    settings.alpha_p = 0.3;
    settings.n_non_orth_correctors = 1;
    using cfd::fv::BoundaryCondition;
    using cfd::fv::FlowBoundary;
    cfd::fv::FlowSolver flow(mesh,
                             {{"inlet", BoundaryCondition{FlowBoundary::VelocityInlet}},
                              {"outlet", BoundaryCondition{FlowBoundary::PressureOutlet}},
                              {"walls", BoundaryCondition{FlowBoundary::Wall}},
                              {"cylinder", BoundaryCondition{FlowBoundary::Wall}},
                              {"front", BoundaryCondition{FlowBoundary::Empty}},
                              {"back", BoundaryCondition{FlowBoundary::Empty}}},
                             settings);
    const auto* inlet = mesh.find_patch("inlet");
    std::vector<Vec3> profile;
    for (int q = 0; q < inlet->n_faces; ++q) {
        const double y = mesh.face_centres()[static_cast<std::size_t>(inlet->start_face + q)].y();
        profile.emplace_back(4.0 * u_max * y * (height - y) / (height * height), 0.0, 0.0);
    }
    flow.set_boundary_velocity_profile("inlet", profile);

    const auto t0 = std::chrono::steady_clock::now();
    int it = 0;
    for (; it < max_iterations; ++it) {
        const auto r = flow.simple_iteration();
        if (it % 200 == 0) {
            std::printf("iteration %d  residual %.2e\n", it, r.max());
        }
        if (r.max() < 1e-7) {
            break;
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const auto force = flow.patch_force("cylinder");
    const double scale = 2.0 / (u_mean * u_mean * diameter * depth);
    const Vec3 c = force.total() * scale;

    // Pressure at a point: nearest cell, linearly extrapolated with the cell gradient.
    const auto grad_p = flow.pressure_gradient();
    auto pressure_at = [&](const Vec3& x) {
        std::size_t best = 0;
        double best_d = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < mesh.cell_centres().size(); ++i) {
            const double d = (mesh.cell_centres()[i] - x).head<2>().squaredNorm();
            if (d < best_d) {
                best_d = d;
                best = i;
            }
        }
        Vec3 dx = x - mesh.cell_centres()[best];
        dx.z() = 0.0;
        return flow.pressure()[best] + grad_p[best].dot(dx);
    };
    const double dp =
        pressure_at(Vec3(0.15, 0.2, 0.5 * depth)) - pressure_at(Vec3(0.25, 0.2, 0.5 * depth));

    std::printf("cells %d, iterations %d, %.1f s\n", mesh.n_cells(), it, seconds);
    std::printf("Cd = %.5f  (Schaefer-Turek: 5.57953, pressure part %.5f)\n", c.x(),
                force.pressure.x() * scale);
    std::printf("Cl = %.5f  (Schaefer-Turek: 0.010618)\n", c.y());
    std::printf("dp = %.5f  (Schaefer-Turek: 0.11752)\n", dp);
    return 0;
}
