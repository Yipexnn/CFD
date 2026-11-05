// Laminar vortex shedding behind a cylinder, Schaefer & Turek benchmark 2D-2 (Re = 100).
// Channel 2.2 x 0.41, cylinder D = 0.1 at (0.2, 0.2), parabolic inflow U_max = 1.5
// (mean 1), nu = 1e-3. PISO with BDF2 in time; forces from the periodic state.
// Reference ranges: Cd_max 3.22..3.24, Cl_max 0.99..1.01, St = f D / U 0.295..0.305.
//
// Usage: cylinder_schaefer_turek_2d2 [dt=0.002] [end_time=10] [size=0.015] [size_near=0.002]
#include <algorithm>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/geometry/geometry.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

using cfd::core::Scalar;
using cfd::core::Vec3;

int main(int argc, char** argv) {
    namespace geo = cfd::geometry;
    const Scalar dt = argc > 1 ? std::atof(argv[1]) : 0.002;
    const Scalar end_time = argc > 2 ? std::atof(argv[2]) : 10.0;
    const Scalar height = 0.41;
    const Scalar diameter = 0.1;
    const Scalar depth = 0.01;
    const Scalar u_max = 1.5;
    const Scalar u_mean = 1.0;

    geo::GeometrySpec spec;
    spec.two_d = true;
    spec.domain_max = Vec3(2.2, height, depth);
    geo::Body cylinder;
    cylinder.kind = geo::Body::Kind::Cylinder;
    cylinder.a = Vec3(0.2, 0.2, 0.0);
    cylinder.radius = 0.5 * diameter;
    cylinder.patch = "cylinder";
    spec.bodies.push_back(cylinder);
    spec.size = argc > 3 ? std::atof(argv[3]) : 0.015;
    spec.size_near = argc > 4 ? std::atof(argv[4]) : 0.002;
    spec.refine_distance = 0.3;
    spec.bl_layers = 8;
    spec.bl_first = 2e-4;
    spec.bl_ratio = 1.2;
    spec.domain_patch_names = {"inlet", "outlet", "walls", "walls", "front", "back"};
    const auto mesh = geo::generate_mesh(spec, std::filesystem::temp_directory_path() /
                                                   "cfd_cylinder_schaefer_turek_2d2");

    cfd::fv::FlowSettings settings;
    settings.nu = 1e-3;
    settings.convection = cfd::fv::ConvectionScheme::LinearUpwind;
    settings.time_scheme = cfd::fv::TimeScheme::Backward;
    settings.n_piso_correctors = 2;
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
        const Scalar y = mesh.face_centres()[static_cast<std::size_t>(inlet->start_face + q)].y();
        profile.emplace_back(4.0 * u_max * y * (height - y) / (height * height), 0.0, 0.0);
    }
    flow.set_boundary_velocity_profile("inlet", profile);

    const Scalar scale = 2.0 / (u_mean * u_mean * diameter * depth);
    std::vector<Scalar> times;
    std::vector<Scalar> cd;
    std::vector<Scalar> cl;
    const auto t0 = std::chrono::steady_clock::now();
    const int steps = static_cast<int>(end_time / dt + 0.5);
    std::ofstream csv("cylinder_2d2_forces.csv");
    csv << "time,Cd,Cl\n" << std::setprecision(10);
    for (int n = 1; n <= steps; ++n) {
        (void)flow.piso_step(dt);
        const Vec3 c = flow.patch_force("cylinder").total() * scale;
        times.push_back(n * dt);
        cd.push_back(c.x());
        cl.push_back(c.y());
        csv << n * dt << ',' << c.x() << ',' << c.y() << '\n';
        if (n % 500 == 0) {
            std::printf("t = %.3f  Cd %.4f  Cl %+.4f\n", n * dt, c.x(), c.y());
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Periodic state: last 30 % of the run. Period from upward zero crossings of Cl - mean.
    const std::size_t first = times.size() * 7 / 10;
    Scalar cl_mean = 0.0;
    for (std::size_t i = first; i < cl.size(); ++i) {
        cl_mean += cl[i];
    }
    cl_mean /= static_cast<Scalar>(cl.size() - first);
    std::vector<Scalar> crossings;
    for (std::size_t i = first + 1; i < cl.size(); ++i) {
        const Scalar a = cl[i - 1] - cl_mean;
        const Scalar b = cl[i] - cl_mean;
        if (a < 0.0 && b >= 0.0) {
            crossings.push_back(times[i - 1] + dt * a / (a - b));
        }
    }
    const Scalar cd_max =
        *std::max_element(cd.begin() + static_cast<std::ptrdiff_t>(first), cd.end());
    const Scalar cl_max =
        *std::max_element(cl.begin() + static_cast<std::ptrdiff_t>(first), cl.end());
    const Scalar cl_min =
        *std::min_element(cl.begin() + static_cast<std::ptrdiff_t>(first), cl.end());
    Scalar strouhal = 0.0;
    if (crossings.size() >= 2) {
        const Scalar period =
            (crossings.back() - crossings.front()) / static_cast<Scalar>(crossings.size() - 1);
        strouhal = diameter / (period * u_mean);
    }
    std::printf("cells %d, dt %.4f, %d steps, %.1f s\n", mesh.n_cells(), dt, steps, seconds);
    std::printf("Cd_max = %.4f  (Schaefer-Turek: 3.22 .. 3.24)\n", cd_max);
    std::printf("Cl_max = %.4f  (0.99 .. 1.01), Cl_min = %.4f\n", cl_max, cl_min);
    std::printf("St     = %.4f  (0.295 .. 0.305), %zu periods\n", strouhal,
                crossings.empty() ? 0 : crossings.size() - 1);
    return 0;
}
