#include <algorithm>
#include <cfd/fv/compressible.hpp>
#include <cfd/fv/energy.hpp>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/fv/k_omega_sst.hpp>
#include <cfd/fv/multiphase.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/parallel/decomposition.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <utility>

using cfd::core::Scalar;
using cfd::core::Vec3;
using namespace cfd::fv;
namespace par = cfd::parallel;

namespace {

const par::Comm& comm() {
    return par::Comm::world();
}

// Max |a - b| over the owned cells of all ranks.
Scalar max_difference(const par::LocalMesh& lm, const std::vector<Vec3>& local,
                      const std::vector<Vec3>& global) {
    Scalar d = 0.0;
    for (cfd::core::Label c = 0; c < lm.halo.n_owned; ++c) {
        const auto g = static_cast<std::size_t>(lm.global_cell[static_cast<std::size_t>(c)]);
        d = std::max(d, (local[static_cast<std::size_t>(c)] - global[g]).norm());
    }
    return comm().max(d);
}

TEST(ParallelSolver, RunsOnSeveralProcesses) {
    EXPECT_GE(comm().size(), 2) << "run with mpiexec -n 2 or more";
}

TEST(ParallelSolver, CavityMatchesSerial) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 24;
    spec.ny = 24;
    spec.nz = 1;
    spec.max = Vec3(1.0, 1.0, 0.1);
    spec.patch_types[4] = cfd::mesh::PatchType::Symmetry;
    spec.patch_types[5] = cfd::mesh::PatchType::Symmetry;
    const auto global = cfd::mesh::make_box_mesh(spec);
    const std::map<std::string, BoundaryCondition> bcs{
        {"ymax", {FlowBoundary::Wall, Vec3(1, 0, 0), 0.0}}};
    FlowSettings settings;
    settings.nu = 0.01;
    settings.convection = ConvectionScheme::LimitedLinearUpwind;

    FlowSolver serial(global, bcs, settings);
    (void)serial.solve_steady(1e-9, 3000);

    const auto parts = par::partition_rcb(global, comm().size());
    const auto lm = par::decompose(global, parts, comm().rank(), comm());
    FlowSolver parallel(lm.mesh, bcs, settings, &lm.halo);
    const int its = parallel.solve_steady(1e-9, 3000);
    EXPECT_LT(its, 3000);

    Scalar u_max = 0.0;
    for (const auto& u : serial.velocity()) {
        u_max = std::max(u_max, u.norm());
    }
    EXPECT_LT(max_difference(lm, parallel.velocity(), serial.velocity()), 1e-5 * u_max);
    // Pressure is only defined up to a constant (no outlet): compare gradients via forces.
    const Vec3 fs = serial.patch_force("ymax").total();
    const Vec3 fp = parallel.patch_force("ymax").total();
    EXPECT_LT((fs - fp).norm(), 1e-5 * fs.norm());
}

// Fully developed half channel with k-omega SST and wall functions (Re_tau = 2000),
// split along y so that processor boundaries cross the boundary layer.
TEST(ParallelSolver, SstChannelMatchesSerial) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 1;
    spec.ny = 24;
    spec.nz = 1;
    spec.max = Vec3(0.1, 1.0, 0.1);
    const auto global = cfd::mesh::make_box_mesh(spec);
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const std::map<std::string, BoundaryCondition> bcs{
        {"xmin", empty},
        {"xmax", empty},
        {"zmin", empty},
        {"zmax", empty},
        {"ymax", {FlowBoundary::Symmetry, Vec3::Zero(), 0.0}}};
    FlowSettings settings;
    settings.nu = 1.0 / 2000.0;
    settings.body_force = Vec3(1.0, 0.0, 0.0);
    settings.alpha_u = 0.95;
    settings.alpha_p = 0.5;

    auto solve = [&](FlowSolver& flow) {
        KOmegaSst sst(flow, SstSettings{1.0, 10.0});
        flow.set_uniform_velocity(Vec3(15.0, 0.0, 0.0));
        flow.set_turbulence_model(&sst);
        int it = 0;
        while (it < 20000 && flow.simple_iteration().max() > 1e-8) {
            ++it;
        }
        flow.set_turbulence_model(nullptr);
        return std::make_pair(it, sst.k());
    };
    FlowSolver serial(global, bcs, settings);
    const auto [serial_its, serial_k] = solve(serial);
    ASSERT_LT(serial_its, 20000);

    const auto parts = par::partition_rcb(global, comm().size());
    const auto lm = par::decompose(global, parts, comm().rank(), comm());
    FlowSolver parallel(lm.mesh, bcs, settings, &lm.halo);
    const auto [parallel_its, parallel_k] = solve(parallel);
    EXPECT_LT(parallel_its, 20000);

    EXPECT_LT(max_difference(lm, parallel.velocity(), serial.velocity()), 1e-5 * 20.0);
    Scalar dk = 0.0;
    for (cfd::core::Label c = 0; c < lm.halo.n_owned; ++c) {
        const auto g = static_cast<std::size_t>(lm.global_cell[static_cast<std::size_t>(c)]);
        dk = std::max(dk, std::abs(parallel_k[static_cast<std::size_t>(c)] - serial_k[g]) /
                              serial_k[g]);
    }
    EXPECT_LT(comm().max(dk), 1e-4);
}

// Natural convection (Boussinesq) on several processes matches the serial solution.
TEST(ParallelSolver, HeatedCavityMatchesSerial) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 24;
    spec.ny = 24;
    spec.nz = 1;
    spec.max = Vec3(1.0, 1.0, 1.0 / 24.0);
    const auto global = cfd::mesh::make_box_mesh(spec);
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const std::map<std::string, BoundaryCondition> bcs{{"zmin", empty}, {"zmax", empty}};
    FlowSettings settings;
    settings.nu = 0.71;
    settings.convection = ConvectionScheme::LimitedLinearUpwind;
    EnergySettings es;
    es.alpha = 1.0;
    es.beta = 1.0;
    es.t_ref = 0.5;
    es.gravity = Vec3(0.0, -1e4 * 0.71, 0.0);
    const std::map<std::string, ThermalBoundary> thermal{
        {"xmin", {ThermalBoundary::Kind::FixedTemperature, 1.0}},
        {"xmax", {ThermalBoundary::Kind::FixedTemperature, 0.0}}};
    auto solve = [&](FlowSolver& flow) {
        EnergyEquation energy(flow, es, thermal);
        for (int it = 0; it < 5000; ++it) {
            flow.set_cell_acceleration(energy.buoyancy());
            Residuals r = flow.simple_iteration();
            r.t = energy.correct(flow, 0.0);
            if (r.max() < 1e-9) {
                break;
            }
        }
        return energy.patch_heat_flow(flow, "xmin");
    };
    FlowSolver serial(global, bcs, settings);
    const Scalar q_serial = solve(serial);
    const auto parts = par::partition_rcb(global, comm().size());
    const auto lm = par::decompose(global, parts, comm().rank(), comm());
    FlowSolver parallel(lm.mesh, bcs, settings, &lm.halo);
    const Scalar q_parallel = solve(parallel);
    EXPECT_NEAR(q_parallel, q_serial, 1e-5 * q_serial);
    EXPECT_LT(max_difference(lm, parallel.velocity(), serial.velocity()), 1e-4);
}

// Conjugate heat transfer (fluid zone + solid plate) on several processes = serial.
TEST(ParallelSolver, ConjugatePlateMatchesSerial) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 30;
    spec.ny = 12;
    spec.nz = 1;
    spec.max = Vec3(3.0, 1.2, 0.1);
    spec.patch_types[0] = cfd::mesh::PatchType::Inlet;
    spec.patch_types[1] = cfd::mesh::PatchType::Outlet;
    const auto global = cfd::mesh::make_box_mesh(spec);
    const std::vector<std::string> names{"fluid", "plate"};
    std::vector<int> global_zone(static_cast<std::size_t>(global.n_cells()));
    for (std::size_t c = 0; c < global_zone.size(); ++c) {
        global_zone[c] = global.cell_centres()[c].y() < 0.2 ? 1 : 0;
    }
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const std::map<std::string, BoundaryCondition> bcs{
        {"xmin", {FlowBoundary::VelocityInlet, Vec3(1, 0, 0), 0.0}},
        {"zmin", empty},
        {"zmax", empty}};
    FlowSettings settings;
    settings.nu = 0.02;
    EnergySettings es;
    es.alpha = 0.02 / 0.71;
    const std::map<std::string, ThermalBoundary> thermal{
        {"ymin", {ThermalBoundary::Kind::FixedTemperature, 1.0}}};

    auto solve = [&](const cfd::mesh::Mesh& full, const std::vector<int>& zones,
                     const par::Halo* halo) {
        ConjugateMesh cm;
        cm.zone_of_cell = zones;
        cm.zone_names = names;
        cm.solids = {{"plate", 0.25, 1.0}};
        const par::SubMesh fluid = par::extract_zone(full, zones, names, 0, halo);
        const FvGeometry geo(full, halo);
        cm.geometry = &geo;
        cm.fluid = &fluid;
        FlowSolver flow(fluid.mesh, bcs, settings, halo != nullptr ? &fluid.halo : nullptr);
        EnergyEquation energy(flow, cm, es, thermal);
        for (int it = 0; it < 3000; ++it) {
            Residuals r = flow.simple_iteration();
            r.t = energy.correct(flow, 0.0);
            if (r.max() < 1e-9) {
                break;
            }
        }
        return std::make_pair(energy.patch_heat_flow(flow, "ymin"),
                              energy.patch_heat_flow(flow, "plate"));
    };
    const auto [q_serial, plate_serial] = solve(global, global_zone, nullptr);
    const auto parts = par::partition_rcb(global, comm().size());
    const auto lm = par::decompose(global, parts, comm().rank(), comm());
    std::vector<int> local_zone(lm.global_cell.size());
    for (std::size_t c = 0; c < local_zone.size(); ++c) {
        local_zone[c] = global_zone[static_cast<std::size_t>(lm.global_cell[c])];
    }
    const auto [q_parallel, plate_parallel] = solve(lm.mesh, local_zone, &lm.halo);
    EXPECT_GT(q_serial, 0.0);
    EXPECT_NEAR(q_parallel, q_serial, 1e-6 * q_serial);
    EXPECT_NEAR(plate_parallel, plate_serial, 1e-6 * q_serial);
    EXPECT_NEAR(plate_serial, q_serial, 1e-5 * q_serial);
}

// VOF dam break on several processes: same volume and front as serial.
TEST(ParallelSolver, DamBreakMatchesSerial) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 32;
    spec.ny = 16;
    spec.nz = 1;
    spec.max = Vec3(0.28, 0.14, 0.01);
    const auto global = cfd::mesh::make_box_mesh(spec);
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const std::map<std::string, BoundaryCondition> bcs{{"zmin", empty}, {"zmax", empty}};
    MultiphaseSettings s;
    s.gravity = Vec3(0, -9.81, 0);
    s.sigma = 0.07;
    auto solve = [&](const cfd::mesh::Mesh& mesh, const par::Halo* halo) {
        MultiphaseSolver solver(mesh, bcs, s, halo);
        std::vector<Scalar> alpha(static_cast<std::size_t>(mesh.n_cells()));
        for (std::size_t c = 0; c < alpha.size(); ++c) {
            const Vec3& x = mesh.cell_centres()[c];
            alpha[c] = x.x() < 0.06 && x.y() < 0.12 ? 1.0 : 0.0;
        }
        solver.set_alpha(alpha);
        for (int n = 0; n < 60; ++n) {
            (void)solver.step(5e-4);
        }
        return std::make_pair(solver.phase1_volume(), solver.alpha());
    };
    const auto [v_serial, a_serial] = solve(global, nullptr);
    const auto parts = par::partition_rcb(global, comm().size());
    const auto lm = par::decompose(global, parts, comm().rank(), comm());
    const auto [v_parallel, a_parallel] = solve(lm.mesh, &lm.halo);
    EXPECT_NEAR(v_parallel, v_serial, 1e-12);
    Scalar da = 0.0;
    for (cfd::core::Label c = 0; c < lm.halo.n_owned; ++c) {
        const auto g = static_cast<std::size_t>(lm.global_cell[static_cast<std::size_t>(c)]);
        da = std::max(da, std::abs(a_parallel[static_cast<std::size_t>(c)] - a_serial[g]));
    }
    EXPECT_LT(comm().max(da), 1e-3);
}

// Sod shock tube on several processes = serial.
TEST(ParallelSolver, SodShockTubeMatchesSerial) {
    cfd::mesh::BoxSpec spec;
    spec.nx = 100;
    spec.ny = 1;
    spec.nz = 1;
    spec.max = Vec3(1.0, 0.01, 0.01);
    const auto global = cfd::mesh::make_box_mesh(spec);
    const BoundaryCondition empty{FlowBoundary::Empty, Vec3::Zero(), 0.0};
    const BoundaryCondition out{FlowBoundary::PressureOutlet, Vec3::Zero(), 0.0};
    const std::map<std::string, BoundaryCondition> bcs{{"ymin", empty}, {"ymax", empty},
                                                       {"zmin", empty}, {"zmax", empty},
                                                       {"xmin", out},   {"xmax", out}};
    auto solve = [&](const cfd::mesh::Mesh& mesh, const par::Halo* halo) {
        CompressibleSolver solver(mesh, bcs, CompressibleSettings{}, halo);
        std::vector<GasState> init(static_cast<std::size_t>(mesh.n_cells()));
        for (std::size_t c = 0; c < init.size(); ++c) {
            init[c] = mesh.cell_centres()[c].x() < 0.5 ? GasState{1.0, Vec3::Zero(), 1.0}
                                                       : GasState{0.125, Vec3::Zero(), 0.1};
        }
        solver.set_state(init);
        for (int n = 0; n < 100; ++n) {
            (void)solver.step(1e-3);
        }
        return solver.density();
    };
    const auto serial = solve(global, nullptr);
    const auto parts = par::partition_rcb(global, comm().size());
    const auto lm = par::decompose(global, parts, comm().rank(), comm());
    const auto parallel = solve(lm.mesh, &lm.halo);
    Scalar d = 0.0;
    for (cfd::core::Label c = 0; c < lm.halo.n_owned; ++c) {
        const auto g = static_cast<std::size_t>(lm.global_cell[static_cast<std::size_t>(c)]);
        d = std::max(d, std::abs(parallel[static_cast<std::size_t>(c)] - serial[g]));
    }
    EXPECT_LT(comm().max(d), 1e-12);
}

} // namespace
