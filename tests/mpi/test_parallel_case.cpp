#include <cfd/case/case_runner.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/parallel/comm.hpp>
#include <cfd/parallel/decomposition.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

namespace par = cfd::parallel;
using cfd::cases::CaseRunner;

namespace {

cfd::cases::CaseConfig cavity(int iterations, const std::filesystem::path& dir) {
    return cfd::cases::parse_case(R"({
        "name": "cavity",
        "mesh": {"type": "box", "cells": [20, 20, 1], "max": [1, 1, 0.1],
                 "patch_types": {"zmin": "symmetry", "zmax": "symmetry"}},
        "physics": {"nu": 0.01},
        "boundaries": {"ymax": {"type": "wall", "velocity": [1, 0, 0]}},
        "solver": {"max_iterations": )" +
                                      std::to_string(iterations) +
                                      R"(, "tolerance": 1e-12},
        "forces": {"patches": ["ymax"]}
    })",
                                  dir);
}

// Restarting from a checkpoint written by the parallel run continues the run. Not bitwise:
// both processes of a processor face compute its flux with their own (flipped) geometry, so
// values differ by rounding and linear solves may stop one iteration apart.
TEST(ParallelCase, CheckpointRestartContinuesRun) {
    const par::Comm& comm = par::Comm::world();
    const auto dir =
        std::filesystem::temp_directory_path() / ("cfd_mpi_case_" + std::to_string(comm.size()));
    if (comm.is_root()) {
        std::filesystem::remove_all(dir);
    }
    comm.barrier();

    CaseRunner full(cavity(40, dir / "full"));
    (void)full.run();
    const cfd::core::Vec3 f_full = full.force("ymax");

    CaseRunner first(cavity(20, dir / "restart"));
    (void)first.run();
    auto config = cavity(40, dir / "restart");
    config.restart = "results/checkpoint.bin";
    CaseRunner second(config);
    EXPECT_EQ(second.step(), 20); // restart at the AMG rebuild age (20)
    (void)second.run();
    const cfd::core::Vec3 f_restart = second.force("ymax");
    EXPECT_NEAR((f_full - f_restart).norm(), 0.0, 1e-5 * f_full.norm());

    EXPECT_EQ(full.global_cell_count(), 400);
    EXPECT_EQ(full.has_global_mesh(), comm.is_root()); // only the root holds the global mesh
    ASSERT_NE(full.local_mesh(), nullptr);
    EXPECT_LT(full.mesh().n_cells(), 400);
    if (comm.is_root()) {
        EXPECT_TRUE(std::filesystem::exists(dir / "full" / "results" / "cavity.vtu"));
        EXPECT_TRUE(std::filesystem::exists(dir / "full" / "results" / "forces.csv"));
    }
}

} // namespace

// Parallel mesh distribution: the local meshes received from the root equal the ones each
// process would decompose itself; pack / unpack round-trips.
TEST(ParallelCase, DistributeMatchesDecompose) {
    const auto& comm = cfd::parallel::Comm::world();
    cfd::mesh::BoxSpec spec;
    spec.nx = 12;
    spec.ny = 9;
    spec.nz = 4;
    const auto global = cfd::mesh::make_box_mesh(spec);
    const auto received = cfd::parallel::distribute(comm.is_root() ? &global : nullptr, comm);
    const auto parts = cfd::parallel::partition_rcb(global, comm.size());
    const auto expected = cfd::parallel::decompose(global, parts, comm.rank(), comm);
    EXPECT_EQ(received.mesh.n_cells(), expected.mesh.n_cells());
    EXPECT_EQ(received.mesh.n_faces(), expected.mesh.n_faces());
    EXPECT_EQ(received.global_cell, expected.global_cell);
    EXPECT_EQ(received.global_face, expected.global_face);
    EXPECT_EQ(received.halo.n_owned, expected.halo.n_owned);
    ASSERT_EQ(received.halo.neighbours.size(), expected.halo.neighbours.size());
    for (std::size_t i = 0; i < expected.halo.neighbours.size(); ++i) {
        EXPECT_EQ(received.halo.neighbours[i].send, expected.halo.neighbours[i].send);
        EXPECT_EQ(received.halo.neighbours[i].recv, expected.halo.neighbours[i].recv);
    }
    for (std::size_t c = 0; c < expected.mesh.cell_centres().size(); ++c) {
        EXPECT_TRUE(received.mesh.cell_centres()[c].isApprox(expected.mesh.cell_centres()[c]));
    }
    const auto back =
        cfd::parallel::unpack_local_mesh(cfd::parallel::pack_local_mesh(expected), comm);
    EXPECT_EQ(back.mesh.patches().size(), expected.mesh.patches().size());
    EXPECT_EQ(back.face_flipped, expected.face_flipped);
}
