#include <cfd/case/case_runner.hpp>
#include <cfd/parallel/comm.hpp>
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

    EXPECT_EQ(full.global_mesh().n_cells(), 400);
    ASSERT_NE(full.local_mesh(), nullptr);
    EXPECT_LT(full.mesh().n_cells(), 400);
    if (comm.is_root()) {
        EXPECT_TRUE(std::filesystem::exists(dir / "full" / "results" / "cavity.vtu"));
        EXPECT_TRUE(std::filesystem::exists(dir / "full" / "results" / "forces.csv"));
    }
}

} // namespace
