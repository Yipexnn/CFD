#include <cfd/case/case_runner.hpp>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

using cfd::cases::CaseConfig;
using cfd::cases::CaseRunner;
using cfd::cases::parse_case;

namespace {

std::filesystem::path test_dir(const std::string& name) {
    const auto dir = std::filesystem::temp_directory_path() / ("cfd_case_runner_" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

CaseConfig cavity(int n, int iterations, double tolerance, const std::string& out,
                  const std::filesystem::path& base) {
    const std::string json = R"({
        "name": "cavity",
        "mesh": {"type": "box", "cells": [)" +
                             std::to_string(n) + "," + std::to_string(n) + R"(, 1],
                 "max": [1, 1, 0.1], "patch_types": {"zmin": "symmetry", "zmax": "symmetry"}},
        "physics": {"nu": 0.01},
        "boundaries": {"ymax": {"type": "wall", "velocity": [1, 0, 0]}},
        "solver": {"max_iterations": )" +
                             std::to_string(iterations) + R"(, "tolerance": )" +
                             std::to_string(tolerance) + R"(},
        "output": {"directory": ")" +
                             out + R"("}
    })";
    return parse_case(json, base);
}

CaseConfig channel_sst(int iterations, const std::string& out, const std::filesystem::path& base) {
    const std::string json = R"({
        "name": "channel",
        "mesh": {"type": "box", "cells": [1, 40, 1], "max": [0.1, 1, 0.1], "grading": [1, 10, 1],
                 "patch_types": {"ymax": "symmetry"}},
        "physics": {"nu": 0.0025316456, "body_force": [1, 0, 0], "turbulence": "kOmegaSST"},
        "boundaries": {"xmin": {"type": "empty"}, "xmax": {"type": "empty"},
                       "zmin": {"type": "empty"}, "zmax": {"type": "empty"}},
        "initial": {"velocity": [15, 0, 0], "k": 1, "omega": 10},
        "solver": {"max_iterations": )" +
                             std::to_string(iterations) +
                             R"(, "tolerance": 1e-14, "alpha_u": 0.95, "alpha_p": 0.5},
        "output": {"directory": ")" +
                             out + R"("}
    })";
    return parse_case(json, base);
}

int count_lines(const std::filesystem::path& file) {
    std::ifstream in(file);
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        ++n;
    }
    return n;
}

} // namespace

TEST(CaseRunner, SteadyCavityWritesOutputs) {
    const auto dir = test_dir("steady");
    CaseRunner runner(cavity(12, 2000, 1e-5, "out", dir));
    const auto result = runner.run();
    EXPECT_TRUE(result.converged);
    EXPECT_FALSE(result.stopped);
    EXPECT_LT(result.steps, 2000);
    EXPECT_EQ(result.result_file, dir / "out" / "cavity.vtu");
    EXPECT_TRUE(std::filesystem::exists(dir / "out" / "cavity.vtu"));
    EXPECT_TRUE(std::filesystem::exists(dir / "out" / "checkpoint.bin"));
    EXPECT_EQ(count_lines(dir / "out" / "residuals.csv"), result.steps + 1);
}

TEST(CaseRunner, IntermediateResultsWritten) {
    const auto dir = test_dir("interval");
    CaseConfig config = cavity(8, 10, 1e-12, "out", dir);
    config.output.write_interval = 5;
    CaseRunner runner(config);
    (void)runner.run();
    EXPECT_TRUE(std::filesystem::exists(dir / "out" / "cavity_000005.vtu"));
    EXPECT_TRUE(std::filesystem::exists(dir / "out" / "cavity_000010.vtu"));
}

TEST(CaseRunner, LaminarRestartIsExact) {
    const auto dir = test_dir("restart_laminar");
    CaseRunner full(cavity(12, 40, 1e-12, "full", dir));
    (void)full.run();

    CaseRunner first(cavity(12, 20, 1e-12, "first", dir));
    (void)first.run();
    CaseConfig second_config = cavity(12, 40, 1e-12, "second", dir);
    second_config.restart = "first/checkpoint.bin";
    CaseRunner second(second_config);
    EXPECT_EQ(second.step(), 20);
    const auto result = second.run();
    EXPECT_EQ(result.steps, 40);

    const auto& a = full.flow().velocity();
    const auto& b = second.flow().velocity();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t c = 0; c < a.size(); ++c) {
        EXPECT_NEAR((a[c] - b[c]).norm(), 0.0, 1e-12);
        EXPECT_NEAR(full.flow().pressure()[c], second.flow().pressure()[c], 1e-12);
    }
}

TEST(CaseRunner, TurbulentRestartIsExact) {
    const auto dir = test_dir("restart_sst");
    CaseRunner full(channel_sst(60, "full", dir));
    (void)full.run();

    CaseRunner first(channel_sst(30, "first", dir));
    (void)first.run();
    CaseConfig second_config = channel_sst(60, "second", dir);
    second_config.restart = "first/checkpoint.bin";
    CaseRunner second(second_config);
    (void)second.run();

    ASSERT_NE(full.sst(), nullptr);
    ASSERT_NE(second.sst(), nullptr);
    for (std::size_t c = 0; c < full.flow().velocity().size(); ++c) {
        EXPECT_NEAR(full.flow().velocity()[c].x(), second.flow().velocity()[c].x(), 1e-10);
        EXPECT_NEAR(full.sst()->k()[c], second.sst()->k()[c], 1e-10);
        EXPECT_NEAR(full.sst()->nut()[c], second.sst()->nut()[c], 1e-12);
    }
}

TEST(CaseRunner, TransientRunReachesEndTime) {
    const auto dir = test_dir("transient");
    CaseConfig config = cavity(8, 100, 1e-6, "out", dir);
    config.solver.mode = cfd::cases::SolverMode::Transient;
    config.solver.dt = 0.01;
    config.solver.end_time = 0.1;
    CaseRunner runner(config);
    const auto result = runner.run();
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.steps, 10);
    EXPECT_NEAR(result.time, 0.1, 1e-12);
}

TEST(CaseRunner, ProgressCallbackCanStop) {
    const auto dir = test_dir("stop");
    CaseRunner runner(cavity(8, 1000, 1e-12, "out", dir));
    int calls = 0;
    const auto result = runner.run([&](int step, double, const cfd::fv::Residuals&) {
        ++calls;
        return step < 5;
    });
    EXPECT_TRUE(result.stopped);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.steps, 5);
    EXPECT_EQ(calls, 5);
    EXPECT_TRUE(std::filesystem::exists(result.result_file));
}

TEST(CaseRunner, CheckpointMeshMismatchThrows) {
    const auto dir = test_dir("mismatch");
    CaseRunner first(cavity(12, 2, 1e-12, "first", dir));
    (void)first.run();
    CaseConfig other = cavity(10, 2, 1e-12, "other", dir);
    other.restart = "first/checkpoint.bin";
    EXPECT_THROW(CaseRunner{other}, std::runtime_error);
}

TEST(CaseRunner, MissingGmshFileThrows) {
    const auto dir = test_dir("gmsh");
    const CaseConfig config =
        parse_case(R"({"mesh": {"type": "gmsh", "file": "missing.msh"}})", dir);
    EXPECT_THROW(CaseRunner{config}, std::runtime_error);
}
