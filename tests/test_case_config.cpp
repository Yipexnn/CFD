#include <cfd/case/case_config.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace cc = cfd::cases;
namespace cfd_fv = cfd::fv;
namespace cfd_mesh = cfd::mesh;

namespace {

constexpr auto kCavityCase = R"({
    "name": "cavity",
    "mesh": {
        "type": "box",
        "cells": [16, 16, 1],
        "max": [1, 1, 0.0625],
        "patch_types": {"zmin": "symmetry", "zmax": "symmetry"}
    },
    "physics": {"nu": 0.01, "turbulence": "laminar"},
    "boundaries": {"ymax": {"type": "wall", "velocity": [1, 0, 0]}},
    "solver": {
        "mode": "steady",
        "max_iterations": 500,
        "tolerance": 1e-5,
        "alpha_u": 0.7,
        "alpha_p": 0.3
    },
    "output": {"write_interval": 100}
})";

} // namespace

TEST(CaseConfig, FullCavityCase) {
    const cc::CaseConfig c = cc::parse_case(kCavityCase, "/work/cavity");

    EXPECT_EQ(c.name, "cavity");
    EXPECT_EQ(c.base_directory, std::filesystem::path("/work/cavity"));
    EXPECT_EQ(c.mesh.source, cc::MeshSource::Box);
    EXPECT_EQ(c.mesh.box.nx, 16);
    EXPECT_EQ(c.mesh.box.ny, 16);
    EXPECT_EQ(c.mesh.box.nz, 1);
    EXPECT_DOUBLE_EQ(c.mesh.box.max[0], 1.0);
    EXPECT_DOUBLE_EQ(c.mesh.box.max[1], 1.0);
    EXPECT_DOUBLE_EQ(c.mesh.box.max[2], 0.0625);
    EXPECT_EQ(c.mesh.box.patch_types[4], cfd_mesh::PatchType::Symmetry);
    EXPECT_EQ(c.mesh.box.patch_types[5], cfd_mesh::PatchType::Symmetry);
    EXPECT_EQ(c.mesh.box.patch_types[3], cfd_mesh::PatchType::Wall);

    EXPECT_EQ(c.turbulence, cc::TurbulenceKind::Laminar);
    ASSERT_EQ(c.boundaries.count("ymax"), 1U);
    const cfd_fv::BoundaryCondition& ymax = c.boundaries.at("ymax");
    EXPECT_EQ(ymax.kind, cfd_fv::FlowBoundary::Wall);
    EXPECT_DOUBLE_EQ(ymax.velocity[0], 1.0);
    EXPECT_DOUBLE_EQ(ymax.velocity[1], 0.0);
    EXPECT_DOUBLE_EQ(ymax.velocity[2], 0.0);

    EXPECT_DOUBLE_EQ(c.solver.flow.nu, 0.01);
    EXPECT_EQ(c.solver.mode, cc::SolverMode::Steady);
    EXPECT_EQ(c.solver.max_iterations, 500);
    EXPECT_DOUBLE_EQ(c.solver.tolerance, 1e-5);
    EXPECT_DOUBLE_EQ(c.solver.flow.alpha_u, 0.7);
    EXPECT_DOUBLE_EQ(c.solver.flow.alpha_p, 0.3);

    EXPECT_EQ(c.output.write_interval, 100);
    EXPECT_FALSE(c.restart.has_value());
}

TEST(CaseConfig, Defaults) {
    const cc::CaseConfig c = cc::parse_case(R"({"mesh": {"type": "box"}})");

    EXPECT_EQ(c.name, "case");
    EXPECT_EQ(c.turbulence, cc::TurbulenceKind::Laminar);
    EXPECT_EQ(c.solver.mode, cc::SolverMode::Steady);
    EXPECT_EQ(c.solver.max_iterations, 1000);
    EXPECT_EQ(c.output.directory, std::filesystem::path("results"));
    EXPECT_FALSE(c.restart.has_value());
    EXPECT_TRUE(c.boundaries.empty());
}

TEST(CaseConfig, GmshMeshAndResolve) {
    const cc::CaseConfig c =
        cc::parse_case(R"({"mesh": {"type": "gmsh", "file": "m.msh"}})", "/tmp/x");

    EXPECT_EQ(c.mesh.source, cc::MeshSource::Gmsh);
    EXPECT_EQ(c.mesh.file, std::filesystem::path("m.msh"));
    EXPECT_EQ(c.resolve(c.mesh.file), std::filesystem::path("/tmp/x/m.msh"));
    EXPECT_EQ(c.resolve("/abs/m.msh"), std::filesystem::path("/abs/m.msh"));
}

TEST(CaseConfig, TurbulentInlet) {
    const cc::CaseConfig c = cc::parse_case(R"({
        "mesh": {"type": "box"},
        "physics": {"turbulence": "kOmegaSST"},
        "boundaries": {
            "inlet": {
                "type": "velocity_inlet",
                "velocity": [2, 0, 0],
                "turbulence_intensity": 0.1,
                "viscosity_ratio": 5
            }
        },
        "initial": {"k": 0.01, "omega": 10}
    })");

    EXPECT_EQ(c.turbulence, cc::TurbulenceKind::KOmegaSst);

    ASSERT_EQ(c.boundaries.count("inlet"), 1U);
    const cfd_fv::BoundaryCondition& inlet = c.boundaries.at("inlet");
    EXPECT_EQ(inlet.kind, cfd_fv::FlowBoundary::VelocityInlet);
    EXPECT_DOUBLE_EQ(inlet.velocity[0], 2.0);
    EXPECT_DOUBLE_EQ(inlet.velocity[1], 0.0);
    EXPECT_DOUBLE_EQ(inlet.velocity[2], 0.0);
    EXPECT_DOUBLE_EQ(inlet.turbulence_intensity, 0.1);
    EXPECT_DOUBLE_EQ(inlet.viscosity_ratio, 5.0);

    EXPECT_DOUBLE_EQ(c.initial.k, 0.01);
    EXPECT_DOUBLE_EQ(c.initial.omega, 10.0);
}

TEST(CaseConfig, TransientSettings) {
    const cc::CaseConfig c = cc::parse_case(R"({
        "mesh": {"type": "box"},
        "solver": {
            "mode": "transient",
            "dt": 0.01,
            "end_time": 2,
            "piso_correctors": 3,
            "convection": "upwind"
        }
    })");

    EXPECT_EQ(c.solver.mode, cc::SolverMode::Transient);
    EXPECT_DOUBLE_EQ(c.solver.dt, 0.01);
    EXPECT_DOUBLE_EQ(c.solver.end_time, 2.0);
    EXPECT_EQ(c.solver.flow.n_piso_correctors, 3);
    EXPECT_EQ(c.solver.flow.convection, cfd_fv::ConvectionScheme::Upwind);
}

TEST(CaseConfig, CommentsAllowed) {
    const cc::CaseConfig c = cc::parse_case(R"({
        // leading comment
        "name": "commented", // trailing comment
        "mesh": {"type": "box"}
    })");

    EXPECT_EQ(c.name, "commented");
    EXPECT_EQ(c.mesh.source, cc::MeshSource::Box);
}

TEST(CaseConfig, Errors) {
    EXPECT_THROW((void)cc::parse_case(R"({"mesh": )"), std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({"meshh": {"type": "box"}})"), std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({"name": "no_mesh"})"), std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({
        "mesh": {"type": "box"},
        "boundaries": {"inlet": {"type": "inflow"}}
    })"),
                 std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({
        "mesh": {"type": "box"},
        "boundaries": {"inlet": {"type": "velocity_inlet"}}
    })"),
                 std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({
        "mesh": {"type": "box"},
        "boundaries": {"inlet": {"type": "velocity_inlet", "velocity": [1, 0]}}
    })"),
                 std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({
        "mesh": {"type": "box"},
        "physics": {"nu": -0.1}
    })"),
                 std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({
        "mesh": {"type": "box"},
        "solver": {"alpha_u": 1.5}
    })"),
                 std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({
        "mesh": {"type": "box", "patch_types": {"left": "wall"}}
    })"),
                 std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({
        "mesh": {"type": "box"},
        "physics": {"turbulence": "kEpsilon"}
    })"),
                 std::runtime_error);

    EXPECT_THROW((void)cc::parse_case(R"({"mesh": {"type": "gmsh"}})"), std::runtime_error);
}

TEST(CaseConfig, ErrorMessageMentionsPath) {
    try {
        static_cast<void>(cc::parse_case(R"({
            "mesh": {"type": "box"},
            "boundaries": {"inlet": {"type": "inflow"}}
        })"));
        ADD_FAILURE() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("boundaries.inlet.type"), std::string::npos) << message;
    }
}
