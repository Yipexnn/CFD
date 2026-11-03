#include <cfd/case/case_config.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <system_error>

namespace cc = cfd::cases;
namespace cfd_fv = cfd::fv;
namespace cfd_mesh = cfd::mesh;

namespace {

void expect_vec3_eq(const cfd::core::Vec3& a, const cfd::core::Vec3& b) {
    EXPECT_DOUBLE_EQ(a.x(), b.x());
    EXPECT_DOUBLE_EQ(a.y(), b.y());
    EXPECT_DOUBLE_EQ(a.z(), b.z());
}

void expect_config_eq(const cc::CaseConfig& a, const cc::CaseConfig& b) {
    EXPECT_EQ(a.name, b.name);

    EXPECT_EQ(a.mesh.source, b.mesh.source);
    EXPECT_EQ(a.mesh.file, b.mesh.file);
    EXPECT_EQ(a.mesh.box.nx, b.mesh.box.nx);
    EXPECT_EQ(a.mesh.box.ny, b.mesh.box.ny);
    EXPECT_EQ(a.mesh.box.nz, b.mesh.box.nz);
    expect_vec3_eq(a.mesh.box.min, b.mesh.box.min);
    expect_vec3_eq(a.mesh.box.max, b.mesh.box.max);
    expect_vec3_eq(a.mesh.box.grading, b.mesh.box.grading);
    EXPECT_EQ(a.mesh.box.patch_names, b.mesh.box.patch_names);
    EXPECT_EQ(a.mesh.box.patch_types, b.mesh.box.patch_types);

    EXPECT_EQ(a.turbulence, b.turbulence);

    ASSERT_EQ(a.boundaries.size(), b.boundaries.size());
    for (const auto& [name, bc_a] : a.boundaries) {
        const auto it = b.boundaries.find(name);
        ASSERT_NE(it, b.boundaries.end()) << name;
        const cfd_fv::BoundaryCondition& bc_b = it->second;
        EXPECT_EQ(bc_a.kind, bc_b.kind) << name;
        expect_vec3_eq(bc_a.velocity, bc_b.velocity);
        EXPECT_DOUBLE_EQ(bc_a.pressure, bc_b.pressure) << name;
        EXPECT_DOUBLE_EQ(bc_a.turbulence_intensity, bc_b.turbulence_intensity) << name;
        EXPECT_DOUBLE_EQ(bc_a.viscosity_ratio, bc_b.viscosity_ratio) << name;
    }

    expect_vec3_eq(a.initial.velocity, b.initial.velocity);
    EXPECT_DOUBLE_EQ(a.initial.k, b.initial.k);
    EXPECT_DOUBLE_EQ(a.initial.omega, b.initial.omega);

    EXPECT_EQ(a.solver.mode, b.solver.mode);
    EXPECT_EQ(a.solver.max_iterations, b.solver.max_iterations);
    EXPECT_DOUBLE_EQ(a.solver.tolerance, b.solver.tolerance);
    EXPECT_DOUBLE_EQ(a.solver.dt, b.solver.dt);
    EXPECT_DOUBLE_EQ(a.solver.end_time, b.solver.end_time);
    EXPECT_DOUBLE_EQ(a.solver.flow.nu, b.solver.flow.nu);
    expect_vec3_eq(a.solver.flow.body_force, b.solver.flow.body_force);
    EXPECT_EQ(a.solver.flow.convection, b.solver.flow.convection);
    EXPECT_DOUBLE_EQ(a.solver.flow.alpha_u, b.solver.flow.alpha_u);
    EXPECT_DOUBLE_EQ(a.solver.flow.alpha_p, b.solver.flow.alpha_p);
    EXPECT_EQ(a.solver.flow.n_non_orth_correctors, b.solver.flow.n_non_orth_correctors);
    EXPECT_EQ(a.solver.flow.n_piso_correctors, b.solver.flow.n_piso_correctors);

    EXPECT_EQ(a.output.directory, b.output.directory);
    EXPECT_EQ(a.output.write_interval, b.output.write_interval);
    EXPECT_EQ(a.output.log_interval, b.output.log_interval);
    EXPECT_EQ(a.output.checkpoint_interval, b.output.checkpoint_interval);

    ASSERT_EQ(a.restart.has_value(), b.restart.has_value());
    if (a.restart) {
        EXPECT_EQ(*a.restart, *b.restart);
    }
}

cc::CaseConfig populated_box_case() {
    cc::CaseConfig c;
    c.name = "roundtrip";
    c.mesh.source = cc::MeshSource::Box;
    c.mesh.box.nx = 12;
    c.mesh.box.ny = 8;
    c.mesh.box.nz = 4;
    c.mesh.box.min = {0.0, -1.0, 0.5};
    c.mesh.box.max = {2.0, 1.0, 3.5};
    c.mesh.box.grading = {1.0, 2.0, 0.5};
    c.mesh.box.patch_types = {cfd_mesh::PatchType::Inlet, cfd_mesh::PatchType::Outlet,
                              cfd_mesh::PatchType::Wall,  cfd_mesh::PatchType::Symmetry,
                              cfd_mesh::PatchType::Patch, cfd_mesh::PatchType::Wall};
    c.turbulence = cc::TurbulenceKind::KOmegaSst;

    cfd_fv::BoundaryCondition inlet;
    inlet.kind = cfd_fv::FlowBoundary::VelocityInlet;
    inlet.velocity = {1.5, 0.0, 0.0};
    inlet.turbulence_intensity = 0.03;
    inlet.viscosity_ratio = 5.0;
    c.boundaries["xmin"] = inlet;

    cfd_fv::BoundaryCondition outlet;
    outlet.kind = cfd_fv::FlowBoundary::PressureOutlet;
    outlet.pressure = 0.25;
    c.boundaries["xmax"] = outlet;

    c.initial.velocity = {0.1, 0.2, 0.0};
    c.initial.k = 0.01;
    c.initial.omega = 5.0;

    c.solver.mode = cc::SolverMode::Transient;
    c.solver.max_iterations = 250;
    c.solver.tolerance = 1e-7;
    c.solver.dt = 2e-4;
    c.solver.end_time = 0.5;
    c.solver.flow.nu = 2e-4;
    c.solver.flow.body_force = {0.0, 0.0, -9.81};
    c.solver.flow.convection = cfd_fv::ConvectionScheme::LimitedLinearUpwind;
    c.solver.flow.alpha_u = 0.8;
    c.solver.flow.alpha_p = 0.25;
    c.solver.flow.n_non_orth_correctors = 1;
    c.solver.flow.n_piso_correctors = 3;

    c.output.directory = "out/run1";
    c.output.write_interval = 20;
    c.output.log_interval = 5;
    c.output.checkpoint_interval = 100;

    c.restart = std::filesystem::path("checkpoints/state_50.json");
    return c;
}

} // namespace

TEST(CaseWriter, FullyPopulatedRoundTrip) {
    const cc::CaseConfig c = populated_box_case();
    const cc::CaseConfig back = cc::parse_case(cc::case_to_json(c));
    expect_config_eq(c, back);
}

TEST(CaseWriter, GmshMeshRoundTrip) {
    cc::CaseConfig c;
    c.name = "wing";
    c.mesh.source = cc::MeshSource::Gmsh;
    c.mesh.file = "meshes/wing.msh";
    c.turbulence = cc::TurbulenceKind::KOmegaSst;

    const cc::CaseConfig back = cc::parse_case(cc::case_to_json(c));
    EXPECT_EQ(back.mesh.source, cc::MeshSource::Gmsh);
    EXPECT_EQ(back.mesh.file, std::filesystem::path("meshes/wing.msh"));
    expect_config_eq(c, back);
}

TEST(CaseWriter, DefaultConfigRoundTrip) {
    const cc::CaseConfig c;
    expect_config_eq(c, cc::parse_case(cc::case_to_json(c)));
}

TEST(CaseWriter, SaveAndLoadFile) {
    const cc::CaseConfig c = populated_box_case();
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "cfd_case_writer_roundtrip.json";

    cc::save_case(c, file);
    const cc::CaseConfig back = cc::load_case(file);
    expect_config_eq(c, back);

    std::error_code ec;
    static_cast<void>(std::filesystem::remove(file, ec));
}

TEST(CaseWriter, SaveToMissingDirectoryThrows) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "cfd_case_writer_no_such_dir" / "case.json";
    EXPECT_THROW(cc::save_case(populated_box_case(), file), std::runtime_error);
}
