#include <cfd/case/case_config.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace cfd::cases {

namespace {

using nlohmann::ordered_json;

ordered_json vec3_json(const Vec3& v) {
    return ordered_json::array({v.x(), v.y(), v.z()});
}

const char* patch_type_name(mesh::PatchType t) {
    switch (t) {
    case mesh::PatchType::Wall:
        return "wall";
    case mesh::PatchType::Inlet:
        return "inlet";
    case mesh::PatchType::Outlet:
        return "outlet";
    case mesh::PatchType::Symmetry:
        return "symmetry";
    case mesh::PatchType::Patch:
        return "patch";
    }
    return "patch";
}

const char* boundary_type_name(fv::FlowBoundary kind) {
    switch (kind) {
    case fv::FlowBoundary::VelocityInlet:
        return "velocity_inlet";
    case fv::FlowBoundary::Wall:
        return "wall";
    case fv::FlowBoundary::PressureOutlet:
        return "pressure_outlet";
    case fv::FlowBoundary::Symmetry:
        return "symmetry";
    case fv::FlowBoundary::Empty:
        return "empty";
    }
    return "wall";
}

const char* convection_name(fv::ConvectionScheme s) {
    switch (s) {
    case fv::ConvectionScheme::Upwind:
        return "upwind";
    case fv::ConvectionScheme::LinearUpwind:
        return "linear_upwind";
    case fv::ConvectionScheme::LimitedLinearUpwind:
        return "limited_linear_upwind";
    }
    return "linear_upwind";
}

ordered_json mesh_json(const MeshConfig& m) {
    ordered_json j = ordered_json::object();
    if (m.source == MeshSource::Gmsh) {
        j["type"] = "gmsh";
        j["file"] = m.file.generic_string();
        return j;
    }
    const mesh::BoxSpec& b = m.box;
    j["type"] = "box";
    j["cells"] = ordered_json::array({b.nx, b.ny, b.nz});
    j["min"] = vec3_json(b.min);
    j["max"] = vec3_json(b.max);
    j["grading"] = vec3_json(b.grading);
    ordered_json patch_types = ordered_json::object();
    for (std::size_t i = 0; i < b.patch_names.size(); ++i) {
        patch_types[b.patch_names[i]] = patch_type_name(b.patch_types[i]);
    }
    j["patch_types"] = patch_types;
    return j;
}

ordered_json boundaries_json(const std::map<std::string, fv::BoundaryCondition>& boundaries) {
    ordered_json j = ordered_json::object();
    for (const auto& [name, bc] : boundaries) {
        ordered_json b = ordered_json::object();
        b["type"] = boundary_type_name(bc.kind);
        b["velocity"] = vec3_json(bc.velocity);
        b["pressure"] = bc.pressure;
        b["turbulence_intensity"] = bc.turbulence_intensity;
        b["viscosity_ratio"] = bc.viscosity_ratio;
        j[name] = b;
    }
    return j;
}

} // namespace

std::string case_to_json(const CaseConfig& config) {
    ordered_json root = ordered_json::object();
    root["name"] = config.name;
    root["mesh"] = mesh_json(config.mesh);

    const fv::FlowSettings& flow = config.solver.flow;
    ordered_json physics = ordered_json::object();
    physics["nu"] = flow.nu;
    physics["body_force"] = vec3_json(flow.body_force);
    physics["turbulence"] =
        config.turbulence == TurbulenceKind::KOmegaSst ? "kOmegaSST" : "laminar";
    root["physics"] = physics;

    root["boundaries"] = boundaries_json(config.boundaries);

    ordered_json initial = ordered_json::object();
    initial["velocity"] = vec3_json(config.initial.velocity);
    initial["k"] = config.initial.k;
    initial["omega"] = config.initial.omega;
    root["initial"] = initial;

    const SolverConfig& s = config.solver;
    ordered_json solver = ordered_json::object();
    solver["mode"] = s.mode == SolverMode::Transient ? "transient" : "steady";
    solver["max_iterations"] = s.max_iterations;
    solver["tolerance"] = s.tolerance;
    solver["dt"] = s.dt;
    solver["end_time"] = s.end_time;
    solver["alpha_u"] = flow.alpha_u;
    solver["alpha_p"] = flow.alpha_p;
    solver["convection"] = convection_name(flow.convection);
    solver["non_orthogonal_correctors"] = flow.n_non_orth_correctors;
    solver["piso_correctors"] = flow.n_piso_correctors;
    root["solver"] = solver;

    ordered_json output = ordered_json::object();
    output["directory"] = config.output.directory.generic_string();
    output["write_interval"] = config.output.write_interval;
    output["log_interval"] = config.output.log_interval;
    output["checkpoint_interval"] = config.output.checkpoint_interval;
    root["output"] = output;

    if (config.restart) {
        root["restart"] = config.restart->generic_string();
    }
    return root.dump(2);
}

void save_case(const CaseConfig& config, const std::filesystem::path& file) {
    std::ofstream out(file);
    if (!out) {
        throw std::runtime_error("Case file: cannot write " + file.string());
    }
    out << case_to_json(config) << '\n';
    out.flush();
    if (!out) {
        throw std::runtime_error("Case file: cannot write " + file.string());
    }
}

} // namespace cfd::cases
