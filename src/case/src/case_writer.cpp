#include <array>
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

ordered_json body_json(const geometry::Body& b) {
    ordered_json j = ordered_json::object();
    switch (b.kind) {
    case geometry::Body::Kind::Box:
        j["type"] = "box";
        j["min"] = vec3_json(b.a);
        j["max"] = vec3_json(b.b);
        break;
    case geometry::Body::Kind::Cylinder:
        j["type"] = "cylinder";
        j["base"] = vec3_json(b.a);
        j["axis"] = vec3_json(b.b);
        j["radius"] = b.radius;
        break;
    case geometry::Body::Kind::Sphere:
        j["type"] = "sphere";
        j["centre"] = vec3_json(b.a);
        j["radius"] = b.radius;
        break;
    case geometry::Body::Kind::File:
        j["type"] = "file";
        j["file"] = b.file.generic_string();
        break;
    }
    j["patch"] = b.patch;
    return j;
}

ordered_json geometry_json(const geometry::GeometrySpec& g) {
    ordered_json j = ordered_json::object();
    j["type"] = "geometry";
    j["min"] = vec3_json(g.domain_min);
    j["max"] = vec3_json(g.domain_max);
    ordered_json bodies = ordered_json::array();
    for (const auto& b : g.bodies) {
        bodies.push_back(body_json(b));
    }
    j["bodies"] = bodies;
    j["internal"] = g.internal;
    j["two_d"] = g.two_d;
    j["size"] = g.size;
    j["size_near"] = g.size_near;
    j["refine_distance"] = g.refine_distance;
    j["boundary_layer"] = {{"layers", g.bl_layers}, {"first", g.bl_first}, {"ratio", g.bl_ratio}};
    j["quads"] = g.quads;
    const std::array<const char*, 6> faces{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    ordered_json dp = ordered_json::object();
    for (std::size_t i = 0; i < faces.size(); ++i) {
        dp[faces[i]] = g.domain_patch_names[i];
    }
    j["domain_patches"] = dp;
    ordered_json patches = ordered_json::array();
    for (const auto& p : g.patches) {
        patches.push_back({{"name", p.name}, {"min", vec3_json(p.min)}, {"max", vec3_json(p.max)}});
    }
    j["patches"] = patches;
    return j;
}

ordered_json mesh_json(const MeshConfig& m) {
    ordered_json j = ordered_json::object();
    if (m.source == MeshSource::Geometry) {
        return geometry_json(m.geometry);
    }
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

ordered_json boundaries_json(const std::map<std::string, fv::BoundaryCondition>& boundaries,
                             const std::map<std::string, InletProfile>& profiles,
                             const std::map<std::string, fv::ThermalBoundary>& thermal) {
    ordered_json j = ordered_json::object();
    for (const auto& [name, bc] : boundaries) {
        ordered_json b = ordered_json::object();
        b["type"] = boundary_type_name(bc.kind);
        b["velocity"] = vec3_json(bc.velocity);
        b["pressure"] = bc.pressure;
        b["turbulence_intensity"] = bc.turbulence_intensity;
        b["viscosity_ratio"] = bc.viscosity_ratio;
        if (const auto it = profiles.find(name);
            it != profiles.end() && bc.kind == fv::FlowBoundary::VelocityInlet) {
            b["profile"] = "parabolic";
            b["profile_axes"] = it->second.axes;
        }
        j[name] = b;
    }
    for (const auto& [name, tb] : thermal) {
        ordered_json t = ordered_json::object();
        t["type"] = tb.kind == fv::ThermalBoundary::Kind::FixedTemperature ? "fixed"
                    : tb.kind == fv::ThermalBoundary::Kind::HeatFlux       ? "heat_flux"
                                                                           : "adiabatic";
        t["value"] = tb.value;
        if (!j.contains(name)) {
            j[name] = ordered_json::object();
        }
        j[name]["thermal"] = t;
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
    if (config.energy) {
        const fv::EnergySettings& e = *config.energy;
        ordered_json energy = ordered_json::object();
        energy["alpha"] = e.alpha;
        energy["prandtl_t"] = e.prandtl_t;
        energy["beta"] = e.beta;
        energy["t_ref"] = e.t_ref;
        energy["gravity"] = vec3_json(e.gravity);
        physics["energy"] = energy;
    }
    root["physics"] = physics;

    root["boundaries"] = boundaries_json(config.boundaries, config.profiles, config.thermal);

    ordered_json initial = ordered_json::object();
    initial["velocity"] = vec3_json(config.initial.velocity);
    initial["k"] = config.initial.k;
    initial["omega"] = config.initial.omega;
    if (config.initial.temperature) {
        initial["temperature"] = *config.initial.temperature;
    }
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

    if (!config.forces.patches.empty()) {
        ordered_json forces = ordered_json::object();
        forces["patches"] = config.forces.patches;
        forces["density"] = config.forces.density;
        forces["reference_velocity"] = config.forces.reference_velocity;
        forces["reference_area"] = config.forces.reference_area;
        root["forces"] = forces;
    }

    if (!config.heat.patches.empty()) {
        ordered_json heat = ordered_json::object();
        heat["patches"] = config.heat.patches;
        heat["rho_cp"] = config.heat.rho_cp;
        heat["reference_length"] = config.heat.reference_length;
        heat["reference_delta_t"] = config.heat.reference_delta_t;
        heat["reference_area"] = config.heat.reference_area;
        root["heat"] = heat;
    }

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
