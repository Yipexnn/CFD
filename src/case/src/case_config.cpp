#include <cfd/case/case_config.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cfd::cases {

namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& path, const std::string& what) {
    throw std::runtime_error("Case file: " + path + ": " + what);
}

void check_keys(const json& j, const std::string& path, const std::set<std::string>& allowed) {
    if (!j.is_object()) {
        fail(path, "expected an object");
    }
    for (const auto& [key, value] : j.items()) {
        if (allowed.count(key) == 0) {
            std::string list;
            for (const auto& a : allowed) {
                list += (list.empty() ? "" : ", ") + a;
            }
            fail(path, "unknown key '" + key + "' (allowed: " + list + ")");
        }
    }
}

Scalar get_number(const json& j, const std::string& key, const std::string& path, Scalar def) {
    if (!j.contains(key)) {
        return def;
    }
    if (!j[key].is_number()) {
        fail(path + "." + key, "expected a number");
    }
    return j[key].get<Scalar>();
}

int get_int(const json& j, const std::string& key, const std::string& path, int def) {
    if (!j.contains(key)) {
        return def;
    }
    if (!j[key].is_number_integer()) {
        fail(path + "." + key, "expected an integer");
    }
    return j[key].get<int>();
}

std::string get_string(const json& j, const std::string& key, const std::string& path,
                       const std::string& def) {
    if (!j.contains(key)) {
        return def;
    }
    if (!j[key].is_string()) {
        fail(path + "." + key, "expected a string");
    }
    return j[key].get<std::string>();
}

Vec3 get_vec3(const json& j, const std::string& key, const std::string& path, const Vec3& def) {
    if (!j.contains(key)) {
        return def;
    }
    const json& v = j[key];
    if (!v.is_array() || v.size() != 3 || !v[0].is_number() || !v[1].is_number() ||
        !v[2].is_number()) {
        fail(path + "." + key, "expected an array of 3 numbers");
    }
    return {v[0].get<Scalar>(), v[1].get<Scalar>(), v[2].get<Scalar>()};
}

mesh::PatchType parse_patch_type(const std::string& s, const std::string& path) {
    if (s == "wall") {
        return mesh::PatchType::Wall;
    }
    if (s == "inlet") {
        return mesh::PatchType::Inlet;
    }
    if (s == "outlet") {
        return mesh::PatchType::Outlet;
    }
    if (s == "symmetry") {
        return mesh::PatchType::Symmetry;
    }
    if (s == "patch") {
        return mesh::PatchType::Patch;
    }
    fail(path, "unknown patch type '" + s + "' (wall, inlet, outlet, symmetry, patch)");
}

MeshConfig parse_mesh(const json& j) {
    const std::string path = "mesh";
    MeshConfig m;
    const std::string type = get_string(j, "type", path, "box");
    if (type == "gmsh") {
        check_keys(j, path, {"type", "file"});
        m.source = MeshSource::Gmsh;
        m.file = get_string(j, "file", path, "");
        if (m.file.empty()) {
            fail(path + ".file", "required for gmsh meshes");
        }
        return m;
    }
    if (type != "box") {
        fail(path + ".type", "unknown mesh type '" + type + "' (box, gmsh)");
    }
    check_keys(j, path, {"type", "cells", "min", "max", "grading", "patch_types"});
    m.source = MeshSource::Box;
    if (j.contains("cells")) {
        const json& c = j["cells"];
        if (!c.is_array() || c.size() != 3 || !c[0].is_number_integer() ||
            !c[1].is_number_integer() || !c[2].is_number_integer()) {
            fail(path + ".cells", "expected an array of 3 integers");
        }
        m.box.nx = c[0].get<int>();
        m.box.ny = c[1].get<int>();
        m.box.nz = c[2].get<int>();
    }
    m.box.min = get_vec3(j, "min", path, m.box.min);
    m.box.max = get_vec3(j, "max", path, m.box.max);
    m.box.grading = get_vec3(j, "grading", path, m.box.grading);
    if (j.contains("patch_types")) {
        const json& pt = j["patch_types"];
        if (!pt.is_object()) {
            fail(path + ".patch_types", "expected an object");
        }
        for (const auto& [name, value] : pt.items()) {
            std::size_t index = m.box.patch_names.size();
            for (std::size_t i = 0; i < m.box.patch_names.size(); ++i) {
                if (m.box.patch_names[i] == name) {
                    index = i;
                }
            }
            if (index == m.box.patch_names.size()) {
                fail(path + ".patch_types",
                     "unknown box patch '" + name + "' (xmin, xmax, ymin, ymax, zmin, zmax)");
            }
            if (!value.is_string()) {
                fail(path + ".patch_types." + name, "expected a string");
            }
            m.box.patch_types[index] =
                parse_patch_type(value.get<std::string>(), path + ".patch_types." + name);
        }
    }
    return m;
}

fv::BoundaryCondition parse_boundary(const json& j, const std::string& path) {
    check_keys(j, path,
               {"type", "velocity", "pressure", "turbulence_intensity", "viscosity_ratio"});
    fv::BoundaryCondition bc;
    const std::string type = get_string(j, "type", path, "");
    if (type == "velocity_inlet") {
        bc.kind = fv::FlowBoundary::VelocityInlet;
        if (!j.contains("velocity")) {
            fail(path + ".velocity", "required for velocity_inlet");
        }
    } else if (type == "wall") {
        bc.kind = fv::FlowBoundary::Wall;
    } else if (type == "pressure_outlet") {
        bc.kind = fv::FlowBoundary::PressureOutlet;
    } else if (type == "symmetry") {
        bc.kind = fv::FlowBoundary::Symmetry;
    } else if (type == "empty") {
        bc.kind = fv::FlowBoundary::Empty;
    } else {
        fail(path + ".type", "unknown boundary type '" + type +
                                 "' (velocity_inlet, wall, pressure_outlet, symmetry, empty)");
    }
    bc.velocity = get_vec3(j, "velocity", path, Vec3::Zero());
    bc.pressure = get_number(j, "pressure", path, 0.0);
    bc.turbulence_intensity = get_number(j, "turbulence_intensity", path, bc.turbulence_intensity);
    bc.viscosity_ratio = get_number(j, "viscosity_ratio", path, bc.viscosity_ratio);
    return bc;
}

} // namespace

CaseConfig parse_case(const std::string& json_text, const std::filesystem::path& base_directory) {
    json root;
    try {
        root = json::parse(json_text, nullptr, true, true); // allow comments
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("Case file: invalid JSON: ") + e.what());
    }
    check_keys(root, "case",
               {"name", "mesh", "physics", "boundaries", "initial", "solver", "output", "restart"});

    CaseConfig c;
    c.base_directory = base_directory;
    c.name = get_string(root, "name", "case", c.name);

    if (!root.contains("mesh")) {
        fail("case", "missing 'mesh'");
    }
    c.mesh = parse_mesh(root["mesh"]);

    fv::FlowSettings& flow = c.solver.flow;
    if (root.contains("physics")) {
        const json& p = root["physics"];
        check_keys(p, "physics", {"nu", "body_force", "turbulence"});
        flow.nu = get_number(p, "nu", "physics", flow.nu);
        if (!(flow.nu > 0.0)) {
            fail("physics.nu", "must be positive");
        }
        flow.body_force = get_vec3(p, "body_force", "physics", flow.body_force);
        const std::string turb = get_string(p, "turbulence", "physics", "laminar");
        if (turb == "laminar") {
            c.turbulence = TurbulenceKind::Laminar;
        } else if (turb == "kOmegaSST") {
            c.turbulence = TurbulenceKind::KOmegaSst;
        } else {
            fail("physics.turbulence", "unknown model '" + turb + "' (laminar, kOmegaSST)");
        }
    }

    if (root.contains("boundaries")) {
        const json& b = root["boundaries"];
        if (!b.is_object()) {
            fail("boundaries", "expected an object");
        }
        for (const auto& [name, value] : b.items()) {
            c.boundaries[name] = parse_boundary(value, "boundaries." + name);
        }
    }

    if (root.contains("initial")) {
        const json& i = root["initial"];
        check_keys(i, "initial", {"velocity", "k", "omega"});
        c.initial.velocity = get_vec3(i, "velocity", "initial", c.initial.velocity);
        c.initial.k = get_number(i, "k", "initial", c.initial.k);
        c.initial.omega = get_number(i, "omega", "initial", c.initial.omega);
    }

    if (root.contains("solver")) {
        const json& s = root["solver"];
        const std::string path = "solver";
        check_keys(s, path,
                   {"mode", "max_iterations", "tolerance", "dt", "end_time", "alpha_u", "alpha_p",
                    "convection", "non_orthogonal_correctors", "piso_correctors"});
        const std::string mode = get_string(s, "mode", path, "steady");
        if (mode == "steady") {
            c.solver.mode = SolverMode::Steady;
        } else if (mode == "transient") {
            c.solver.mode = SolverMode::Transient;
        } else {
            fail(path + ".mode", "unknown mode '" + mode + "' (steady, transient)");
        }
        c.solver.max_iterations = get_int(s, "max_iterations", path, c.solver.max_iterations);
        c.solver.tolerance = get_number(s, "tolerance", path, c.solver.tolerance);
        c.solver.dt = get_number(s, "dt", path, c.solver.dt);
        c.solver.end_time = get_number(s, "end_time", path, c.solver.end_time);
        flow.alpha_u = get_number(s, "alpha_u", path, flow.alpha_u);
        flow.alpha_p = get_number(s, "alpha_p", path, flow.alpha_p);
        const std::string conv = get_string(s, "convection", path, "linear_upwind");
        if (conv == "linear_upwind") {
            flow.convection = fv::ConvectionScheme::LinearUpwind;
        } else if (conv == "upwind") {
            flow.convection = fv::ConvectionScheme::Upwind;
        } else if (conv == "limited_linear_upwind") {
            flow.convection = fv::ConvectionScheme::LimitedLinearUpwind;
        } else {
            fail(path + ".convection",
                 "unknown scheme '" + conv + "' (upwind, linear_upwind, limited_linear_upwind)");
        }
        flow.n_non_orth_correctors =
            get_int(s, "non_orthogonal_correctors", path, flow.n_non_orth_correctors);
        flow.n_piso_correctors = get_int(s, "piso_correctors", path, flow.n_piso_correctors);
        if (c.solver.max_iterations < 1 || !(c.solver.dt > 0.0) || !(c.solver.end_time > 0.0) ||
            !(flow.alpha_u > 0.0 && flow.alpha_u <= 1.0) ||
            !(flow.alpha_p > 0.0 && flow.alpha_p <= 1.0) || flow.n_non_orth_correctors < 0 ||
            flow.n_piso_correctors < 1) {
            fail(path, "invalid value (iterations >= 1, dt/end_time > 0, 0 < alpha <= 1, "
                       "correctors >= 0 / >= 1)");
        }
    }

    if (root.contains("output")) {
        const json& o = root["output"];
        check_keys(o, "output",
                   {"directory", "write_interval", "log_interval", "checkpoint_interval"});
        c.output.directory = get_string(o, "directory", "output", c.output.directory.string());
        c.output.write_interval = get_int(o, "write_interval", "output", c.output.write_interval);
        c.output.log_interval = get_int(o, "log_interval", "output", c.output.log_interval);
        c.output.checkpoint_interval =
            get_int(o, "checkpoint_interval", "output", c.output.checkpoint_interval);
    }

    if (root.contains("restart")) {
        c.restart = std::filesystem::path(get_string(root, "restart", "case", ""));
    }
    return c;
}

CaseConfig load_case(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) {
        throw std::runtime_error("Case file: cannot open " + file.string());
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    CaseConfig c = parse_case(buffer.str(), file.parent_path());
    if (c.name == "case") {
        c.name = file.stem().string();
    }
    return c;
}

} // namespace cfd::cases
