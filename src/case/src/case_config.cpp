#include <algorithm>
#include <cfd/case/case_config.hpp>
#include <cfd/gpu/krylov.hpp>
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

bool get_bool(const json& j, const std::string& key, const std::string& path, bool def) {
    if (!j.contains(key)) {
        return def;
    }
    if (!j[key].is_boolean()) {
        fail(path + "." + key, "expected true or false");
    }
    return j[key].get<bool>();
}

geometry::Body parse_body(const json& j, const std::string& path,
                          std::vector<SolidConfig>& solids) {
    geometry::Body b;
    const std::string type = get_string(j, "type", path, "");
    if (type == "box") {
        check_keys(j, path, {"type", "min", "max", "patch", "solid"});
        b.kind = geometry::Body::Kind::Box;
        b.a = get_vec3(j, "min", path, b.a);
        b.b = get_vec3(j, "max", path, b.b);
    } else if (type == "cylinder") {
        check_keys(j, path, {"type", "base", "axis", "radius", "patch", "solid"});
        b.kind = geometry::Body::Kind::Cylinder;
        b.a = get_vec3(j, "base", path, b.a);
        b.b = get_vec3(j, "axis", path, Vec3(0, 0, 1));
        b.radius = get_number(j, "radius", path, 0.0);
    } else if (type == "sphere") {
        check_keys(j, path, {"type", "centre", "radius", "patch", "solid"});
        b.kind = geometry::Body::Kind::Sphere;
        b.a = get_vec3(j, "centre", path, b.a);
        b.radius = get_number(j, "radius", path, 0.0);
    } else if (type == "file") {
        check_keys(j, path, {"type", "file", "patch", "solid"});
        b.kind = geometry::Body::Kind::File;
        b.file = get_string(j, "file", path, "");
        if (b.file.empty()) {
            fail(path + ".file", "required for file bodies");
        }
    } else {
        fail(path + ".type", "unknown body type '" + type + "' (box, cylinder, sphere, file)");
    }
    if (b.kind != geometry::Body::Kind::Box && b.kind != geometry::Body::Kind::File &&
        !(b.radius > 0.0)) {
        fail(path + ".radius", "must be positive");
    }
    b.patch = get_string(j, "patch", path, b.patch);
    if (j.contains("solid")) {
        const json& sj = j["solid"];
        const std::string sp = path + ".solid";
        check_keys(sj, sp, {"conductivity", "rho_cp"});
        SolidConfig sc;
        sc.name = b.patch;
        sc.conductivity = get_number(sj, "conductivity", sp, sc.conductivity);
        sc.rho_cp = get_number(sj, "rho_cp", sp, sc.rho_cp);
        if (!(sc.conductivity > 0.0) || !(sc.rho_cp > 0.0)) {
            fail(sp, "conductivity and rho_cp must be positive");
        }
        b.solid = true;
        solids.push_back(sc);
    }
    return b;
}

geometry::GeometrySpec parse_geometry(const json& j, const std::string& path,
                                      std::vector<SolidConfig>& solids) {
    check_keys(j, path,
               {"type", "min", "max", "bodies", "internal", "two_d", "size", "size_near",
                "refine_distance", "refine_boxes", "boundary_layer", "quads", "domain_patches",
                "patches"});
    geometry::GeometrySpec g;
    g.domain_min = get_vec3(j, "min", path, g.domain_min);
    g.domain_max = get_vec3(j, "max", path, g.domain_max);
    if (j.contains("bodies")) {
        if (!j["bodies"].is_array()) {
            fail(path + ".bodies", "expected an array");
        }
        for (std::size_t i = 0; i < j["bodies"].size(); ++i) {
            g.bodies.push_back(
                parse_body(j["bodies"][i], path + ".bodies[" + std::to_string(i) + "]", solids));
        }
    }
    g.internal = get_bool(j, "internal", path, g.internal);
    g.two_d = get_bool(j, "two_d", path, g.two_d);
    g.size = get_number(j, "size", path, g.size);
    g.size_near = get_number(j, "size_near", path, g.size_near);
    g.refine_distance = get_number(j, "refine_distance", path, g.refine_distance);
    if (j.contains("refine_boxes")) {
        if (!j["refine_boxes"].is_array()) {
            fail(path + ".refine_boxes", "expected an array");
        }
        for (std::size_t i = 0; i < j["refine_boxes"].size(); ++i) {
            const json& b = j["refine_boxes"][i];
            const std::string bp = path + ".refine_boxes[" + std::to_string(i) + "]";
            check_keys(b, bp, {"min", "max", "size"});
            geometry::RefineBox box;
            box.min = get_vec3(b, "min", bp, box.min);
            box.max = get_vec3(b, "max", bp, box.max);
            box.size = get_number(b, "size", bp, box.size);
            if (!(box.size > 0.0) || (box.max - box.min).minCoeff() <= 0.0) {
                fail(bp, "needs size > 0 and max > min");
            }
            g.refine_boxes.push_back(box);
        }
    }
    g.quads = get_bool(j, "quads", path, g.quads);
    if (!(g.size > 0.0)) {
        fail(path + ".size", "must be positive");
    }
    if (j.contains("boundary_layer")) {
        const json& bl = j["boundary_layer"];
        const std::string blp = path + ".boundary_layer";
        check_keys(bl, blp, {"layers", "first", "ratio"});
        g.bl_layers = get_int(bl, "layers", blp, g.bl_layers);
        g.bl_first = get_number(bl, "first", blp, g.bl_first);
        g.bl_ratio = get_number(bl, "ratio", blp, g.bl_ratio);
    }
    if (j.contains("domain_patches")) {
        const json& dp = j["domain_patches"];
        const std::array<std::string, 6> faces{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        check_keys(dp, path + ".domain_patches", {faces.begin(), faces.end()});
        for (std::size_t i = 0; i < faces.size(); ++i) {
            g.domain_patch_names[i] =
                get_string(dp, faces[i], path + ".domain_patches", g.domain_patch_names[i]);
        }
    }
    if (j.contains("patches")) {
        if (!j["patches"].is_array()) {
            fail(path + ".patches", "expected an array");
        }
        for (std::size_t i = 0; i < j["patches"].size(); ++i) {
            const json& ps = j["patches"][i];
            const std::string pp = path + ".patches[" + std::to_string(i) + "]";
            check_keys(ps, pp, {"name", "min", "max"});
            geometry::PatchSelector sel;
            sel.name = get_string(ps, "name", pp, "");
            if (sel.name.empty()) {
                fail(pp + ".name", "required");
            }
            sel.min = get_vec3(ps, "min", pp, sel.min);
            sel.max = get_vec3(ps, "max", pp, sel.max);
            g.patches.push_back(sel);
        }
    }
    return g;
}

MeshConfig parse_mesh(const json& j, std::vector<SolidConfig>& solids) {
    const std::string path = "mesh";
    MeshConfig m;
    const std::string type = get_string(j, "type", path, "box");
    if (type == "geometry") {
        m.source = MeshSource::Geometry;
        m.geometry = parse_geometry(j, path, solids);
        return m;
    }
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
        fail(path + ".type", "unknown mesh type '" + type + "' (box, gmsh, geometry)");
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

// Temperature-dependent property: {"law": "constant" | "polynomial" | "sutherland" | "power",
// "reference", "coefficients", "t_ref", "s", "exponent"}.
fv::PropertyLaw parse_property_law(const json& j, const std::string& path, Scalar reference) {
    if (!j.is_object()) {
        fail(path, "expected an object with \"law\"");
    }
    check_keys(j, path, {"law", "reference", "coefficients", "t_ref", "s", "exponent"});
    fv::PropertyLaw law;
    law.reference = get_number(j, "reference", path, reference);
    law.t_ref = get_number(j, "t_ref", path, law.t_ref);
    law.s = get_number(j, "s", path, law.s);
    law.exponent = get_number(j, "exponent", path, law.exponent);
    const std::string kind = get_string(j, "law", path, "constant");
    if (kind == "constant") {
        law.kind = fv::PropertyLaw::Kind::Constant;
    } else if (kind == "polynomial") {
        law.kind = fv::PropertyLaw::Kind::Polynomial;
        if (!j.contains("coefficients") || !j["coefficients"].is_array() ||
            j["coefficients"].empty()) {
            fail(path + ".coefficients", "polynomial needs coefficients [c0, c1, ...]");
        }
        for (const auto& v : j["coefficients"]) {
            if (!v.is_number()) {
                fail(path + ".coefficients", "expected numbers");
            }
            law.coefficients.push_back(v.get<Scalar>());
        }
    } else if (kind == "sutherland") {
        law.kind = fv::PropertyLaw::Kind::Sutherland;
    } else if (kind == "power") {
        law.kind = fv::PropertyLaw::Kind::PowerLaw;
    } else {
        fail(path + ".law", "unknown law '" + kind + "' (constant, polynomial, sutherland, power)");
    }
    if (law.kind != fv::PropertyLaw::Kind::Polynomial && !(law.reference > 0.0)) {
        fail(path + ".reference", "must be positive");
    }
    if (!(law.t_ref > 0.0)) {
        fail(path + ".t_ref", "must be positive");
    }
    return law;
}

fv::BoundaryCondition parse_boundary(const json& j, const std::string& path) {
    check_keys(j, path,
               {"type", "velocity", "pressure", "turbulence_intensity", "viscosity_ratio",
                "profile", "profile_axes", "thermal", "alpha", "temperature", "contact_angle"});
    fv::BoundaryCondition bc;
    const std::string type = get_string(j, "type", path, "");
    if (type == "far_field") { // compressible: characteristic free-stream boundary
        bc.kind = fv::FlowBoundary::VelocityInlet;
        bc.far_field = true;
    } else if (type == "velocity_inlet") {
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
        fail(path + ".type",
             "unknown boundary type '" + type +
                 "' (velocity_inlet, far_field, wall, pressure_outlet, symmetry, empty)");
    }
    bc.velocity = get_vec3(j, "velocity", path, Vec3::Zero());
    bc.pressure = get_number(j, "pressure", path, 0.0);
    bc.turbulence_intensity = get_number(j, "turbulence_intensity", path, bc.turbulence_intensity);
    bc.viscosity_ratio = get_number(j, "viscosity_ratio", path, bc.viscosity_ratio);
    bc.temperature = get_number(j, "temperature", path, bc.temperature);
    bc.contact_angle = get_number(j, "contact_angle", path, bc.contact_angle);
    if (!(bc.contact_angle > 0.0 && bc.contact_angle < 180.0)) {
        fail(path + ".contact_angle", "must be between 0 and 180 degrees");
    }
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
               {"name", "mesh", "physics", "boundaries", "initial", "solver", "output", "forces",
                "heat", "solids", "restart", "decomposed_mesh"});

    CaseConfig c;
    c.base_directory = base_directory;
    c.decomposed_mesh = get_bool(root, "decomposed_mesh", "case", false);
    c.name = get_string(root, "name", "case", c.name);

    if (!root.contains("mesh")) {
        fail("case", "missing 'mesh'");
    }
    std::vector<SolidConfig> body_solids;
    c.mesh = parse_mesh(root["mesh"], body_solids);

    fv::FlowSettings& flow = c.solver.flow;
    if (root.contains("physics")) {
        const json& p = root["physics"];
        check_keys(p, "physics",
                   {"nu", "body_force", "turbulence", "energy", "multiphase", "compressible"});
        if (p.contains("compressible")) {
            const json& cp = p["compressible"];
            const std::string cpp = "physics.compressible";
            check_keys(cp, cpp,
                       {"gamma", "r_gas", "mu", "prandtl", "cfl", "second_order", "limiter",
                        "venkat_k", "implicit", "krylov_dim", "krylov_tolerance", "prandtl_t",
                        "viscosity"});
            fv::CompressibleSettings cs;
            cs.gamma = get_number(cp, "gamma", cpp, cs.gamma);
            cs.r_gas = get_number(cp, "r_gas", cpp, cs.r_gas);
            cs.mu = get_number(cp, "mu", cpp, cs.mu);
            cs.prandtl = get_number(cp, "prandtl", cpp, cs.prandtl);
            cs.cfl = get_number(cp, "cfl", cpp, cs.cfl);
            cs.second_order = get_bool(cp, "second_order", cpp, cs.second_order);
            const std::string limiter = get_string(cp, "limiter", cpp, "barthJespersen");
            if (limiter == "barthJespersen") {
                cs.limiter = fv::Limiter::BarthJespersen;
            } else if (limiter == "venkatakrishnan") {
                cs.limiter = fv::Limiter::Venkatakrishnan;
            } else {
                fail(cpp + ".limiter",
                     "unknown limiter '" + limiter + "' (barthJespersen, venkatakrishnan)");
            }
            cs.venkat_k = get_number(cp, "venkat_k", cpp, cs.venkat_k);
            cs.implicit = get_bool(cp, "implicit", cpp, cs.implicit);
            cs.krylov_dim = static_cast<int>(get_number(cp, "krylov_dim", cpp, cs.krylov_dim));
            cs.krylov_tolerance = get_number(cp, "krylov_tolerance", cpp, cs.krylov_tolerance);
            cs.prandtl_t = get_number(cp, "prandtl_t", cpp, cs.prandtl_t);
            if (cp.contains("viscosity")) {
                cs.mu_law = parse_property_law(cp["viscosity"], cpp + ".viscosity", cs.mu);
                if (!cs.mu_law.is_constant() && !(cs.mu > 0.0)) {
                    cs.mu = cs.mu_law.kind == fv::PropertyLaw::Kind::Polynomial
                                ? cs.mu_law(cs.mu_law.t_ref)
                                : cs.mu_law.reference;
                }
            }
            if (cs.krylov_dim < 0 || !(cs.krylov_tolerance > 0.0) || !(cs.prandtl_t > 0.0) ||
                !(cs.venkat_k > 0.0)) {
                fail(cpp, "needs krylov_dim >= 0, krylov_tolerance, prandtl_t, venkat_k > 0");
            }
            if (!(cs.gamma > 1.0 && cs.r_gas > 0.0 && cs.prandtl > 0.0 && cs.cfl > 0.0) ||
                cs.mu < 0.0) {
                fail(cpp, "needs gamma > 1, r_gas, prandtl, cfl > 0, mu >= 0");
            }
            c.compressible = cs;
        }
        if (p.contains("multiphase")) {
            const json& mp = p["multiphase"];
            const std::string mpp = "physics.multiphase";
            check_keys(mp, mpp, {"rho1", "nu1", "rho2", "nu2", "sigma", "gravity", "compression"});
            fv::MultiphaseSettings ms;
            ms.rho1 = get_number(mp, "rho1", mpp, ms.rho1);
            ms.nu1 = get_number(mp, "nu1", mpp, ms.nu1);
            ms.rho2 = get_number(mp, "rho2", mpp, ms.rho2);
            ms.nu2 = get_number(mp, "nu2", mpp, ms.nu2);
            ms.sigma = get_number(mp, "sigma", mpp, ms.sigma);
            ms.gravity = get_vec3(mp, "gravity", mpp, ms.gravity);
            ms.compression = get_number(mp, "compression", mpp, ms.compression);
            if (!(ms.rho1 > 0.0 && ms.rho2 > 0.0 && ms.nu1 > 0.0 && ms.nu2 > 0.0) ||
                ms.sigma < 0.0) {
                fail(mpp, "densities and viscosities must be positive, sigma >= 0");
            }
            c.multiphase = ms;
        }
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
        if (p.contains("energy")) {
            const json& e = p["energy"];
            const std::string ep = "physics.energy";
            check_keys(e, ep,
                       {"alpha", "prandtl", "prandtl_t", "beta", "t_ref", "gravity", "rho_cp",
                        "properties", "radiation"});
            fv::EnergySettings es;
            if (e.contains("alpha") && e.contains("prandtl")) {
                fail(ep, "give either alpha or prandtl");
            }
            es.alpha = e.contains("alpha") ? get_number(e, "alpha", ep, 0.0)
                                           : flow.nu / get_number(e, "prandtl", ep, 0.71);
            es.prandtl_t = get_number(e, "prandtl_t", ep, es.prandtl_t);
            es.beta = get_number(e, "beta", ep, es.beta);
            es.t_ref = get_number(e, "t_ref", ep, es.t_ref);
            es.gravity = get_vec3(e, "gravity", ep, es.gravity);
            es.rho_cp = get_number(e, "rho_cp", ep, es.rho_cp);
            if (!(es.alpha > 0.0) || !(es.prandtl_t > 0.0)) {
                fail(ep, "alpha / prandtl and prandtl_t must be positive");
            }
            if (e.contains("properties")) {
                const json& pr = e["properties"];
                const std::string pp = ep + ".properties";
                check_keys(pr, pp, {"nu", "alpha"});
                if (pr.contains("nu")) {
                    es.nu_law = parse_property_law(pr["nu"], pp + ".nu", flow.nu);
                }
                if (pr.contains("alpha")) {
                    es.alpha_law = parse_property_law(pr["alpha"], pp + ".alpha", es.alpha);
                }
            }
            if (e.contains("radiation")) {
                const json& r = e["radiation"];
                const std::string rp = ep + ".radiation";
                check_keys(r, rp,
                           {"model", "absorption", "scattering", "anisotropy", "emissivity",
                            "patch_emissivity"});
                if (get_string(r, "model", rp, "P1") != "P1") {
                    fail(rp + ".model", "only \"P1\" is available");
                }
                fv::RadiationSettings rs;
                rs.absorption = get_number(r, "absorption", rp, rs.absorption);
                rs.scattering = get_number(r, "scattering", rp, rs.scattering);
                rs.anisotropy = get_number(r, "anisotropy", rp, rs.anisotropy);
                rs.emissivity = get_number(r, "emissivity", rp, rs.emissivity);
                if (r.contains("patch_emissivity")) {
                    const json& pe = r["patch_emissivity"];
                    if (!pe.is_object()) {
                        fail(rp + ".patch_emissivity", "expected {\"patch\": emissivity}");
                    }
                    for (const auto& [name, v] : pe.items()) {
                        if (!v.is_number()) {
                            fail(rp + ".patch_emissivity." + name, "expected a number");
                        }
                        rs.patch_emissivity[name] = v.get<Scalar>();
                    }
                }
                const bool eps_ok =
                    std::all_of(rs.patch_emissivity.begin(), rs.patch_emissivity.end(),
                                [](const auto& kv) { return kv.second > 0.0 && kv.second <= 1.0; });
                if (!(rs.absorption > 0.0) || rs.scattering < 0.0 ||
                    std::abs(rs.anisotropy) > 1.0 ||
                    !(rs.emissivity > 0.0 && rs.emissivity <= 1.0) || !eps_ok) {
                    fail(rp, "needs absorption > 0, scattering >= 0, |anisotropy| <= 1, "
                             "emissivities in (0, 1]");
                }
                c.radiation = rs;
            }
            c.energy = es;
        }
    }

    if (root.contains("boundaries")) {
        const json& b = root["boundaries"];
        if (!b.is_object()) {
            fail("boundaries", "expected an object");
        }
        for (const auto& [name, value] : b.items()) {
            const std::string path = "boundaries." + name;
            if (value.is_object() && value.contains("thermal")) {
                const json& t = value["thermal"];
                const std::string tp = path + ".thermal";
                check_keys(t, tp, {"type", "value"});
                const std::string kind = get_string(t, "type", tp, "adiabatic");
                fv::ThermalBoundary tb;
                if (kind == "fixed") {
                    tb.kind = fv::ThermalBoundary::Kind::FixedTemperature;
                } else if (kind == "heat_flux") {
                    tb.kind = fv::ThermalBoundary::Kind::HeatFlux;
                } else if (kind != "adiabatic") {
                    fail(tp + ".type",
                         "unknown thermal type '" + kind + "' (fixed, heat_flux, adiabatic)");
                }
                tb.value = get_number(t, "value", tp, 0.0);
                c.thermal[name] = tb;
                if (value.size() == 1) {
                    continue; // thermal condition only: flow condition from the patch type
                }
            }
            c.boundaries[name] = parse_boundary(value, path);
            if (value.contains("alpha")) {
                c.inlet_alpha[name] = get_number(value, "alpha", path, 0.0);
            }
            const std::string profile = get_string(value, "profile", path, "uniform");
            if (profile == "parabolic") {
                if (c.boundaries[name].kind != fv::FlowBoundary::VelocityInlet) {
                    fail(path + ".profile", "only for velocity_inlet");
                }
                InletProfile ip;
                ip.axes = get_string(value, "profile_axes", path, ip.axes);
                if (ip.axes.empty() || ip.axes.find_first_not_of("xyz") != std::string::npos) {
                    fail(path + ".profile_axes", "expected axis letters, e.g. \"y\" or \"yz\"");
                }
                c.profiles[name] = ip;
            } else if (profile != "uniform") {
                fail(path + ".profile", "unknown profile '" + profile + "' (uniform, parabolic)");
            }
        }
    }

    if (root.contains("initial")) {
        const json& i = root["initial"];
        check_keys(
            i, "initial",
            {"velocity", "k", "omega", "temperature", "alpha_regions", "pressure", "regions"});
        c.initial.pressure = get_number(i, "pressure", "initial", c.initial.pressure);
        c.initial.temperature_gas =
            get_number(i, "temperature", "initial", c.initial.temperature_gas);
        if (i.contains("regions")) {
            if (!i["regions"].is_array()) {
                fail("initial.regions", "expected an array");
            }
            for (std::size_t k = 0; k < i["regions"].size(); ++k) {
                const json& r = i["regions"][k];
                const std::string rp = "initial.regions[" + std::to_string(k) + "]";
                check_keys(r, rp, {"min", "max", "velocity", "pressure", "temperature"});
                InitialConfig::GasRegion g;
                g.min = get_vec3(r, "min", rp, Vec3::Zero());
                g.max = get_vec3(r, "max", rp, Vec3::Zero());
                g.velocity = get_vec3(r, "velocity", rp, c.initial.velocity);
                g.pressure = get_number(r, "pressure", rp, c.initial.pressure);
                g.temperature = get_number(r, "temperature", rp, c.initial.temperature_gas);
                c.initial.regions.push_back(g);
            }
        }
        if (i.contains("alpha_regions")) {
            if (!i["alpha_regions"].is_array()) {
                fail("initial.alpha_regions", "expected an array of {min, max}");
            }
            for (std::size_t k = 0; k < i["alpha_regions"].size(); ++k) {
                const json& r = i["alpha_regions"][k];
                const std::string rp = "initial.alpha_regions[" + std::to_string(k) + "]";
                check_keys(r, rp, {"min", "max"});
                c.initial.alpha_regions.emplace_back(get_vec3(r, "min", rp, Vec3::Zero()),
                                                     get_vec3(r, "max", rp, Vec3::Zero()));
            }
        }
        if (i.contains("temperature")) {
            c.initial.temperature = get_number(i, "temperature", "initial", 0.0);
        }
        c.initial.velocity = get_vec3(i, "velocity", "initial", c.initial.velocity);
        c.initial.k = get_number(i, "k", "initial", c.initial.k);
        c.initial.omega = get_number(i, "omega", "initial", c.initial.omega);
    }

    if (root.contains("solver")) {
        const json& s = root["solver"];
        const std::string path = "solver";
        check_keys(s, path,
                   {"mode", "max_iterations", "tolerance", "dt", "end_time", "alpha_u", "alpha_p",
                    "convection", "non_orthogonal_correctors", "piso_correctors", "time_scheme",
                    "max_courant", "device", "gradient"});
        c.solver.max_courant = get_number(s, "max_courant", path, c.solver.max_courant);
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
        const std::string gradient = get_string(s, "gradient", path, "gauss");
        if (gradient == "gauss") {
            flow.gradient = fv::GradientScheme::Gauss;
        } else if (gradient == "least_squares") {
            flow.gradient = fv::GradientScheme::LeastSquares;
        } else {
            fail(path + ".gradient", "unknown scheme '" + gradient + "' (gauss, least_squares)");
        }
        const std::string device = get_string(s, "device", path, "auto");
        if (device == "auto") {
            c.solver.device = ComputeDevice::Auto;
        } else if (device == "cpu") {
            c.solver.device = ComputeDevice::Cpu;
        } else if (device == "gpu") {
            c.solver.device = ComputeDevice::Gpu;
        } else {
            fail(path + ".device", "unknown device '" + device + "' (auto, cpu, gpu)");
        }
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
        const std::string ts = get_string(s, "time_scheme", path, "euler");
        if (ts == "euler") {
            flow.time_scheme = fv::TimeScheme::Euler;
        } else if (ts == "backward") {
            flow.time_scheme = fv::TimeScheme::Backward;
        } else {
            fail(path + ".time_scheme", "unknown scheme '" + ts + "' (euler, backward)");
        }
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

    if (root.contains("forces")) {
        const json& f = root["forces"];
        check_keys(f, "forces", {"patches", "density", "reference_velocity", "reference_area"});
        if (!f.contains("patches") || !f["patches"].is_array()) {
            fail("forces.patches", "expected an array of patch names");
        }
        for (const auto& name : f["patches"]) {
            if (!name.is_string()) {
                fail("forces.patches", "expected an array of patch names");
            }
            c.forces.patches.push_back(name.get<std::string>());
        }
        c.forces.density = get_number(f, "density", "forces", c.forces.density);
        c.forces.reference_velocity =
            get_number(f, "reference_velocity", "forces", c.forces.reference_velocity);
        c.forces.reference_area =
            get_number(f, "reference_area", "forces", c.forces.reference_area);
        if (!(c.forces.density > 0.0) || !(c.forces.reference_velocity > 0.0) ||
            !(c.forces.reference_area > 0.0)) {
            fail("forces", "density and reference values must be positive");
        }
    }

    if (root.contains("heat")) {
        const json& h = root["heat"];
        check_keys(
            h, "heat",
            {"patches", "rho_cp", "reference_length", "reference_delta_t", "reference_area"});
        if (!h.contains("patches") || !h["patches"].is_array()) {
            fail("heat.patches", "expected an array of patch names");
        }
        for (const auto& name : h["patches"]) {
            if (!name.is_string()) {
                fail("heat.patches", "expected an array of patch names");
            }
            c.heat.patches.push_back(name.get<std::string>());
        }
        c.heat.rho_cp = get_number(h, "rho_cp", "heat", c.heat.rho_cp);
        c.heat.reference_length = get_number(h, "reference_length", "heat", 1.0);
        c.heat.reference_delta_t = get_number(h, "reference_delta_t", "heat", 1.0);
        c.heat.reference_area = get_number(h, "reference_area", "heat", 1.0);
    }
    if (root.contains("solids")) {
        const json& sl = root["solids"];
        if (!sl.is_array()) {
            fail("solids", "expected an array");
        }
        for (std::size_t i = 0; i < sl.size(); ++i) {
            const std::string sp = "solids[" + std::to_string(i) + "]";
            check_keys(sl[i], sp, {"name", "min", "max", "conductivity", "rho_cp"});
            SolidConfig sc;
            sc.name = get_string(sl[i], "name", sp, "");
            if (sc.name.empty() || sc.name == "fluid") {
                fail(sp + ".name", "required (and not 'fluid')");
            }
            if (sl[i].contains("min") != sl[i].contains("max")) {
                fail(sp, "give both min and max (box selection) or neither (mesh zone)");
            }
            if (sl[i].contains("min")) {
                sc.min = get_vec3(sl[i], "min", sp, Vec3::Zero());
                sc.max = get_vec3(sl[i], "max", sp, Vec3::Zero());
            }
            sc.conductivity = get_number(sl[i], "conductivity", sp, sc.conductivity);
            sc.rho_cp = get_number(sl[i], "rho_cp", sp, sc.rho_cp);
            if (!(sc.conductivity > 0.0) || !(sc.rho_cp > 0.0)) {
                fail(sp, "conductivity and rho_cp must be positive");
            }
            c.solids.push_back(sc);
        }
    }
    c.solids.insert(c.solids.end(), body_solids.begin(), body_solids.end());
    for (std::size_t i = 0; i < c.solids.size(); ++i) {
        for (std::size_t k = i + 1; k < c.solids.size(); ++k) {
            if (c.solids[i].name == c.solids[k].name) {
                fail("solids", "duplicate solid '" + c.solids[i].name + "'");
            }
        }
    }
    if (c.compressible) {
        c.initial.temperature.reset(); // the gas temperature (initial.temperature_gas)
    }
    if (!c.energy && (!c.solids.empty() || !c.thermal.empty() || !c.heat.patches.empty() ||
                      c.initial.temperature)) {
        fail("physics.energy", "thermal settings need the energy equation (physics.energy)");
    }

    if (!c.compressible) {
        for (const auto& [name, bc] : c.boundaries) {
            if (bc.far_field) {
                fail("boundaries." + name, "far_field needs physics.compressible");
            }
        }
    }
    if (c.compressible) {
        if (c.energy || !c.solids.empty() || c.multiphase) {
            fail("physics.compressible", "not combinable with energy, solids or multiphase");
        }
        c.compressible->turbulence = c.turbulence == TurbulenceKind::KOmegaSst;
        if (c.compressible->turbulence && !(c.compressible->mu > 0.0)) {
            fail("physics.compressible", "turbulence needs mu > 0");
        }
    }
    apply_device(c.solver);
    if (c.multiphase) {
        if (c.energy || !c.solids.empty()) {
            fail("physics.multiphase", "not combinable with energy or solids (yet)");
        }
        c.multiphase->turbulence = c.turbulence == TurbulenceKind::KOmegaSst;
        c.multiphase->p_solver.type = c.solver.flow.p_solver.type == fv::LinearSolverType::GpuCg
                                          ? fv::LinearSolverType::GpuCg
                                          : c.multiphase->p_solver.type;
        c.multiphase->u_solver.type = c.solver.flow.u_solver.type;
        c.multiphase->sst.k_init = c.initial.k;
        c.multiphase->sst.omega_init = c.initial.omega;
        if (c.solver.mode != SolverMode::Transient) {
            fail("physics.multiphase", "needs solver.mode \"transient\"");
        }
    } else if (!c.initial.alpha_regions.empty() || !c.inlet_alpha.empty()) {
        fail("physics.multiphase", "alpha settings need physics.multiphase");
    }

    if (root.contains("restart")) {
        c.restart = std::filesystem::path(get_string(root, "restart", "case", ""));
    }
    return c;
}

bool uses_gpu(ComputeDevice device) {
    switch (device) {
    case ComputeDevice::Gpu:
        return true;
    case ComputeDevice::Cpu:
        return false;
    case ComputeDevice::Auto:
        break;
    }
    return gpu::device_available();
}

void apply_device(SolverConfig& solver) {
    const bool gpu_solvers = uses_gpu(solver.device);
    solver.flow.p_solver.type =
        gpu_solvers ? fv::LinearSolverType::GpuCg : fv::LinearSolverType::AmgCg;
    solver.flow.u_solver.type =
        gpu_solvers ? fv::LinearSolverType::GpuBiCgStab : fv::LinearSolverType::BiCgStab;
}

const char* device_name(ComputeDevice device) {
    switch (device) {
    case ComputeDevice::Cpu:
        return "cpu";
    case ComputeDevice::Gpu:
        return "gpu";
    case ComputeDevice::Auto:
        break;
    }
    return "auto";
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
