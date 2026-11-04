#pragma once

#include <array>
#include <cfd/mesh/mesh.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace cfd::geometry {

using core::Scalar;
using core::Vec3;

// Solid body inside the flow domain (subtracted from it) or, for internal flow, the
// fluid volume itself (Kind::File with GeometrySpec::internal).
struct Body {
    enum class Kind { Box, Cylinder, Sphere, File };
    Kind kind = Kind::Cylinder;
    Vec3 a = Vec3::Zero();          // Box: min corner; Cylinder: base centre; Sphere: centre
    Vec3 b = Vec3::Zero();          // Box: max corner; Cylinder: axis vector (length = height)
    Scalar radius = 0.0;            // Cylinder, Sphere
    std::filesystem::path file;     // File: .step/.stp (OpenCASCADE) or .stl (triangulation)
    std::string patch = "obstacle"; // boundary patch name of the body surface
};

// Named boundary patch selected by an axis-aligned box: every boundary surface entirely
// inside [min, max] belongs to the patch (overrides the default naming).
struct PatchSelector {
    std::string name;
    Vec3 min = Vec3::Zero();
    Vec3 max = Vec3::Zero();
};

// Geometry + meshing parameters for Gmsh.
//
// External flow (default): fluid = box [domain_min, domain_max] minus the bodies. The six
// box faces are named domain_patch_names (xmin, xmax, ymin, ymax, zmin, zmax), body
// surfaces by Body::patch.
// Internal flow (internal = true): fluid = the single File body (STEP solid); its surfaces
// are named "walls" unless a PatchSelector claims them.
// two_d: the geometry is built in the z = domain_min.z plane (bodies: Box -> rectangle,
// Cylinder -> disk using a and radius), meshed in 2D (optionally with quad boundary
// layers around the bodies) and extruded by one cell layer of thickness
// domain_max.z - domain_min.z (hexes / prisms), for use with "empty" z patches.
struct GeometrySpec {
    Vec3 domain_min = Vec3::Zero();
    Vec3 domain_max = Vec3::Ones();
    std::vector<Body> bodies;
    bool internal = false;
    bool two_d = false;

    Scalar size = 0.05;           // maximum cell size
    Scalar size_near = 0.0;       // cell size at body surfaces (<= 0: same as size)
    Scalar refine_distance = 0.0; // distance over which size_near grows to size

    int bl_layers = 0;     // boundary-layer cells on body surfaces (two_d only)
    Scalar bl_first = 0.0; // first boundary-layer cell height
    Scalar bl_ratio = 1.2; // growth ratio
    bool quads = true;     // two_d: recombine triangles into quads

    std::array<std::string, 6> domain_patch_names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    std::vector<PatchSelector> patches;
};

// Gmsh .geo script for the spec (meshing options, physical groups, MSH 4.1 ASCII output).
// Throws std::invalid_argument for inconsistent specs.
[[nodiscard]] std::string write_geo(const GeometrySpec& spec);

// Gmsh executable: $CFD_GMSH, else "gmsh" (gmsh.exe on Windows) from PATH. Empty if not found.
[[nodiscard]] std::string find_gmsh();

// Writes <work_dir>/geometry.geo, runs Gmsh and reads <work_dir>/geometry.msh.
// Throws std::runtime_error (with the Gmsh log tail) if meshing fails.
[[nodiscard]] mesh::Mesh generate_mesh(const GeometrySpec& spec,
                                       const std::filesystem::path& work_dir,
                                       const std::string& gmsh_executable = find_gmsh());

// Reads <work_dir>/geometry.msh written by generate_mesh (e.g. on other MPI processes)
// and applies the same patch types.
[[nodiscard]] mesh::Mesh read_generated_mesh(const GeometrySpec& spec,
                                             const std::filesystem::path& work_dir);

} // namespace cfd::geometry
