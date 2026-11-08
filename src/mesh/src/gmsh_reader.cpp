#include <algorithm>
#include <cctype>
#include <cfd/mesh/gmsh_reader.hpp>
#include <cfd/mesh/mesh_builder.hpp>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cfd::mesh {

namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("Gmsh reader: " + what);
}

template <typename T> T read(std::istream& in, const char* what) {
    T value{};
    if (!(in >> value)) {
        fail(std::string("unexpected end of input while reading ") + what);
    }
    return value;
}

void skip_section(std::istream& in, const std::string& name) {
    const std::string end = "$End" + name.substr(1);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind(end, 0) == 0) {
            return;
        }
    }
    fail("missing " + end);
}

void expect_end(std::istream& in, const std::string& end) {
    std::string token;
    if (!(in >> token) || token != end) {
        fail("expected " + end);
    }
}

std::optional<CellShape> volume_shape(int type) {
    switch (type) {
    case 4:
        return CellShape::Tet;
    case 5:
        return CellShape::Hex;
    case 6:
        return CellShape::Prism;
    case 7:
        return CellShape::Pyramid;
    default:
        return std::nullopt;
    }
}

int surface_nodes(int type) {
    switch (type) {
    case 2:
        return 3;
    case 3:
        return 4;
    default:
        return 0;
    }
}

struct RawData {
    std::map<std::pair<int, int>, std::string> physical_names; // (dim, tag) -> name
    std::unordered_map<int, int> surface_physical;             // surface entity -> phys tag
    std::unordered_map<int, int> volume_physical;              // volume entity -> phys tag
    std::vector<Vec3> points;
    std::unordered_map<long long, Label> node_index; // node tag -> point index
    struct Element {
        int dim = 0;
        int entity = 0;
        int type = 0;
        std::vector<Label> nodes;
    };
    std::vector<Element> elements;
};

void read_format(std::istream& in) {
    const auto version = read<std::string>(in, "format version");
    const auto file_type = read<int>(in, "file type");
    read<int>(in, "data size");
    if (version.rfind("4.1", 0) != 0) {
        fail("unsupported MSH version " + version + " (only 4.1 is supported)");
    }
    if (file_type != 0) {
        fail("binary MSH files are not supported (save as ASCII)");
    }
    expect_end(in, "$EndMeshFormat");
}

void read_physical_names(std::istream& in, RawData& data) {
    const auto n = read<int>(in, "number of physical names");
    for (int i = 0; i < n; ++i) {
        const auto dim = read<int>(in, "physical dimension");
        const auto tag = read<int>(in, "physical tag");
        std::string rest;
        std::getline(in, rest);
        const auto first = rest.find('"');
        const auto last = rest.rfind('"');
        if (first == std::string::npos || last == first) {
            fail("malformed physical name");
        }
        data.physical_names[{dim, tag}] = rest.substr(first + 1, last - first - 1);
    }
    expect_end(in, "$EndPhysicalNames");
}

void read_entities(std::istream& in, RawData& data) {
    const auto n_points = read<long long>(in, "number of point entities");
    const auto n_curves = read<long long>(in, "number of curve entities");
    const auto n_surfaces = read<long long>(in, "number of surface entities");
    const auto n_volumes = read<long long>(in, "number of volume entities");

    auto read_physicals = [&]() {
        const auto count = read<long long>(in, "number of physical tags");
        std::vector<int> tags;
        for (long long i = 0; i < count; ++i) {
            tags.push_back(read<int>(in, "physical tag"));
        }
        return tags;
    };
    auto skip_bounding = [&]() {
        const auto count = read<long long>(in, "number of bounding entities");
        for (long long i = 0; i < count; ++i) {
            read<int>(in, "bounding entity tag");
        }
    };

    for (long long i = 0; i < n_points; ++i) {
        read<int>(in, "point tag");
        for (int k = 0; k < 3; ++k) {
            read<double>(in, "point coordinate");
        }
        read_physicals();
    }
    for (long long i = 0; i < n_curves + n_surfaces + n_volumes; ++i) {
        const auto tag = read<int>(in, "entity tag");
        for (int k = 0; k < 6; ++k) {
            // As text: extruded / discrete entities can have +-1.797693134862316e+308 (rounded
            // DBL_MAX), which overflows when parsed as double. The box is not used.
            read<std::string>(in, "bounding box");
        }
        const auto phys = read_physicals();
        skip_bounding();
        const bool is_surface = i >= n_curves && i < n_curves + n_surfaces;
        if (is_surface && !phys.empty()) {
            data.surface_physical[tag] = phys.front();
        }
        if (i >= n_curves + n_surfaces && !phys.empty()) {
            data.volume_physical[tag] = phys.front();
        }
    }
    expect_end(in, "$EndEntities");
}

void read_nodes(std::istream& in, RawData& data) {
    const auto n_blocks = read<long long>(in, "number of node blocks");
    const auto n_nodes = read<long long>(in, "number of nodes");
    read<long long>(in, "min node tag");
    read<long long>(in, "max node tag");
    data.points.reserve(static_cast<std::size_t>(n_nodes));
    data.node_index.reserve(static_cast<std::size_t>(n_nodes));

    for (long long b = 0; b < n_blocks; ++b) {
        const auto dim = read<int>(in, "node block dimension");
        read<int>(in, "node block entity");
        const auto parametric = read<int>(in, "parametric flag");
        const auto count = read<long long>(in, "nodes in block");
        std::vector<long long> tags(static_cast<std::size_t>(count));
        for (auto& t : tags) {
            t = read<long long>(in, "node tag");
        }
        const int extra = parametric != 0 ? dim : 0;
        for (const long long t : tags) {
            Vec3 p;
            p.x() = read<double>(in, "node x");
            p.y() = read<double>(in, "node y");
            p.z() = read<double>(in, "node z");
            for (int k = 0; k < extra; ++k) {
                read<double>(in, "parametric coordinate");
            }
            data.node_index[t] = static_cast<Label>(data.points.size());
            data.points.push_back(p);
        }
    }
    expect_end(in, "$EndNodes");
}

void read_elements(std::istream& in, RawData& data) {
    const auto n_blocks = read<long long>(in, "number of element blocks");
    read<long long>(in, "number of elements");
    read<long long>(in, "min element tag");
    read<long long>(in, "max element tag");

    for (long long b = 0; b < n_blocks; ++b) {
        const auto dim = read<int>(in, "element block dimension");
        const auto entity = read<int>(in, "element block entity");
        const auto type = read<int>(in, "element type");
        const auto count = read<long long>(in, "elements in block");

        int nodes_per_element = 0;
        if (dim == 3) {
            const auto shape = volume_shape(type);
            if (!shape) {
                fail("unsupported 3D element type " + std::to_string(type) +
                     " (only first-order tet/hex/prism/pyramid)");
            }
            nodes_per_element = n_nodes(*shape);
        } else if (dim == 2) {
            nodes_per_element = surface_nodes(type);
            if (nodes_per_element == 0) {
                fail("unsupported 2D element type " + std::to_string(type) +
                     " (only first-order triangles/quads)");
            }
        }

        std::string line;
        std::getline(in, line); // rest of block header
        for (long long e = 0; e < count; ++e) {
            if (!std::getline(in, line)) {
                fail("unexpected end of input in element block");
            }
            if (dim < 2) {
                continue; // points and lines are not needed
            }
            std::istringstream ls(line);
            read<long long>(ls, "element tag");
            RawData::Element element{dim, entity, type, {}};
            element.nodes.reserve(static_cast<std::size_t>(nodes_per_element));
            for (int k = 0; k < nodes_per_element; ++k) {
                const auto tag = read<long long>(ls, "element node");
                const auto it = data.node_index.find(tag);
                if (it == data.node_index.end()) {
                    fail("element references unknown node " + std::to_string(tag));
                }
                element.nodes.push_back(it->second);
            }
            data.elements.push_back(std::move(element));
        }
    }
    expect_end(in, "$EndElements");
}

} // namespace

PatchType infer_patch_type(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    auto starts = [&](const char* prefix) { return lower.rfind(prefix, 0) == 0; };
    if (starts("inlet")) {
        return PatchType::Inlet;
    }
    if (starts("outlet")) {
        return PatchType::Outlet;
    }
    if (starts("wall")) {
        return PatchType::Wall;
    }
    if (starts("sym")) {
        return PatchType::Symmetry;
    }
    return PatchType::Patch;
}

Mesh read_gmsh(std::istream& in) {
    RawData data;
    bool have_format = false;
    bool have_nodes = false;
    bool have_elements = false;

    std::string token;
    while (in >> token) {
        if (token == "$MeshFormat") {
            read_format(in);
            have_format = true;
        } else if (token == "$PhysicalNames") {
            read_physical_names(in, data);
        } else if (token == "$Entities") {
            read_entities(in, data);
        } else if (token == "$Nodes") {
            read_nodes(in, data);
            have_nodes = true;
        } else if (token == "$Elements") {
            if (!have_nodes) {
                fail("$Elements before $Nodes");
            }
            read_elements(in, data);
            have_elements = true;
        } else if (!token.empty() && token[0] == '$') {
            skip_section(in, token);
        } else {
            fail("unexpected token '" + token + "'");
        }
    }
    if (!have_format || !have_nodes || !have_elements) {
        fail("missing $MeshFormat, $Nodes or $Elements section");
    }

    MeshBuilder builder(std::move(data.points));

    // Patches in order of physical tag for a deterministic layout.
    for (const auto& [key, name] : data.physical_names) {
        if (key.first == 2) {
            builder.add_patch(name, infer_patch_type(name));
        }
    }
    auto patch_name = [&](int entity) -> std::optional<std::string> {
        const auto it = data.surface_physical.find(entity);
        if (it == data.surface_physical.end()) {
            return std::nullopt;
        }
        const auto name = data.physical_names.find({2, it->second});
        return name != data.physical_names.end() ? name->second
                                                 : "physical_" + std::to_string(it->second);
    };

    // Cell zones from the physical volumes (in element order = cell order).
    std::vector<std::string> zone_names;
    std::map<std::string, int> zone_index;
    std::vector<int> zones;
    for (const auto& e : data.elements) {
        if (e.dim == 3) {
            builder.add_cell(*volume_shape(e.type), e.nodes);
            std::string zone = "default";
            if (const auto it = data.volume_physical.find(e.entity);
                it != data.volume_physical.end()) {
                const auto name = data.physical_names.find({3, it->second});
                zone = name != data.physical_names.end() ? name->second
                                                         : "physical_" + std::to_string(it->second);
            }
            const auto [pos, inserted] =
                zone_index.emplace(zone, static_cast<int>(zone_names.size()));
            if (inserted) {
                zone_names.push_back(zone);
            }
            zones.push_back(pos->second);
        }
    }
    if (builder.n_cells() == 0) {
        fail("no 3D elements found");
    }
    for (const auto& e : data.elements) {
        if (e.dim == 2) {
            if (const auto name = patch_name(e.entity)) {
                builder.add_boundary_face(e.nodes, *name);
            }
        }
    }
    Mesh mesh = builder.build();
    if (zone_names.size() > 1) {
        mesh.set_cell_zones(std::move(zone_names), std::move(zones));
    }
    return mesh;
}

Mesh read_gmsh(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) {
        fail("cannot open " + file.string());
    }
    return read_gmsh(in);
}

} // namespace cfd::mesh
