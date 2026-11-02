#include <cfd/core/build_info.hpp>
#include <cfd/core/version.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/mesh/gmsh_reader.hpp>
#include <cfd/mesh/mesh.hpp>
#include <cfd/mesh/quality.hpp>
#include <charconv>
#include <exception>
#include <iostream>
#include <string_view>
#include <system_error>

namespace {

void print_usage(std::ostream& os) {
    os << "Usage: cfd <command>\n"
       << "  --version                  Print version\n"
       << "  --info                     Print build information\n"
       << "  --help                     Show this help\n"
       << "  mesh-box <nx> <ny> <nz>    Create a unit box mesh and print statistics\n"
       << "  check-mesh <file.msh>      Read a Gmsh 4.1 mesh and print statistics\n";
}

// Parses a strictly positive integer. Returns false on non-numeric input or values < 1.
bool parse_positive(std::string_view text, cfd::mesh::Label& value) {
    cfd::mesh::Label parsed = 0;
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), last, parsed);
    if (ec != std::errc{} || ptr != last || parsed < 1) {
        return false;
    }
    value = parsed;
    return true;
}

void print_mesh_summary(const cfd::mesh::Mesh& mesh) {
    std::cout << "Points:          " << mesh.n_points() << "\n"
              << "Faces:           " << mesh.n_faces() << " (internal " << mesh.n_internal_faces()
              << ", boundary " << mesh.n_boundary_faces() << ")\n"
              << "Cells:           " << mesh.n_cells() << "\n"
              << "Patches:\n";
    for (const auto& patch : mesh.patches()) {
        std::cout << "  " << patch.name << " (" << cfd::mesh::to_string(patch.type)
                  << "): " << patch.n_faces << " faces\n";
    }
    std::cout << "\n" << cfd::mesh::to_string(cfd::mesh::compute_quality(mesh)) << "\n";
}

int run_mesh_box(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "cfd: error: mesh-box expects 3 arguments\n";
        print_usage(std::cerr);
        return 1;
    }

    cfd::mesh::BoxSpec spec;
    if (!parse_positive(argv[2], spec.nx) || !parse_positive(argv[3], spec.ny) ||
        !parse_positive(argv[4], spec.nz)) {
        std::cerr << "cfd: error: mesh-box dimensions must be integers >= 1\n";
        print_usage(std::cerr);
        return 1;
    }

    print_mesh_summary(cfd::mesh::make_box_mesh(spec));
    return 0;
}

int run_check_mesh(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "cfd: error: check-mesh expects 1 argument\n";
        print_usage(std::cerr);
        return 1;
    }

    print_mesh_summary(cfd::mesh::read_gmsh(argv[2]));
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(std::cout);
        return 0;
    }

    const std::string_view arg{argv[1]};

    if (arg == "--help" || arg == "-h") {
        print_usage(std::cout);
        return 0;
    }
    if (arg == "--version") {
        std::cout << "cfd " << cfd::core::k_version_string << "\n";
        return 0;
    }
    if (arg == "--info") {
        std::cout << cfd::core::build_info_string();
        return 0;
    }

    if (arg == "mesh-box" || arg == "check-mesh") {
        try {
            return arg == "mesh-box" ? run_mesh_box(argc, argv) : run_check_mesh(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "cfd: error: " << e.what() << "\n";
            return 1;
        }
    }

    std::cerr << "cfd: unknown argument '" << arg << "'\n";
    print_usage(std::cerr);
    return 1;
}
