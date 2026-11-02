#include <cfd/core/build_info.hpp>
#include <cfd/core/version.hpp>
#include <iostream>
#include <string_view>

namespace {

void print_usage(std::ostream& os) {
    os << "Usage: cfd [--version | --info | --help]\n"
       << "  --version  Print version\n"
       << "  --info     Print build information\n"
       << "  --help     Show this help\n";
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

    std::cerr << "cfd: unknown argument '" << arg << "'\n";
    print_usage(std::cerr);
    return 1;
}
