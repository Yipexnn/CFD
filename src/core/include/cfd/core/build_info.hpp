#pragma once

#include <string>

namespace cfd::core {

struct BuildInfo {
    std::string version;
    std::string compiler;
    std::string build_type;
    bool openmp_enabled = false;
    int openmp_max_threads = 1;
    std::string eigen_version;
};

[[nodiscard]] BuildInfo build_info();

// Multi-line human-readable summary of build_info().
[[nodiscard]] std::string build_info_string();

} // namespace cfd::core
