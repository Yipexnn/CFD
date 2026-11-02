#include <Eigen/Core>
#include <cfd/core/build_info.hpp>
#include <cfd/core/version.hpp>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef CFD_BUILD_TYPE
#define CFD_BUILD_TYPE "unknown"
#endif

namespace cfd::core {

namespace {

std::string compiler_string() {
#if defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) +
           "." + std::to_string(__clang_patchlevel__);
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_FULL_VER);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "unknown";
#endif
}

std::string eigen_string() {
#ifdef EIGEN_VERSION_STRING // Eigen >= 5
    return EIGEN_VERSION_STRING;
#else
    return std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
           std::to_string(EIGEN_MINOR_VERSION);
#endif
}

} // namespace

BuildInfo build_info() {
    BuildInfo info;
    info.version = std::string(k_version_string);
    info.compiler = compiler_string();
    info.build_type = CFD_BUILD_TYPE;
#ifdef _OPENMP
    info.openmp_enabled = true;
    info.openmp_max_threads = omp_get_max_threads();
#else
    info.openmp_enabled = false;
    info.openmp_max_threads = 1;
#endif
    info.eigen_version = eigen_string();
    return info;
}

std::string build_info_string() {
    const BuildInfo info = build_info();
    std::ostringstream os;
    os << "cfd " << info.version << '\n'
       << "Compiler:   " << info.compiler << '\n'
       << "Build type: " << info.build_type << '\n'
       << "OpenMP:     ";
    if (info.openmp_enabled) {
        os << "enabled (" << info.openmp_max_threads << " threads)";
    } else {
        os << "disabled";
    }
    os << '\n' << "Eigen:      " << info.eigen_version << '\n';
    return os.str();
}

} // namespace cfd::core
