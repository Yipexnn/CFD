cmake_minimum_required(VERSION 3.25)

project(cfd VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Use a preset: cmake --preset <name>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CFD_BUILD_APP "Build the cfd command-line application" ON)
option(CFD_BUILD_TESTS "Build the unit tests" ON)
option(CFD_ENABLE_OPENMP "Enable OpenMP parallelism" ON)
option(CFD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(CFD_BUILD_GUI "Build the Qt/VTK desktop application" OFF)
option(CFD_BUILD_VALIDATION "Build validation programs (slow benchmark cases)" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CfdTargetDefaults)

find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
if(CFD_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

add_subdirectory(src/core)
add_subdirectory(src/mesh)
add_subdirectory(src/fv)
add_subdirectory(src/geometry)
add_subdirectory(src/case)

if(CFD_BUILD_APP)
    add_subdirectory(src/app)
endif()

if(CFD_BUILD_GUI)
    add_subdirectory(src/gui)
endif()

if(CFD_BUILD_VALIDATION)
    add_subdirectory(validation)
endif()

if(CFD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
