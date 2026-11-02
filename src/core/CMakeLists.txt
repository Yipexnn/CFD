configure_file(
    include/cfd/core/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/include/cfd/core/version.hpp
    @ONLY
)

add_library(cfd_core STATIC
    src/build_info.cpp
)
add_library(cfd::core ALIAS cfd_core)

target_include_directories(cfd_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated/include
)

target_link_libraries(cfd_core PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)
if(CFD_ENABLE_OPENMP)
    target_link_libraries(cfd_core PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(cfd_core PRIVATE CFD_BUILD_TYPE="$<CONFIG>")

cfd_target_defaults(cfd_core)
