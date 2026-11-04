add_library(cfd_parallel STATIC
    src/comm.cpp
    src/decomposition.cpp
)
add_library(cfd::parallel ALIAS cfd_parallel)

target_include_directories(cfd_parallel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cfd_parallel PUBLIC cfd::mesh)
if(CFD_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_link_libraries(cfd_parallel PRIVATE MPI::MPI_CXX)
    target_compile_definitions(cfd_parallel PRIVATE CFD_HAS_MPI)
endif()

cfd_target_defaults(cfd_parallel)
