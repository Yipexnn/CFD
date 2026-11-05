add_library(cfd_fv STATIC
    src/fv_geometry.cpp
    src/ldu_matrix.cpp
    src/amg.cpp
    src/flow_solver.cpp
    src/scalar_transport.cpp
    src/wall_distance.cpp
    src/k_omega_sst.cpp
    src/energy.cpp
    src/parallel_amg.cpp
)
add_library(cfd::fv ALIAS cfd_fv)

target_include_directories(cfd_fv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cfd_fv PUBLIC cfd::mesh cfd::parallel)

cfd_target_defaults(cfd_fv)
