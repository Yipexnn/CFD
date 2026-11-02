add_library(cfd_fv STATIC
    src/fv_geometry.cpp
    src/ldu_matrix.cpp
    src/flow_solver.cpp
)
add_library(cfd::fv ALIAS cfd_fv)

target_include_directories(cfd_fv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cfd_fv PUBLIC cfd::mesh)

cfd_target_defaults(cfd_fv)
