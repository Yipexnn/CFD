add_library(cfd_mesh STATIC
    src/mesh.cpp
    src/mesh_builder.cpp
    src/box_mesher.cpp
    src/gmsh_reader.cpp
    src/quality.cpp
)
add_library(cfd::mesh ALIAS cfd_mesh)

target_include_directories(cfd_mesh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cfd_mesh PUBLIC cfd::core)

cfd_target_defaults(cfd_mesh)
