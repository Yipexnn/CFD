add_library(cfd_geometry STATIC
    src/geometry.cpp
)
add_library(cfd::geometry ALIAS cfd_geometry)

target_include_directories(cfd_geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cfd_geometry PUBLIC cfd::mesh)

cfd_target_defaults(cfd_geometry)
