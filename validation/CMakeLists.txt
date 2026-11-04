add_executable(bfs_gartling bfs_gartling.cpp)
target_link_libraries(bfs_gartling PRIVATE cfd::fv)
cfd_target_defaults(bfs_gartling)

add_executable(cylinder_schaefer_turek cylinder_schaefer_turek.cpp)
target_link_libraries(cylinder_schaefer_turek PRIVATE cfd::fv cfd::geometry)
cfd_target_defaults(cylinder_schaefer_turek)
