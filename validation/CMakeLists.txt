add_executable(bfs_gartling bfs_gartling.cpp)
target_link_libraries(bfs_gartling PRIVATE cfd::fv)
cfd_target_defaults(bfs_gartling)

add_executable(cylinder_schaefer_turek cylinder_schaefer_turek.cpp)
target_link_libraries(cylinder_schaefer_turek PRIVATE cfd::fv cfd::geometry)
cfd_target_defaults(cylinder_schaefer_turek)

add_executable(cavity_de_vahl_davis cavity_de_vahl_davis.cpp)
target_link_libraries(cavity_de_vahl_davis PRIVATE cfd::fv)
cfd_target_defaults(cavity_de_vahl_davis)
