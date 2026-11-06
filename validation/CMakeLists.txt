add_executable(bfs_gartling bfs_gartling.cpp)
target_link_libraries(bfs_gartling PRIVATE cfd::fv)
cfd_target_defaults(bfs_gartling)

add_executable(cylinder_schaefer_turek cylinder_schaefer_turek.cpp)
target_link_libraries(cylinder_schaefer_turek PRIVATE cfd::fv cfd::geometry)
cfd_target_defaults(cylinder_schaefer_turek)

add_executable(cavity_de_vahl_davis cavity_de_vahl_davis.cpp)
target_link_libraries(cavity_de_vahl_davis PRIVATE cfd::fv)
cfd_target_defaults(cavity_de_vahl_davis)

add_executable(cylinder_schaefer_turek_2d2 cylinder_schaefer_turek_2d2.cpp)
target_link_libraries(cylinder_schaefer_turek_2d2 PRIVATE cfd::fv cfd::geometry)
cfd_target_defaults(cylinder_schaefer_turek_2d2)

add_executable(dam_break dam_break.cpp)
target_link_libraries(dam_break PRIVATE cfd::fv)
cfd_target_defaults(dam_break)

add_executable(sod_shock_tube sod_shock_tube.cpp)
target_link_libraries(sod_shock_tube PRIVATE cfd::fv)
cfd_target_defaults(sod_shock_tube)

add_executable(supersonic_ramp supersonic_ramp.cpp)
target_link_libraries(supersonic_ramp PRIVATE cfd::fv)
cfd_target_defaults(supersonic_ramp)

add_executable(flat_plate_turbulent flat_plate_turbulent.cpp)
target_link_libraries(flat_plate_turbulent PRIVATE cfd::fv)
cfd_target_defaults(flat_plate_turbulent)
