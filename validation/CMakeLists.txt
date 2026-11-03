add_executable(bfs_gartling bfs_gartling.cpp)
target_link_libraries(bfs_gartling PRIVATE cfd::fv)
cfd_target_defaults(bfs_gartling)
