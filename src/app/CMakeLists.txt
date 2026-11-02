add_executable(cfd main.cpp)
target_link_libraries(cfd PRIVATE cfd::core cfd::mesh)
cfd_target_defaults(cfd)
