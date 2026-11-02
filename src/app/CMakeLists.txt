add_executable(cfd main.cpp)
target_link_libraries(cfd PRIVATE cfd::core cfd::mesh cfd::fv cfd::case)
cfd_target_defaults(cfd)
