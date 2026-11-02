find_package(GTest CONFIG REQUIRED)

add_executable(cfd_tests
    test_build_info.cpp
    test_dependencies.cpp
    test_mesh_builder.cpp
    test_box_mesher.cpp
    test_gmsh_reader.cpp
    test_ldu_matrix.cpp
    test_flow_validation.cpp
)
target_link_libraries(cfd_tests PRIVATE cfd::core cfd::mesh cfd::fv GTest::gtest_main)
cfd_target_defaults(cfd_tests)

include(GoogleTest)
gtest_discover_tests(cfd_tests DISCOVERY_MODE PRE_TEST)
