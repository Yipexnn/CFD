find_package(GTest CONFIG REQUIRED)

add_executable(cfd_tests
    test_build_info.cpp
    test_dependencies.cpp
    test_mesh_builder.cpp
    test_box_mesher.cpp
    test_gmsh_reader.cpp
    test_ldu_matrix.cpp
    test_flow_validation.cpp
    test_turbulence.cpp
    test_scalar_transport.cpp
    test_case_config.cpp
    test_case_writer.cpp
    test_case_runner.cpp
    test_amg.cpp
    test_forces.cpp
    test_geometry.cpp
    test_case_geometry.cpp
    test_decomposition.cpp
    test_energy.cpp
    test_time_scheme.cpp
    test_conjugate.cpp
    test_multiphase.cpp
    test_compressible.cpp
    test_gpu.cpp
    test_properties.cpp
    test_radiation.cpp
)
target_link_libraries(cfd_tests PRIVATE cfd::core cfd::mesh cfd::fv cfd::parallel cfd::geometry cfd::case GTest::gtest_main)
cfd_target_defaults(cfd_tests)

include(GoogleTest)
gtest_discover_tests(cfd_tests DISCOVERY_MODE PRE_TEST)

if(CFD_BUILD_GUI)
    add_executable(cfd_gui_tests test_vtk_convert.cpp)
    target_link_libraries(cfd_gui_tests PRIVATE cfd::gui_core GTest::gtest_main)
    cfd_target_defaults(cfd_gui_tests)
    vtk_module_autoinit(TARGETS cfd_gui_tests MODULES ${VTK_LIBRARIES})
    gtest_discover_tests(cfd_gui_tests DISCOVERY_MODE PRE_TEST)

    find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
    add_executable(cfd_case_editor_tests test_case_editor.cpp ${CMAKE_SOURCE_DIR}/src/gui/case_editor.cpp)
    set_target_properties(cfd_case_editor_tests PROPERTIES AUTOMOC ON)
    target_include_directories(cfd_case_editor_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/gui)
    target_link_libraries(cfd_case_editor_tests PRIVATE cfd::case cfd::gui_core
                          Qt${QT_VERSION_MAJOR}::Widgets GTest::gtest_main)
    cfd_target_defaults(cfd_case_editor_tests)
    gtest_discover_tests(cfd_case_editor_tests DISCOVERY_MODE PRE_TEST)
endif()

# Parallel tests: run with mpiexec on 2 and 4 processes.
if(CFD_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_executable(cfd_mpi_tests mpi/main.cpp mpi/test_parallel_solver.cpp mpi/test_parallel_case.cpp)
    target_link_libraries(cfd_mpi_tests PRIVATE cfd::fv cfd::parallel cfd::case GTest::gtest)
    cfd_target_defaults(cfd_mpi_tests)
    foreach(n 2 4)
        add_test(NAME cfd_mpi_tests_np${n}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
                         $<TARGET_FILE:cfd_mpi_tests> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(cfd_mpi_tests_np${n} PROPERTIES TIMEOUT 300 LABELS mpi)
    endforeach()
endif()
