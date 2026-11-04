# Desktop GUI: Qt 6 (preferred) or Qt 5.15, plus VTK with Qt support.
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets OpenGL)
if(QT_VERSION_MAJOR EQUAL 6)
    find_package(Qt6 REQUIRED COMPONENTS OpenGLWidgets)
endif()
# Some distro VTK packages (e.g. Ubuntu 24.04) reference MPI targets in their exports.
enable_language(C)
find_package(MPI QUIET COMPONENTS C)
find_package(VTK REQUIRED COMPONENTS
    CommonCore CommonDataModel CommonExecutionModel
    FiltersCore FiltersGeometry FiltersSources FiltersFlowPaths
    RenderingCore RenderingOpenGL2 RenderingAnnotation RenderingFreeType
    InteractionStyle GUISupportQt IOImage
)
message(STATUS "GUI: Qt ${QT_VERSION}, VTK ${VTK_VERSION}")

# VTK conversion helpers (no Qt widgets): unit tested separately.
add_library(cfd_gui_core STATIC vtk_convert.cpp)
add_library(cfd::gui_core ALIAS cfd_gui_core)
target_include_directories(cfd_gui_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cfd_gui_core PUBLIC cfd::mesh ${VTK_LIBRARIES})
cfd_target_defaults(cfd_gui_core)

add_executable(cfd_gui WIN32
    main.cpp
    main_window.cpp
    case_editor.cpp
    post_panel.cpp
    render_view.cpp
    residual_plot.cpp
    solver_worker.cpp
    mpi_runner.cpp
)
set_target_properties(cfd_gui PROPERTIES AUTOMOC ON)
target_link_libraries(cfd_gui PRIVATE cfd::case cfd::gui_core Qt${QT_VERSION_MAJOR}::Widgets
                                      Qt${QT_VERSION_MAJOR}::OpenGL)
if(QT_VERSION_MAJOR EQUAL 6)
    target_link_libraries(cfd_gui PRIVATE Qt6::OpenGLWidgets)
endif()
cfd_target_defaults(cfd_gui)
# Parallel runs start the cfd command-line program; remember where it is built.
if(TARGET cfd)
    target_compile_definitions(cfd_gui PRIVATE CFD_CLI_PATH="$<TARGET_FILE:cfd>")
    add_dependencies(cfd_gui cfd)
endif()
vtk_module_autoinit(TARGETS cfd_gui cfd_gui_core MODULES ${VTK_LIBRARIES})
