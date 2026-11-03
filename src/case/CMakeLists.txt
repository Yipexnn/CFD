add_library(cfd_case STATIC
    src/case_config.cpp
    src/case_writer.cpp
    src/case_runner.cpp
)
add_library(cfd::case ALIAS cfd_case)

target_include_directories(cfd_case PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cfd_case PUBLIC cfd::fv PRIVATE nlohmann_json::nlohmann_json)

cfd_target_defaults(cfd_case)
