cmake_minimum_required(VERSION 3.20)
project(vol_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
    add_compile_options(/EHsc)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)

add_library(vol_engine STATIC engine.cpp)
set_target_properties(vol_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(vol_engine PRIVATE /W4 /fp:strict)
else()
    target_compile_options(vol_engine PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

# Normalize line endings so Git CRLF conversion does not alter the source ID.
set(SOURCE_TEXT "")
foreach(NAME engine.hpp engine.cpp bindings.cpp CMakeLists.txt build_info.hpp.in)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}" CONTENT)
    string(REPLACE "\r\n" "\n" CONTENT "${CONTENT}")
    string(APPEND SOURCE_TEXT "${NAME}\n${CONTENT}\n")
endforeach()
string(SHA256 VOL_SOURCE_SHA "${SOURCE_TEXT}")
configure_file(build_info.hpp.in build_info.hpp.configured @ONLY)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/build_info.hpp"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/build_info.hpp.configured")
pybind11_add_module(_vol_native bindings.cpp)
target_include_directories(_vol_native PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
target_link_libraries(_vol_native PRIVATE vol_engine)
# An optional local extension, never a prerequisite for installing the app.
get_filename_component(DEFAULT_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/../src" ABSOLUTE)
set(VOL_OUTPUT_DIR "${DEFAULT_OUTPUT}" CACHE PATH "Extension output directory")
set_target_properties(_vol_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${VOL_OUTPUT_DIR}/$<0:>"
    RUNTIME_OUTPUT_DIRECTORY "${VOL_OUTPUT_DIR}/$<0:>")

enable_testing()
add_executable(vol_engine_tests test_engine.cpp)
target_link_libraries(vol_engine_tests PRIVATE vol_engine)
add_test(NAME vol_engine_tests COMMAND vol_engine_tests)
