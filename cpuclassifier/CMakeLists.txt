cmake_minimum_required(VERSION 3.5)

project(cpuclassifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(../generator)
file(GLOB SOURCES "src/*pp")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
add_executable(cpuclassifier ${SOURCES} ../generator/generator.hpp)
