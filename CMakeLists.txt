cmake_minimum_required(VERSION 3.10)

# Set naming scheme 
set(PROJECT_NAME nbc4gpu)
project(${PROJECT_NAME})

set(PROJECT_DESCRIPTION "naive bayes classifier for GPU")

set(EXECUTABLE ${PROJECT_NAME}-program)
set(PERFORMANCE ${PROJECT_NAME}-performance)
set(COMPARE nbc4cpu-performance)
set(TESTGPU ${PROJECT_NAME}-test)
set(TESTCPU nbc4cpu-test)

# Add custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules/finders")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

# Find external tools
find_package(Doxygen)

# NOTE : checking is set for individual targets
set(CLANG_TIDY_CHECKS "cppcoreguidelines-*,performance-*,portability-*,-clang-analyzer-*,readability-*,cert-*,bugprone-*")

# Add Clang Format
include(UseClangFormat)

# Add CTest
include(CTest)
enable_testing()

# Add external libs
find_package(BoostCompute REQUIRED)
find_package(Boost COMPONENTS program_options unit_test_framework REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)

set(EXTERNAL_LIBS "Boost::program_options" "Boost::unit_test_framework" "Threads::Threads" ${OpenCL_LIBRARY})


# CXX standard
set(CMAKE_CXX_STANDARD 17)

# Configure Toolchain
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
    if (USE_THREAD_CONTROL)
        set(CODE_CONTROL_FLAGS ${CODE_CONTROL_FLAGS} -Wthread-safety -fsanitize=thread)
    endif ()
    if (USE_CODE_CONTROL)
        set(CODE_CONTROL_FLAGS ${CODE_CONTROL_FLAGS} -fsanitize=memory -fsanitize=address -fsanitize=undefined)
    endif ()

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g ${CODE_CONTROL_FLAGS} -Wextra-tokens -Wambiguous-member-template -Wshadow -Wall -pedantic -Wextra -Wold-style-cast")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -finline-functions -fmerge-all-constants -fvectorize -flto")

    # find toolchain
    include(LLVMtoolchain)

    # configure toolchain
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld)

    set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_FINISH true)
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_FINISH true)

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_BUILD_TYPE EQUAL Release)
        cmake_policy(SET CMP0069 NEW)
        include(CheckIPOSupported)
        check_ipo_supported(IPO_SUPPORT)
        if (IPO_SUPPORT)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif ()
    endif ()

    # using GCC
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}   -O0 -Og -g -Wshadow -Wall -pedantic -Wextra -Wold-style-cast")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -finline-functions -fdevirtualize -fmerge-constants")
    # find and set toolchain
    include(GNUtoolchain)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

    # using MSVC (experimental)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /W2")
endif ()

message(STATUS "C Compiler executable: ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler executable: ${CMAKE_CXX_COMPILER}")
message(STATUS "Linker executable: ${CMAKE_LINKER}")

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})

# include dirs
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${BoostCompute_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(generator)
link_directories(${OpenCL_LIBRARY})

# find source files
file(GLOB_RECURSE LIB_INC ${PROJECT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE LIB_TMP ${PROJECT_SOURCE_DIR}/src/gpuclassifier/*.hpp)
file(GLOB_RECURSE CPU_TMP ${PROJECT_SOURCE_DIR}/src/cpuclassifier/*.hpp)
file(GLOB_RECURSE TEST_GPU_SRC ${PROJECT_SOURCE_DIR}/test/gpuTest/*.cpp)
file(GLOB_RECURSE TEST_CPU_SRC ${PROJECT_SOURCE_DIR}/test/cpuTest/*.cpp)

# Include src and test for library and test-check
add_subdirectory("test")

# Program executable
add_executable(${COMPARE} performanceTestCPU.cpp src/generator/Generator.hpp ${CPU_TMP})
add_executable(${PERFORMANCE} performanceTestGPU.cpp src/generator/Generator.hpp ${LIB_TMP})
target_link_libraries(${COMPARE} ${EXTERNAL_LIBS})
target_link_libraries(${PERFORMANCE} ${EXTERNAL_LIBS})

# Clang Format
clangformat_setup(${LIB_SRC} ${LIB_INC} ${TEST_GPU_SRC} ${TEST_CPU_SRC} ${LIB_TMP} ${CPU_TMP} src/generator/Generator.hpp performanceTestCPU.cpp performanceTestGPU.cpp)
add_custom_target(build-all DEPENDS format-project ${PROGRAM} ${LIB} ${CHECK} ${TESTGPU} ${TESTCPU})

# Doxygen
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/doc/doxygen)

    set(DOXYGEN_PROJECT_NAME ${PROJECT_NAME})
    set(DOXYGEN_PROJECT_BRIEF ${PROJECT_DESCRIPTION})

    set(DOXYGEN_HTML_COLORSTYLE_HUE 123)
    set(DOXYGEN_HTML_COLORSTYLE_SAT 69)
    set(DOXYGEN_HTML_COLORSTYLE_GAMMA 240)

    set(DOXYGEN_HTML_DYNAMIC_MENUS YES)
    set(DOXYGEN_HTML_DYNAMIC_SECTIONS YES)

    set(DOXYGEN_FILE_PATTERNS *.h *.c *.cpp *.hpp *.cc *.cxx *.hxx *.hh)
    set(DOXYGEN_EXTRACT_ALL YES)

    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_MARKDOWN_SUPPORT YES)

    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_MAN NO)

    doxygen_add_docs(${PROJECT_NAME}-doxygen ${LIB_SRC} ${LIB_INC} ${TEST_GPU_SRC} ${TEST_CPU_SRC} ${LIB_TMP} ${CPU_TMP})
endif()
