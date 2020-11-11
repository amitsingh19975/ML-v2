cmake_minimum_required(VERSION 3.15)

# Setting project name 
project(dataframe CXX)
# Including cmake settings

include(cmake/VendorConfig.cmake)

include(cmake/StandardProjectSettings.cmake)
include(cmake/OpenMP.cmake)
include(cmake/Matplot.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${PROJECT_SOURCE_DIR}/include")

# Linking Library
add_library(project_options INTERFACE)

add_library(project_warnings INTERFACE)

include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# enable sanitizer options supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

include(cmake/Doxygen.cmake)
enable_doxygen()

include(cmake/StaticAnalyzers.cmake)

option(BUILD_SHARED_LIBS "Enable Compilation of shared libraries" OFF)
option(ENABLE_TESTING "Enable Test Builds" ON)

option(ENABLE_PCH "Enable Precompiled Header" OFF)
if (ENABLE_PCH)
    target_precompile_headers(project_options INTERFACE <iostream> <vector> <map> <utility>)
endif()

set(CONAN_EXTRA_REQUIRES "")
set(CONAN_EXTRA_OPTIONS "")

include(cmake/Conan.cmake)
run_conan()

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")

if (ENABLE_TESTING)
    enable_testing()
    message(
        "Building Tests."
    )
    add_subdirectory(test)
endif()

if (ENABLE_FUZZING)
    message(
        "Building Fuzz Tests."
    )
    add_subdirectory(fuzz_test)
endif()

add_subdirectory(src)
add_subdirectory(examples)
