cmake_minimum_required(VERSION 3.24)

project(warpkep LANGUAGES CXX)

add_library(warpkep STATIC src/version.cpp)
set_target_properties(warpkep PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(warpkep PUBLIC include)
target_compile_features(warpkep PUBLIC cxx_std_17)

add_executable(warpkep_smoke src/main.cpp)
target_link_libraries(warpkep_smoke PRIVATE warpkep)

option(WARPKEP_BUILD_PYTHON "Build the Python extension" OFF)

if(WARPKEP_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED
        COMPONENTS Interpreter Development.Module
    )

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )

    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core bindings/module.cpp)
    target_link_libraries(_core PRIVATE warpkep)

    install(TARGETS _core LIBRARY DESTINATION warpkep)
endif()