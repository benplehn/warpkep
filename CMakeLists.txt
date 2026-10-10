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

option(WARPKEP_ENABLE_CUDA "Build the CUDA library" OFF)

option(
    WARPKEP_RUN_GPU_TESTS
    "Register GPU tests requiring an available CUDA device"
    OFF
)

if(WARPKEP_RUN_GPU_TESTS AND NOT WARPKEP_ENABLE_CUDA)
    message(FATAL_ERROR
        "WARPKEP_RUN_GPU_TESTS requires WARPKEP_ENABLE_CUDA=ON"
    )
endif()

if(WARPKEP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(warpkep_cuda STATIC
        src/cuda/kepler.cu
    )

    add_library(warpkep::cuda ALIAS warpkep_cuda)

    target_link_libraries(warpkep_cuda PUBLIC
        warpkep
        CUDA::cudart
    )

    set_target_properties(warpkep_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_SEPARABLE_COMPILATION OFF
    )

    # Compilation coverage of the internal host/device primitives.
    add_library(warpkep_cuda_compile_smoke OBJECT
        examples/cuda_smoke.cu
    )

    target_link_libraries(warpkep_cuda_compile_smoke PRIVATE
        warpkep
        CUDA::cudart
    )

    set_target_properties(warpkep_cuda_compile_smoke PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_RUNTIME_LIBRARY Shared
    )

    # An ordinary C++ consumer of the compiled CUDA library.
    add_executable(warpkep_gpu_smoke
        examples/kepler_gpu_smoke.cpp
    )

    target_link_libraries(warpkep_gpu_smoke PRIVATE
        warpkep::cuda
    )
endif()


option(WARPKEP_BUILD_BENCHMARKS "Build CPU benchmarks" OFF)

if(WARPKEP_BUILD_BENCHMARKS)
    find_package(Threads REQUIRED)

    add_executable(warpkep_cpu_baseline
        benchmarks/cpu_baseline.cpp
    )

    target_link_libraries(warpkep_cpu_baseline PRIVATE
        warpkep
        Threads::Threads
    )
endif()


include(CTest)

if(BUILD_TESTING)
    add_executable(warpkep_test_stumpff tests/test_stumpff.cpp)
    target_link_libraries(warpkep_test_stumpff PRIVATE warpkep)
    add_test(NAME stumpff COMMAND warpkep_test_stumpff)

    add_executable(warpkep_test_universal_kepler tests/test_universal_kepler.cpp)
    target_link_libraries(warpkep_test_universal_kepler PRIVATE warpkep)
    add_test(NAME universal_kepler_equation COMMAND warpkep_test_universal_kepler)

    add_executable(warpkep_test_kepler_solver tests/test_kepler_solver.cpp)
    target_link_libraries(warpkep_test_kepler_solver PRIVATE warpkep)
    add_test(NAME universal_kepler_solver COMMAND warpkep_test_kepler_solver)

    add_executable(warpkep_test_kepler_propagation tests/test_kepler_propagation.cpp)
    target_link_libraries(warpkep_test_kepler_propagation PRIVATE warpkep)
    add_test(NAME kepler_propagation COMMAND warpkep_test_kepler_propagation)


    add_executable(warpkep_test_kepler_precision tests/test_kepler_precision.cpp)
    target_link_libraries(warpkep_test_kepler_precision PRIVATE warpkep)
    add_test(NAME kepler_precision COMMAND warpkep_test_kepler_precision)

    add_executable(warpkep_cpu_samples validation/cpu_propagation_samples.cpp)
    target_link_libraries(warpkep_cpu_samples PRIVATE warpkep)
endif()


if(WARPKEP_RUN_GPU_TESTS)
    if(NOT BUILD_TESTING)
        message(FATAL_ERROR
            "WARPKEP_RUN_GPU_TESTS requires BUILD_TESTING=ON"
        )
    endif()

    add_test(
        NAME kepler_gpu_smoke
        COMMAND warpkep_gpu_smoke
    )

    set_tests_properties(kepler_gpu_smoke PROPERTIES
        LABELS "gpu"
    )
endif()


