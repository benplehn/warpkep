#pragma once

#include <cuda_runtime.h>
#include <warpkep/detail/cartesian_soa.hpp>
#include <warpkep/detail/kepler_cpu.hpp>

#include <cstddef>

cudaError_t launch_kepler_soa_double(
    warpkep::detail::CartesianSoAConstView<double> input,
    const double* durations,
    double mu,
    warpkep::detail::CartesianSoAView<double> output,
    warpkep::detail::KeplerPropagationStatus* statuses,
    int* iterations,
    std::size_t n,
    cudaStream_t stream
);