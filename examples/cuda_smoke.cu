#include <cuda_runtime.h>

#include <warpkep/detail/lambert_multi_rev_cpu.hpp>
#include <warpkep/detail/kepler_cpu.hpp>
#include <warpkep/detail/solve_universal_kepler.hpp>
#include <warpkep/detail/stumpff.hpp>

// Compile-only coverage of the internal host/device primitives.

__global__ void square_kernel(const double* input, double* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * input[i];
    }
}

__global__ void stumpff_double_kernel(
    const double* z,
    double* c,
    double* s,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const auto stumpff_values = warpkep::detail::stumpff(z[i]);
        c[i] = stumpff_values.c;
        s[i] = stumpff_values.s;
    }
}

__global__ void stumpff_float_kernel(
    const float* z,
    float* c,
    float* s,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const auto stumpff_values = warpkep::detail::stumpff(z[i]);
        c[i] = stumpff_values.c;
        s[i] = stumpff_values.s;
    }
}

__global__ void kepler_solver_double_kernel(
    const warpkep::detail::UniversalKeplerParameters<double>* parameters,
    warpkep::detail::KeplerSolveResult<double>* results,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        results[i] = warpkep::detail::solve_universal_kepler(parameters[i]);
    }
}

__global__ void kepler_solver_float_kernel(
    const warpkep::detail::UniversalKeplerParameters<float>* parameters,
    warpkep::detail::KeplerSolveResult<float>* results,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        results[i] = warpkep::detail::solve_universal_kepler(parameters[i]);
    }
}

__global__ void kepler_propagation_double_kernel(
    const warpkep::detail::CartesianState<double>* initial_states,
    const double* durations,
    double mu,
    warpkep::detail::KeplerPropagationResult<double>* results,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        results[i] = warpkep::detail::propagate_kepler_cpu(
            initial_states[i], durations[i], mu);
    }
}

__global__ void lambert_multi_rev_compile_kernel(
    const warpkep::detail::Vector3<double>* departures,
    const warpkep::detail::Vector3<double>* arrivals,
    const double* durations,
    double mu,
    int revolutions,
    warpkep::detail::LambertMultiRevResult<double>* results,
    std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    results[i] = warpkep::detail::solve_lambert_multi_rev_cpu(
        departures[i],
        arrivals[i],
        durations[i],
        mu,
        warpkep::detail::LambertArc::short_arc,
        revolutions);

}
