#include <cuda_runtime.h>
#include <warpkep/detail/stumpff.hpp>
#include <warpkep/detail/solve_universal_kepler.hpp>
#include <warpkep/detail/kepler_cpu.hpp>

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
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
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
        results[i] = warpkep::detail::propagate_kepler_cpu(initial_states[i], durations[i], mu);
    }
}




