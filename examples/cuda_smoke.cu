#include <cuda_runtime.h>
#include <warpkep/detail/stumpff.hpp>
#include <warpkep/detail/solve_universal_kepler.hpp>
#include <warpkep/detail/kepler_cpu.hpp>
#include <warpkep/detail/cartesian_soa.hpp>
#include "kepler_launch.hpp"
#include <cstddef>
#include <limits>

namespace wd = warpkep::detail;


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


template <typename T>
__global__ void kepler_propagation_soa_kernel(
    wd::CartesianSoAConstView<T> input,
    const T* durations,                 // une durée par trajectoire : un tableau
    T mu,                               // le même mu pour toutes : un simple nombre
    wd::CartesianSoAView<T> output,
    wd::KeplerPropagationStatus* statuses,
    int* iterations,
    std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }
    // A. Reconstruite l etat initial de la trajectoire i.
    const wd::CartesianState<T> initial{
        {input.r_x[i], input.r_y[i], input.r_z[i]},
        {input.v_x[i], input.v_y[i], input.v_z[i]}
    };

    // B. Appel du propagateur
    const auto result = wd::propagate_kepler_cpu(initial, durations[i], mu);

    // C. Écrire les six composantes du résultat
    output.r_x[i] = result.state.position[0];
    output.r_y[i] = result.state.position[1];
    output.r_z[i] = result.state.position[2];
    output.v_x[i] = result.state.velocity[0];
    output.v_y[i] = result.state.velocity[1];
    output.v_z[i] = result.state.velocity[2];

    // D. Ecrire le statut et le nombre d'iterations
    statuses[i] = result.status;
    iterations[i] = result.iterations;
}


template __global__ void kepler_propagation_soa_kernel<double>(
    wd::CartesianSoAConstView<double>,
    const double*,
    double,
    wd::CartesianSoAView<double>,
    wd::KeplerPropagationStatus*,
    int*,
    std::size_t
);

template __global__ void kepler_propagation_soa_kernel<float>(
    wd::CartesianSoAConstView<float>,
    const float*,
    float,
    wd::CartesianSoAView<float>,
    wd::KeplerPropagationStatus*,
    int*,
    std::size_t
);



// Submits the SoA Kepler propagation kernel (FP64) to the caller's stream.
// The function is asynchronous: cudaSuccess means "work submitted",
// not "results ready". The caller must synchronize the stream before
// reading the outputs or reusing the arrays.
// All pointers must refer to GPU memory, hold at least n elements,
// and outputs must not overlap the inputs or each other.
cudaError_t launch_kepler_soa_double(
    wd::CartesianSoAConstView<double> input,
    const double* durations,
    double mu,
    wd::CartesianSoAView<double> output,
    wd::KeplerPropagationStatus* statuses,
    int* iterations,
    std::size_t n,
    cudaStream_t stream
) {
    // 1. Empty batch: nothing to launch, no array is touched.
    if (n == 0) {
        return cudaSuccess;
    }

    // 2. Grid size: one thread per trajectory, grouped in blocks of 256.
    // Ceiling division without computing n + 255, which could overflow.
    constexpr unsigned int threads_per_block = 256;
    const std::size_t block_count =
        n / threads_per_block
        + (n % threads_per_block != 0);

    // 3. CUDA expects an unsigned int block count: check before converting,
    // otherwise the conversion would silently truncate the value.
    if (block_count > std::numeric_limits<unsigned int>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    const auto blocks = static_cast<unsigned int>(block_count);

    // 4. Submit the kernel to the stream:
    // <<<blocks, threads per block, dynamic shared memory bytes, stream>>>.
    kepler_propagation_soa_kernel<double>
        <<<blocks, threads_per_block, 0, stream>>>(
            input,
            durations,
            mu,
            output,
            statuses,
            iterations,
            n
        );

    // 5. Report launch errors (invalid configuration, missing kernel, ...).
    // This also reads and clears any earlier asynchronous error on this thread.
    return cudaGetLastError();
}




