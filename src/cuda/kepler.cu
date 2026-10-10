#include <warpkep/cuda/kepler.hpp>

#include <warpkep/detail/kepler_cpu.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

namespace warpkep {

namespace {

namespace wd = ::warpkep::detail;

// One thread per trajectory: gather -> propagate -> scatter.
template <typename T>
__global__ void kepler_propagation_soa_kernel(
    CartesianSoAConstView<T> input,
    const T* durations,  // one duration per trajectory
    T mu,                // shared by all trajectories
    CartesianSoAView<T> output,
    KeplerPropagationStatus* statuses,
    int* iterations,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    // A. Gather the initial state of trajectory i.
    const wd::CartesianState<T> initial{
        {input.r_x[i], input.r_y[i], input.r_z[i]},
        {input.v_x[i], input.v_y[i], input.v_z[i]}
    };

    // B. Propagate (internal scalar code).
    const auto result = wd::propagate_kepler_cpu(initial, durations[i], mu);

    // C. Scatter the six output components.
    output.r_x[i] = result.state.position[0];
    output.r_y[i] = result.state.position[1];
    output.r_z[i] = result.state.position[2];
    output.v_x[i] = result.state.velocity[0];
    output.v_y[i] = result.state.velocity[1];
    output.v_z[i] = result.state.velocity[2];

    // D. Status and iteration count.
    statuses[i] = result.status;
    iterations[i] = result.iterations;
}

// Keep both precisions compiled; only FP64 is public for now.
template __global__ void kepler_propagation_soa_kernel<double>(
    CartesianSoAConstView<double>,
    const double*,
    double,
    CartesianSoAView<double>,
    KeplerPropagationStatus*,
    int*,
    std::size_t
);

template __global__ void kepler_propagation_soa_kernel<float>(
    CartesianSoAConstView<float>,
    const float*,
    float,
    CartesianSoAView<float>,
    KeplerPropagationStatus*,
    int*,
    std::size_t
);

} // namespace

// Contract documented in <warpkep/cuda/kepler.hpp>.
cudaError_t launch_kepler_soa_double(
    CartesianSoAConstView<double> input,
    const double* durations,
    double mu,
    CartesianSoAView<double> output,
    KeplerPropagationStatus* statuses,
    int* iterations,
    std::size_t n,
    cudaStream_t stream
) {
    // 1. Empty batch: no pointer accessed, no work submitted.
    if (n == 0) {
        return cudaSuccess;
    }

    // 2. Ceiling division without computing n + 255 (could overflow).
    constexpr unsigned int threads_per_block = 256;
    const std::size_t block_count =
        n / threads_per_block
        + (n % threads_per_block != 0);

    // 3. CUDA expects an unsigned int: check before converting.
    if (block_count > std::numeric_limits<unsigned int>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    const auto blocks = static_cast<unsigned int>(block_count);

    // 4. Submit to the caller's stream.
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

    // Returns and clears the CUDA runtime's last error on this host thread.
    // It may also report errors from earlier asynchronous work.
    // Successful return does not establish execution completion.
    return cudaGetLastError();
}

} // namespace warpkep
