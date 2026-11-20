#include <warpkep/lambert_soa.hpp>
#include <warpkep/detail/lambert_multi_rev_cpu.hpp>
#include <warpkep/cuda/lambert.hpp>

#include <cuda_runtime.h>
#include <cstddef>

#include <limits>


namespace warpkep{

namespace {

namespace wd = ::warpkep::detail;


__global__ void lambert_multi_rev_soa_kernel(
    Vector3SoAConstView<double> departures,
    Vector3SoAConstView<double> arrivals,
    const double* durations,
    double mu,
    wd::LambertArc arc,
    int revolutions,
    wd::LambertMultiRevOptions options,
    LambertMultiRevSoAView<double> output,
    std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    // A. Gather departure and arrival into two Vector3<double> values
    const wd::Vector3<double> departure{departures.x[i], departures.y[i], departures.z[i]};
    const wd::Vector3<double> arrival{arrivals.x[i], arrivals.y[i], arrivals.z[i]};

    // B. Solve this transfer with the shared scalar implementation
    auto result = wd::solve_lambert_multi_rev_cpu<double>(departure, arrival, durations[i], mu, arc, revolutions, options);

    // C. Preserve the request-level diagnostics and partial successes
    output.statuses[i] = result.status;
    output.slot_count[i] = result.slot_count;
    output.solution_count[i] = result.solution_count;
    output.complete[i] = result.complete ? 1 : 0;

    // D. Write both slots, explicitly initializing inactive slots
    const double nan = wd::NumericLimits<double>::quiet_NaN();
    for (int slot = 0; slot < 2; ++slot) {
        const bool active = slot < result.slot_count;

        // The inactive slot is never read from the scalar result.
        const auto solution = active ? result.solutions[slot].result : wd::lambert_failure<double>(LambertStatus::unsupported_case);
        const double x = active ? result.solutions[slot].x : nan;
        const auto& destination = output.solutions[slot];

        
        destination.departure_velocity.x[i] = solution.departure_velocity[0];
        destination.departure_velocity.y[i] = solution.departure_velocity[1];
        destination.departure_velocity.z[i] = solution.departure_velocity[2];

        destination.arrival_velocity.x[i] = solution.arrival_velocity[0];
        destination.arrival_velocity.y[i] = solution.arrival_velocity[1];
        destination.arrival_velocity.z[i] = solution.arrival_velocity[2];

        destination.x[i] = x;
        destination.statuses[i] = solution.status;
        destination.iterations[i] = solution.iterations;
    }
}

} // namespace




cudaError_t launch_lambert_multi_rev_soa_double(
    Vector3SoAConstView<double> departures,
    Vector3SoAConstView<double> arrivals,
    const double* durations,
    double mu,
    detail::LambertArc arc,
    int revolutions,
    detail::LambertMultiRevOptions options,
    LambertMultiRevSoAView<double> output,
    std::size_t n,
    cudaStream_t stream
) {
    // 1. Empty batch: no pointer accessed, no work submitted
    if (n == 0) {
        return cudaSuccess;
    }

    // 2. Ceiling division without computing n + 255 (could overflow)
    constexpr unsigned int threads_per_block = 256;
    // Ceiling division without overflowing n + threads_per_block - 1
    const std::size_t block_count = n / threads_per_block + (n % threads_per_block != 0);

    if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidConfiguration;
    }

    const auto blocks = static_cast<unsigned int>(block_count);

     // 4. Submit to the caller's stream
    lambert_multi_rev_soa_kernel<<<blocks, threads_per_block, 0, stream>>>(
        departures,
        arrivals,
        durations,
        mu,
        arc,
        revolutions,
        options,
        output,
        n
    );

    // Returns and clears the CUDA runtime's last error on this host thread.
    // It may also report errors from earlier asynchronous work.
    // Successful return does not establish execution completion.
    return cudaGetLastError();
}

} // namespace warpkep