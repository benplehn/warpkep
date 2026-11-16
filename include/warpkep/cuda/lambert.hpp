#pragma once


#include <warpkep/lambert_soa.hpp>
#include <warpkep/detail/lambert_multi_rev_cpu.hpp>

#include <cuda_runtime.h>
#include <cstddef>


namespace warpkep {


// Asynchronous FP64 launch; mu, arc, M and options are shared by the batch.
// n == 0: no pointers accessed, no work submitted.
// Otherwise, all arrays must contain at least n elements on the current GPU, including both output slots. Writes must not overlap inputs or other outputs.
// Buffers must remain valid until execution completes.
// Success reports submission only; numerical outcomes are in output.statuses
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
);


} // namespace warpkep