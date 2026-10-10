#pragma once

#include <cuda_runtime.h>

#include <warpkep/cartesian_soa.hpp>
#include <warpkep/kepler_status.hpp>

#include <cstddef>

namespace warpkep{



    // Experimental FP64 batch interface

    // For n > 0:
    // - All arrays must be accessible on the selected CUDA device
    // - Every array must contain at least n elements
    // - Outputs must not overlap inputs or each other
    // - The caller owns the arrays and the stream
    // - Storage must remain valid until execution completes.

    // No allocation, transfer or synchronization is performed.
    // cudaSuccess does not mean that ouput data is ready
    // Execution errors must also be checked at completion

    // For n == 0, returns cudaSuccess without accessing pointers or submitting work; nullptrs are accepted

    cudaError_t launch_kepler_soa_double(
        CartesianSoAConstView<double> input,
        const double* duations,
        double mu,
        CartesianSoAView<double> output,
        KeplerPropagationStatus* statuses,
        int* iterations,
        std::size_t n,
        cudaStream_t stream
    );


    





} // namespace warpkep