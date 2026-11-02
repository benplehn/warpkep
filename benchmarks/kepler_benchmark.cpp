#include <cuda_runtime.h>
#include "kepler_workload.hpp"
#include <warpkep/cuda/kepler.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// RAII owners: each object acquires one CUDA resource in its constructor and
// releases it in its destructor. The destructor runs automatically when the
// object goes out of scope (closing brace, return or exception), so nothing
// can leak and nothing is freed twice.
// ---------------------------------------------------------------------------

// GPU memory (VRAM). The CPU may keep and pass the address, never read it.
template <typename T>
struct DeviceArray {
    T* ptr = nullptr;                  // GPU address, null until allocated
    cudaError_t error = cudaSuccess;   // allocation result, checked by require()

    // Acquire: reserve count elements of type T on the GPU.
    explicit DeviceArray(std::size_t count) {
        error = cudaMalloc(&ptr, count * sizeof(T));
    }

    // Release: runs automatically at end of scope.
    // Never throws; release errors are ignored here on purpose.
    ~DeviceArray() noexcept {
        if (ptr != nullptr) {          // nothing to free after a failed allocation
            cudaFree(ptr);
        }
    }

    // No copies: two copies would hold the same address and free it twice.
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// Pinned (page-locked) CPU memory: fixed in RAM, so the GPU can copy it
// directly over PCIe. Faster, truly asynchronous transfers.
// The CPU reads and writes it normally through ptr.
template <typename T>
struct PinnedArray {
    T* ptr = nullptr;                  // CPU address, usable on the CPU
    cudaError_t error = cudaSuccess;

    explicit PinnedArray(std::size_t count) {
        error = cudaMallocHost(&ptr, count * sizeof(T));
    }

    // Release: runs automatically at end of scope.
    // Never throws; release errors are ignored here on purpose.
    ~PinnedArray() noexcept {
        if (ptr != nullptr) {          // nothing to free after a failed allocation
            cudaFreeHost(ptr);
        }
    }

    PinnedArray(const PinnedArray&) = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;
};

// One CUDA stream: an ordered queue of GPU work (copies, kernels).
// Non-blocking: no implicit synchronization with the legacy default stream.
struct CudaStream {
    cudaStream_t handle = nullptr;
    cudaError_t error = cudaSuccess;

    CudaStream() {
        error = cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking);
    }

    ~CudaStream() {
        if (handle != nullptr) {
            cudaStreamDestroy(handle);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

// A CUDA error is not a measurement: it aborts the benchmark.
void require(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + " failed: " + cudaGetErrorString(error));
    }
}


int main() {

    try {

        
        const std::size_t n = 1000;
        constexpr double mu = 1.0;  // the shared workload is normalized with mu = 1


        // 1. Prepare 
        const auto inputs = warpkep_bench::make_inputs(n);

        PinnedArray<double> h_in(7 * n);   // [r_x|r_y|r_z|v_x|v_y|v_z|dt]
        PinnedArray<double> h_out(6 * n);  // [r_x|r_y|r_z|v_x|v_y|v_z]
        PinnedArray<warpkep::KeplerPropagationStatus> h_status(n);
        PinnedArray<int> h_iter(n);

        DeviceArray<double> d_in(7 * n);
        DeviceArray<double> d_out(6 * n);
        DeviceArray<warpkep::KeplerPropagationStatus> d_status(n);
        DeviceArray<int> d_iter(n);

        CudaStream stream;

        // 1a. Every allocation can fail: stop with a clear message.
        require(h_in.error, "cudaMallocHost (inputs and durations)");
        require(h_out.error, "cudaMallocHost (output states)");
        require(h_status.error, "cudaMallocHost (statuses)");
        require(h_iter.error, "cudaMallocHost (iterations)");

        require(d_in.error, "cudaMalloc (inputs and durations)");
        require(d_out.error, "cudaMalloc (output states)");
        require(d_status.error, "cudaMalloc (statuses)");
        require(d_iter.error, "cudaMalloc (iterations)");

        require(stream.error, "cudaStreamCreateWithFlags");


        // 1b. Pack AoS inputs into the 7-slice SoA buffer (CPU, pinned memory).
        //     Slice k starts at k * n; trajectory i sits at offset i in each slice.
        //     h_in: [ r_x | r_y | r_z | v_x | v_y | v_z | dt ]
        //            0*n   1*n   2*n   3*n   4*n   5*n   6*n
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < 3; ++k) {
                h_in.ptr[k * n + i] = inputs[i].state.position[k];        // slices 0, 1, 2
                h_in.ptr[(3 + k) * n + i] = inputs[i].state.velocity[k];  // slices 3, 4, 5
            }
            h_in.ptr[6 * n + i] = inputs[i].dt;                           // slice 6
        }


        // CPU reference on the same inputs (outside any timed region).
        std::vector<warpkep::detail::KeplerPropagationResult<double>> cpu(n);
        warpkep_bench::run_batch(inputs, cpu, 1);


        try {
            
            // 2. Transfer: one copy CPU -> GPU, queued in the stream.
            //    The CPU only queues the order and continues immediately.
            require(cudaMemcpyAsync(d_in.ptr, h_in.ptr, 7 * n * sizeof(double), cudaMemcpyHostToDevice, stream.handle), "cudaMemcpyAsync (inputs to device)");

            // 3a. Input view: six GPU addresses, one per component slice of d_in.
            //     Pure address arithmetic on the CPU: nothing is read on the GPU.
            //     Thread i will read input.r_x[i], ..., input.v_z[i].
            //     Slice 6 (durations) is passed separately to the launcher.
            warpkep::CartesianSoAConstView<double> input{};
            input.r_x = d_in.ptr + 0 * n;
            input.r_y = d_in.ptr + 1 * n;
            input.r_z = d_in.ptr + 2 * n;
            input.v_x = d_in.ptr + 3 * n;
            input.v_y = d_in.ptr + 4 * n;
            input.v_z = d_in.ptr + 5 * n;

            // 3b. Same slicing for the outputs, written by the kernel.
            warpkep::CartesianSoAView<double> output{};
            output.r_x = d_out.ptr + 0 * n;
            output.r_y = d_out.ptr + 1 * n;
            output.r_z = d_out.ptr + 2 * n;
            output.v_x = d_out.ptr + 3 * n;
            output.v_y = d_out.ptr + 4 * n;
            output.v_z = d_out.ptr + 5 * n;

            // 3c. Queue the kernel in the stream, after the input copy.
            //     The launcher runs on the CPU: it computes the grid and returns.
            //     Success here means "queued", not "finished": execution errors
            //     appear at the synchronization (step 5).
            require(warpkep::launch_kepler_soa_double(input, d_in.ptr + 6 * n, mu, output, d_status.ptr , d_iter.ptr, n, stream.handle), "launch_kepler_soa_double");

            // 4. Retrieve: three copies GPU -> CPU, queued after the kernel.
            //    The pinned arrays are NOT filled yet when these lines return.
            require(cudaMemcpyAsync(h_out.ptr, d_out.ptr, 6 * n * sizeof(double), cudaMemcpyDeviceToHost, stream.handle), "cudaMemcpyAsync (states to host)");
            require(cudaMemcpyAsync(h_status.ptr, d_status.ptr, n * sizeof(warpkep::KeplerPropagationStatus), cudaMemcpyDeviceToHost, stream.handle),  "cudaMemcpyAsync (statuses to host)");
            require(cudaMemcpyAsync(h_iter.ptr, d_iter.ptr, n * sizeof(int), cudaMemcpyDeviceToHost, stream.handle), "cudaMemcpyAsync (iterations to host)");


            // 5. Wait: the only wait of the chain. Also reports execution errors.
            //    Only after this line may the CPU read h_out, h_status, h_iter,
            //    and only after it may the destructors free the buffers.
            require(cudaStreamSynchronize(stream.handle), "cudaStreamSynchronize");


        } catch (...) {
            const auto error = cudaStreamSynchronize(stream.handle);
            if (error != cudaSuccess) {
                std::cerr << "Cleanup: " << cudaGetErrorString(error) << '\n';
            }
            throw; // Rethrow the original error to the outer catch.
        }

    // 6. Compare every GPU result with the CPU reference.
    //    A success status alone does not prove the values are right.
    //    Iteration counts may differ between CPU and GPU: not compared.
    constexpr double tolerance = 1e-11;
    double max_error_r = 0.0;
    double max_error_v = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const auto& ref = cpu[i];

        // a. Both sides must report success.
        if (h_status.ptr[i] != warpkep::KeplerPropagationStatus::success || ref.status != warpkep::KeplerPropagationStatus::success) {
            throw std::runtime_error( "Propagation failed at index " + std::to_string(i) + ": GPU " + warpkep::propagation_status_name(h_status.ptr[i]) + ", CPU " + warpkep::propagation_status_name(ref.status));
        }

        // b, c. Component differences: GPU slice k (position) and 3 + k (velocity).
        const double dx = h_out.ptr[0 * n + i] - ref.state.position[0];
        const double dy = h_out.ptr[1 * n + i] - ref.state.position[1];
        const double dz = h_out.ptr[2 * n + i] - ref.state.position[2];
        const double dvx = h_out.ptr[3 * n + i] - ref.state.velocity[0];
        const double dvy = h_out.ptr[4 * n + i] - ref.state.velocity[1];
        const double dvz = h_out.ptr[5 * n + i] - ref.state.velocity[2];

        // d. Euclidean errors.
        const double error_r = std::hypot(dx, dy, dz);
        const double error_v = std::hypot(dvx, dvy, dvz);

        // e. !(e <= tol) also rejects NaN, so non-finite outputs fail here.
        if (!(error_r <= tolerance) || !(error_v <= tolerance)) {
            throw std::runtime_error( "GPU/CPU mismatch at index " + std::to_string(i) + ": error_r = " + std::to_string(error_r) + ", error_v = " + std::to_string(error_v));
        }

        // f. Keep the largest errors for the report.
        max_error_r = std::max(max_error_r, error_r);
        max_error_v = std::max(max_error_v, error_v);
    }

    std::cout << "GPU vs CPU: " << n << " / " << n << " trajectories within " << tolerance << '\n'  << "Max position error: " << max_error_r << '\n' << "Max velocity error: " << max_error_v << '\n';
    return 0;

    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
