// CPU/GPU benchmark of the FP64 Kepler batch.
// Current stage: kernel time with data already on the GPU (CUDA events,
// median of 9 repeats of 32 launches), results checked against the CPU.

#include "kepler_workload.hpp"

#include <warpkep/cuda/kepler.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Status = warpkep::KeplerPropagationStatus;

// RAII owners: the constructor acquires one CUDA resource, the destructor releases it.
// RAII does not wait for pending GPU work: synchronize before buffers go out of scope.

// GPU memory. The CPU may keep and pass the address, never dereference it.
template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    cudaError_t error = cudaSuccess;

    explicit DeviceArray(std::size_t count) { 
        error = cudaMalloc(&ptr, count * sizeof(T)); 
    }
    ~DeviceArray() noexcept { 
        if (ptr != nullptr) cudaFree(ptr); 
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// Pinned (page-locked) host memory: direct, truly asynchronous PCIe transfers.
template <typename T>
struct PinnedArray {
    T* ptr = nullptr;
    cudaError_t error = cudaSuccess;

    explicit PinnedArray(std::size_t count) { 
        error = cudaMallocHost(&ptr, count * sizeof(T)); 
    }
    ~PinnedArray() noexcept { 
        if (ptr != nullptr) cudaFreeHost(ptr); 
    }

    PinnedArray(const PinnedArray&) = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;
};

// Ordered queue of GPU work; non-blocking with respect to the legacy default stream.
struct CudaStream {
    cudaStream_t handle = nullptr;
    cudaError_t error = cudaSuccess;

    CudaStream() {
        error = cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking); 
    }
    ~CudaStream() noexcept { 
        if (handle != nullptr) cudaStreamDestroy(handle); 
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

// Marker in a stream: the GPU records a timestamp when it reaches it.
struct CudaEvent {
    cudaEvent_t handle = nullptr;
    cudaError_t error = cudaSuccess;

    CudaEvent() { 
        error = cudaEventCreate(&handle); 
    }
    ~CudaEvent() noexcept { 
        if (handle != nullptr) cudaEventDestroy(handle); 
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};

// A CUDA error is not a measurement: it aborts the benchmark.
void require(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(error));
}

int main() {
    try {
        constexpr std::size_t n = 1000;
        constexpr double mu = 1.0;                // shared workload is normalized with mu = 1
        constexpr int timed_repeats = 9;          // median of 9 measurements
        constexpr int launches_per_repeat = 32;   // amortize event resolution
        constexpr double tolerance = 1e-11;

        // 1. Prepare: every allocation happens before any measurement.
        const auto inputs = warpkep_bench::make_inputs(n);

        PinnedArray<double> h_in(7 * n);          // [r_x|r_y|r_z|v_x|v_y|v_z|dt]
        PinnedArray<double> h_out(6 * n);         // [r_x|r_y|r_z|v_x|v_y|v_z]
        PinnedArray<Status> h_status(n);
        PinnedArray<int> h_iter(n);
        DeviceArray<double> d_in(7 * n);
        DeviceArray<double> d_out(6 * n);
        DeviceArray<Status> d_status(n);
        DeviceArray<int> d_iter(n);
        CudaStream stream;
        CudaEvent start;
        CudaEvent stop;

        require(h_in.error, "cudaMallocHost (inputs)");
        require(h_out.error, "cudaMallocHost (states)");
        require(h_status.error, "cudaMallocHost (statuses)");
        require(h_iter.error, "cudaMallocHost (iterations)");
        require(d_in.error, "cudaMalloc (inputs)");
        require(d_out.error, "cudaMalloc (states)");
        require(d_status.error, "cudaMalloc (statuses)");
        require(d_iter.error, "cudaMalloc (iterations)");
        require(stream.error, "cudaStreamCreateWithFlags");
        require(start.error, "cudaEventCreate (start)");
        require(stop.error, "cudaEventCreate (stop)");

        // Pack AoS inputs into SoA slices: component k of trajectory i at k * n + i, durations in slice 6.
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < 3; ++k) {
                h_in.ptr[k * n + i] = inputs[i].state.position[k];
                h_in.ptr[(3 + k) * n + i] = inputs[i].state.velocity[k];
            }
            h_in.ptr[6 * n + i] = inputs[i].dt;
        }

        // Views: start address of each component slice (address arithmetic only).
        warpkep::CartesianSoAConstView<double> input{};
        input.r_x = d_in.ptr + 0 * n;
        input.r_y = d_in.ptr + 1 * n;
        input.r_z = d_in.ptr + 2 * n;
        input.v_x = d_in.ptr + 3 * n;
        input.v_y = d_in.ptr + 4 * n;
        input.v_z = d_in.ptr + 5 * n;
        const double* d_dt = d_in.ptr + 6 * n;

        warpkep::CartesianSoAView<double> output{};
        output.r_x = d_out.ptr + 0 * n;
        output.r_y = d_out.ptr + 1 * n;
        output.r_z = d_out.ptr + 2 * n;
        output.v_x = d_out.ptr + 3 * n;
        output.v_y = d_out.ptr + 4 * n;
        output.v_z = d_out.ptr + 5 * n;

        // CPU reference on the same inputs, outside any timed region.
        std::vector<warpkep::detail::KeplerPropagationResult<double>> cpu(n);
        warpkep_bench::run_batch(inputs, cpu, 1);

        std::vector<double> kernel_ms;            // ms per launch, one value per repeat
        kernel_ms.reserve(timed_repeats);

        const auto launch = [&] { 
            return warpkep::launch_kepler_soa_double(input, d_dt, mu, output, d_status.ptr, d_iter.ptr, n, stream.handle); 
        };

        // 2. GPU work. The inner catch waits for queued work while the buffers still exist.
        try {
            require(cudaMemcpyAsync(d_in.ptr, h_in.ptr, 7 * n * sizeof(double), cudaMemcpyHostToDevice, stream.handle), "cudaMemcpyAsync (inputs)");

            // Warm-up, not timed.
            require(launch(), "launch (warm-up)");
            require(cudaStreamSynchronize(stream.handle), "cudaStreamSynchronize (warm-up)");

            // Timed: [start] 32 launches [stop], no copy inside the interval.
            for (int repeat = 0; repeat < timed_repeats; ++repeat) {
                require(cudaEventRecord(start.handle, stream.handle), "cudaEventRecord (start)");
                for (int k = 0; k < launches_per_repeat; ++k) require(launch(), "launch (timed)");
                require(cudaEventRecord(stop.handle, stream.handle), "cudaEventRecord (stop)");
                require(cudaEventSynchronize(stop.handle), "cudaEventSynchronize");

                float elapsed_ms = 0.0f;
                require(cudaEventElapsedTime(&elapsed_ms, start.handle, stop.handle), "cudaEventElapsedTime");
                if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
                    throw std::runtime_error("Invalid CUDA event duration");
                }
                kernel_ms.push_back(static_cast<double>(elapsed_ms) / launches_per_repeat);
            }

            // Results of the last timed launch; not ready before the synchronization.
            require(cudaMemcpyAsync(h_out.ptr, d_out.ptr, 6 * n * sizeof(double), cudaMemcpyDeviceToHost, stream.handle), "cudaMemcpyAsync (states)");
            require(cudaMemcpyAsync(h_status.ptr, d_status.ptr, n * sizeof(Status), cudaMemcpyDeviceToHost, stream.handle), "cudaMemcpyAsync (statuses)");
            require(cudaMemcpyAsync(h_iter.ptr, d_iter.ptr, n * sizeof(int), cudaMemcpyDeviceToHost, stream.handle), "cudaMemcpyAsync (iterations)");
            require(cudaStreamSynchronize(stream.handle), "cudaStreamSynchronize");
        } catch (...) {
            const cudaError_t cleanup = cudaStreamSynchronize(stream.handle);
            if (cleanup != cudaSuccess) std::cerr << "Cleanup synchronization failed: " << cudaGetErrorString(cleanup) << '\n';
            throw;
        }

        // 3. Check every GPU result against the CPU. Iteration counts may differ: not compared.
        double max_error_r = 0.0;
        double max_error_v = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            const auto& ref = cpu[i];
            if (h_status.ptr[i] != Status::success || ref.status != Status::success) {
                throw std::runtime_error("Propagation failed at index " + std::to_string(i) + ": GPU " + warpkep::propagation_status_name(h_status.ptr[i]) + ", CPU " + warpkep::propagation_status_name(ref.status));
            }

            const double error_r = std::hypot(h_out.ptr[0 * n + i] - ref.state.position[0], h_out.ptr[1 * n + i] - ref.state.position[1], h_out.ptr[2 * n + i] - ref.state.position[2]);
            const double error_v = std::hypot(h_out.ptr[3 * n + i] - ref.state.velocity[0], h_out.ptr[4 * n + i] - ref.state.velocity[1], h_out.ptr[5 * n + i] - ref.state.velocity[2]);

            // !(e <= tol) also rejects NaN and infinity.
            if (!(error_r <= tolerance) || !(error_v <= tolerance)) {
                throw std::runtime_error("GPU/CPU mismatch at index " + std::to_string(i) + ": error_r = " + std::to_string(error_r) + ", error_v = " + std::to_string(error_v));
            }
            max_error_r = std::max(max_error_r, error_r);
            max_error_v = std::max(max_error_v, error_v);
        }

        // 4. Report.
        const double median_ms = warpkep_bench::median(kernel_ms);
        std::cout << "Trajectories: " << n << '\n';
        std::cout << "GPU vs CPU: all within " << tolerance << " (max error_r " << max_error_r << ", max error_v " << max_error_v << ")\n";
        std::cout << "Kernel only, median of " << timed_repeats << " x " << launches_per_repeat << " launches: " << median_ms * 1000.0 << " us per launch\n";
        std::cout << "Kernel throughput: " << static_cast<double>(n) / (median_ms / 1000.0) << " propagations/s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << '\n';
        return 1;
    }
}