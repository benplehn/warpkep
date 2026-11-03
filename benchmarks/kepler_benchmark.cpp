// CPU/GPU benchmark of the FP64 Kepler batch on a shared workload.
// Four timings (median of 9 repeats): CPU serial, CPU parallel, GPU kernel only
// (CUDA events, 32 launches per repeat), GPU with transfers (steady_clock).
// Every timed result is checked outside the timed intervals.

#include "kepler_workload.hpp"

#include <warpkep/cuda/kepler.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Status = warpkep::KeplerPropagationStatus;
using CpuResults = std::vector<warpkep::detail::KeplerPropagationResult<double>>;

// RAII owners: the constructor acquires one CUDA resource, the destructor releases it.
// RAII does not wait for pending GPU work: synchronize before buffers go out of scope.

// GPU memory. The CPU may keep and pass the address, never dereference it.
template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    cudaError_t error = cudaSuccess;

    explicit DeviceArray(std::size_t count) { error = cudaMalloc(&ptr, count * sizeof(T)); }
    ~DeviceArray() noexcept { if (ptr != nullptr) cudaFree(ptr); }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// Pinned (page-locked) host memory: direct, truly asynchronous PCIe transfers.
template <typename T>
struct PinnedArray {
    T* ptr = nullptr;
    cudaError_t error = cudaSuccess;

    explicit PinnedArray(std::size_t count) { error = cudaMallocHost(&ptr, count * sizeof(T)); }
    ~PinnedArray() noexcept { if (ptr != nullptr) cudaFreeHost(ptr); }

    PinnedArray(const PinnedArray&) = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;
};

// Ordered queue of GPU work; non-blocking with respect to the legacy default stream.
struct CudaStream {
    cudaStream_t handle = nullptr;
    cudaError_t error = cudaSuccess;

    CudaStream() { error = cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking); }
    ~CudaStream() noexcept { if (handle != nullptr) cudaStreamDestroy(handle); }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

// Marker in a stream: the GPU records a timestamp when it reaches it.
struct CudaEvent {
    cudaEvent_t handle = nullptr;
    cudaError_t error = cudaSuccess;

    CudaEvent() { error = cudaEventCreate(&handle); }
    ~CudaEvent() noexcept { if (handle != nullptr) cudaEventDestroy(handle); }

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
        CpuResults cpu(n);
        warpkep_bench::run_batch(inputs, cpu, 1);

        // CPU threads for the parallel baseline: all hardware threads, at least 1.
        constexpr std::size_t threads = 2;
        // Used later: const std::size_t threads = std::max(1u, std::thread::hardware_concurrency());

        // Timed CPU runs write here; cpu stays the untouched reference.
        CpuResults cpu_serial(n);
        CpuResults cpu_parallel(n);

        // One duration per repeat, in milliseconds.
        std::vector<double> cpu_serial_ms;
        std::vector<double> cpu_parallel_ms;
        std::vector<double> kernel_ms;            // per launch
        std::vector<double> gpu_total_ms;
        cpu_serial_ms.reserve(timed_repeats);
        cpu_parallel_ms.reserve(timed_repeats);
        kernel_ms.reserve(timed_repeats);
        gpu_total_ms.reserve(timed_repeats);

        // CPU results must match the reference bit for bit: same code, same inputs, same processor.
        const auto check_cpu = [&](const CpuResults& results, const char* label) {
            for (std::size_t i = 0; i < n; ++i) {
                const auto& got = results[i];
                const auto& ref = cpu[i];
                bool same = got.status == ref.status;
                for (std::size_t k = 0; k < 3; ++k) {
                    same = same && got.state.position[k] == ref.state.position[k] && got.state.velocity[k] == ref.state.velocity[k];
                }
                if (!same) throw std::runtime_error(std::string(label) + " differs from the reference at index " + std::to_string(i));
            }
        };

        // CPU serial: 9 timed runs. The reference run above served as warm-up.
        for (int repeat = 0; repeat < timed_repeats; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            warpkep_bench::run_batch(inputs, cpu_serial, 1);
            const auto end = std::chrono::steady_clock::now();
            cpu_serial_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            check_cpu(cpu_serial, "CPU serial");  // outside the timed interval
        }

        // CPU parallel: one untimed warm-up (thread creation, caches), then 9 timed runs.
        warpkep_bench::run_batch(inputs, cpu_parallel, threads);
        for (int repeat = 0; repeat < timed_repeats; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            warpkep_bench::run_batch(inputs, cpu_parallel, threads);
            const auto end = std::chrono::steady_clock::now();
            cpu_parallel_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            check_cpu(cpu_parallel, "CPU parallel");  // outside the timed interval
        }

        // GPU results must match the CPU reference within tolerance (other processor: not bitwise).
        double max_error_r = 0.0;
        double max_error_v = 0.0;
        const auto check_gpu = [&] {
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
        };

        const auto launch = [&] {
            return warpkep::launch_kepler_soa_double(input, d_dt, mu, output, d_status.ptr, d_iter.ptr, n, stream.handle);
        };

        // Transfers, queued in the stream.
        const auto copy_in = [&] {
            require(cudaMemcpyAsync(d_in.ptr, h_in.ptr, 7 * n * sizeof(double), cudaMemcpyHostToDevice, stream.handle), "cudaMemcpyAsync (inputs)");
        };
        const auto copy_out = [&] {
            require(cudaMemcpyAsync(h_out.ptr, d_out.ptr, 6 * n * sizeof(double), cudaMemcpyDeviceToHost, stream.handle), "cudaMemcpyAsync (states)");
            require(cudaMemcpyAsync(h_status.ptr, d_status.ptr, n * sizeof(Status), cudaMemcpyDeviceToHost, stream.handle), "cudaMemcpyAsync (statuses)");
            require(cudaMemcpyAsync(h_iter.ptr, d_iter.ptr, n * sizeof(int), cudaMemcpyDeviceToHost, stream.handle), "cudaMemcpyAsync (iterations)");
        };

        // One full pass as a user would run it: inputs in, one launch, results out, wait.
        const auto full_pass = [&] {
            copy_in();
            require(launch(), "launch (with transfers)");
            copy_out();
            require(cudaStreamSynchronize(stream.handle), "cudaStreamSynchronize (with transfers)");
        };

        // Overwrite host outputs before a pass, so stale correct values cannot pass the check.
        const auto poison_host_outputs = [&] {
            std::fill(h_out.ptr, h_out.ptr + 6 * n, std::numeric_limits<double>::quiet_NaN());
            std::fill(h_status.ptr, h_status.ptr + n, Status::invalid_input);
        };

        // 2. GPU work. The inner catch waits for queued work while the buffers still exist.
        try {
            // Kernel only: data already on the GPU. One untimed warm-up first.
            copy_in();
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
                if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) throw std::runtime_error("Invalid CUDA event duration");
                kernel_ms.push_back(static_cast<double>(elapsed_ms) / launches_per_repeat);
            }

            // Check the results of the last kernel-only launch.
            poison_host_outputs();
            copy_out();
            require(cudaStreamSynchronize(stream.handle), "cudaStreamSynchronize (kernel only)");
            check_gpu();

            // With transfers: one untimed warm-up, then 9 timed full passes (one launch each, no division).
            full_pass();
            for (int repeat = 0; repeat < timed_repeats; ++repeat) {
                poison_host_outputs();  // outside the timed interval
                const auto begin = std::chrono::steady_clock::now();
                full_pass();
                const auto end = std::chrono::steady_clock::now();
                gpu_total_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
                check_gpu();            // outside the timed interval
            }
        } catch (...) {
            const cudaError_t cleanup = cudaStreamSynchronize(stream.handle);
            if (cleanup != cudaSuccess) std::cerr << "Cleanup synchronization failed: " << cudaGetErrorString(cleanup) << '\n';
            throw;
        }

        // 3. Report: medians of 9 repeats, and throughput for the same n trajectories.
        const auto report = [&](const char* label, const std::vector<double>& times_ms) {
            for (double duration : times_ms) {
                if (!std::isfinite(duration) || duration <= 0.0) {
                    throw std::runtime_error(
                        std::string(label) + ": invalid duration");
                }
            }
            const double ms = warpkep_bench::median(times_ms);
            std::cout << "  " << std::left << std::setw(22) << label << std::right << std::setw(12) << ms << " ms  " << std::setw(14) << static_cast<double>(n) / (ms / 1000.0) << " propagations/s\n";
        };

        std::cout << "Workload: " << n << " trajectories, normalized two-body, mu = 1, FP64\n";
        std::cout << "CPU threads (parallel): " << threads << '\n';
        std::cout << "Checks: CPU runs identical to the reference; GPU within " << tolerance << " (max error_r " << max_error_r << ", max error_v " << max_error_v << ")\n";
        std::cout << "Median of " << timed_repeats << " repeats:\n";
        std::cout << std::fixed << std::setprecision(4);
        report("CPU serial", cpu_serial_ms);
        report("CPU parallel", cpu_parallel_ms);
        report("GPU kernel only", kernel_ms);
        report("GPU with transfers", gpu_total_ms);
        std::cout << std::defaultfloat;
        std::cout << "GPU kernel only: data already on the GPU, CUDA events, " << launches_per_repeat << " launches per repeat\n";
        std::cout << "GPU with transfers: copy in + 1 launch + 3 copies out + sync (pinned buffers; excludes AoS -> SoA packing)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << '\n';
        return 1;
    }
}