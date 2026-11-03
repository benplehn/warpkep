// CPU/GPU benchmark of the FP64 Kepler batch on a shared workload, for several batch sizes.
// Four timings per size (median of 9 repeats): CPU serial, CPU parallel, GPU kernel only
// (CUDA events, 32 launches per repeat), GPU with transfers (steady_clock).
// Every timed result is checked outside the timed intervals.
// stdout: CSV, one row per size. stderr: diagnostics and raw samples.
// Source commit, injected by CMake at configure time (see CMakeLists.txt).
#ifndef WARPKEP_GIT_COMMIT
#define WARPKEP_GIT_COMMIT "unknown"
#endif

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

constexpr double mu = 1.0;                // shared workload is normalized with mu = 1
constexpr int timed_repeats = 9;          // median of 9 measurements
constexpr int launches_per_repeat = 32;   // amortize event resolution
constexpr double tolerance = 1e-11;
constexpr std::size_t max_threads = 2;    // CPU parallel baseline, fixed for reproducibility

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

// Median of valid durations; a zero, negative or non-finite duration aborts the benchmark.
double median_of(const char* label, const std::vector<double>& times_ms) {
    for (double t : times_ms) {
        if (!std::isfinite(t) || t <= 0.0) throw std::runtime_error(std::string(label) + ": invalid duration");
    }
    return warpkep_bench::median(times_ms);
}

// Raw samples, on stderr, for diagnostics.
void print_samples(std::size_t n, const char* label, const std::vector<double>& times_ms) {
    std::cerr << "  n=" << n << ' ' << std::left << std::setw(20) << label << std::right << std::fixed << std::setprecision(4);
    for (double t : times_ms) std::cerr << ' ' << t;
    std::cerr << std::defaultfloat << '\n';
}

// Execution metadata, on stderr: without it, a timing cannot be interpreted.
void print_metadata() {
    // 1. GPU: which device, and its main characteristics.
    int device = 0;
    require(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    require(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::cerr << "GPU: " << prop.name << " | compute " << prop.major << '.' << prop.minor << " | " << prop.multiProcessorCount << " SMs | " << std::fixed << std::setprecision(1) << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GiB" << std::defaultfloat << '\n';

    // 2. CUDA: driver (installed on the machine) and runtime (linked into the program).
    //    Both are encoded as 1000 * major + 10 * minor, e.g. 12040 -> 12.4.
    int driver = 0;
    int runtime = 0;
    require(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    require(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    const auto version = [](int v) { return std::to_string(v / 1000) + '.' + std::to_string((v % 1000) / 10); };
    std::cerr << "CUDA: driver " << version(driver) << " | runtime " << version(runtime) << '\n';

    // 3. CPU side: host compiler, threads, source commit.
    std::cerr << "Host compiler: " << __VERSION__ << '\n';
    std::cerr << "CPU threads: " << max_threads << " used at most | " << std::thread::hardware_concurrency() << " available\n";
    std::cerr << "Commit: " << WARPKEP_GIT_COMMIT << '\n';
}

// One CSV row.
struct SizeResult {
    std::size_t n = 0;
    std::size_t threads = 0;
    double cpu_serial_ms = 0.0;
    double cpu_parallel_ms = 0.0;
    double gpu_kernel_ms = 0.0;
    double gpu_total_ms = 0.0;
    double max_error_r = 0.0;
    double max_error_v = 0.0;
};

// Full chain for one batch size: allocations, preparation, warm-ups, four timings and checks.
SizeResult run_size(std::size_t n) {
    const std::size_t threads = std::min(max_threads, n);

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

    // CPU reference on the same inputs, outside any timed region; timed CPU runs write elsewhere.
    CpuResults cpu(n);
    CpuResults cpu_serial(n);
    CpuResults cpu_parallel(n);
    warpkep_bench::run_batch(inputs, cpu, 1);

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

    // 2. CPU serial: 9 timed runs. The reference run above served as warm-up.
    for (int repeat = 0; repeat < timed_repeats; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        warpkep_bench::run_batch(inputs, cpu_serial, 1);
        const auto end = std::chrono::steady_clock::now();
        cpu_serial_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        check_cpu(cpu_serial, "CPU serial");  // outside the timed interval
    }

    // 3. CPU parallel: one untimed warm-up (thread creation, caches), then 9 timed runs.
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

    // 4. GPU work. The inner catch waits for queued work while the buffers still exist.
    try {
        // Warm-up, not timed: full passes for 1 s, so GPU clocks are up and d_in is filled.
        const auto warm_begin = std::chrono::steady_clock::now();
        do {
            full_pass();
        } while (std::chrono::steady_clock::now() - warm_begin < std::chrono::seconds(1));

        // Timed: [start] 32 launches [stop], no copy inside the interval.
        for (int repeat = 0; repeat < timed_repeats; ++repeat) {
            require(cudaEventRecord(start.handle, stream.handle), "cudaEventRecord (start)");
            for (int k = 0; k < launches_per_repeat; ++k) require(launch(), "launch (timed)");
            require(cudaEventRecord(stop.handle, stream.handle), "cudaEventRecord (stop)");
            require(cudaEventSynchronize(stop.handle), "cudaEventSynchronize");

            float elapsed_ms = 0.0f;
            require(cudaEventElapsedTime(&elapsed_ms, start.handle, stop.handle), "cudaEventElapsedTime");
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

    // 5. Diagnostics on stderr, result for the CSV.
    print_samples(n, "CPU serial", cpu_serial_ms);
    print_samples(n, "CPU parallel", cpu_parallel_ms);
    print_samples(n, "GPU kernel only", kernel_ms);
    print_samples(n, "GPU with transfers", gpu_total_ms);

    SizeResult result;
    result.n = n;
    result.threads = threads;
    result.cpu_serial_ms = median_of("CPU serial", cpu_serial_ms);
    result.cpu_parallel_ms = median_of("CPU parallel", cpu_parallel_ms);
    result.gpu_kernel_ms = median_of("GPU kernel only", kernel_ms);
    result.gpu_total_ms = median_of("GPU with transfers", gpu_total_ms);
    result.max_error_r = max_error_r;
    result.max_error_v = max_error_v;
    return result;
}

int main() {
    try {
        print_metadata();
        constexpr std::size_t sizes[] = {1, 256, 257, 1000, 16384, 65536, 262144, 1048576};

        std::cerr << "Workload: normalized two-body, mu = 1, FP64 | CPU threads: up to " << max_threads << " (hardware: " << std::thread::hardware_concurrency() << ")\n";
        std::cerr << "GPU kernel only: data on the GPU, CUDA events, " << launches_per_repeat << " launches per repeat\n";
        std::cerr << "GPU with transfers: copy in + 1 launch + 3 copies out + sync (pinned buffers; excludes AoS -> SoA packing)\n";

        // CSV on stdout; each row is flushed as soon as its size is done.
        std::cout << "n,threads,cpu_serial_ms,cpu_parallel_ms,gpu_kernel_ms,gpu_total_ms,max_error_r,max_error_v\n" << std::flush;
        for (std::size_t n : sizes) {
            const SizeResult r = run_size(n);
            std::cout << std::setprecision(9) << r.n << ',' << r.threads << ',' << r.cpu_serial_ms << ',' << r.cpu_parallel_ms << ',' << r.gpu_kernel_ms << ',' << r.gpu_total_ms << ',' << r.max_error_r << ',' << r.max_error_v << '\n' << std::flush;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << '\n';
        return 1;
    }
}