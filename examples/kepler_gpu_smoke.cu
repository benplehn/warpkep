#include "kepler_launch.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>

namespace wd = warpkep::detail;

// Owns one GPU array of `count` elements and frees it automatically
// when the variable goes out of scope (end of main or early return).
template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    cudaError_t error = cudaSuccess;

    explicit DeviceArray(std::size_t count) {
        error = cudaMalloc(&ptr, count * sizeof(T));
    }
    ~DeviceArray() {
        cudaFree(ptr); // cudaFree(nullptr) is a no-op
    }

    // Copying would free the same GPU memory twice: forbid it.
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// Owns one CUDA stream (a GPU work queue) and destroys it automatically.
// cudaStreamNonBlocking: no implicit synchronization with the default stream.
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

    // Copying would destroy the same stream twice: forbid it.
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

// Prints a readable message if a CUDA call failed.
bool check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::cerr << what << " failed: " << cudaGetErrorString(error) << '\n';
        return false;
    }
    return true;
}

// Enqueues a copy of a CPU std::array into a GPU DeviceArray of the same size.
// Asynchronous: the copy is only guaranteed complete after synchronizing the
// stream, so `src` must stay alive and unchanged until then.
template <typename T, std::size_t N>
cudaError_t copy_to_gpu(DeviceArray<T>& dst, const std::array<T, N>& src, cudaStream_t stream) {
    return cudaMemcpyAsync(
        dst.ptr, src.data(), N * sizeof(T), cudaMemcpyHostToDevice, stream
    );
}

// Enqueues a copy of a GPU DeviceArray back into a CPU std::array.
// `dst` must not be read before the stream has been synchronized.
template <typename T, std::size_t N>
cudaError_t copy_to_cpu(std::array<T, N>& dst, const DeviceArray<T>& src, cudaStream_t stream) {
    return cudaMemcpyAsync(
        dst.data(), src.ptr, N * sizeof(T), cudaMemcpyDeviceToHost, stream
    );
}

int main() {
    // --- Test 1: empty batch (no kernel is launched) ---
    const cudaError_t status = launch_kepler_soa_double(
        {}, nullptr, 1.0, {}, nullptr, nullptr, 0, nullptr
    );
    if (status != cudaSuccess) {
        std::cerr << "Empty batch failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    std::cout << "Empty batch passed\n";

    // The caller owns the stream; the launcher only submits work to it.
    CudaStream stream;
    if (!check(stream.error, "Stream creation")) {
        return 1;
    }

    // --- Test 2: analytical cases, mu = 1 ---
    using State = wd::CartesianState<double>;
    const double pi = std::acos(-1.0);
    const double sqrt3 = std::sqrt(3.0);

    // The three reference cases: {position}, {velocity}.
    const std::array<State, 3> initial_cases{
        State{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},   // circle forward
        State{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},   // circle backward
        State{{0.5, 0.0, 0.0}, {0.0, sqrt3, 0.0}}  // ellipse a = 1, e = 0.5
    };

    // Propagation duration of each reference case.
    const std::array<double, 3> duration_cases{
        pi / 2.0,   // quarter turn forward
        -pi / 2.0,  // quarter turn backward
        pi          // half period: periapsis -> apoapsis
    };

    // Analytical result of each reference case.
    const std::array<State, 3> expected_cases{
        State{{0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}},
        State{{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}},
        State{{-1.5, 0.0, 0.0}, {0.0, -1.0 / sqrt3, 0.0}}
    };

    // 257 trajectories = 2 blocks of 256 threads: the second block has a
    // single active thread, the other 255 must exit through `if (i >= n)`.
    constexpr std::size_t n = 257;

    std::array<State, n> initial_states{};
    std::array<double, n> durations{};
    std::array<State, n> expected_states{};

    // Repeat the three cases: case_index = 0, 1, 2, 0, 1, 2, ...
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t case_index = i % initial_cases.size();
        initial_states[i] = initial_cases[case_index];
        durations[i] = duration_cases[case_index];
        expected_states[i] = expected_cases[case_index];
    }

    // --- Per-trajectory exceptions inside the same batch ---
    // Valid neighbours must keep succeeding next to invalid trajectories.
    using Status = wd::KeplerPropagationStatus;

    std::array<Status, n> expected_statuses{};
    expected_statuses.fill(Status::success);

    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Index 1: position at the origin -> invalid input.
    initial_states[1].position = {0.0, 0.0, 0.0};
    expected_statuses[1] = Status::invalid_input;

    // Index 2: one velocity component is NaN -> invalid input.
    initial_states[2].velocity[1] = nan;
    expected_statuses[2] = Status::invalid_input;

    // Index 3: zero duration on a valid state -> success, state unchanged.
    durations[3] = 0.0;
    expected_states[3] = initial_states[3];

    // Index 256 (second block): velocity parallel to position, i.e. radial
    // motion with a nonzero duration -> outside the supported domain.
    initial_states[256] = State{{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    durations[256] = 1.0;
    expected_statuses[256] = Status::unsupported_case;

    // --- 3. Split the states into six SoA arrays (h_ = host, CPU memory) ---
    std::array<double, n> h_r_x{}, h_r_y{}, h_r_z{};
    std::array<double, n> h_v_x{}, h_v_y{}, h_v_z{};

    for (std::size_t i = 0; i < n; ++i) {
        h_r_x[i] = initial_states[i].position[0];
        h_r_y[i] = initial_states[i].position[1];
        h_r_z[i] = initial_states[i].position[2];
        h_v_x[i] = initial_states[i].velocity[0];
        h_v_y[i] = initial_states[i].velocity[1];
        h_v_z[i] = initial_states[i].velocity[2];
    }

    // --- 4. Allocate the GPU arrays (d_ = device, GPU memory) ---
    DeviceArray<double> d_r_x(n), d_r_y(n), d_r_z(n);
    DeviceArray<double> d_v_x(n), d_v_y(n), d_v_z(n);
    DeviceArray<double> d_durations(n);
    DeviceArray<double> d_out_r_x(n), d_out_r_y(n), d_out_r_z(n);
    DeviceArray<double> d_out_v_x(n), d_out_v_y(n), d_out_v_z(n);
    DeviceArray<wd::KeplerPropagationStatus> d_statuses(n);
    DeviceArray<int> d_iterations(n);

    for (const cudaError_t error : {
             d_r_x.error, d_r_y.error, d_r_z.error, d_v_x.error, d_v_y.error,
             d_v_z.error, d_durations.error, d_out_r_x.error, d_out_r_y.error,
             d_out_r_z.error, d_out_v_x.error, d_out_v_y.error, d_out_v_z.error,
             d_statuses.error, d_iterations.error}) {
        if (!check(error, "GPU allocation")) {
            return 1;
        }
    }
    std::cout << "GPU allocation passed\n";

    // Everything below goes into ONE queue, in this order:
    //   input copies -> kernel -> output copies -> single final wait.
    // The stream itself guarantees the order: no intermediate CPU wait.
    const cudaStream_t s = stream.handle;

    // --- 5. Enqueue the input copies (CPU -> GPU) ---
    for (const cudaError_t error : {
             copy_to_gpu(d_r_x, h_r_x, s), copy_to_gpu(d_r_y, h_r_y, s),
             copy_to_gpu(d_r_z, h_r_z, s), copy_to_gpu(d_v_x, h_v_x, s),
             copy_to_gpu(d_v_y, h_v_y, s), copy_to_gpu(d_v_z, h_v_z, s),
             copy_to_gpu(d_durations, durations, s)}) {
        if (!check(error, "Copy to GPU")) {
            return 1;
        }
    }

    // --- 6. Enqueue the kernel, after the input copies in the same stream ---
    // The views bundle the six GPU addresses of each side.
    const wd::CartesianSoAConstView<double> input{
        d_r_x.ptr, d_r_y.ptr, d_r_z.ptr, d_v_x.ptr, d_v_y.ptr, d_v_z.ptr
    };
    const wd::CartesianSoAView<double> output{
        d_out_r_x.ptr, d_out_r_y.ptr, d_out_r_z.ptr,
        d_out_v_x.ptr, d_out_v_y.ptr, d_out_v_z.ptr
    };
    if (!check(launch_kepler_soa_double(
                   input, d_durations.ptr, 1.0, output,
                   d_statuses.ptr, d_iterations.ptr, n, s),
               "Kernel launch")) {
        return 1;
    }

    // --- 7. Enqueue the output copies (GPU -> CPU), after the kernel ---
    std::array<double, n> h_out_r_x{}, h_out_r_y{}, h_out_r_z{};
    std::array<double, n> h_out_v_x{}, h_out_v_y{}, h_out_v_z{};
    std::array<wd::KeplerPropagationStatus, n> h_statuses{};
    std::array<int, n> h_iterations{};

    for (const cudaError_t error : {
             copy_to_cpu(h_out_r_x, d_out_r_x, s), copy_to_cpu(h_out_r_y, d_out_r_y, s),
             copy_to_cpu(h_out_r_z, d_out_r_z, s), copy_to_cpu(h_out_v_x, d_out_v_x, s),
             copy_to_cpu(h_out_v_y, d_out_v_y, s), copy_to_cpu(h_out_v_z, d_out_v_z, s),
             copy_to_cpu(h_statuses, d_statuses, s),
             copy_to_cpu(h_iterations, d_iterations, s)}) {
        if (!check(error, "Copy to CPU")) {
            return 1;
        }
    }

    // Single wait: the CPU blocks until copies, kernel and copies back are done.
    // It also reports errors that happened while the kernel was running.
    // The h_out_* arrays must not be read before this point.
    if (!check(cudaStreamSynchronize(s), "Stream execution")) {
        return 1;
    }

    // --- 8. Compare every trajectory to its analytical reference ---
    // Absolute tolerance for these cases in normalized units.
    constexpr double tolerance = 1e-12;
    std::size_t passed_count = 0;
    double max_position_error = 0.0;
    double max_velocity_error = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        // Expected failure: check its status and its six NaN outputs,
        // then move on (there is no reference state to compare to).
        if (expected_statuses[i] != Status::success) {
            const bool all_nan = std::isnan(h_out_r_x[i]) && std::isnan(h_out_r_y[i])
                && std::isnan(h_out_r_z[i]) && std::isnan(h_out_v_x[i])
                && std::isnan(h_out_v_y[i]) && std::isnan(h_out_v_z[i]);

            const bool passed = h_statuses[i] == expected_statuses[i] && all_nan;

            if (passed) {
                ++passed_count;
            } else {
                std::cerr << "case " << i << " FAIL: expected="
                          << wd::propagation_status_name(expected_statuses[i])
                          << " actual=" << wd::propagation_status_name(h_statuses[i])
                          << " all_nan=" << all_nan << '\n';
            }
            continue;
        }

        // Expected success: compare to the analytical reference.
        const auto& expected = expected_states[i];

        const double position_error = std::hypot(
            h_out_r_x[i] - expected.position[0],
            h_out_r_y[i] - expected.position[1],
            h_out_r_z[i] - expected.position[2]
        );
        const double velocity_error = std::hypot(
            h_out_v_x[i] - expected.velocity[0],
            h_out_v_y[i] - expected.velocity[1],
            h_out_v_z[i] - expected.velocity[2]
        );

        // isfinite first: a NaN error must fail, never pass silently.
        const bool passed = h_statuses[i] == Status::success
            && std::isfinite(position_error) && std::isfinite(velocity_error)
            && position_error <= tolerance && velocity_error <= tolerance;

        if (passed) {
            ++passed_count;
            max_position_error = std::fmax(max_position_error, position_error);
            max_velocity_error = std::fmax(max_velocity_error, velocity_error);
        } else {
            // Only failures are detailed, to keep the output readable.
            std::cerr << "case " << i << " FAIL"
                      << " status=" << wd::propagation_status_name(h_statuses[i])
                      << " position_error=" << position_error
                      << " velocity_error=" << velocity_error
                      << " iterations=" << h_iterations[i] << '\n';
        }
    }

    std::cout << passed_count << " / " << n << " trajectories passed"
              << " (max position error " << max_position_error
              << ", max velocity error " << max_velocity_error << ")\n";

    if (passed_count != n) {
        std::cerr << "GPU mixed batch checks failed\n";
        return 1;
    }
    std::cout << "GPU mixed batch checks passed\n";
    return 0;
}
