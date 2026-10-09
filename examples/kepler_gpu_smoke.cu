#include "kepler_launch.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>

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

// Prints a readable message if a CUDA call failed.
bool check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::cerr << what << " failed: " << cudaGetErrorString(error) << '\n';
        return false;
    }
    return true;
}

// Copies a CPU std::array into a GPU DeviceArray of the same size.
template <typename T, std::size_t N>
cudaError_t copy_to_gpu(DeviceArray<T>& dst, const std::array<T, N>& src) {
    return cudaMemcpy(dst.ptr, src.data(), N * sizeof(T), cudaMemcpyHostToDevice);
}

// Copies a GPU DeviceArray back into a CPU std::array of the same size.
template <typename T, std::size_t N>
cudaError_t copy_to_cpu(std::array<T, N>& dst, const DeviceArray<T>& src) {
    return cudaMemcpy(dst.data(), src.ptr, N * sizeof(T), cudaMemcpyDeviceToHost);
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

    // --- 5. Copy the inputs from CPU to GPU ---
    for (const cudaError_t error : {
             copy_to_gpu(d_r_x, h_r_x), copy_to_gpu(d_r_y, h_r_y),
             copy_to_gpu(d_r_z, h_r_z), copy_to_gpu(d_v_x, h_v_x),
             copy_to_gpu(d_v_y, h_v_y), copy_to_gpu(d_v_z, h_v_z),
             copy_to_gpu(d_durations, durations)}) {
        if (!check(error, "Copy to GPU")) {
            return 1;
        }
    }

    // --- 6. Launch the kernel ---
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
                   d_statuses.ptr, d_iterations.ptr, n, nullptr),
               "Kernel launch")) {
        return 1;
    }

    // --- 7. Wait for the GPU, then copy the results back ---
    // The launch is asynchronous: wait until the GPU has finished.
    // This also reports errors that happened while the kernel was running.
    if (!check(cudaStreamSynchronize(nullptr), "Kernel execution")) {
        return 1;
    }

    std::array<double, n> h_out_r_x{}, h_out_r_y{}, h_out_r_z{};
    std::array<double, n> h_out_v_x{}, h_out_v_y{}, h_out_v_z{};
    std::array<wd::KeplerPropagationStatus, n> h_statuses{};
    std::array<int, n> h_iterations{};

    for (const cudaError_t error : {
             copy_to_cpu(h_out_r_x, d_out_r_x), copy_to_cpu(h_out_r_y, d_out_r_y),
             copy_to_cpu(h_out_r_z, d_out_r_z), copy_to_cpu(h_out_v_x, d_out_v_x),
             copy_to_cpu(h_out_v_y, d_out_v_y), copy_to_cpu(h_out_v_z, d_out_v_z),
             copy_to_cpu(h_statuses, d_statuses), copy_to_cpu(h_iterations, d_iterations)}) {
        if (!check(error, "Copy to CPU")) {
            return 1;
        }
    }

    // --- 8. Compare every trajectory to its analytical reference ---
    // Absolute tolerance for these cases in normalized units.
    constexpr double tolerance = 1e-12;
    std::size_t passed_count = 0;
    double max_position_error = 0.0;
    double max_velocity_error = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
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
        const bool passed = h_statuses[i] == wd::KeplerPropagationStatus::success
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
        std::cerr << "GPU analytical checks failed\n";
        return 1;
    }
    std::cout << "GPU analytical checks passed\n";
    return 0;
}
