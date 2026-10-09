#include "kepler_launch.hpp"

#include <initializer_list>
#include <array>
#include <cmath>
#include <iostream>
#include <cstddef>

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

    // --- Test 2: three analytical cases, mu = 1 ---
    using State = wd::CartesianState<double>;
    const double pi = std::acos(-1.0);
    const double sqrt3 = std::sqrt(3.0);

    // Initial states: {position}, {velocity}.
    const std::array<State, 3> initial_states{
        State{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},   // circle forward
        State{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},   // circle backward
        State{{0.5, 0.0, 0.0}, {0.0, sqrt3, 0.0}}  // ellipse a = 1, e = 0.5
    };

    // Propagation duration of each case.
    const std::array<double, 3> durations{
        pi / 2.0,   // quarter turn forward
        -pi / 2.0,  // quarter turn backward
        pi          // half period: periapsis -> apoapsis
    };

    // Analytical references after propagation.
    const std::array<State, 3> expected_states{
        State{{0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}},
        State{{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}},
        State{{-1.5, 0.0, 0.0}, {0.0, -1.0 / sqrt3, 0.0}}
    };


    constexpr std::size_t n = 3;

    // h_ means  « host » : CPU memory
    std::array<double, n> h_r_x{};
    std::array<double, n> h_r_y{};
    std::array<double, n> h_r_z{};
    std::array<double, n> h_v_x{};
    std::array<double, n> h_v_y{};
    std::array<double, n> h_v_z{};

    for (std::size_t i = 0; i < n; ++i) {
        h_r_x[i] = initial_states[i].position[0];
        h_r_y[i] = initial_states[i].position[1];
        h_r_z[i] = initial_states[i].position[2];
        h_v_x[i] = initial_states[i].velocity[0];
        h_v_y[i] = initial_states[i].velocity[1];
        h_v_z[i] = initial_states[i].velocity[2];
    }


    // d_ means "device": GPU memory.
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
        if (error != cudaSuccess) {
            std::cerr << "GPU allocation failed: " << cudaGetErrorString(error) << '\n';
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

    // Temporary display, replaced by the comparison in step 8.
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "case " << i << ": " << wd::propagation_status_name(h_statuses[i])
                  << " r=(" << h_out_r_x[i] << ", " << h_out_r_y[i] << ", " << h_out_r_z[i]
                  << ") v=(" << h_out_v_x[i] << ", " << h_out_v_y[i] << ", " << h_out_v_z[i]
                  << ") iterations=" << h_iterations[i] << '\n';
    }

    return 0;
}