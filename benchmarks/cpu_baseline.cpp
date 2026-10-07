#include <warpkep/detail/kepler_cpu.hpp>
#include <warpkep/version.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using warpkep::detail::CartesianState;
using warpkep::detail::KeplerPropagationResult;
using warpkep::detail::KeplerPropagationStatus;

struct Input {
    CartesianState state;
    double dt;
};

// Fixed, normalized workload: circular, elliptic and hyperbolic states,
// with different orientations, radii and positive/negative durations.
// Generation is excluded from the measured interval.


// Generate a vector of inputs for the benchmark.
std::vector<Input> make_inputs(std::size_t count) {
    const double pi = std::acos(-1.0);
    std::vector<Input> inputs(count);

    // Populate the inputs with a variety of orbital states and durations.
    for (std::size_t i = 0; i < count; ++i) {
        const double angle = 2.0 * pi * static_cast<double>(i % 997) / 997.0;
        const double radius = 0.8 + 1.2 * static_cast<double>(i % 251) / 251.0;
        const double inclination = 0.6 * static_cast<double>(i % 127) / 127.0;
        const double speed_factor = (i % 3 == 0) ? 1.0 : (i % 3 == 1 ? 0.75 : 1.65);
        const double speed = speed_factor / std::sqrt(radius); // mu = 1
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        inputs[i].state = {
            {radius * c, radius * s, 0.0},
            {-speed * s * std::cos(inclination),
              speed * c * std::cos(inclination),
              speed * std::sin(inclination)}
        };

        // Assign a duration that varies with the index, alternating between positive and negative values.
        const double duration = 0.2 + 1.8 * static_cast<double>(i % 509) / 509.0;
        inputs[i].dt = (i % 2 == 0) ? duration : -duration;
    }
    return inputs;
}

// Each call writes only [begin, end); inputs are shared read-only.
void propagate_range(
    const std::vector<Input>& inputs,
    std::vector<KeplerPropagationResult>& outputs,
    std::size_t begin, std::size_t end
) {
    for (std::size_t i = begin; i < end; ++i) {
        outputs[i] = warpkep::detail::propagate_kepler_cpu(
            inputs[i].state, inputs[i].dt, 1.0
        );
    }
}

// Run the batch of propagations in parallel using the specified number of threads.
void run_batch(
    const std::vector<Input>& inputs,
    std::vector<KeplerPropagationResult>& outputs,
    std::size_t thread_count
) {
    if (thread_count == 1) {
        propagate_range(inputs, outputs, 0, inputs.size());
        return;
    }

    // Create a vector of threads to handle the workload in parallel.
    std::vector<std::thread> workers;
    workers.reserve(thread_count - 1);
    const std::size_t base = inputs.size() / thread_count;
    const std::size_t extra = inputs.size() % thread_count;
    auto boundary = [&](std::size_t t) {
        return t * base + std::min(t, extra);
    };

    // The caller handles one chunk, so the total is thread_count threads.
    try {
        for (std::size_t t = 0; t + 1 < thread_count; ++t) {
            const std::size_t begin = boundary(t);
            const std::size_t end = boundary(t + 1);
            workers.emplace_back([&inputs, &outputs, begin, end] {
                propagate_range(inputs, outputs, begin, end);
            });
        }

        // The last thread is handled by the caller to avoid an extra join.
        propagate_range(inputs, outputs, boundary(thread_count - 1), inputs.size());
    } catch (...) {
        for (auto& worker : workers) {
            worker.join();
        }
        throw;
    }
    for (auto& worker : workers) {
        worker.join();
    }
}


// Check that the serial and parallel results match, throwing an exception if they do not.
void check_results(
    const std::vector<KeplerPropagationResult>& serial,
    const std::vector<KeplerPropagationResult>& parallel
) {
    for (std::size_t i = 0; i < serial.size(); ++i) {
        const auto& a = serial[i];
        const auto& b = parallel[i];
        if (a.status != b.status || a.iterations != b.iterations) {
            throw std::runtime_error("Status/iteration mismatch at index " + std::to_string(i));
        }
        if (a.status != KeplerPropagationStatus::success) {
            throw std::runtime_error(
                "Propagation failed at index " + std::to_string(i) + ": "
                + warpkep::detail::propagation_status_name(a.status)
            );
        }
        for (std::size_t k = 0; k < 3; ++k) {
            if (!std::isfinite(a.state.position[k]) || !std::isfinite(a.state.velocity[k])
                || a.state.position[k] != b.state.position[k]
                || a.state.velocity[k] != b.state.velocity[k]) {
                throw std::runtime_error("State mismatch at index " + std::to_string(i));
            }
        }
    }
}


// Measure the time taken to run the batch of propagations and return the elapsed time in seconds.
double measure(
    const std::vector<Input>& inputs,
    std::vector<KeplerPropagationResult>& outputs,
    std::size_t threads
) {
    const auto start = std::chrono::steady_clock::now();
    run_batch(inputs, outputs, threads);
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

// Compute the median of a vector of times, which is used to report the typical performance of the benchmark.
double median(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    const std::size_t middle = times.size() / 2;
    return (times.size() % 2 == 0)
        ? 0.5 * (times[middle - 1] + times[middle]) : times[middle];
}


// Parse a string as a positive integer and return it as a size_t, throwing an exception for invalid input.
std::size_t positive_integer(const char* text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Arguments must be positive integers");
    }
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0
        || parsed > static_cast<unsigned long long>(static_cast<std::size_t>(-1))) {
        throw std::invalid_argument("Arguments must fit a positive size_t");
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            throw std::invalid_argument("Usage: warpkep_cpu_baseline [count] [threads] [repeats]");
        }
        const unsigned int hint = std::thread::hardware_concurrency();
        const std::size_t count = argc > 1 ? positive_integer(argv[1]) : 200000;
        const std::size_t threads = argc > 2 ? positive_integer(argv[2])
            : std::min<std::size_t>({std::max(1u, hint), count, 256});
        const std::size_t repeats = argc > 3 ? positive_integer(argv[3]) : 5;
        if (threads > count || threads > 256) {
            throw std::invalid_argument("Thread count must be <= count and <= 256");
        }

        const auto inputs = make_inputs(count);
        std::vector<KeplerPropagationResult> serial(count), parallel(count);
        std::vector<double> serial_times, parallel_times;
        serial_times.reserve(repeats);
        parallel_times.reserve(repeats);

        // Warm both paths and check partitioning before timing.
        run_batch(inputs, serial, 1);
        run_batch(inputs, parallel, threads);
        check_results(serial, parallel);

        for (std::size_t trial = 0; trial < repeats; ++trial) {
            // Alternate order to reduce a systematic thermal/order bias.
            if (trial % 2 == 0) {
                serial_times.push_back(measure(inputs, serial, 1));
                parallel_times.push_back(measure(inputs, parallel, threads));
            } else {
                parallel_times.push_back(measure(inputs, parallel, threads));
                serial_times.push_back(measure(inputs, serial, 1));
            }
            check_results(serial, parallel); // Outside the timed interval.
        }

        const double serial_seconds = median(serial_times);
        const double parallel_seconds = median(parallel_times);
        if (serial_seconds <= 0.0 || parallel_seconds <= 0.0) {
            throw std::runtime_error("Batch too short for the clock: increase count");
        }
        std::cout << "warpkep " << warpkep::version() << '\n'
                  << "Compiler: " << __VERSION__ << '\n'
                  << "Workload: normalized two-body, mu=1, FP64\n"
                  << "Propagations per batch: " << count << '\n'
                  << "Hardware concurrency hint: " << hint << '\n'
                  << "Threads used: " << threads << '\n'
                  << "Timed repeats: " << repeats << '\n'
                  << "Failures: 0\n"
                  << "Serial/parallel states, statuses and iterations: identical\n"
                  << "Timing includes thread creation and joining; excludes generation,\n"
                  << "output allocation, warm-up, checks and printing.\n"
                  << std::fixed << std::setprecision(3)
                  << "Serial median: " << serial_seconds * 1000.0 << " ms\n"
                  << "Parallel median: " << parallel_seconds * 1000.0 << " ms\n"
                  << "Serial throughput: " << static_cast<double>(count) / serial_seconds << " propagations/s\n"
                  << "Parallel throughput: " << static_cast<double>(count) / parallel_seconds << " propagations/s\n"
                  << "Speedup: " << serial_seconds / parallel_seconds << "x\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU baseline failed: " << error.what() << '\n';
        return 1;
    }
}
