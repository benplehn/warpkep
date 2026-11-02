#include "kepler_workload.hpp"

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

using warpkep::detail::KeplerPropagationResult;
using warpkep::detail::KeplerPropagationStatus;


void check_results(
    const std::vector<KeplerPropagationResult<double>>& serial,
    const std::vector<KeplerPropagationResult<double>>& parallel
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

double measure(
    const std::vector<warpkep_bench::Input>& inputs,
    std::vector<KeplerPropagationResult<double>>& outputs,
    std::size_t threads
) {
    const auto start = std::chrono::steady_clock::now();
    warpkep_bench::run_batch(inputs, outputs, threads);
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}



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

        const auto inputs = warpkep_bench::make_inputs(count);
        std::vector<KeplerPropagationResult<double>> serial(count), parallel(count);
        std::vector<double> serial_times, parallel_times;
        serial_times.reserve(repeats);
        parallel_times.reserve(repeats);

        // Warm both paths and check partitioning before timing.
        warpkep_bench::run_batch(inputs, serial, 1);
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

        const double serial_seconds = warpkep_bench::median(serial_times);
        const double parallel_seconds = warpkep_bench::median(parallel_times);
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
                  << "Serial throughput: " << static_cast<double>(count) / serial_seconds
                  << " propagations/s\n"
                  << "Parallel throughput: " << static_cast<double>(count) / parallel_seconds
                  << " propagations/s\n"
                  << "Speedup: " << serial_seconds / parallel_seconds << "x\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU baseline failed: " << error.what() << '\n';
        return 1;
    }
}
