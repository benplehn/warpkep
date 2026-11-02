#pragma once

#include <warpkep/detail/kepler_cpu.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

// Shared benchmark workload: the CPU and GPU benchmarks use the same inputs.
namespace warpkep_bench {

using warpkep::detail::CartesianState;
using warpkep::detail::KeplerPropagationResult;

struct Input {
    CartesianState<double> state;
    double dt;
};

// Fixed, normalized workload: circular, elliptic and hyperbolic states,
// with different orientations, radii and positive/negative durations.
// Generation is excluded from the measured interval.
inline std::vector<Input> make_inputs(std::size_t count) {
    const double pi = std::acos(-1.0);
    std::vector<Input> inputs(count);

    for (std::size_t i = 0; i < count; ++i) {
        const double angle = 2.0 * pi * static_cast<double>(i % 997) / 997.0;
        const double radius = 0.8 + 1.2 * static_cast<double>(i % 251) / 251.0;
        const double inclination = 0.6 * static_cast<double>(i % 127) / 127.0;
        const double speed_factor = (i % 3 == 0) ? 1.0 : (i % 3 == 1 ? 0.75 : 1.65);
        const double speed = speed_factor / std::sqrt(radius);  // mu = 1
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        inputs[i].state = {
            {radius * c, radius * s, 0.0},
            {-speed * s * std::cos(inclination),
             speed * c * std::cos(inclination),
             speed * std::sin(inclination)}
        };

        const double duration = 0.2 + 1.8 * static_cast<double>(i % 509) / 509.0;
        inputs[i].dt = (i % 2 == 0) ? duration : -duration;
    }
    return inputs;
}

// Each call writes only [begin, end); inputs are shared read-only.
inline void propagate_range(
    const std::vector<Input>& inputs,
    std::vector<KeplerPropagationResult<double>>& outputs,
    std::size_t begin, std::size_t end
) {
    for (std::size_t i = begin; i < end; ++i) {
        outputs[i] = warpkep::detail::propagate_kepler_cpu<double>(
            inputs[i].state, inputs[i].dt, 1.0
        );
    }
}


// Preconditions:
// - thread_count >= 1
// - outputs.size() == inputs.size()
inline void run_batch(const std::vector<Input>& inputs, std::vector<KeplerPropagationResult<double>>& outputs, std::size_t thread_count) {
    if (thread_count == 1) {
        propagate_range(inputs, outputs, 0, inputs.size());
        return;
    }

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

} // namespace warpkep_bench