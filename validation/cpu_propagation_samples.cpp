#include <warpkep/detail/kepler_cpu.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace wd = warpkep::detail;

struct Sample {
    const char* name;
    wd::CartesianState initial;
    double dt;
    double mu;
};

void write_vector(const std::array<double, 3>& v) {
    std::cout << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
}

int main() {

    // Define a set of test samples for Kepler propagation.
    const std::array<Sample, 4> samples{{
        {"circular", {{1, 0, 0}, {0, 1, 0}}, std::acos(-1.0) / 2, 1},
        {"elliptic", {{1.5, 0.1, -0.2}, {-0.15, 0.6, 0.25}}, 1.7, 1},
        {"hyperbolic", {{1, 0, 0}, {0.2, 1.6, 0.1}}, 0.7, 1},
        {"backward", {{1.5, 0.1, -0.2}, {-0.15, 0.6, 0.25}}, -1.7, 1}
    }};

    // Set the output precision and format for JSON-like output.
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "[\n";

    // Iterate over the samples, propagate each one, and output the results.
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        const auto result = wd::propagate_kepler_cpu(sample.initial, sample.dt, sample.mu);
        if (result.status != wd::KeplerPropagationStatus::success) {
            std::cerr << sample.name << ": " << wd::propagation_status_name(result.status) << '\n';
            return 1;
        }
        if (i != 0) std::cout << ",\n";
        std::cout << "{\"name\":\"" << sample.name << "\",\"mu\":" << sample.mu
                  << ",\"dt\":" << sample.dt << ",\"r0\":";
        write_vector(sample.initial.position);
        std::cout << ",\"v0\":";
        write_vector(sample.initial.velocity);
        std::cout << ",\"r1\":";
        write_vector(result.state.position);
        std::cout << ",\"v1\":";
        write_vector(result.state.velocity);
        std::cout << ",\"status\":\"success\",\"iterations\":" << result.iterations << '}';
    }
    std::cout << "\n]\n";
    return 0;
}
