#include "kepler_cases.hpp"

#include <warpkep/detail/kepler_cpu.hpp>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>

namespace wd = warpkep::detail;

void write_vector(const wd::Vector3<double>& v) {
    std::cout << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
}

int main() {
    const auto samples = kepler_validation_cases();

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "[\n";

    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        const auto result =
            wd::propagate_kepler_cpu(sample.initial, sample.dt, sample.mu);

        if (result.status != wd::KeplerPropagationStatus::success) {
            std::cerr << sample.name << ": "
                      << wd::propagation_status_name(result.status) << '\n';
            return 1;
        }

        if (i != 0) {
            std::cout << ",\n";
        }
        std::cout << "{\"name\":\"" << sample.name << "\""
                  << ",\"mu\":" << sample.mu
                  << ",\"dt\":" << sample.dt
                  << ",\"length_scale\":" << sample.length_scale
                  << ",\"velocity_scale\":" << sample.velocity_scale
                  << ",\"tolerance\":" << sample.tolerance
                  << ",\"r0\":";
        write_vector(sample.initial.position);
        std::cout << ",\"v0\":";
        write_vector(sample.initial.velocity);
        std::cout << ",\"r1\":";
        write_vector(result.state.position);
        std::cout << ",\"v1\":";
        write_vector(result.state.velocity);
        std::cout << ",\"status\":\"success\",\"iterations\":"
                  << result.iterations << '}';
    }

    std::cout << "\n]\n";
    return 0;
}