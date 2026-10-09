#include <warpkep/detail/kepler_cpu.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>

namespace wd = warpkep::detail;

struct AnalyticalCase {
    const char* name;
    wd::CartesianState initial;
    double dt;
    double mu;
    wd::CartesianState expected;
};

double distance(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

int main() {
    const double pi = std::acos(-1.0);
    const double sqrt3 = std::sqrt(3.0);
    const std::array<AnalyticalCase, 7> cases{{
        {"circular quarter", {{1, 0, 0}, {0, 1, 0}}, pi / 2, 1,
         {{0, 1, 0}, {-1, 0, 0}}},
        {"circular backward", {{1, 0, 0}, {0, 1, 0}}, -pi / 2, 1,
         {{0, -1, 0}, {1, 0, 0}}},
        {"circular full turn", {{1, 0, 0}, {0, 1, 0}}, 2 * pi, 1,
         {{1, 0, 0}, {0, 1, 0}}},
        {"inclined circle", {{0, 1, 0}, {0, 0, 1}}, pi / 2, 1,
         {{0, 0, 1}, {0, -1, 0}}},
        {"scaled circle", {{2, 0, 0}, {0, 2, 0}}, pi / 2, 8,
         {{0, 2, 0}, {-2, 0, 0}}},
        // a=1, e=0.5, mu=1: periapsis to apoapsis in half a period.
        {"elliptic half turn", {{0.5, 0, 0}, {0, sqrt3, 0}}, pi, 1,
         {{-1.5, 0, 0}, {0, -1 / sqrt3, 0}}},
        {"zero duration, zero velocity", {{1, 2, 3}, {0, 0, 0}}, 0, 1,
         {{1, 2, 3}, {0, 0, 0}}}
    }};
    bool passed = true;
    constexpr double tolerance = 1e-12; // These normalized analytical cases only.
    std::cout << std::scientific << std::setprecision(3);
    for (const auto& reference : cases) {
        const auto result =
            wd::propagate_kepler_cpu(reference.initial, reference.dt, reference.mu);
        const double er = distance(result.state.position, reference.expected.position);
        const double ev = distance(result.state.velocity, reference.expected.velocity);
        std::cout << reference.name
                  << ": status=" << wd::propagation_status_name(result.status)
                  << " error_r=" << er << " error_v=" << ev << '\n';
        if (result.status != wd::KeplerPropagationStatus::success
            || !std::isfinite(er) || !std::isfinite(ev)
            || er > tolerance || ev > tolerance) {
            passed = false;
        }
    }

    const auto expect_failure = [&passed](
        const char* name, const wd::KeplerPropagationResult& result,
        wd::KeplerPropagationStatus expected
    ) {
        bool matches = result.status == expected;
        for (std::size_t i = 0; i < 3; ++i) {
            matches = matches && std::isnan(result.state.position[i])
                      && std::isnan(result.state.velocity[i]);
        }
        std::cout << name << ": " << (matches ? "passed" : "FAILED") << '\n';
        passed = passed && matches;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    expect_failure("zero radius",
        wd::propagate_kepler_cpu({{0, 0, 0}, {0, 1, 0}}, 1, 1),
        wd::KeplerPropagationStatus::invalid_input);
    expect_failure("invalid mu at zero duration",
        wd::propagate_kepler_cpu({{1, 0, 0}, {0, 1, 0}}, 0, -1),
        wd::KeplerPropagationStatus::invalid_input);
    expect_failure("nonfinite state",
        wd::propagate_kepler_cpu({{1, 0, 0}, {0, nan, 0}}, 1, 1),
        wd::KeplerPropagationStatus::invalid_input);
    expect_failure("radial trajectory",
        wd::propagate_kepler_cpu({{1, 0, 0}, {0.2, 0, 0}}, 1, 1),
        wd::KeplerPropagationStatus::unsupported_case);
    expect_failure("zero velocity",
        wd::propagate_kepler_cpu({{1, 0, 0}, {0, 0, 0}}, 1, 1),
        wd::KeplerPropagationStatus::unsupported_case);

    wd::KeplerSolveOptions limited;
    limited.max_iterations = 1;
    expect_failure("iteration limit",
        wd::propagate_kepler_cpu({{1.5, 0.1, -0.2}, {-0.15, 0.6, 0.25}}, 1.7, 1, limited),
        wd::KeplerPropagationStatus::not_converged);

    if (!passed) {
        std::cerr << "Kepler propagation checks failed.\n";
        return 1;
    }
    std::cout << "Kepler propagation checks passed.\n";
    return 0;
}
