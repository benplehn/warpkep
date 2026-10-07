#include <warpkep/detail/solve_universal_kepler.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace wd = warpkep::detail;

struct RootReference {
    const char* name;
    wd::UniversalKeplerParameters parameters;
    double expected_chi;
};

int main() {
    const double pi = std::acos(-1.0);

    // Nontrivial roots computed independently by 80-digit decimal bisection.
    const std::array<RootReference, 8> cases{{
        {"circular quarter", {1.0, 0.0, 1.0, 1.0, pi / 2.0}, pi / 2.0},
        {"circular full turn", {1.0, 0.0, 1.0, 1.0, 2.0 * pi}, 2.0 * pi},
        {"circular backward", {1.0, 0.0, 1.0, 1.0, -pi / 2.0}, -pi / 2.0},
        {"elliptic", {1.5, 0.2, 0.3, 1.0, 0.7},
         0.44788600439244110256465008531324187},
        {"hyperbolic backward", {1.5, 0.2, -0.3, 1.0, -0.7},
         -0.46491201073344649248040788022872934},
        {"parabolic", {1.5, 0.2, 0.0, 1.0, 0.7},
         0.44382121606468919253239244293891638},
        {"different mu", {1.5, 0.4, 0.3, 2.0, 0.35},
         0.44788600439244110256465008531324187},
        {"zero duration", {1.5, 0.2, 0.3, 1.0, 0.0}, 0.0}
    }};

    bool passed = true;
    constexpr double tolerance = 1e-13;
    std::cout << std::scientific << std::setprecision(3);

    // Iterate over the test cases, solve the universal Kepler equation, and check the results against the expected roots.
    for (const auto& reference : cases) {
        const auto result = wd::solve_universal_kepler(reference.parameters);
        const double error = std::abs(result.chi - reference.expected_chi);
        std::cout << reference.name << ": error_chi=" << error
                  << " iterations=" << result.iterations << '\n';
        if (result.status != wd::KeplerSolveStatus::success
            || !std::isfinite(result.chi) || error > tolerance) {
            passed = false;
        }
    }

    const auto expect_failure = [&passed](
        const char* name, const wd::KeplerSolveResult& result,
        wd::KeplerSolveStatus expected_status
    ) {
        const bool matches = result.status == expected_status && std::isnan(result.chi);
        std::cout << name << ": " << (matches ? "passed" : "FAILED") << '\n';
        passed = passed && matches;
    };

    const double nan = std::numeric_limits<double>::quiet_NaN();
    expect_failure("zero initial radius",
        wd::solve_universal_kepler({0.0, 0.0, 1.0, 1.0, 0.7}),
        wd::KeplerSolveStatus::invalid_input);
    expect_failure("invalid input at zero duration",
        wd::solve_universal_kepler({0.0, 0.0, 1.0, 1.0, 0.0}),
        wd::KeplerSolveStatus::invalid_input);
    expect_failure("nonfinite duration",
        wd::solve_universal_kepler({1.0, 0.0, 1.0, 1.0, nan}),
        wd::KeplerSolveStatus::invalid_input);

    wd::KeplerSolveOptions limited;
    limited.max_iterations = 1;
    expect_failure("iteration limit",
        wd::solve_universal_kepler({1.5, 0.2, 0.3, 1.0, 0.7}, limited),
        wd::KeplerSolveStatus::not_converged);
    expect_failure("overflow",
        wd::solve_universal_kepler({1.0, 0.0, 1.0, 2.0, 1e308}),
        wd::KeplerSolveStatus::numerical_failure);

    if (!passed) {
        std::cerr << "Universal Kepler solver checks failed.\n";
        return 1;
    }
    std::cout << "Universal Kepler solver checks passed.\n";
    return 0;
}
