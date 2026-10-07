#include <warpkep/detail/universal_kepler.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

struct EquationReference {
    const char* name;
    warpkep::detail::UniversalKeplerParameters parameters;
    double chi;
    double value;
    double derivative;
};

int main() {
    const double quarter_period = std::acos(-1.0) / 2.0;

    // Nontrivial constants evaluated with independent 80-digit arithmetic.
    const std::array<EquationReference, 5> cases{{
        {"circular root", {1.0, 0.0, 1.0, 1.0, quarter_period},
         quarter_period, 0.0, 1.0},
        {"zero anomaly", {1.5, 0.2, 0.3, 1.0, 0.7},
         0.0, -0.7, 1.5},
        {"elliptic", {1.5, 0.2, 0.3, 1.0, 0.7},
         0.8, 0.609467358731604254372639, 1.82813088864382830753115},
        {"hyperbolic backward", {1.5, 0.2, -0.3, 1.0, -0.7},
         -0.8, -0.559896041847198345719271, 1.80630229955671493394354},
        {"parabolic", {1.5, 0.2, 0.0, 1.0, 0.7},
         0.8, 0.649333333333333333333333, 1.98}
    }};

    constexpr double tolerance = 5e-15;
    bool passed = true;
    std::cout << std::scientific << std::setprecision(3);

    for (const auto& reference : cases) {
        const auto result = warpkep::detail::evaluate_universal_kepler(
            reference.chi, reference.parameters
        );
        const double error_value = std::abs(result.value - reference.value);
        const double error_derivative =
            std::abs(result.derivative - reference.derivative);

        std::cout << reference.name
                  << ": error_F=" << error_value
                  << " error_dF=" << error_derivative << '\n';

        if (!std::isfinite(result.value) || !std::isfinite(result.derivative)
            || error_value > tolerance || error_derivative > tolerance) {
            passed = false;
        }
    }

    if (!passed) {
        std::cerr << "Universal Kepler equation checks failed.\n";
        return 1;
    }

    std::cout << "Universal Kepler equation checks passed.\n";
    return 0;
}
