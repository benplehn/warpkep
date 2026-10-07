#pragma once

#include <warpkep/detail/universal_kepler.hpp>

#include <cmath>
#include <limits>

namespace warpkep::detail {

enum class KeplerSolveStatus {
    success,
    invalid_input,
    not_converged,
    numerical_failure
};

struct KeplerSolveOptions {
    int max_iterations = 64;
    double relative_tolerance = 8.0 * std::numeric_limits<double>::epsilon();
};

struct KeplerSolveResult {
    KeplerSolveStatus status;
    double chi;
    int iterations;
};

// Internal CPU prototype. The stopping tolerance applies to the equation
// residual relative to sqrt(mu) * abs(dt), not to position or velocity errors.
// Physically consistent parameters are derived from a validated Cartesian state.
inline KeplerSolveResult solve_universal_kepler(
    const UniversalKeplerParameters& p,
    const KeplerSolveOptions& options = {}
) {
    const double nan = std::numeric_limits<double>::quiet_NaN();

    if (!std::isfinite(p.radius0) || p.radius0 <= 0.0
        || !std::isfinite(p.r0_dot_v0) || !std::isfinite(p.alpha)
        || !std::isfinite(p.sqrt_mu) || p.sqrt_mu <= 0.0
        || !std::isfinite(p.dt) || options.max_iterations <= 0
        || !std::isfinite(options.relative_tolerance)
        || options.relative_tolerance <= 0.0) {
        return {KeplerSolveStatus::invalid_input, nan, 0};
    }

    if (p.dt == 0.0) {
        return {KeplerSolveStatus::success, 0.0, 0};
    }

    const double direction = p.dt > 0.0 ? 1.0 : -1.0;
    const double target = p.sqrt_mu * std::abs(p.dt);
    const double tolerance = options.relative_tolerance * target;
    double lower = 0.0;
    double upper = target / p.radius0;

    if (!std::isfinite(target) || !std::isfinite(tolerance)
        || !std::isfinite(upper) || upper <= 0.0) {
        return {KeplerSolveStatus::numerical_failure, nan, 0};
    }

    // Work with x >= 0 and chi = direction * x, including backward propagation.
    bool bracket_found = false;
    for (int expansion = 0; expansion < 64; ++expansion) {
        const auto evaluation = evaluate_universal_kepler(direction * upper, p);
        if (!std::isfinite(evaluation.value)
            || !std::isfinite(evaluation.derivative)
            || evaluation.derivative <= 0.0) {
            return {KeplerSolveStatus::numerical_failure, nan, 0};
        }
        if (std::abs(evaluation.value) <= tolerance) {
            return {KeplerSolveStatus::success, direction * upper, 0};
        }
        if (direction * evaluation.value > 0.0) {
            bracket_found = true;
            break;
        }
        lower = upper;
        upper *= 2.0;
        if (!std::isfinite(upper)) {
            return {KeplerSolveStatus::numerical_failure, nan, 0};
        }
    }

    if (!bracket_found) {
        return {KeplerSolveStatus::not_converged, nan, 0};
    }

    double x = lower + 0.5 * (upper - lower);
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        const auto evaluation = evaluate_universal_kepler(direction * x, p);
        if (!std::isfinite(evaluation.value)
            || !std::isfinite(evaluation.derivative)
            || evaluation.derivative <= 0.0) {
            return {KeplerSolveStatus::numerical_failure, nan, iteration};
        }
        if (std::abs(evaluation.value) <= tolerance) {
            return {KeplerSolveStatus::success, direction * x, iteration};
        }

        const double value = direction * evaluation.value;
        if (value < 0.0) {
            lower = x;
        } else {
            upper = x;
        }

        double candidate = x - value / evaluation.derivative;
        if (!std::isfinite(candidate) || candidate < lower || candidate > upper
            || candidate == x) {
            candidate = lower + 0.5 * (upper - lower);
        }
        if (candidate == x) {
            return {KeplerSolveStatus::not_converged, nan, iteration};
        }
        x = candidate;
    }

    return {KeplerSolveStatus::not_converged, nan, options.max_iterations};
}

} // namespace warpkep::detail
