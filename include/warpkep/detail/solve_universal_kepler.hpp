#pragma once

#include <warpkep/detail/universal_kepler.hpp>
#include <warpkep/detail/config.hpp>
#include <cmath>
#include <warpkep/detail/numeric_limits.hpp>

namespace warpkep::detail {

enum class KeplerSolveStatus { success, invalid_input, not_converged, numerical_failure };

template <typename T>
struct KeplerSolveOptions {
    int max_iterations = 64;
    T relative_tolerance = T{8} * NumericLimits<T>::epsilon();
};

template <typename T>
struct KeplerSolveResult {
    KeplerSolveStatus status;
    T chi;
    int iterations;
};

// Internal CPU prototype. The stopping tolerance applies to the equation
// residual relative to sqrt(mu) * abs(dt), not to position or velocity errors.
// Physically consistent parameters are derived from a validated Cartesian state.
template <typename T>
WARPKEP_HD inline KeplerSolveResult<T> solve_universal_kepler(
    const UniversalKeplerParameters<T>& p, const KeplerSolveOptions<T>& options = {}
) {
    const T nan = NumericLimits<T>::quiet_NaN();

    if (!std::isfinite(p.radius0) || p.radius0 <= T{0} || !std::isfinite(p.r0_dot_v0)
        || !std::isfinite(p.alpha) || !std::isfinite(p.sqrt_mu) || p.sqrt_mu <= T{0}
        || !std::isfinite(p.dt) || options.max_iterations <= 0
        || !std::isfinite(options.relative_tolerance) || options.relative_tolerance <= T{0}) {
        return {KeplerSolveStatus::invalid_input, nan, 0};
    }

    if (p.dt == T{0}) {
        return {KeplerSolveStatus::success, T{0}, 0};
    }

    const T direction = p.dt > T{0} ? T{1} : T{-1};
    const T target = p.sqrt_mu * std::abs(p.dt);
    const T tolerance = options.relative_tolerance * target;
    T lower = T{0};
    T upper = target / p.radius0;

    if (!std::isfinite(target) || !std::isfinite(tolerance) || !std::isfinite(upper)
        || upper <= T{0}) {
        return {KeplerSolveStatus::numerical_failure, nan, 0};
    }

    // Work with x >= 0 and chi = direction * x, including backward propagation.
    bool bracket_found = false;
    for (int expansion = 0; expansion < 64; ++expansion) {
        const auto evaluation = evaluate_universal_kepler(direction * upper, p);
        if (!std::isfinite(evaluation.value) || !std::isfinite(evaluation.derivative)
            || evaluation.derivative <= T{0}) {
            return {KeplerSolveStatus::numerical_failure, nan, 0};
        }
        if (std::abs(evaluation.value) <= tolerance) {
            return {KeplerSolveStatus::success, direction * upper, 0};
        }
        if (direction * evaluation.value > T{0}) {
            bracket_found = true;
            break;
        }
        lower = upper;
        upper *= T{2};
        if (!std::isfinite(upper)) {
            return {KeplerSolveStatus::numerical_failure, nan, 0};
        }
    }

    if (!bracket_found) {
        return {KeplerSolveStatus::not_converged, nan, 0};
    }

    T x = lower + (upper - lower) / T{2};
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        const auto evaluation = evaluate_universal_kepler(direction * x, p);
        if (!std::isfinite(evaluation.value) || !std::isfinite(evaluation.derivative)
            || evaluation.derivative <= T{0}) {
            return {KeplerSolveStatus::numerical_failure, nan, iteration};
        }
        if (std::abs(evaluation.value) <= tolerance) {
            return {KeplerSolveStatus::success, direction * x, iteration};
        }

        const T value = direction * evaluation.value;
        if (value < T{0}) {
            lower = x;
        } else {
            upper = x;
        }

        T candidate = x - value / evaluation.derivative;
        if (!std::isfinite(candidate) || candidate < lower || candidate > upper || candidate == x) {
            candidate = lower + (upper - lower) / T{2};
        }
        if (candidate == x) {
            return {KeplerSolveStatus::not_converged, nan, iteration};
        }
        x = candidate;
    }

    return {KeplerSolveStatus::not_converged, nan, options.max_iterations};
}

} // namespace warpkep::detail
