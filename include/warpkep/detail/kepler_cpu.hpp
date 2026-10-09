#pragma once

#include <warpkep/detail/solve_universal_kepler.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace warpkep::detail {

struct CartesianState {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
};

enum class KeplerPropagationStatus {
    success,
    invalid_input,
    unsupported_case,
    not_converged,
    numerical_failure
};

inline const char* propagation_status_name(KeplerPropagationStatus status) {
    switch (status) {
        case KeplerPropagationStatus::success: return "success";
        case KeplerPropagationStatus::invalid_input: return "invalid_input";
        case KeplerPropagationStatus::unsupported_case: return "unsupported_case";
        case KeplerPropagationStatus::not_converged: return "not_converged";
        case KeplerPropagationStatus::numerical_failure: return "numerical_failure";
    }
    return "unknown";
}

struct KeplerPropagationResult {
    CartesianState state;
    KeplerPropagationStatus status;
    int iterations;
};

inline KeplerPropagationResult kepler_failure(KeplerPropagationStatus status, int iterations = 0) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {{{nan, nan, nan}, {nan, nan, nan}}, status, iterations};
}

// Internal scalar CPU prototype; units must be consistent.
// Exactly radial motion at nonzero dt is deliberately unsupported.
// Extended-domain validation and tuning are still required.
inline KeplerPropagationResult propagate_kepler_cpu(
    const CartesianState& initial, double dt, double mu,
    const KeplerSolveOptions& options = {}
) {
    using Status = KeplerPropagationStatus;
    if (!std::isfinite(mu) || mu <= 0.0 || !std::isfinite(dt)
        || options.max_iterations <= 0
        || !std::isfinite(options.relative_tolerance)
        || options.relative_tolerance <= 0.0) {
        return kepler_failure(Status::invalid_input);
    }

    bool nonzero_position = false;
    for (std::size_t i = 0; i < 3; ++i) {
        if (!std::isfinite(initial.position[i]) || !std::isfinite(initial.velocity[i])) {
            return kepler_failure(Status::invalid_input);
        }
        nonzero_position = nonzero_position || initial.position[i] != 0.0;
    }
    if (!nonzero_position) {
        return kepler_failure(Status::invalid_input);
    }

    // Input validation comes first: dt = 0 does not make an invalid state valid.
    if (dt == 0.0) {
        return {initial, Status::success, 0};
    }

    const auto& r = initial.position;
    const auto& v = initial.velocity;
    const double radius0 = std::hypot(r[0], r[1], r[2]);
    const double speed0 = std::hypot(v[0], v[1], v[2]);
    if (!std::isfinite(radius0) || !std::isfinite(speed0)) {
        return kepler_failure(Status::numerical_failure);
    }
    if (speed0 == 0.0) {
        return kepler_failure(Status::unsupported_case);
    }

    // Scale before the cross product to avoid overflow in the radial check.
    const std::array<double, 3> ur{r[0] / radius0, r[1] / radius0, r[2] / radius0};
    const std::array<double, 3> uv{v[0] / speed0, v[1] / speed0, v[2] / speed0};
    const double cross_norm = std::hypot(
        ur[1] * uv[2] - ur[2] * uv[1],
        ur[2] * uv[0] - ur[0] * uv[2],
        ur[0] * uv[1] - ur[1] * uv[0]
    );
    if (cross_norm == 0.0) {
        return kepler_failure(Status::unsupported_case);
    }

    const double sqrt_mu = std::sqrt(mu);
    const double scaled_speed = speed0 / sqrt_mu;
    const double alpha = 2.0 / radius0 - scaled_speed * scaled_speed;
    const double rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
    if (!std::isfinite(alpha) || !std::isfinite(rv)) {
        return kepler_failure(Status::numerical_failure);
    }

    const auto solution = solve_universal_kepler({radius0, rv, alpha, sqrt_mu, dt}, options);
    switch (solution.status) {
        case KeplerSolveStatus::invalid_input:
            return kepler_failure(Status::invalid_input, solution.iterations);
        case KeplerSolveStatus::not_converged:
            return kepler_failure(Status::not_converged, solution.iterations);
        case KeplerSolveStatus::numerical_failure:
            return kepler_failure(Status::numerical_failure, solution.iterations);
        case KeplerSolveStatus::success:
            break;
    }

    // Universal Lagrange coefficients f, g, fdot and gdot.
    const double chi = solution.chi;
    const double chi2 = chi * chi;
    const double z = alpha * chi2;
    const auto cs = stumpff(z);
    const double f = 1.0 - chi2 * cs.c / radius0;
    const double g = dt - chi2 * chi * cs.s / sqrt_mu;
    if (!std::isfinite(f) || !std::isfinite(g)) {
        return kepler_failure(Status::numerical_failure, solution.iterations);
    }

    CartesianState final{};
    for (std::size_t i = 0; i < 3; ++i) {
        final.position[i] = f * r[i] + g * v[i];
        if (!std::isfinite(final.position[i])) {
            return kepler_failure(Status::numerical_failure, solution.iterations);
        }
    }
    const double radius1 = std::hypot(final.position[0], final.position[1], final.position[2]);
    if (!std::isfinite(radius1) || radius1 <= 0.0) {
        return kepler_failure(Status::numerical_failure, solution.iterations);
    }

    const double fdot = (sqrt_mu * chi / radius0 / radius1) * (z * cs.s - 1.0);
    const double gdot = 1.0 - chi2 * cs.c / radius1;
    if (!std::isfinite(fdot) || !std::isfinite(gdot)) {
        return kepler_failure(Status::numerical_failure, solution.iterations);
    }
    for (std::size_t i = 0; i < 3; ++i) {
        final.velocity[i] = fdot * r[i] + gdot * v[i];
        if (!std::isfinite(final.velocity[i])) {
            return kepler_failure(Status::numerical_failure, solution.iterations);
        }
    }

    return {final, Status::success, solution.iterations};
}

} // namespace warpkep::detail
