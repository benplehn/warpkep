#pragma once

#include <warpkep/detail/solve_universal_kepler.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace warpkep::detail {

// Position and velocity of a body, in consistent units.
// std::array has a fixed size known at compile time: no dynamic allocation.
template <typename T>
struct CartesianState {
    std::array<T, 3> position;
    std::array<T, 3> velocity;
};

// Outcome of a propagation. enum class values must be written with their
// scope (KeplerPropagationStatus::success) and never convert silently to int.
enum class KeplerPropagationStatus {
    success,
    invalid_input,
    unsupported_case,
    not_converged,
    numerical_failure
};

// Human-readable status name, used by tests and diagnostics.
inline const char* propagation_status_name(KeplerPropagationStatus status) {
    switch (status) {
    case KeplerPropagationStatus::success:
        return "success";
    case KeplerPropagationStatus::invalid_input:
        return "invalid_input";
    case KeplerPropagationStatus::unsupported_case:
        return "unsupported_case";
    case KeplerPropagationStatus::not_converged:
        return "not_converged";
    case KeplerPropagationStatus::numerical_failure:
        return "numerical_failure";
    }
    return "unknown";
}

template <typename T>
struct KeplerPropagationResult {
    CartesianState<T> state;
    KeplerPropagationStatus status;
    int iterations;
};

// Builds a failed result whose state is filled with NaN, so that a caller
// Failed states are filled with NaN.
// The status remains the authoritative indication of failure.
template <typename T>
inline KeplerPropagationResult<T>
kepler_failure(KeplerPropagationStatus status, int iterations = 0) {
    const T nan = std::numeric_limits<T>::quiet_NaN();
    return {{{nan, nan, nan}, {nan, nan, nan}}, status, iterations};
}

// Internal scalar CPU prototype; units must be consistent.
// Propagates a Cartesian state over dt on a two-body Keplerian orbit:
//   1. validate inputs,
//   2. compute orbital quantities (r0, v0, alpha = 1/a, r.v),
//   3. solve the universal Kepler equation for chi,
//   4. build the new position with the Lagrange coefficients f and g,
//   5. build the new velocity with fdot and gdot.
// Radial motion is deliberately outside this prototype's supported domain.
// Extended-domain validation and tuning are still required.
template <typename T>
inline KeplerPropagationResult<T> propagate_kepler_cpu(
    const CartesianState<T>& initial, T dt, T mu, const KeplerSolveOptions<T>& options = {}
) {
    using Status = KeplerPropagationStatus; // Local shorthand.

    // --- 1. Input validation -------------------------------------------------
    if (!std::isfinite(mu) || mu <= T{0} || !std::isfinite(dt) || options.max_iterations <= 0
        || !std::isfinite(options.relative_tolerance) || options.relative_tolerance <= T{0}) {
        return kepler_failure<T>(Status::invalid_input);
    }

    // Every component must be finite and the position must not be the origin.
    bool nonzero_position = false;
    for (std::size_t i = 0; i < 3; ++i) {
        if (!std::isfinite(initial.position[i]) || !std::isfinite(initial.velocity[i])) {
            return kepler_failure<T>(Status::invalid_input);
        }
        nonzero_position = nonzero_position || initial.position[i] != T{0};
    }
    if (!nonzero_position) {
        return kepler_failure<T>(Status::invalid_input);
    }

    // Input validation comes first: dt = 0 does not make an invalid state valid.
    if (dt == T{0}) {
        return {initial, Status::success, 0};
    }

    // --- 2. Orbital quantities -----------------------------------------------
    // References (const auto&) alias the arrays without copying them.
    const auto& r = initial.position;
    const auto& v = initial.velocity;

    // hypot computes sqrt(x^2 + y^2 + z^2) without intermediate overflow.
    const T radius0 = std::hypot(r[0], r[1], r[2]);
    const T speed0 = std::hypot(v[0], v[1], v[2]);
    if (!std::isfinite(radius0) || !std::isfinite(speed0)) {
        return kepler_failure<T>(Status::numerical_failure);
    }
    if (speed0 == T{0}) {
        return kepler_failure<T>(Status::unsupported_case);
    }

    // Radial check: if r and v are parallel, |r x v| = 0 and the universal
    // formulation is not supported here.
    // Scale before the cross product to avoid overflow in the radial check.
    const std::array<T, 3> ur{r[0] / radius0, r[1] / radius0, r[2] / radius0};
    const std::array<T, 3> uv{v[0] / speed0, v[1] / speed0, v[2] / speed0};
    const T cross_norm = std::hypot(
        ur[1] * uv[2] - ur[2] * uv[1], ur[2] * uv[0] - ur[0] * uv[2], ur[0] * uv[1] - ur[1] * uv[0]
    );
    if (cross_norm == T{0}) {
        return kepler_failure<T>(Status::unsupported_case);
    }

    // alpha = 2/r0 - v0^2/mu = 1/a:
    // alpha > 0 elliptic, alpha = 0 parabolic, alpha < 0 hyperbolic.
    const T sqrt_mu = std::sqrt(mu);
    const T scaled_speed = speed0 / sqrt_mu;
    const T alpha = T{2} / radius0 - scaled_speed * scaled_speed;
    const T rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]; // Dot product r . v.
    if (!std::isfinite(alpha) || !std::isfinite(rv)) {
        return kepler_failure<T>(Status::numerical_failure);
    }

    // --- 3. Universal Kepler equation ----------------------------------------
    const auto solution = solve_universal_kepler<T>({radius0, rv, alpha, sqrt_mu, dt}, options);

    // Translate the solver status into a propagation status.
    switch (solution.status) {
    case KeplerSolveStatus::invalid_input:
        return kepler_failure<T>(Status::invalid_input, solution.iterations);
    case KeplerSolveStatus::not_converged:
        return kepler_failure<T>(Status::not_converged, solution.iterations);
    case KeplerSolveStatus::numerical_failure:
        return kepler_failure<T>(Status::numerical_failure, solution.iterations);
    case KeplerSolveStatus::success:
        break;
    }

    // --- 4. New position: r1 = f * r0 + g * v0 --------------------------------
    // Universal Lagrange coefficients f, g, fdot and gdot.
    const T chi = solution.chi;
    const T chi2 = chi * chi;
    const T z = alpha * chi2;
    const auto cs = stumpff(z);
    const T f = T{1} - chi2 * cs.c / radius0;
    const T g = dt - chi2 * chi * cs.s / sqrt_mu;
    if (!std::isfinite(f) || !std::isfinite(g)) {
        return kepler_failure<T>(Status::numerical_failure, solution.iterations);
    }

    CartesianState<T> final_state{};
    for (std::size_t i = 0; i < 3; ++i) {
        final_state.position[i] = f * r[i] + g * v[i];
        if (!std::isfinite(final_state.position[i])) {
            return kepler_failure<T>(Status::numerical_failure, solution.iterations);
        }
    }

    // fdot depends on the new radius r1, so it is computed after the position.
    const T radius1 =
        std::hypot(final_state.position[0], final_state.position[1], final_state.position[2]);
    if (!std::isfinite(radius1) || radius1 <= T{0}) {
        return kepler_failure<T>(Status::numerical_failure, solution.iterations);
    }

    // --- 5. New velocity: v1 = fdot * r0 + gdot * v0 --------------------------
    const T fdot = (sqrt_mu * chi / radius0 / radius1) * (z * cs.s - T{1});
    const T gdot = T{1} - chi2 * cs.c / radius1;
    if (!std::isfinite(fdot) || !std::isfinite(gdot)) {
        return kepler_failure<T>(Status::numerical_failure, solution.iterations);
    }
    for (std::size_t i = 0; i < 3; ++i) {
        final_state.velocity[i] = fdot * r[i] + gdot * v[i];
        if (!std::isfinite(final_state.velocity[i])) {
            return kepler_failure<T>(Status::numerical_failure, solution.iterations);
        }
    }

    return {final_state, Status::success, solution.iterations};
}

} // namespace warpkep::detail