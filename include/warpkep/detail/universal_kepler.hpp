#pragma once

#include <warpkep/detail/stumpff.hpp>

namespace warpkep::detail {

struct UniversalKeplerParameters {
    double radius0;
    double r0_dot_v0;
    double alpha;
    double sqrt_mu;
    double dt;
};

struct UniversalKeplerEvaluation {
    double value;
    double derivative;
};

// Standard universal Kepler equation with z = alpha * chi^2.
// https://orbital-mechanics.space/time-since-periapsis-and-keplers-equation/universal-variables.html
// Parameters are validated by the caller before evaluating the equation.
inline UniversalKeplerEvaluation evaluate_universal_kepler(
    double chi, const UniversalKeplerParameters& parameters
) {
    const double chi2 = chi * chi;
    const double z = parameters.alpha * chi2;
    const auto cs = stumpff(z);
    const double a = parameters.r0_dot_v0 / parameters.sqrt_mu;
    const double b = 1.0 - parameters.alpha * parameters.radius0;

    const double value =
        a * chi2 * cs.c
        + b * chi2 * chi * cs.s
        + parameters.radius0 * chi
        - parameters.sqrt_mu * parameters.dt;

    const double derivative =
        a * chi * (1.0 - z * cs.s)
        + b * chi2 * cs.c
        + parameters.radius0;

    return {value, derivative};
}

} // namespace warpkep::detail
