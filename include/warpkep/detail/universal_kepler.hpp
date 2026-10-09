#pragma once

#include <warpkep/detail/stumpff.hpp>
#include <warpkep/detail/config.hpp>

namespace warpkep::detail {

template <typename T>
struct UniversalKeplerParameters {
    T radius0;
    T r0_dot_v0;
    T alpha;
    T sqrt_mu;
    T dt;
};

template <typename T>
struct UniversalKeplerEvaluation {
    T value;
    T derivative;
};

// Standard universal Kepler equation with z = alpha * chi^2.
// https://orbital-mechanics.space/time-since-periapsis-and-keplers-equation/universal-variables.html
// Parameters are validated by the caller before evaluating the equation.
template <typename T>
WARPKEP_HD inline UniversalKeplerEvaluation<T> evaluate_universal_kepler(T chi, const UniversalKeplerParameters<T>& parameters) {
    const T chi2 = chi * chi;
    const T z = parameters.alpha * chi2;
    const auto cs = stumpff(z);
    const T a = parameters.r0_dot_v0 / parameters.sqrt_mu;
    const T b = T{1} - parameters.alpha * parameters.radius0;

    const T value = a * chi2 * cs.c + b * chi2 * chi * cs.s + parameters.radius0 * chi
                    - parameters.sqrt_mu * parameters.dt;

    const T derivative = a * chi * (T{1} - z * cs.s) + b * chi2 * cs.c + parameters.radius0;

    return {value, derivative};
}

} // namespace warpkep::detail
