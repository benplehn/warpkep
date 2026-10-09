#pragma once

#include <cmath>

namespace warpkep::detail {

struct StumpffValues {
    double c;
    double s;
};

// C = c_2(z), S = c_3(z). The caller checks for nonfinite results.
inline StumpffValues stumpff(double z) {
    if (std::abs(z) <= 1.0) {
        double term_c = 0.5;
        double term_s = 1.0 / 6.0;
        double c = term_c;
        double s = term_s;

        for (int k = 1; k <= 10; ++k) {
            term_c *= -z / ((2.0 * k + 1.0) * (2.0 * k + 2.0));
            term_s *= -z / ((2.0 * k + 2.0) * (2.0 * k + 3.0));
            c += term_c;
            s += term_s;
        }

        return {c, s};
    }

    if (z > 0.0) {
        const double q = std::sqrt(z);
        const double half_sine = std::sin(0.5 * q);
        return {
            2.0 * half_sine * half_sine / z,
            (q - std::sin(q)) / (q * q * q)
        };
    }

    const double q = std::sqrt(-z);
    const double half_sinh = std::sinh(0.5 * q);
    return {
        2.0 * half_sinh * half_sinh / (-z),
        (std::sinh(q) - q) / (q * q * q)
    };
}

} // namespace warpkep::detail
