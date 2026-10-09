#pragma once

#include <cmath>
#include <type_traits>

namespace warpkep::detail {

template <typename T>
struct StumpffValues {
    T c;
    T s;
};

// C = c_2(z), S = c_3(z). The caller checks for nonfinite results.
template <typename T>
inline StumpffValues<T> stumpff(T z) {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, double>,
        "stumpff requires float or double type"
    );
    if (std::abs(z) <= T{1}) {
        T term_c = T{1} / T{2};
        T term_s = T{1} / T{6};
        T c = term_c;
        T s = term_s;

        for (int k = 1; k <= 10; ++k) {
            const T kt = static_cast<T>(k);
            term_c *= -z / ((T{2} * kt + T{1}) * (T{2} * kt + T{2}));
            term_s *= -z / ((T{2} * kt + T{2}) * (T{2} * kt + T{3}));
            c += term_c;
            s += term_s;
        }

        return {c, s};
    }

    if (z > T{0}) {
        const T q = std::sqrt(z);
        const T half_sine = std::sin(q / T{2});

        return {T{2} * half_sine * half_sine / z, (q - std::sin(q)) / (q * q * q)};
    }

    const T q = std::sqrt(-z);
    const T half_sinh = std::sinh(q / T{2});
    return {T{2} * half_sinh * half_sinh / (-z), (std::sinh(q) - q) / (q * q * q)};
}

} // namespace warpkep::detail
