#include <warpkep/detail/stumpff.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

struct ReferenceCase {
    double z;
    double c;
    double s;
};

int main() {
    // Constants evaluated independently with 80-digit decimal arithmetic.
    const std::array<ReferenceCase, 7> cases{
        {{0.0, 0.5, 0.16666666666666666667},
         {1.0, 0.45969769413186028260, 0.15852901519210349335},
         {-1.0, 0.54308063481524377848, 0.17520119364380145688},
         {4.0, 0.35403670913678559675, 0.13633782164678978808},
         {-4.0, 0.69054892277090786489, 0.20335755098087734596},
         {1e-12, 0.49999999999995833333, 0.16666666666665833333},
         {-1e-12, 0.50000000000004166667, 0.16666666666667500000}}
    };

    constexpr double tolerance = 1e-15;
    bool passed = true;
    std::cout << std::scientific << std::setprecision(3);

    for (const auto& reference : cases) {
        const auto result = warpkep::detail::stumpff(reference.z);
        const double error_c = std::abs(result.c - reference.c);
        const double error_s = std::abs(result.s - reference.s);

        std::cout << "z=" << reference.z << " error_C=" << error_c << " error_S=" << error_s
                  << '\n';

        if (!std::isfinite(result.c) || !std::isfinite(result.s) || error_c > tolerance
            || error_s > tolerance) {
            passed = false;
        }
    }

    if (!passed) {
        std::cerr << "Stumpff checks failed.\n";
        return 1;
    }

    std::cout << "Stumpff checks passed.\n";
    return 0;
}
