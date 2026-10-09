#include <cmath>
#include <iomanip>
#include <iostream>
#include <warpkep/detail/kepler_cpu.hpp>

namespace {

namespace wd = warpkep::detail;

// propagate one case with mu=1 and compares it to analytical reference.
// Returns true if the test passed, false otherwise.
template <typename T>
bool check_case(
    const char* name,
    const warpkep::detail::CartesianState<T>& initial,
    T dt,
    const warpkep::detail::CartesianState<T>& expected,
    T tolerance
) {

    const auto result = warpkep::detail::propagate_kepler_cpu(initial, dt, T{1});

    if (result.status != warpkep::detail::KeplerPropagationStatus::success) {
        std::cerr << "Propagation failed with status: "
                  << warpkep::detail::propagation_status_name(result.status) << "\n";
        return false;
    }

    const T position_error = std::hypot(
        result.state.position[0] - expected.position[0],
        result.state.position[1] - expected.position[1],
        result.state.position[2] - expected.position[2]
    );
    const T velocity_error = std::hypot(
        result.state.velocity[0] - expected.velocity[0],
        result.state.velocity[1] - expected.velocity[1],
        result.state.velocity[2] - expected.velocity[2]
    );

    std::cout << "  " << name << ": position_error=" << position_error
              << " velocity_error=" << velocity_error << " iterations=" << result.iterations
              << '\n';

    const bool passed = std::isfinite(position_error) && std::isfinite(velocity_error)
                        && position_error <= tolerance && velocity_error <= tolerance;
    if (!passed) {
        std::cerr << name << ": tolerance exceeded.\n";
    }
    return passed;
}

// run 3 analytical cases in precision T
// Constants are computed in T so that the reference itself has type T
template <typename T>
bool run_cases(const char* type_name, T tolerance) {

    using State = warpkep::detail::CartesianState<T>;
    const T pi = std::acos(T{-1});
    const T sqrt3 = std::sqrt(T{3});

    std::cout << "Running precision tests for type " << type_name << " with tolerance " << tolerance
              << "\n";

    bool ok = true;

    ok = check_case<T>(
             "circular quarter",
             State{{T{1}, T{0}, T{0}}, {T{0}, T{1}, T{0}}},
             pi / T{2},
             State{{T{0}, T{1}, T{0}}, {T{-1}, T{0}, T{0}}},
             tolerance
         )
         && ok;

    ok = check_case<T>(
             "circular backward",
             State{{T{1}, T{0}, T{0}}, {T{0}, T{1}, T{0}}},
             -pi / T{2},
             State{{T{0}, T{-1}, T{0}}, {T{1}, T{0}, T{0}}},
             tolerance
         )
         && ok;

    // Ellipse with a = 1, e = 0.5: periapsis (r = 0.5) to apoapsis (r = 1.5)
    // in half a period (pi). Apoapsis speed = 0.5 * sqrt(3) / 1.5 = 1 / sqrt(3).
    ok = check_case<T>(
             "ellipse peri->apo",
             State{{T{1} / T{2}, T{0}, T{0}}, {T{0}, sqrt3, T{0}}},
             pi,
             State{{T{-3} / T{2}, T{0}, T{0}}, {T{0}, T{-1} / sqrt3, T{0}}},
             tolerance
         )
         && ok;

    return ok;
}

} // namespace

int main() {
    std::cout << std::scientific << std::setprecision(3);

    // Two separate statements: the float cases run even if double fails.
    const bool double_ok = run_cases<double>("double", 1e-12);
    const bool float_ok = run_cases<float>("float", 5e-6f);

    if (!double_ok || !float_ok) {
        std::cerr << "Kepler precision checks failed.\n";
        return 1;
    }
    std::cout << "Kepler precision checks passed.\n";
    return 0;
}