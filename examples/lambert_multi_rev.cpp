// CPU-only, normalized units: mu = 1, length scale = 1.
#include <warpkep/detail/kepler_cpu.hpp>
#include <warpkep/detail/lambert_multi_rev_cpu.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace wd = warpkep::detail;
using Status = warpkep::LambertStatus;

const char* status_name(Status status) {
    switch (status) {
    case Status::success: return "success";
    case Status::invalid_input: return "invalid_input";
    case Status::unsupported_case: return "unsupported_case";
    case Status::not_converged: return "not_converged";
    case Status::numerical_failure: return "numerical_failure";
    case Status::no_solution: return "no_solution";
    case Status::near_minimum: return "near_minimum";
    }
    return "unknown";
}

double distance(const wd::Vector3<double>& a, const wd::Vector3<double>& b) {
    return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

int main() {
    // [solve-begin]
    const wd::Vector3<double> r1{1.0, 0.0, 0.0};
    const wd::Vector3<double> r2{0.0, 1.0, 0.0};
    const double pi = std::acos(-1.0);
    const double mu = 1.0;
    const double dt = pi / 2.0 + 2.0 * pi;
    const auto arc = wd::LambertArc::short_arc;
    const auto transfer = wd::solve_lambert_multi_rev_cpu<double>(
        r1, r2, dt, mu, arc, 1
    );
    // [solve-end]

    std::cout << std::setprecision(12);
    std::cout << "M=1: " << status_name(transfer.status)
              << ", solutions=" << transfer.solution_count
              << ", complete=" << std::boolalpha << transfer.complete << '\n';

    // [slots-begin]
    for (int i = 0; i < transfer.slot_count; ++i) {
        const auto& solution = transfer.solutions[i];
        if (solution.result.status != Status::success) {
            std::cerr << "Slot " << i << ": "
                      << status_name(solution.result.status) << '\n';
            continue;
        }
        const auto& v1 = solution.result.departure_velocity;
        const auto& v2 = solution.result.arrival_velocity;
        const auto closure = wd::propagate_kepler_cpu<double>({r1, v1}, dt, mu);
        if (closure.status != warpkep::KeplerPropagationStatus::success) {
            std::cerr << "Kepler closure failed\n";
            return 1;
        }
        const double position_error = distance(closure.state.position, r2);
        const double velocity_error = distance(closure.state.velocity, v2);
        std::cout << "  " << (solution.branch == wd::LambertBranch::left ? "left" : "right")
                  << ": x=" << solution.x
                  << ", closure_r=" << position_error
                  << ", closure_v=" << velocity_error << '\n';
        // Absolute tolerances for this normalized example only.
        if (!(position_error <= 1e-10 && velocity_error <= 1e-10)) {
            return 1;
        }
    }
    // [slots-end]

    if (!transfer.complete || transfer.solution_count != 2) {
        return 1;
    }

    // [enumerate-begin]
    const auto prepared = wd::prepare_lambert_geometry(r1, r2, dt, mu, arc);
    if (prepared.status != Status::success) {
        return 1;
    }
    for (int revolutions = 0; revolutions <= 3; ++revolutions) {
        const auto family = wd::solve_lambert_multi_rev_prepared(
            prepared.geometry, mu, arc, revolutions
        );
        std::cout << "M=" << revolutions << ": " << status_name(family.status)
                  << ", solutions=" << family.solution_count << '\n';
        // no_solution is a completed classification for this M.
        // near_minimum or a numerical failure needs further investigation.
        if (!family.complete) {
            return 1;
        }
    }
    // [enumerate-end]
}
