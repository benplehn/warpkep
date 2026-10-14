#pragma once

#include <warpkep/detail/kepler_cpu.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// One validation case shared by the CPU sample generator and the GPU test.
// Data only: nothing here calls the propagator.
struct KeplerValidationCase {
    std::string name;       // owns its text, safe to build in loops
    warpkep::detail::CartesianState<double> initial;
    double dt;
    double mu;
    double length_scale;    // L used for position errors
    double velocity_scale;  // V used for velocity errors
    double tolerance;       // on scaled errors
};

// Periapsis state with r_p = 1 and mu = 1:
// r = (1, 0, 0), v = (0, sqrt(1 + e), 0).
inline KeplerValidationCase periapsis_case(std::string name, double e, double dt) {
    return {
        std::move(name),
        {{1.0, 0.0, 0.0}, {0.0, std::sqrt(1.0 + e), 0.0}},
        dt,
        1.0,    // mu
        1.0,    // length_scale
        1.0,    // velocity_scale
        1e-11   // initial target, not tuned
    };
}

// Cyclic shift (x, y, z) -> (z, x, y): same orbit, other orientation.
inline KeplerValidationCase permute_axes(KeplerValidationCase c) {
    const auto r = c.initial.position;  // copies: read before overwrite
    const auto v = c.initial.velocity;

    c.initial.position[0] = r[2];
    c.initial.position[1] = r[0];
    c.initial.position[2] = r[1];

    c.initial.velocity[0] = v[2];
    c.initial.velocity[1] = v[0];
    c.initial.velocity[2] = v[1];

    c.name += "_zxy";
    return c;
}

// Same dimensionless problem in other units:
// r' = L r, v' = (L/T) v, dt' = T dt, mu' = (L^3 / T^2) mu.
inline KeplerValidationCase rescale(KeplerValidationCase c, double L, double T) {
    const double V = L / T;

    for (std::size_t k = 0; k < 3; ++k) {
        c.initial.position[k] *= L;
        c.initial.velocity[k] *= V;
    }
    c.dt *= T;
    c.mu *= L * L * L / (T * T);
    c.length_scale *= L;
    c.velocity_scale *= V;

    c.name += "_scaled";
    return c;
}

// inline: this header is included by several programs.
inline std::vector<KeplerValidationCase> kepler_validation_cases() {
    const double pi = std::acos(-1.0);

    std::vector<KeplerValidationCase> cases{
        // Legacy samples: unit scales, tolerance 1e-12.
        // name, {position, velocity}, dt, mu, L, V, tolerance
        {"circular", {{1, 0, 0}, {0, 1, 0}}, pi / 2, 1.0, 1.0, 1.0, 1e-12},
        {"elliptic", {{1.5, 0.1, -0.2}, {-0.15, 0.6, 0.25}}, 1.7, 1.0, 1.0, 1.0, 1e-12},
        {"hyperbolic", {{1, 0, 0}, {0.2, 1.6, 0.1}}, 0.7, 1.0, 1.0, 1.0, 1e-12},
        {"backward", {{1.5, 0.1, -0.2}, {-0.15, 0.6, 0.25}}, -1.7, 1.0, 1.0, 1.0, 1e-12},
    };

    // Literal labels: std::to_string(1 - 1e-8) would print "1.000000",
    // the same as std::to_string(1.0).
    struct Labeled {
        const char* label;
        double value;
    };

    const Labeled eccentricities[] = {
        // Ellipses
        {"e0.5", 0.5}, {"e0.9", 0.9}, {"e0.99", 0.99},
        // Near-parabolic
        {"e1-1e-8", 1.0 - 1e-8}, {"e1", 1.0}, {"e1+1e-8", 1.0 + 1e-8},
        // Hyperbolas
        {"e1.2", 1.2}, {"e2", 2.0},
    };

    const Labeled durations[] = {
        {"dt-1", -1.0}, {"dt0", 0.0}, {"dt+1", 1.0},
    };

    // 8 eccentricities x 3 durations = 24 cases.
    for (const auto& e : eccentricities) {
        for (const auto& dt : durations) {
            cases.push_back(periapsis_case(
                std::string("periapsis_") + e.label + "_" + dt.label,
                e.value,
                dt.value));
        }
    }

    // Ten revolutions, forward and backward: a = r_p / (1 - e), P = 2 pi sqrt(a^3 / mu).
    {
        const double e = 0.5;
        const double a = 1.0 / (1.0 - e);
        const double period = 2.0 * pi * std::sqrt(a * a * a / 1.0);

        cases.push_back(periapsis_case("periapsis_e0.5_10periods_forward", e, 10.0 * period));
        cases.push_back(periapsis_case("periapsis_e0.5_10periods_backward", e, -10.0 * period));
    }

    // Orientation: one ellipse, one hyperbola.
    cases.push_back(permute_axes(periapsis_case("periapsis_e0.5_dt+1", 0.5, 1.0)));
    cases.push_back(permute_axes(periapsis_case("periapsis_e2_dt+1", 2.0, 1.0)));

    // Units: L = 1000, T = 1000 -> mu' = 1000 mu, V = 1.
    cases.push_back(rescale(periapsis_case("periapsis_e0.5_dt+1", 0.5, 1.0), 1000.0, 1000.0));

    // 4 legacy + 24 + 2 + 2 + 1 = 33 cases.
    return cases;
}