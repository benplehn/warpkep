#pragma once

#include <warpkep/lambert_status.hpp>
#include <warpkep/vector3_soa.hpp>

namespace warpkep {

    template <typename T>
    struct LambertSolutionSoAView {
        Vector3SoAView<T> departure_velocity;
        Vector3SoAView<T> arrival_velocity;
        T* x{};
        LambertStatus* statuses{};
        int* iterations{};
    };

    template <typename T>
    struct LambertMultiRevSoAView {
        LambertSolutionSoAView<T> solutions[2]{};
        LambertStatus* statuses{};
        int* slot_count{};
        int* solution_count{};
        int* complete{}; // Integer flag: 0 or 1
    };


} // namespace warpkep
