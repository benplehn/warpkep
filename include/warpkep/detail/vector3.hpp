#pragma once
 
#include <warpkep/detail/config.hpp>
#include <cstddef>

namespace warpkep::detail {


template <typename T>
struct Vector3 {
    T values[3];

    WARPKEP_HD constexpr T& operator[](std::size_t i) noexcept {
        return values[i];
    }

    WARPKEP_HD constexpr const T& operator[](std::size_t i) const noexcept {
        return values[i];
    }
};


} // namespace warpkep::detail
