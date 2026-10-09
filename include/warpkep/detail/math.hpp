#pragma once

#include <warpkep/detail/config.hpp>

#include <cmath>
#include <type_traits>

#if defined(__CUDACC__)
    #include  <cuda_runtime.h>
#endif

namespace warpkep::detail {


template <typename T> 
WARPKEP_HD inline T norm3(T x, T y, T z) {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, double>, "norm3 requires a float or a double"
    );


    #if defined(__CUDA_ARCH__)
        if constexpr (std::is_same_v<T, float>) {
            return ::norm3df(x, y, z);
        } else {
            return ::norm3d(x, y, z);
        }
    #else 
        return std::hypot(x, y, z);
    #endif
}

} // namespace warpkep::detail