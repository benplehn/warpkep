#pragma once

#if defined(__CUDACC__)
    #include <cuda/std/limits>
#else
    #include <limits>
#endif

namespace warpkep::detail {

template <typename T>
using NumericLimits =
#if defined(__CUDACC__)
    cuda::std::numeric_limits<T>;
#else
    std::numeric_limits<T>;
#endif



} // namespace warpkep::detail