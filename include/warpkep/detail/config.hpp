#pragma once 

#if defined(__CUDACC__)
    #define WARPKEP_HD __host__ __device__
#else
    #define WARPKEP_HD
#endif