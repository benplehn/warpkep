#include <cuda_runtime.h>

__global__ void square_kernel(const double* input, double* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * input[i];
    }
}
