#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace warpkep::test{

inline void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}


template <typename T>
struct DeviceArray{
    T* ptr{};

    explicit DeviceArray(std::size_t count){
        if (count == 0){
            return;
        }

        cuda_check(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceArray() noexcept {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

};


struct CudaStream {
    cudaStream_t handle{};

    CudaStream() {
        cuda_check(
            cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags"
        );
    }

    ~CudaStream() noexcept {
        if (handle != nullptr) {
            cudaStreamDestroy(handle);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};


} // namespace warpkep::test