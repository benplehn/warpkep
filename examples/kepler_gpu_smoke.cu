#include "kepler_launch.hpp"

#include <iostream>

int main() {
    const cudaError_t status = launch_kepler_soa_double(
        {},
        nullptr,
        1.0,
        {},
        nullptr,
        nullptr,
        0,
        nullptr
    );

    if (status != cudaSuccess) {
        std::cerr << "Empty batch failed: "
                  << cudaGetErrorString(status) << '\n';
        return 1;
    }

    std::cout << "Empty batch passed\n";
    return 0;
}