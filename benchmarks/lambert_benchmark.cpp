#include <warpkep/cuda/lambert.hpp>
#include "cuda_test_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd = warpkep::detail;
using warpkep::test::cuda_check;
using warpkep::test::CudaStream;
using warpkep::test::DeviceArray;



void benchmark(std::size_t n) {

        //////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////// Create input and necessary buffers ////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////

        // Define use-case
        constexpr std::array<double, 3> departure{1.0, 0.0, 0.0};
        constexpr std::array<double, 3> arrival{0.0, 1.0, 0.0};
        constexpr double pi = 3.14159265358979323846;
        constexpr double duration = 5.0 * pi / 2.0;

    

        // Create buffer and initialize it
        std::vector<double> inputs(7 * n, 0.0);
        std::fill_n(inputs.begin(), n, departure[0]);
        std::fill_n(inputs.begin() + 4 * n, n, arrival[1]);
        std::fill_n(inputs.begin() + 6 * n, n, duration);

        // Check input
        const double expected[7] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, duration};
        for (std::size_t block = 0; block < 7; ++block) {
            const double first = inputs[block * n];
            const double last = inputs[block * n + n - 1];

            if (first != expected[block] || last != expected[block]) {
                throw std::runtime_error("Unexpected value in Lambert input buffer");
            }
        }

        // Allocate space for the input batch on the GPU.
        warpkep::test::DeviceArray<double> device_inputs(inputs.size());

        // Copy it on GPU
        warpkep::test::cuda_check(cudaMemcpy(device_inputs.ptr, inputs.data(), inputs.size() * sizeof(double), cudaMemcpyHostToDevice) , "Copy Lambert inputs to GPU");
        
        // Create SoA input view
        warpkep::Vector3SoAConstView<double> departures_soa{device_inputs.ptr, device_inputs.ptr + n, device_inputs.ptr + 2 * n};
        warpkep::Vector3SoAConstView<double> arrivals_soa{device_inputs.ptr + 3 * n, device_inputs.ptr + 4 * n, device_inputs.ptr + 5 * n};
        const double* device_durations = device_inputs.ptr + 6 * n;

        // Allocate outputs
        warpkep::test::DeviceArray<double> outputs(14 * n);
        warpkep::test::DeviceArray<warpkep::LambertStatus> lambert_statuses(3 * n);
        warpkep::test::DeviceArray<int> output_ints(5 * n);

        // Create SoA output view
        warpkep::LambertMultiRevSoAView<double> output{};
        output.solutions[0].departure_velocity = {outputs.ptr, outputs.ptr + n, outputs.ptr + 2 * n};
        output.solutions[1].departure_velocity = {outputs.ptr + 7 * n, outputs.ptr + 8 * n, outputs.ptr + 9 * n};
        output.solutions[0].arrival_velocity = {outputs.ptr + 3 * n, outputs.ptr + 4 * n, outputs.ptr + 5 * n};
        output.solutions[1].arrival_velocity = {outputs.ptr + 10 * n, outputs.ptr + 11 * n, outputs.ptr + 12 * n};
        output.solutions[0].x = outputs.ptr + 6 * n;
        output.solutions[1].x = outputs.ptr + 13 * n;


        output.solutions[0].statuses = lambert_statuses.ptr + n;
        output.solutions[1].statuses = lambert_statuses.ptr + 2 * n;
        output.statuses = lambert_statuses.ptr;

        output.solutions[0].iterations = output_ints.ptr;
        output.solutions[1].iterations = output_ints.ptr + n;
        output.slot_count = output_ints.ptr + 2 * n;
        output.solution_count = output_ints.ptr + 3 * n;
        output.complete = output_ints.ptr + 4 * n;

        //////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////// Launch and compute performance ///////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////

        // Number of launches
        constexpr std::size_t sample_count = 9;
        constexpr int launches_per_sample = 32;
        std::array<float, sample_count> samples_ms{};

        // Create lambda function
        const auto launch_lambert = [&] {
            cuda_check(warpkep::launch_lambert_multi_rev_soa_double(departures_soa, arrivals_soa, device_durations, 1.0, wd::LambertArc::short_arc, 1, {}, output, n, nullptr), "Launch Lambert kernel");
        };

        // Create cudaEvents
        cudaEvent_t start{};
        cudaEvent_t stop{};
        cuda_check(cudaEventCreate(&start), "Create start event");
        cuda_check(cudaEventCreate(&stop), "Create stop event");

        // Warm up before recording the first sample.
        launch_lambert();  
        
        // Loop over samples (9 samples, each having 32 launches)
        for (float& sample_ms : samples_ms) {

            // Record event for start
            cuda_check(cudaEventRecord(start, nullptr), "Record start event");

            // Loop over launches
            for (int launch = 0; launch < launches_per_sample; ++launch) {
                // Launch kernel
                launch_lambert();  
            }
        
            // Record stop event: 32 launches are place in the stream
            cuda_check(cudaEventRecord(stop, nullptr), "Record stop event");
            cuda_check(cudaEventSynchronize(stop), "Wait for stop event");
            
            // Compute elapsed time of the sample
            float elapsed_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "Measure elapsed time");
            sample_ms = elapsed_ms / launches_per_sample;
        }
    
        // Destory events
        cuda_check(cudaEventDestroy(start), "Destroy start event");
        cuda_check(cudaEventDestroy(stop), "Destroy stop event");

        // Retrieve median of 9 samples
        std::sort(samples_ms.begin(), samples_ms.end());
        const double median_ms = samples_ms[samples_ms.size() / 2];

        //////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////// Copy GPU -> CPU and analyse results /////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        // Allocate results on CPU
        std::vector<int> v1(n);
        std::vector<int> v2(n);
        
        // Copy results on CPU
        cuda_check(cudaMemcpy(v1.data(), output.solution_count, n * sizeof(int), cudaMemcpyDeviceToHost), "Copy solution counts to CPU");
        cuda_check(cudaMemcpy(v2.data(), output.complete, n * sizeof(int), cudaMemcpyDeviceToHost), "Copy completion flags to CPU");

        // Check results
        for (std::size_t i = 0; i < n; ++i) {
            if (v1[i] != 2 || v2[i] != 1) {
                throw std::runtime_error("Unexpected Lambert result at transfer " + std::to_string(i));
            }
        }


        // Print values (after check of results)
        const double transfers_per_second = 1000.0 * n / median_ms;
        std::cout << "Batch size: " << n << '\n' << "Median GPU time: " << median_ms << " ms per batch\n" << "Throughput: " << transfers_per_second << " transfers/s\n";

    } 


int main() {
    try {
        benchmark(256);
        benchmark(65'536);
        benchmark(1'048'576);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}