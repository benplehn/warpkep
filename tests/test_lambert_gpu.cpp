
#include <warpkep/cuda/lambert.hpp>
#include "cuda_test_utils.hpp"

#include <algorithm>
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


using Vec = wd::Vector3<double>;
using Status = warpkep::LambertStatus;
using Arc = wd::LambertArc;

void check(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}


double norm(const Vec& v) { return std::hypot(v[0], v[1], v[2]); }
double distance(const Vec& a, const Vec& b) {
    return norm(Vec{a[0] - b[0], a[1] - b[1], a[2] - b[2]});
}
bool close(double a, double b) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 2e-11 * (1.0 + std::abs(b));
}
bool all_nan(const Vec& v) {
    return std::isnan(v[0]) && std::isnan(v[1]) && std::isnan(v[2]);
}

struct Case {
    Vec departure, arrival, departure_velocity, arrival_velocity;
    double duration;
    Status expected{Status::success};
};

Case circular_case(std::size_t i, int m, Arc arc, double mu) {
    const double pi = std::acos(-1.0);
    const double radius = 0.7 + 0.1 * static_cast<double>(i % 7);
    const double phase = 0.017 * static_cast<double>(i);
    const double angle = 0.45 + 1.8 * static_cast<double>(i % 257) / 256.0;
    const double direction = arc == Arc::short_arc ? 1.0 : -1.0;
    const double sweep = arc == Arc::short_arc ? angle : 2.0 * pi - angle;

    // Permute coordinates to exercise three orbital planes.
    const auto plane = [i](double x, double y) -> Vec {
        if (i % 3 == 0) { return {x, y, 0.0}; }
        if (i % 3 == 1) { return {0.0, x, y}; }
        return {y, 0.0, x};
    };
    const double speed = direction * std::sqrt(mu / radius);
    return {
        plane(radius * std::cos(phase), radius * std::sin(phase)),
        plane(radius * std::cos(phase + angle), radius * std::sin(phase + angle)),
        plane(-speed * std::sin(phase), speed * std::cos(phase)),
        plane(-speed * std::sin(phase + angle), speed * std::cos(phase + angle)),
        (sweep + 2.0 * pi * m) * radius * std::sqrt(radius / mu)
    };
}

void run_batch(const std::vector<Case>& cases, double mu, Arc arc, int m, wd::LambertMultiRevOptions options = {}) {
    const std::size_t n = cases.size();
    check(n > 0, "run_batch requires a nonempty fixture");
    // Packed SoA storage: 7 input arrays, then 7 output arrays per slot
    // (departure velocity xyz, arrival velocity xyz, x). No arrays overlap.
    std::vector<double> values(21 * n, -12345.0);
    std::vector<Status> statuses(3 * n, Status::numerical_failure);
    // Global slot_count, solution_count, complete; then iterations per slot.
    std::vector<int> integers(5 * n, -12345);
    std::vector<wd::LambertMultiRevResult<double>> reference;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            values[k * n + i] = cases[i].departure[k];
            values[(k + 3) * n + i] = cases[i].arrival[k];
        }
        values[6 * n + i] = cases[i].duration;
        reference.push_back(wd::solve_lambert_multi_rev_cpu( cases[i].departure, cases[i].arrival, cases[i].duration, mu, arc, m, options));
        check(reference.back().status == cases[i].expected, "unexpected CPU fixture status");
    }

    CudaStream stream;
    DeviceArray<double> dv(values.size());
    DeviceArray<Status> ds(statuses.size());
    DeviceArray<int> di(integers.size());
    const auto transfer = [&](void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
        cuda_check(cudaMemcpyAsync(dst, src, bytes, kind, stream.handle), "cudaMemcpyAsync");
    };
    transfer(dv.ptr, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice);
    transfer(ds.ptr, statuses.data(), statuses.size() * sizeof(Status), cudaMemcpyHostToDevice);
    transfer(di.ptr, integers.data(), integers.size() * sizeof(int), cudaMemcpyHostToDevice);

    warpkep::LambertMultiRevSoAView<double> output{};
    output.statuses = ds.ptr;
    output.slot_count = di.ptr;
    output.solution_count = di.ptr + n;
    output.complete = di.ptr + 2 * n;
    for (int slot = 0; slot < 2; ++slot) {
        double* base = dv.ptr + (7 + 7 * slot) * n;
        output.solutions[slot] = {
            {base, base + n, base + 2 * n},
            {base + 3 * n, base + 4 * n, base + 5 * n},
            base + 6 * n, ds.ptr + (1 + slot) * n, di.ptr + (3 + slot) * n
        };
    }
    cuda_check(warpkep::launch_lambert_multi_rev_soa_double(
        {dv.ptr, dv.ptr + n, dv.ptr + 2 * n},
        {dv.ptr + 3 * n, dv.ptr + 4 * n, dv.ptr + 5 * n},
        dv.ptr + 6 * n, mu, arc, m, options, output, n, stream.handle), "Lambert launch");
    transfer(values.data(), dv.ptr, values.size() * sizeof(double), cudaMemcpyDeviceToHost);
    transfer(statuses.data(), ds.ptr, statuses.size() * sizeof(Status), cudaMemcpyDeviceToHost);
    transfer(integers.data(), di.ptr, integers.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check(cudaStreamSynchronize(stream.handle), "Lambert execution/download");

    double max_velocity_difference = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& cpu = reference[i];
        const std::string where = "M=" + std::to_string(m) + " i=" + std::to_string(i);
        check(statuses[i] == cpu.status, where + " global status");
        check(integers[i] == cpu.slot_count && integers[n + i] == cpu.solution_count && integers[2 * n + i] == (cpu.complete ? 1 : 0), where + " counts/completeness");
        bool circular_solution_found = false;
        for (int slot = 0; slot < 2; ++slot) {
            const bool active = slot < cpu.slot_count;
            const Status expected = active ? cpu.solutions[slot].result.status : Status::unsupported_case;
            const std::size_t base = (7 + 7 * slot) * n + i;
            const Vec v1{values[base], values[base + n], values[base + 2 * n]};
            const Vec v2{values[base + 3 * n], values[base + 4 * n], values[base + 5 * n]};
            const double x = values[base + 6 * n];
            const int iterations = integers[(3 + slot) * n + i];
            check(statuses[(1 + slot) * n + i] == expected, where + " slot status");
            check(iterations >= 0 && iterations <= options.roots.max_iterations, where + " iterations");
            if (expected != Status::success) {
                check(all_nan(v1) && all_nan(v2) && std::isnan(x), where + " failure/inactive NaNs");
                if (!active) { check(iterations == 0, where + " inactive iterations"); }
                continue;
            }
            const auto& solution = cpu.solutions[slot];
            check(close(x, solution.x), where + " root x");
            check(std::isfinite(norm(v1)) && std::isfinite(norm(v2)), where + " finite velocities");
            const double difference = std::max(
                distance(v1, solution.result.departure_velocity) / norm(solution.result.departure_velocity),
                distance(v2, solution.result.arrival_velocity) / norm(solution.result.arrival_velocity));
            check(std::isfinite(difference) && difference <= 2e-11, where + " CPU/GPU velocities");
            max_velocity_difference = std::max(max_velocity_difference, difference);

            const auto geometry = wd::prepare_lambert_geometry(
                cases[i].departure, cases[i].arrival, cases[i].duration, mu, arc);
            check(geometry.status == Status::success, where + " prepared geometry");
            const auto time = wd::evaluate_lambert_time(x, geometry.geometry.lambda, geometry.geometry.q, m);
            check(time.status == Status::success
                  && std::abs(time.time - geometry.geometry.target_time) / geometry.geometry.target_time <= 5e-12,
                  where + " time residual");
            const double analytic_error = std::max(
                distance(v1, cases[i].departure_velocity), distance(v2, cases[i].arrival_velocity))
                / norm(cases[i].departure_velocity);
            circular_solution_found = circular_solution_found || analytic_error <= 2e-10;
        }
        if (cases[i].expected == Status::success) {
            check(circular_solution_found, where + " analytic circular solution");
        }
    }
    std::cout << "M=" << m << (arc == Arc::short_arc ? " short" : " long")
              << ", n=" << n << ": OK, max relative CPU/GPU velocity difference="
              << max_velocity_difference << '\n';
}

int main() {
    try {
        cuda_check(warpkep::launch_lambert_multi_rev_soa_double(
            {}, {}, nullptr, 1.0, Arc::short_arc, 0, {}, {}, 0, nullptr), "empty batch");
        std::cout << "Empty batch: OK\n";
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        check(device_count > 0, "An NVIDIA GPU is required to execute this test");

        constexpr double mu = 1.25;
        for (Arc arc : {Arc::short_arc, Arc::long_arc}) {
            for (int m = 0; m <= 2; ++m) {
                std::vector<Case> cases;
                for (std::size_t i = 0; i < 257; ++i) { cases.push_back(circular_case(i, m, arc, mu)); }
                cases[253].duration = -1.0;
                cases[253].expected = Status::invalid_input;
                cases[254].departure[0] = std::numeric_limits<double>::quiet_NaN();
                cases[254].expected = Status::invalid_input;
                for (std::size_t k = 0; k < 3; ++k) { cases[255].arrival[k] = -cases[255].departure[k]; }
                cases[255].expected = Status::unsupported_case;
                if (m > 0) {
                    cases[256].duration = 1e-4;
                    cases[256].expected = Status::no_solution;
                    const auto g = wd::prepare_lambert_geometry(cases[252].departure, cases[252].arrival, 1.0, mu, arc);
                    check(g.status == Status::success, "threshold fixture geometry");
                    const auto minimum = wd::find_lambert_minimum(g.geometry.lambda, g.geometry.q, m);
                    check(minimum.status == Status::success, "threshold fixture minimum");
                    cases[252].duration = minimum.time / g.geometry.target_time;
                    cases[252].expected = Status::near_minimum;
                }
                run_batch(cases, mu, arc, m);
            }
        }
        auto invalid = circular_case(0, 0, Arc::short_arc, mu);
        invalid.expected = Status::invalid_input;
        run_batch({invalid}, 0.0, Arc::short_arc, 0);
        run_batch({invalid}, mu, Arc::short_arc, -1);
        auto exhausted = circular_case(0, 1, Arc::short_arc, mu);
        exhausted.expected = Status::not_converged;
        wd::LambertMultiRevOptions options;
        options.minimum_max_iterations = 1;
        run_batch({exhausted}, mu, Arc::short_arc, 1, options);
        std::cout << "Lambert GPU: all checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Lambert GPU: " << error.what() << '\n';
        return 1;
    }
}
