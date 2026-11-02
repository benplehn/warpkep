// GPU validation of the FP64 Kepler batch.
//
// Checks, on the GPU, through the public library API:
// - shared validation cases (grouped by mu: one launch shares one mu);
// - batch sizes 1, 255, 256, 257, 1025 (block boundaries);
// - empty batch with null pointers;
// - invalid inputs in mixed batches, invalid mu;
// - energy and angular momentum drift;
// - forward/backward round trip of the GPU result.
// Optionally exports GPU results for the independent references:
//   warpkep_test_kepler_gpu --samples gpu_samples.json

#include "kepler_cases.hpp"

#include <warpkep/cuda/kepler.hpp>
#include <warpkep/detail/kepler_cpu.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd = warpkep::detail;

using State = wd::CartesianState<double>;
using Vec3 = wd::Vector3<double>;
using Status = warpkep::KeplerPropagationStatus;

// Initial targets for the physical properties (sans dimension)
constexpr double invariant_tolerance = 1e-11;
constexpr double round_trip_tolerance = 1e-11;

// ---------------------------------------------------------------------------
// CUDA helpers (same mechanisms as examples/kepler_gpu_smoke.cpp)
// ---------------------------------------------------------------------------

// Owns one GPU array and frees it when it goes out of scope.
template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    cudaError_t error = cudaSuccess;

    explicit DeviceArray(std::size_t count) {
        error = cudaMalloc(&ptr, count * sizeof(T));
    }
    ~DeviceArray() {
        cudaFree(ptr);  // cudaFree(nullptr) is a no-op
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// Owns one non-blocking CUDA stream.
struct CudaStream {
    cudaStream_t handle = nullptr;
    cudaError_t error = cudaSuccess;

    CudaStream() {
        error = cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking);
    }
    ~CudaStream() {
        if (handle != nullptr) {
            cudaStreamDestroy(handle);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

// A CUDA error is not a numerical result: it aborts the whole test.
void require(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(error));
    }
}

template <typename T>
void upload(DeviceArray<T>& dst, const std::vector<T>& src, cudaStream_t stream) {
    require(cudaMemcpyAsync(dst.ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync (host to device)");
}

template <typename T>
void download(std::vector<T>& dst, const DeviceArray<T>& src, cudaStream_t stream) {
    require(cudaMemcpyAsync(dst.data(), src.ptr, dst.size() * sizeof(T), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync (device to host)");
}

// ---------------------------------------------------------------------------
// Small vector helpers (host side)
// ---------------------------------------------------------------------------

double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm(const Vec3& a) {
    return std::sqrt(dot(a, a));
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double distance(const Vec3& a, const Vec3& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool all_finite(const State& s) {
    for (std::size_t k = 0; k < 3; ++k) {
        if (!std::isfinite(s.position[k]) || !std::isfinite(s.velocity[k])) {
            return false;
        }
    }
    return true;
}

bool all_nan(const State& s) {
    for (std::size_t k = 0; k < 3; ++k) {
        if (!std::isnan(s.position[k]) || !std::isnan(s.velocity[k])) {
            return false;
        }
    }
    return true;
}

bool same_state(const State& a, const State& b) {
    for (std::size_t k = 0; k < 3; ++k) {
        if (a.position[k] != b.position[k] || a.velocity[k] != b.velocity[k]) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

struct Report {
    int failures = 0;

    void fail(const std::string& message) {
        ++failures;

        std::cout << " FAILED " << message << '\n';
    }
};

// Tracks the largest error and where it happened
struct MaxError {
    double value = 0.0;
    std::string where = "-";

    void update(double error, const std::string& name) {
        if (error > value) {
            value = error;
            where = name;
        }
    }

};

std::string status_text(Status s) {
    return warpkep::propagation_status_name(s);
}

// ---------------------------------------------------------------------------
// One GPU batch: host vectors in, host vectors out
// ---------------------------------------------------------------------------

struct BatchResult {
    std::vector<State> states;
    std::vector<Status> statuses;
    std::vector<int> iterations;
};

BatchResult run_batch(const std::vector<State>& initial, const std::vector<double>& durations, double mu, cudaStream_t stream) {
    const std::size_t n = initial.size();

    if (durations.size() != n) {
        throw std::invalid_argument("run_batch: states and durations differ in size");
    }

    if (n == 0) {
        throw std::invalid_argument("run_batch: empty batch is tested separately");
    }

    // 6 component arrays packed in one buffer: component k is [k*n, (k+1)*n]
    std::vector<double> host_in(6 * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            host_in[k * n + i] = initial[i].position[k];
            host_in[(3 + k) * n + i] = initial[i].velocity[k];
        }
    }

    // --- 1. Reserve GPU memory (runs on CPU, memory lives on GPU) ---
    // Each DeviceArray asks the CUDA driver for space in GPU memory and
    // stores its address in .ptr. The CPU can pass this address around,
    // but must never read or write through it directly.
    DeviceArray<double> d_in(6 * n);    // 6 input components, one after another
    DeviceArray<double> d_out(6 * n);   // 6 output components, same layout
    DeviceArray<double> d_dt(n);        // one duration per trajectory
    DeviceArray<Status> d_status(n);    // one status per trajectory
    DeviceArray<int> d_iter(n);         // one iteration count per trajectory

    // Allocation can fail (no GPU, not enough memory): stop with a clear message.
    require(d_in.error, "cudaMalloc (input states)");
    require(d_out.error, "cudaMalloc (output states)");
    require(d_dt.error, "cudaMalloc (durations)");
    require(d_status.error, "cudaMalloc (statuses)");
    require(d_iter.error, "cudaMalloc (iterations)");

    // --- 2. Prepare CPU memory to receive the results ---
    // Plain CPU vectors, sized to hold exactly what comes back. They are
    // allocated before the first copy so that the cleanup below can wait for
    // the stream before any of them is destroyed.
    std::vector<double> host_out(6 * n);   // same layout as d_out: [r_x|r_y|r_z|v_x|v_y|v_z]
    BatchResult result;
    result.statuses.resize(n);
    result.iterations.resize(n);

    try {
        // --- 3. Queue the copies CPU -> GPU in the stream ---
        // host_in and durations are pageable (non-pinned) memory: the call may
        // block until the data has been staged, it does not necessarily return
        // immediately. The GPU performs the copies in order, before the kernel.
        upload(d_in, host_in, stream);
        upload(d_dt, durations, stream);

        // --- 4. Describe where each component lives (pure address arithmetic) ---
        // d_in is one block of 6*n doubles. The kernel expects six arrays, so we
        // record the start address of each slice of n elements:
        //   [ r_x | r_y | r_z | v_x | v_y | v_z ]
        //     0n    1n    2n    3n    4n    5n
        // This is computed on the CPU and reads nothing on the GPU. host_in was
        // filled with the same layout (host_in[k * n + i]), so the copy lands
        // each component in its own slice.
        warpkep::CartesianSoAConstView<double> input{};
        input.r_x = d_in.ptr + 0 * n;
        input.r_y = d_in.ptr + 1 * n;
        input.r_z = d_in.ptr + 2 * n;
        input.v_x = d_in.ptr + 3 * n;
        input.v_y = d_in.ptr + 4 * n;
        input.v_z = d_in.ptr + 5 * n;

        // Same slicing for the outputs, written by the kernel.
        warpkep::CartesianSoAView<double> output{};
        output.r_x = d_out.ptr + 0 * n;
        output.r_y = d_out.ptr + 1 * n;
        output.r_z = d_out.ptr + 2 * n;
        output.v_x = d_out.ptr + 3 * n;
        output.v_y = d_out.ptr + 4 * n;
        output.v_z = d_out.ptr + 5 * n;

        // --- 5. Queue the kernel (runs on CPU, work runs on GPU) ---
        // The launcher only computes the grid (blocks of 256 threads) and queues
        // the kernel in the stream, after the two uploads. It returns immediately:
        // cudaSuccess means "kernel queued", not "computation finished".
        // We pass GPU addresses only: the kernel will read input/d_dt and write
        // output/d_status/d_iter directly in GPU memory.
        require(warpkep::launch_kepler_soa_double(input, d_dt.ptr, mu, output, d_status.ptr, d_iter.ptr, n, stream), "launch_kepler_soa_double");

        // --- 6. Queue the copies GPU -> CPU ---
        // Queued after the kernel in the same stream, so the GPU runs them only
        // once the kernel has finished writing. With pageable destinations the
        // call may block, but it gives no guarantee that the data is readable:
        // only the synchronization below does.
        download(host_out, d_out, stream);
        download(result.statuses, d_status, stream);
        download(result.iterations, d_iter, stream);

        // --- 7. Wait for the whole queue (the only wait of the batch) ---
        // Blocks the CPU until uploads, kernel and downloads are all done.
        // It also reports errors that happened during execution on the GPU
        // (e.g. invalid memory access), which the launch call could not see.
        // Only after this line may we read host_out, statuses and iterations.
        require(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    } catch (...) {
        // Wait before the local buffers are destroyed.
        const auto cleanup_error = cudaStreamSynchronize(stream);
        if (cleanup_error != cudaSuccess) {
            std::cerr << "Stream cleanup: " << cudaGetErrorString(cleanup_error) << '\n';
        }
        throw;  // keep the original exception
    }

    // --- 8. Rebuild one state per trajectory (CPU only) ---
    // Inverse of the packing done before upload: component k of
    // trajectory i is at host_out[k * n + i].
    //   k = 0, 1, 2 -> position x, y, z  (slices 0n, 1n, 2n)
    //   k = 3, 4, 5 -> velocity x, y, z  (slices 3n, 4n, 5n)
    result.states.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            result.states[i].position[k] = host_out[k * n + i];
            result.states[i].velocity[k] = host_out[(3 + k) * n + i];
        }
    }

    // --- 9. Return; GPU memory is freed here ---
    // d_in, d_out, ... go out of scope: their destructors call cudaFree.
    // This is safe because step 7 (or the cleanup) guaranteed the GPU no longer uses them.
    return result;
}

// ---------------------------------------------------------------------------
// Shared validation cases
// ---------------------------------------------------------------------------

struct CaseResult {
    State state{};
    Status status{};
    int iterations = 0;
};

// One launch shares one mu: group case indices by mu.
std::map<double, std::vector<std::size_t>> group_by_mu(const std::vector<KeplerValidationCase>& cases) {
    std::map<double, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        groups[cases[i].mu].push_back(i);
    }
    return groups;
}

std::vector<CaseResult> run_shared_cases(const std::vector<KeplerValidationCase>& cases, cudaStream_t stream) {
    std::vector<CaseResult> results(cases.size());

    for (const auto& [mu, indices] : group_by_mu(cases)) {
        std::vector<State> states;
        std::vector<double> durations;
        for (std::size_t idx : indices) {
            states.push_back(cases[idx].initial);
            durations.push_back(cases[idx].dt);
        }

        const BatchResult batch = run_batch(states, durations, mu, stream);
        for (std::size_t j = 0; j < indices.size(); ++j) {
            results[indices[j]] = {batch.states[j], batch.statuses[j], batch.iterations[j]};
        }
    }
    return results;
}

// Every shared case is expected to succeed with a finite state.
void check_shared_cases(const std::vector<KeplerValidationCase>& cases, const std::vector<CaseResult>& results, Report& report) {
    std::cout << "Shared cases: " << cases.size() << '\n';
    const int before = report.failures;

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& r = results[i];
        if (r.status != Status::success) {
            report.fail(cases[i].name + ": status " + status_text(r.status));
        } else if (!all_finite(r.state)) {
            report.fail(cases[i].name + ": non-finite state with success status");
        }
    }
    if (report.failures == before) {
        std::cout << "  all succeeded\n";
    }
}

// ---------------------------------------------------------------------------
// Batch sizes around block boundaries (256 threads per block)
// ---------------------------------------------------------------------------

void check_batch_sizes(const std::vector<KeplerValidationCase>& cases, const std::vector<CaseResult>& results, cudaStream_t stream, Report& report) {
    std::cout << "Batch sizes\n";

    // Repeat the largest mu group to fill each batch.
    const auto groups = group_by_mu(cases);
    const std::vector<std::size_t>* group = nullptr;
    for (const auto& [mu, indices] : groups) {
        if (group == nullptr || indices.size() > group->size()) {
            group = &indices;
        }
    }
    const double mu = cases[group->front()].mu;

    for (std::size_t size : {1u, 255u, 256u, 257u, 1025u}) {
        std::vector<State> states(size);
        std::vector<double> durations(size);
        for (std::size_t j = 0; j < size; ++j) {
            const auto& c = cases[(*group)[j % group->size()]];
            states[j] = c.initial;
            durations[j] = c.dt;
        }

        const BatchResult batch = run_batch(states, durations, mu, stream);

        // Each element must match the result of the same case in the first run.
        std::size_t mismatches = 0;
        for (std::size_t j = 0; j < size; ++j) {
            const std::size_t idx = (*group)[j % group->size()];
            const auto& c = cases[idx];
            const auto& ref = results[idx];

            const double e_r = distance(batch.states[j].position, ref.state.position) / c.length_scale;
            const double e_v = distance(batch.states[j].velocity, ref.state.velocity) / c.velocity_scale;
            const bool ok = batch.statuses[j] == ref.status && e_r <= c.tolerance && e_v <= c.tolerance;
            if (!ok) {
                if (mismatches == 0) {
                    report.fail("n=" + std::to_string(size) + ", element " + std::to_string(j) + " (" + c.name + ")");
                }
                ++mismatches;
            }
        }

        std::cout << "  n=" << size << ": " << (mismatches == 0 ? "ok" : std::to_string(mismatches) + " mismatch(es)") << '\n';
    }
}

// n == 0: cudaSuccess, null pointers accepted, nothing submitted.
void check_empty_batch(cudaStream_t stream, Report& report) {
    std::cout << "Empty batch\n";
    const cudaError_t error = warpkep::launch_kepler_soa_double({}, nullptr, 1.0, {}, nullptr, nullptr, 0, stream);
    require(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    if (error != cudaSuccess) {
        report.fail(std::string("n=0 returned ") + cudaGetErrorString(error));
    } else {
        std::cout << "  ok\n";
    }
}

// ---------------------------------------------------------------------------
// Expected failures
// ---------------------------------------------------------------------------

struct FailureCase {
    const char* name;
    State initial;
    double dt;
    Status expected;  // success means: dt = 0 shortcut, unchanged state
};

void check_failures(const std::vector<KeplerValidationCase>& cases, const std::vector<CaseResult>& results, cudaStream_t stream, Report& report) {
    std::cout << "Expected failures (mixed batch, mu = 1)\n";

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    const State valid{{1.5, 0.1, -0.2}, {-0.15, 0.6, 0.25}};

    const std::vector<FailureCase> failures{
        {"origin", {{0, 0, 0}, {0, 1, 0}}, 1.0, Status::invalid_input},
        {"position_nan", {{nan, 0, 0}, {0, 1, 0}}, 1.0, Status::invalid_input},
        {"position_inf", {{1, inf, 0}, {0, 1, 0}}, 1.0, Status::invalid_input},
        {"velocity_nan", {{1, 0, 0}, {0, nan, 0}}, 1.0, Status::invalid_input},
        {"velocity_inf", {{1, 0, 0}, {0, 0, -inf}}, 1.0, Status::invalid_input},
        {"dt_nan", valid, nan, Status::invalid_input},
        {"dt_inf", valid, inf, Status::invalid_input},
        {"dt_minus_inf", valid, -inf, Status::invalid_input},
        {"radial", {{1, 0, 0}, {0.5, 0, 0}}, 1.0, Status::unsupported_case},
        {"zero_velocity", {{1, 0, 0}, {0, 0, 0}}, 1.0, Status::unsupported_case},
        {"valid_dt0", valid, 0.0, Status::success},
        {"invalid_dt0", {{nan, 0, 0}, {0, 1, 0}}, 0.0, Status::invalid_input},
    };

    // Neighbour: shared case 0 (mu = 1), placed before and after each failure.
    const auto& neighbour = cases[0];
    const auto& neighbour_ref = results[0];
    if (neighbour.mu != 1.0) {
        throw std::logic_error("check_failures: shared case 0 must have mu = 1");
    }

    std::vector<State> states{neighbour.initial};
    std::vector<double> durations{neighbour.dt};
    for (const auto& f : failures) {
        states.push_back(f.initial);
        durations.push_back(f.dt);
        states.push_back(neighbour.initial);
        durations.push_back(neighbour.dt);
    }

    const BatchResult batch = run_batch(states, durations, 1.0, stream);
    const int before = report.failures;

    // Even indices: neighbours, unaffected by the failures next to them.
    for (std::size_t j = 0; j < states.size(); j += 2) {
        const double e_r = distance(batch.states[j].position, neighbour_ref.state.position);
        const double e_v = distance(batch.states[j].velocity, neighbour_ref.state.velocity);
        if (batch.statuses[j] != Status::success || !(e_r <= neighbour.tolerance) || !(e_v <= neighbour.tolerance)) {
            report.fail("neighbour at index " + std::to_string(j) + " affected");
        }
    }

    // Odd indices: the failure cases.
    for (std::size_t f = 0; f < failures.size(); ++f) {
        const std::size_t j = 2 * f + 1;
        const auto& expected = failures[f];
        const Status got = batch.statuses[j];
        const State& out = batch.states[j];

        if (got != expected.expected) {
            report.fail(std::string(expected.name) + ": expected " + status_text(expected.expected) + ", got " + status_text(got));
        } else if (expected.expected == Status::success) {
            if (!same_state(out, expected.initial) || batch.iterations[j] != 0) {
                report.fail(std::string(expected.name) + ": dt = 0 must return the initial state with 0 iterations");
            }
        } else if (!all_nan(out)) {
            report.fail(std::string(expected.name) + ": failure must give six NaN");
        }
    }

    // Invalid mu: separate batches, every trajectory must fail.
    for (double bad_mu : {0.0, -1.0, nan, inf, -inf}) {
        const std::vector<State> s{cases[0].initial, valid};
        const std::vector<double> d{1.0, 1.0};
        const BatchResult b = run_batch(s, d, bad_mu, stream);

        for (std::size_t j = 0; j < s.size(); ++j) {
            if (b.statuses[j] != Status::invalid_input || !all_nan(b.states[j])) {
                std::ostringstream name;
                name << "mu=" << bad_mu << ", element " << j << ": got " << status_text(b.statuses[j]);
                report.fail(name.str());
            }
        }
    }

    if (report.failures == before) {
        std::cout << "  " << failures.size() << " failure cases and 5 invalid mu: ok\n";
    }
}

// ---------------------------------------------------------------------------
// Physical properties
// ---------------------------------------------------------------------------

// Energy and angular momentum drift on successful, non-degenerate cases.
void check_invariants(const std::vector<KeplerValidationCase>& cases, const std::vector<CaseResult>& results, Report& report) {
    std::cout << "Invariants (target " << invariant_tolerance << ")\n";
    MaxError max_energy;
    MaxError max_momentum;

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const auto& r = results[i];
        if (r.status != Status::success || c.dt == 0.0) {
            continue;
        }

        const Vec3& r0 = c.initial.position;
        const Vec3& v0 = c.initial.velocity;
        const Vec3& r1 = r.state.position;
        const Vec3& v1 = r.state.velocity;

        const Vec3 h0 = cross(r0, v0);
        const Vec3 h1 = cross(r1, v1);
        if (norm(h0) == 0.0) {
            continue;  // degenerate (radial)
        }

        const double e0 = dot(v0, v0) / 2 - c.mu / norm(r0);
        const double e1 = dot(v1, v1) / 2 - c.mu / norm(r1);

        // Denominator avoids dividing by a near-zero energy near the parabola.
        const double energy_drift = std::abs(e1 - e0) / (dot(v0, v0) / 2 + c.mu / norm(r0));
        const double momentum_drift = distance(h1, h0) / norm(h0);

        max_energy.update(energy_drift, c.name);
        max_momentum.update(momentum_drift, c.name);

        if (!(energy_drift <= invariant_tolerance)) {
            report.fail(c.name + ": energy drift " + std::to_string(energy_drift));
        }
        if (!(momentum_drift <= invariant_tolerance)) {
            report.fail(c.name + ": angular momentum drift " + std::to_string(momentum_drift));
        }
    }

    std::cout << std::scientific << std::setprecision(3) << "  max energy drift: " << max_energy.value << " (" << max_energy.where << ")\n  max angular momentum drift: " << max_momentum.value << " (" << max_momentum.where << ")\n" << std::defaultfloat;
}

// Propagate the GPU result with -dt, compare with the initial state.
void check_round_trip(const std::vector<KeplerValidationCase>& cases, const std::vector<CaseResult>& results, cudaStream_t stream, Report& report) {
    std::cout << "Round trip (target " << round_trip_tolerance << ")\n";
    MaxError max_r;
    MaxError max_v;

    for (const auto& [mu, indices] : group_by_mu(cases)) {
        std::vector<std::size_t> kept;
        std::vector<State> states;
        std::vector<double> durations;
        for (std::size_t idx : indices) {
            if (results[idx].status == Status::success) {
                kept.push_back(idx);
                states.push_back(results[idx].state);  // GPU output, not CPU
                durations.push_back(-cases[idx].dt);
            }
        }
        if (kept.empty()) {
            continue;
        }

        const BatchResult back = run_batch(states, durations, mu, stream);

        for (std::size_t j = 0; j < kept.size(); ++j) {
            const auto& c = cases[kept[j]];
            if (back.statuses[j] != Status::success) {
                report.fail(c.name + ": backward status " + status_text(back.statuses[j]));
                continue;
            }
            const double e_r = distance(back.states[j].position, c.initial.position) / c.length_scale;
            const double e_v = distance(back.states[j].velocity, c.initial.velocity) / c.velocity_scale;
            max_r.update(e_r, c.name);
            max_v.update(e_v, c.name);

            if (!(e_r <= round_trip_tolerance) || !(e_v <= round_trip_tolerance)) {
                report.fail(c.name + ": round trip e_r=" + std::to_string(e_r) + ", e_v=" + std::to_string(e_v));
            }
        }
    }

    std::cout << std::scientific << std::setprecision(3) << "  max e_r: " << max_r.value << " (" << max_r.where << ")\n  max e_v: " << max_v.value << " (" << max_v.where << ")\n" << std::defaultfloat;
}

// ---------------------------------------------------------------------------
// Export (same format as validation/cpu_propagation_samples.cpp)
// ---------------------------------------------------------------------------

void write_vector(std::ostream& out, const Vec3& v) {
    out << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
}

void write_samples(const std::string& path, const std::vector<KeplerValidationCase>& cases, const std::vector<CaseResult>& results) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open " + path);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "[\n";
    bool first = true;
    std::size_t written = 0;

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const auto& r = results[i];
        if (r.status != Status::success || !all_finite(r.state)) {
            throw std::logic_error("write_samples: " + c.name + " did not succeed");
        }
        if (!first) {
            out << ",\n";
        }
        first = false;

        out << "{\"name\":\"" << c.name << "\",\"mu\":" << c.mu << ",\"dt\":" << c.dt << ",\"length_scale\":" << c.length_scale << ",\"velocity_scale\":" << c.velocity_scale << ",\"tolerance\":" << c.tolerance << ",\"r0\":";
        write_vector(out, c.initial.position);
        out << ",\"v0\":";
        write_vector(out, c.initial.velocity);
        out << ",\"r1\":";
        write_vector(out, r.state.position);
        out << ",\"v1\":";
        write_vector(out, r.state.velocity);
        out << ",\"status\":\"success\",\"iterations\":" << r.iterations << '}';
        ++written;
    }
    out << "\n]\n";

    if (!out) {
        throw std::runtime_error("write failed: " + path);
    }
    std::cout << "Wrote " << written << " GPU samples to " << path << '\n';
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string samples_path;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--samples" && a + 1 < argc) {
            samples_path = argv[++a];
        } else {
            std::cerr << "usage: " << argv[0] << " [--samples path.json]\n";
            return 2;
        }
    }

    try {
        CudaStream stream;
        require(stream.error, "cudaStreamCreateWithFlags");

        const auto cases = kepler_validation_cases();
        Report report;

        const auto results = run_shared_cases(cases, stream.handle);
        check_shared_cases(cases, results, report);
        check_batch_sizes(cases, results, stream.handle, report);
        check_empty_batch(stream.handle, report);
        check_failures(cases, results, stream.handle, report);
        check_invariants(cases, results, report);
        check_round_trip(cases, results, stream.handle, report);

        if (report.failures != 0) {
            std::cout << report.failures << " check(s) FAILED\n";
            if (!samples_path.empty()) {
                std::cout << "GPU samples not written: results are incomplete\n";
            }
            return 1;
        }

        if (!samples_path.empty()) {
            write_samples(samples_path, cases, results);
        }
        std::cout << "All GPU validation checks passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
