# Architecture

The numerical routines live in headers, with CUDA and Python wrappers handling batches, memory and execution. Kepler routines compile for CPU and GPU. Lambert currently runs only on the CPU.

## Repository layout

| Path | Purpose |
| --- | --- |
| `include/warpkep/detail/` | Internal vector/math utilities, Kepler primitives and CPU Lambert routines |
| `include/warpkep/cuda/kepler.hpp` | Experimental public CUDA propagation launcher |
| `include/warpkep/cartesian_soa.hpp` | Borrowed six-component state views |
| `include/warpkep/{kepler,lambert}_status.hpp` | Public result-status enums |
| `src/cuda/kepler.cu` | Compiled FP64 batch kernel and host launcher |
| `src/version.cpp` | Core version implementation |
| `bindings/` | nanobind version and CUDA extension modules |
| `python/warpkep/` | Python facade and CuPy allocation/stream adapter |
| `examples/` | CPU Lambert example, CUDA compilation check and GPU smoke program |
| `tests/` | C++ numerical/GPU tests and Python/CuPy tests |
| `validation/` | Shared Kepler cases, sample generation and independent references |
| `benchmarks/` | CPU baseline and CPU/GPU timing harness |
| `docker/` | CUDA compilation and pinned reference environments |
| `docs/` | Sphinx/MyST documentation, independently buildable without CUDA |

## Scalar Kepler path

```text
propagate_kepler_cpu(initial, dt, mu, options)
  validate inputs and supported motion
  compute orbital quantities
  solve_universal_kepler(...)
    evaluate_universal_kepler(...)
      stumpff(...)
  reconstruct state with Lagrange coefficients
  return state, status, iterations
```

The GPU kernel calls `propagate_kepler_cpu` too. The function is marked `WARPKEP_HD`, which expands to host/device attributes under nvcc. FP32 and FP64 share the same source; the public CUDA launcher uses FP64.

## CPU Lambert path

```text
solve_lambert_cpu<double>(r1, r2, dt, mu, arc, options)
  prepare_lambert_geometry(...)          -> triangle, local basis, lambda, q, T
  solve_lambert_root(...)                -> x, y
    bracket + starter
    safeguarded Householder iteration, bisection fallback
  reconstruct_lambert_velocities(...)    -> v1, v2
```

The multi-revolution interface reuses the geometry and velocity reconstruction:

```text
solve_lambert_multi_rev_cpu<double>(..., M, options)
  prepare_lambert_geometry(...)
  solve_lambert_multi_rev_prepared(...)
    M == 0 -> existing zero-revolution root path; one single slot
    M > 0
      find_lambert_minimum(...)          -> signed derivative bracket
      classify time relative to minimum and threshold
        below band -> no_solution
        within band -> near_minimum
        above band
          solve_lambert_branch_root(left)
          solve_lambert_branch_root(right)
          reconstruct velocities for successful roots
          verify separation; report counters and completeness
```

The minimum search applies Halley's method to the time derivative within a bracket. Each branch then uses Householder iteration, falling back to bisection when a step leaves the bracket. If one branch fails, the other can still return a usable solution. If the roots cannot be distinguished near the minimum, neither is returned. Storage is fixed-size, and the solver does not allocate memory or throw exceptions.

Use `solve_lambert_multi_rev_prepared` in a CPU loop to reuse the geometry for different `M` values. Keep the same `mu` and arc. The [Lambert guide](lambert.md) shows how to read the results.

## CUDA propagation path

```text
Caller / host                              Device
allocate buffers and stream
upload inputs
launch_kepler_soa_double(..., stream)
  n == 0 -> cudaSuccess
  compute ceil(n / 256) blocks
  kernel<<<blocks, 256, 0, stream>>> ------> one thread per trajectory
  return cudaGetLastError()                  gather state from SoA
                                             propagate_kepler_cpu<double>
                                             scatter state, status, iterations
download results in stream
synchronize and check completion
```

Each GPU thread processes one trajectory independently. The host launcher submits the kernel without allocating, copying or waiting. Check execution errors when synchronizing the stream, even after the launch returns `cudaSuccess`.

## Python/CuPy path

```text
warpkep.propagate_kepler(...)
  lazy import of cuda.py, CuPy and _cuda
  select the input device
  allocate outputs with CuPy
  wait for the current CuPy stream
  _cuda.propagate_kepler(...)
    validate nanobind arrays and numerical batch parameters
    allocate typed temporary status buffer
    launch_kepler_soa_double(...)
    copy typed statuses to the int32 output on device
    wait for completion
  return (out, statuses, iterations)
```

The binding releases the GIL during execution. CuPy owns the input and output arrays; the C++ adapter owns the temporary status buffer. The Python call includes allocations and waits for completion.

## Build targets

| Target | Built when | Purpose |
| --- | --- | --- |
| `warpkep` | Always | Static core library; propagates include paths and C++17 requirement |
| `warpkep_smoke` | Always | Basic version/environment check |
| `_core` | Python enabled | Version binding |
| `warpkep_cuda` / `warpkep::cuda` | CUDA enabled | Compiled kernel, public core/runtime dependencies |
| `warpkep_cuda_compile_smoke` | CUDA enabled | Device compilation of internal primitives |
| `warpkep_gpu_smoke` | CUDA enabled | Plain C++ consumer of the CUDA library |
| `warpkep_test_kepler_gpu` | CUDA enabled | Structured GPU validation; registered only with GPU tests enabled |
| `_cuda` | Python and CUDA enabled | GPU nanobind extension |
| `warpkep_test_*` CPU targets | Testing enabled | Seven registered CPU tests |
| `warpkep_cpu_samples` | Testing enabled | Kepler reference JSON generator |
| `warpkep_cpu_baseline` | Benchmarks enabled | Serial/parallel CPU timing |
| `warpkep_kepler_benchmark` | Benchmarks and CUDA enabled | CPU/GPU timing with accuracy checks |

The documentation workflow compiles and runs `examples/lambert_multi_rev.cpp` directly, without CUDA.
