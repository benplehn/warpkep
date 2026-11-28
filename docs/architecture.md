# Architecture

Call paths for the scalar Kepler and Lambert solvers and their CUDA batches.

## Files


| File                                                                        | Contents                                                                              | Visibility                   |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------- |
| `include/warpkep/detail/config.hpp`                                         | `WARPKEP_HD` macro (`__host__ __device__` under nvcc)                                 | Internal                     |
| `include/warpkep/detail/vector3.hpp`, `math.hpp`, `numeric_limits.hpp`      | Small vector type,`norm3`, limits for host and device                                 | Internal                     |
| `include/warpkep/detail/stumpff.hpp`                                        | Stumpff functions C(z), S(z)                                                          | Internal                     |
| `include/warpkep/detail/universal_kepler.hpp`, `solve_universal_kepler.hpp` | Universal Kepler equation and its iterative solver                                    | Internal                     |
| `include/warpkep/detail/kepler_cpu.hpp`                                     | `CartesianState`, `KeplerPropagationResult`, scalar propagator `propagate_kepler_cpu` | Internal                     |
| `include/warpkep/detail/lambert_cpu.hpp`                                    | CPU Lambert solver, FP64, zero revolution; entry point`solve_lambert_cpu<double>`     | Internal                     |
| `include/warpkep/lambert_status.hpp`                                        | `LambertStatus`                                                                       | Public                       |
| `include/warpkep/cartesian_soa.hpp`                                         | `CartesianSoAConstView`, `CartesianSoAView`(six pointers, no ownership)               | Public                       |
| `include/warpkep/kepler_status.hpp`                                         | `KeplerPropagationStatus`, `propagation_status_name`                                  | Public                       |
| `include/warpkep/cuda/kepler.hpp`                                           | Declaration of`launch_kepler_soa_double` and its contract                             | Public                       |
| `src/cuda/kepler.cu`                                                        | Kernel`kepler_propagation_soa_kernel<T>`(file-local) and the launcher definition      | Compiled into`warpkep::cuda` |
| `tests/test_lambert.cpp`                                                    | Lambert analytical cases, Kepler closure, parabolic neighbourhood, failure paths      | Test                         |
| `examples/cuda_smoke.cu`                                                    | Compile-only kernels covering the internal primitives on the device                   | Example                      |
| `examples/kepler_gpu_smoke.cpp`                                             | Plain C++ program using the library on a 257-trajectory batch                         | Example / GPU test           |

The Kepler primitives compile for the host and device in FP32 and FP64. The Lambert routines also compile for both, in FP64.

## CPU Lambert call path

```text
solve_lambert_cpu<double>(r1, r2, dt, mu, arc, options)
  1. validate options
  2. prepare_lambert_geometry        → r1, r2, c, s, q, lambda, T_target, local basis
  3. solve_lambert_root              → x, y
       ├─ bracket_lambert_root       → [lower, upper] with T(lower) >= T_target >= T(upper)
       ├─ initial_lambert_guess      → starting point inside the bracket
       └─ loop: evaluate_lambert_derivatives → Householder step, bisection fallback
  4. reconstruct_lambert_velocities  → v1, v2
```

Every stage returns a** **`LambertStatus`; the first failure stops the call and returns six NaN velocities. The computation finishes before returning. The test closes each solution with** **`propagate_kepler_cpu`, which gives a check independent of the Lambert formulas.

## CPU Lambert multirevolution


| File                                                            | Role                                                                                                            |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `<span>include/warpkep/detail/lambert_cpu.hpp</span>`           | Zero-revolution primitives and endpoint velocity reconstruction                                                 |
| `<span>include/warpkep/detail/lambert_multi_rev_cpu.hpp</span>` | Multirevolution evaluation, minimum search, feasibility classification and branch solving                       |
| `<span>include/warpkep/lambert_status.hpp</span>`               | Lambert statuses, including`<span><span>no_solution</span></span>` and `<span><span>near_minimum</span></span>` |
| `<span>tests/test_lambert_multi_rev.cpp</span>`                 | Multirevolution validation; CTest name`<span><span>lambert_multi_rev</span></span>`                             |

`<span><span>solve_lambert_multi_rev_cpu<double></span></span>` prepares the geometry once. For M = 0, it follows the existing root-solving path. For M > 0, it computes one minimum, classifies the requested duration and solves the two branches independently. Both branches reuse the existing velocity reconstruction.

Four-argument overloads of** **`<span><span>evaluate_lambert_time</span></span>` and** **`<span><span>evaluate_lambert_derivatives</span></span>` add the revolution contribution. Existing three-argument calls remain unchanged.

Results use two fixed-capacity slots with individual statuses and an overall completeness flag. A host loop can enumerate M in increasing order through** **`<span><span>solve_lambert_multi_rev_prepared</span></span>`, reusing the prepared geometry. No dynamic container or global cache is introduced into the scalar core.

CMake target** **`<span><span>warpkep_test_lambert_multi_rev</span></span>` is registered with CTest. The existing CPU workflow builds and runs it automatically.

## GPU Lambert call path

`launch_lambert_multi_rev_soa_double` submits the kernel in `src/cuda/lambert.cu` to the caller's stream. Each thread loads one pair of positions and its flight time, calls the shared `solve_lambert_multi_rev_cpu<double>` routine and writes the request status and both output slots. See the [CUDA API](api.md#c-cuda-lambert) for the buffer layout.

## GPU Kepler call path

```text
CPU (user program)                          GPU
------------------                          ---
allocate device arrays
copy inputs (cudaMemcpyAsync, stream)
launch_kepler_soa_double(..., stream)
  ├─ n == 0 → return cudaSuccess
  ├─ blocks = ceil(n / 256)
  ├─ kernel<<<blocks, 256, 0, stream>>> ──▶ one thread per trajectory
  └─ return cudaGetLastError()
copy outputs back (stream)
cudaStreamSynchronize(stream)
check statuses
```

The launcher runs on the host: it computes the grid and submits the kernel, without allocating, copying or waiting.** **`cudaSuccess` means submitted, not finished.

### Inside one thread

```text
kepler_propagation_soa_kernel<T>          (src/cuda/kepler.cu)
  1. i >= n → return
  2. gather   r[i], v[i] from 6 arrays → CartesianState
  3. propagate_kepler_cpu(state, durations[i], mu)
       └─ solve_universal_kepler
            └─ stumpff
  4. scatter  new r, v → 6 arrays
     write    statuses[i], iterations[i]
```

The scalar propagator runs as an ordinary device function; no kernel launches another. Threads do not communicate, and a failed trajectory gets a failure status and NaN outputs without affecting the others.

## CMake targets


| Target                                 | Kind                | Built when                 | Role                                                                       |
| -------------------------------------- | ------------------- | -------------------------- | -------------------------------------------------------------------------- |
| `warpkep`                              | C++ library         | Always                     | Core target: public and internal headers, including the CPU Lambert solver |
| `warpkep_cuda` (alias `warpkep::cuda`) | Static CUDA library | `WARPKEP_ENABLE_CUDA=ON`   | Builds the Kepler and Lambert kernels and links `warpkep` and `CUDA::cudart` publicly  |
| `warpkep_cuda_compile_smoke`           | Object library      | `WARPKEP_ENABLE_CUDA=ON`   | Device compilation check of the primitives; not linked into anything       |
| `warpkep_gpu_smoke`                    | Executable          | `WARPKEP_ENABLE_CUDA=ON`   | Links only`warpkep::cuda`; compiled as plain C++                           |
| CTest`kepler_gpu_smoke`                | Test, label`gpu`    | `WARPKEP_RUN_GPU_TESTS=ON` | Runs`warpkep_gpu_smoke`; needs a CUDA device                               |

Because** **`warpkep_cuda` links its dependencies** **`PUBLIC`, a consumer only needs** **`target_link_libraries(my_app PRIVATE warpkep::cuda)` to get the headers, the compiled kernel and the CUDA runtime.

With** **`WARPKEP_ENABLE_CUDA=OFF`, no CUDA target exists, and the CPU build, Lambert included, needs no CUDA toolkit.

This page shows where each piece of the Kepler batch lives and how a call travels from the CPU to the GPU threads.

## Files


| File                                                                        | Contents                                                                              | Visibility                   |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------- |
| `include/warpkep/detail/config.hpp`                                         | `WARPKEP_HD` macro (`__host__ __device__` under nvcc)                                 | Internal                     |
| `include/warpkep/detail/vector3.hpp`, `math.hpp`, `numeric_limits.hpp`      | Small vector type,`norm3`, limits for host and device                                 | Internal                     |
| `include/warpkep/detail/stumpff.hpp`                                        | Stumpff functions C(z), S(z)                                                          | Internal                     |
| `include/warpkep/detail/universal_kepler.hpp`, `solve_universal_kepler.hpp` | Universal Kepler equation and its iterative solver                                    | Internal                     |
| `include/warpkep/detail/kepler_cpu.hpp`                                     | `CartesianState`, `KeplerPropagationResult`, scalar propagator `propagate_kepler_cpu` | Internal                     |
| `include/warpkep/cartesian_soa.hpp`                                         | `CartesianSoAConstView`, `CartesianSoAView` (six pointers, no ownership)              | Public                       |
| `include/warpkep/kepler_status.hpp`                                         | `KeplerPropagationStatus`, `propagation_status_name`                                  | Public                       |
| `include/warpkep/cuda/kepler.hpp`                                           | Declaration of`launch_kepler_soa_double` and its contract                             | Public                       |
| `src/cuda/kepler.cu`                                                        | Kernel`kepler_propagation_soa_kernel<T>` (file-local) and the launcher definition     | Compiled into`warpkep::cuda` |
| `examples/cuda_smoke.cu`                                                    | Compile-only kernels covering the internal primitives on the device                   | Example                      |
| `examples/kepler_gpu_smoke.cpp`                                             | Plain C++ program using the library on a 257-trajectory batch                         | Example / GPU test           |

The CUDA Kepler batch launcher uses the shared scalar propagator in FP64.

## Call path

```text
CPU (user program)                          GPU
------------------                          ---
allocate device arrays
copy inputs (cudaMemcpyAsync, stream)
launch_kepler_soa_double(..., stream)
  ├─ n == 0 → return cudaSuccess
  ├─ blocks = ceil(n / 256)
  ├─ kernel<<<blocks, 256, 0, stream>>> ──▶ one thread per trajectory
  └─ return cudaGetLastError()
copy outputs back (stream)
cudaStreamSynchronize(stream)
check statuses
```

The launcher runs on the host. It computes the grid and submits the kernel; it does not allocate, copy or wait. Returning `cudaSuccess` means the work was submitted, not finished.

## Inside one thread

Thread `i` handles trajectory `i` from start to end:

```text
kepler_propagation_soa_kernel<T>          (src/cuda/kepler.cu)
  1. i >= n → return
  2. gather  r[i], v[i] from 6 arrays → CartesianState
  3. propagate_kepler_cpu(state, durations[i], mu)
       └─ solve_universal_kepler
            └─ stumpff
  4. scatter  new r, v → 6 arrays
     write    statuses[i], iterations[i]
```

The scalar propagator is called as an ordinary device function; no kernel calls another kernel. Threads do not communicate. A failed trajectory gets a failure status and NaN outputs without affecting the others.

## CMake targets


| Target                                 | Kind                | Built when                 | Role                                                                      |
| -------------------------------------- | ------------------- | -------------------------- | ------------------------------------------------------------------------- |
| `warpkep`                              | C++ library         | Always                     | Core target: public and internal headers                                  |
| `warpkep_cuda` (alias `warpkep::cuda`) | Static CUDA library | `WARPKEP_ENABLE_CUDA=ON`   | Builds the Kepler and Lambert kernels and links `warpkep` and `CUDA::cudart` publicly |
| `warpkep_cuda_compile_smoke`           | Object library      | `WARPKEP_ENABLE_CUDA=ON`   | Device compilation check of the primitives; not linked into anything      |
| `warpkep_gpu_smoke`                    | Executable          | `WARPKEP_ENABLE_CUDA=ON`   | Links only`warpkep::cuda`; compiled as plain C++                          |
| CTest`kepler_gpu_smoke`                | Test, label`gpu`    | `WARPKEP_RUN_GPU_TESTS=ON` | Runs`warpkep_gpu_smoke`; needs a CUDA device                              |

Because `warpkep_cuda` links its dependencies `PUBLIC`, a consumer only writes `target_link_libraries(my_app PRIVATE warpkep::cuda)` to get the headers, the compiled kernel and the CUDA runtime.

With `WARPKEP_ENABLE_CUDA=OFF`, none of the CUDA targets exist and the CPU build needs no CUDA toolkit.
