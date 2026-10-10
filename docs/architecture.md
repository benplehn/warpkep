# Architecture

This page shows where each piece of the Kepler batch lives and how a call travels from the CPU to the GPU threads.

## Files

| File | Contents | Visibility |
| ---- | -------- | ---------- |
| `include/warpkep/detail/config.hpp` | `WARPKEP_HD` macro (`__host__ __device__` under nvcc) | Internal |
| `include/warpkep/detail/vector3.hpp`, `math.hpp`, `numeric_limits.hpp` | Small vector type, `norm3`, limits for host and device | Internal |
| `include/warpkep/detail/stumpff.hpp` | Stumpff functions C(z), S(z) | Internal |
| `include/warpkep/detail/universal_kepler.hpp`, `solve_universal_kepler.hpp` | Universal Kepler equation and its iterative solver | Internal |
| `include/warpkep/detail/kepler_cpu.hpp` | `CartesianState`, `KeplerPropagationResult`, scalar propagator `propagate_kepler_cpu` | Internal |
| `include/warpkep/cartesian_soa.hpp` | `CartesianSoAConstView`, `CartesianSoAView` (six pointers, no ownership) | Public |
| `include/warpkep/kepler_status.hpp` | `KeplerPropagationStatus`, `propagation_status_name` | Public |
| `include/warpkep/cuda/kepler.hpp` | Declaration of `launch_kepler_soa_double` and its contract | Public |
| `src/cuda/kepler.cu` | Kernel `kepler_propagation_soa_kernel<T>` (file-local) and the launcher definition | Compiled into `warpkep::cuda` |
| `examples/cuda_smoke.cu` | Compile-only kernels covering the internal primitives on the device | Example |
| `examples/kepler_gpu_smoke.cpp` | Plain C++ program using the library on a 257-trajectory batch | Example / GPU test |

All scalar code in `detail/` is templated on `T` (float or double) and marked `WARPKEP_HD`, so the same source compiles for the CPU and the GPU. Only the FP64 launcher is public for now.

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

| Target | Kind | Built when | Role |
| ------ | ---- | ---------- | ---- |
| `warpkep` | C++ library | Always | Core target: public and internal headers |
| `warpkep_cuda` (alias `warpkep::cuda`) | Static CUDA library | `WARPKEP_ENABLE_CUDA=ON` | Compiles `src/cuda/kepler.cu`; links `warpkep` and `CUDA::cudart` publicly |
| `warpkep_cuda_compile_smoke` | Object library | `WARPKEP_ENABLE_CUDA=ON` | Device compilation check of the primitives; not linked into anything |
| `warpkep_gpu_smoke` | Executable | `WARPKEP_ENABLE_CUDA=ON` | Links only `warpkep::cuda`; compiled as plain C++ |
| CTest `kepler_gpu_smoke` | Test, label `gpu` | `WARPKEP_RUN_GPU_TESTS=ON` | Runs `warpkep_gpu_smoke`; needs a CUDA device |

Because `warpkep_cuda` links its dependencies `PUBLIC`, a consumer only writes `target_link_libraries(my_app PRIVATE warpkep::cuda)` to get the headers, the compiled kernel and the CUDA runtime.

With `WARPKEP_ENABLE_CUDA=OFF`, none of the CUDA targets exist and the CPU build needs no CUDA toolkit.