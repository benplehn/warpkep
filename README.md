# warpkep

warpkep is a C++/CUDA library in development for orbital propagation and trajectory search in batches, with a Python interface planned.

The aim is to make these building blocks usable in mission design studies and trajectory optimisation. GTOC problems will serve as applications and benchmarks.

## Status

The repository currently contains:

* A generic universal-variable Kepler propagator, written as host/device C++ templates for float and double.
* An experimental FP64 CUDA batch API, built into the library: `warpkep::launch_kepler_soa_double` in `<warpkep/cuda/kepler.hpp>`, CMake target `warpkep::cuda`.
* Five CPU tests and one GPU test registered with CTest.
* Python bindings that only expose the version.

The CPU propagator is checked against pykep and heyoka in CI. GPU validation is currently limited to the cases run by `examples/kepler_gpu_smoke.cpp`. GPU performance has not been measured yet.

See [docs/architecture.md](docs/architecture.md) for the code layout and [docs/numerical\_contract.md](docs/numerical_contract.md) for the API contract.

## Build

Requirements:

* A C++17 compiler
* CMake 3.24 or newer
* For the CUDA library: a CUDA toolkit (an NVIDIA GPU is only needed to run GPU tests)

### CPU only

Works on macOS without CUDA.

```bash
cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DWARPKEP_ENABLE_CUDA=OFF

cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure
./build/cpu/warpkep_smoke
```

Expected output of `warpkep_smoke`:

```text
warpkep: CPU development environment ready
warpkep version: 0.1.0.dev0
```

### CUDA, compile only (no GPU)

Compiles and links the CUDA targets in the development container, without running them.

```bash
docker build --platform linux/amd64 \
  -f docker/Dockerfile.cuda -t warpkep-cuda:dev docker

docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace" -w /workspace warpkep-cuda:dev \
  bash -c '
    set -e
    cmake -S . -B build/cuda -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DWARPKEP_ENABLE_CUDA=ON \
      -DWARPKEP_RUN_GPU_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=80
    cmake --build build/cuda \
      --target warpkep_cuda warpkep_cuda_compile_smoke warpkep_gpu_smoke
  '
```

### CUDA with a GPU

Builds everything and runs the CPU and GPU tests.

```bash
cmake -S . -B build/gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DWARPKEP_ENABLE_CUDA=ON \
  -DWARPKEP_RUN_GPU_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build/gpu --parallel 2
ctest --test-dir build/gpu --output-on-failure
```

Optional memory check:

```bash
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
  ./build/gpu/warpkep_gpu_smoke
```

## CPU baseline

```bash
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DWARPKEP_BUILD_BENCHMARKS=ON
cmake --build build/bench --target warpkep_cpu_baseline
./build/bench/warpkep_cpu_baseline 200000 4
```

Batch of 200,000 two-body propagations (mu = 1, FP64), median of 5 runs, Apple M1 Pro, Apple clang 21:


| Threads | Propagations/s |
| ------- | -------------- |
| 1       | 5.38 M         |
| 4       | 20.27 M        |

## Planned work

* Structured GPU validation: independent references, parabolic and hyperbolic cases, edge cases, invariants and round trips.
* CPU/GPU benchmarks: kernel time and total time including transfers, against the parallel CPU baseline.
* Numerical integration with perturbations and thrust.
* Lambert transfers and rendezvous cost evaluation.
* Beam search for trajectory sequences.
* Python bindings and a batch API for data already on the GPU.

Numerical routines start in double precision and are checked against independent references. Benchmarks will report kernel execution time and total time, including data transfers.

## License

Copyright 2026 Benjamin von Plehn.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
