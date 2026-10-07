# warpkep

warpkep is a C++/CUDA library in development for orbital propagation and trajectory search in batches, with a Python interface planned.

The aim is to make these building blocks usable in mission design studies and trajectory optimisation. GTOC problems will serve as applications and benchmarks.

## Status

The repository contains a small C++ library, Python bindings that only expose the version, and an internal CPU prototype of universal-variable Kepler propagation in FP64. The prototype is checked against pykep and heyoka in CI. CUDA kernels and the public numerical API are not implemented yet.

## Build

Requirements:

- A C++17 compiler
- CMake 3.24 or newer

From the repository root:

```bash
cmake -S . -B build/cpu
cmake --build build/cpu
./build/cpu/warpkep_smoke
```

Expected output:

```text
warpkep: CPU development environment ready
warpkep version: 0.1.0-dev
```

This builds the current C++ skeleton. It works on macOS without CUDA.

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

- Two-body propagation, followed by numerical integration with perturbations and thrust.
- Lambert transfers and rendezvous cost evaluation.
- Beam search for trajectory sequences.
- Python bindings and a batch API for data already on the GPU.

Numerical routines will start in double precision and be checked against independent references. Benchmarks will report kernel execution time and total time, including data transfers.

## License

Copyright 2026 Benjamin von Plehn.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
