# warpkep

warpkep is a C++/CUDA library in development for orbital propagation and trajectory search in batches, with a Python interface planned.

The aim is to make these building blocks usable in mission design studies and trajectory optimisation. GTOC problems will serve as applications and benchmarks.

## Status

The repository currently contains a small C++ library and an executable that prints its version. Orbital algorithms, CUDA kernels and Python bindings are not implemented yet.

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

## Planned work

- Two-body propagation, followed by numerical integration with perturbations and thrust.
- Lambert transfers and rendezvous cost evaluation.
- Beam search for trajectory sequences.
- Python bindings and a batch API for data already on the GPU.

Numerical routines will start in double precision and be checked against independent references. Benchmarks will report kernel execution time and total time, including data transfers.

## License

Copyright 2026 Benjamin von Plehn.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
