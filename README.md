# warpkep

**Two-body astrodynamics in C++17 and CUDA.**

warpkep is a C++17/CUDA library for Kepler propagation and Lambert transfers, including multi-revolution solutions. It is being developed for mission design and trajectory optimisation, with batch propagation on NVIDIA GPUs and an experimental Python/CuPy interface.

[Documentation](docs/index.md) · [Installation](docs/installation.md) · [Lambert guide](docs/lambert.md) · [API reference](docs/api.md) · [Contributing](CONTRIBUTING.md)

## Current capabilities

The current version is `0.1.0.dev0`. The numerical interfaces are experimental, and headers under `warpkep/detail/` may change during development.

| Capability | Interface | Precision | Availability |
| --- | --- | --- | --- |
| Scalar Kepler propagation | Internal C++, host/device templates | FP32 / FP64 | Implemented; FP64 reference validation |
| Batched Kepler propagation | C++ CUDA, device-resident SoA arrays | FP64 | Experimental |
| Batched Kepler propagation | Python / CuPy, synchronous | `float64` | Experimental; CUDA build required |
| Zero-revolution Lambert | Internal C++, short and long arcs | FP64 | Implemented |
| Multi-revolution Lambert | Internal C++, fixed `M`, two branch slots | FP64 | Implemented; analytical tests for `M = 1, 2` |

Lambert runs on the CPU through C++. Its results distinguish a failed computation, a flight time that is too short for the requested revolution count, and a time too close to the minimum to resolve the two branches. The [numerical contract](docs/numerical_contract.md) describes the supported inputs, tolerances and failure behaviour.

## Build and test on CPU

Requires a C++17 compiler and CMake 3.24 or newer. CUDA is optional; the CPU build works on macOS and Linux without a CUDA toolkit.

```bash
cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DWARPKEP_ENABLE_CUDA=OFF
cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure
```

CI runs all seven CPU tests, including zero-revolution and multi-revolution Lambert, on each push and pull request. It also compares Kepler propagation with pykep and heyoka. Independent Lambert comparisons are still planned.

## Solve a multi-revolution Lambert problem

The CPU solver is header-only. Request a revolution count and check the status of each returned solution:

```cpp
#include <warpkep/detail/lambert_multi_rev_cpu.hpp>
#include <cmath>

namespace wd = warpkep::detail;

int main() {
    const wd::Vector3<double> r1{1.0, 0.0, 0.0};
    const wd::Vector3<double> r2{0.0, 1.0, 0.0};
    const double pi = std::acos(-1.0);
    const auto transfer = wd::solve_lambert_multi_rev_cpu<double>(
        r1, r2, pi / 2.0 + 2.0 * pi, 1.0, wd::LambertArc::short_arc, 1
    );

    for (int i = 0; i < transfer.slot_count; ++i) {
        const auto& solution = transfer.solutions[i];
        if (solution.result.status != warpkep::LambertStatus::success) {
            continue;
        }
        // Use solution.result.departure_velocity and arrival_velocity.
        // Identify this solution with (revolutions, arc, branch).
    }
    return transfer.complete ? 0 : 1;
}
```

Use `slot_count` as the loop bound: if only the right branch succeeds, it remains in slot 1. The [complete example](examples/lambert_multi_rev.cpp) checks that both solutions reach the target and shows how to reuse the geometry for several revolution counts.

## Python and CUDA

```bash
# CPU package: version information, without the CUDA extension.
python -m pip install .

# On a machine with a CUDA toolkit; install compatible CuPy separately.
python -m pip install . \
  -Ccmake.define.WARPKEP_ENABLE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native \
  -Ccmake.define.BUILD_TESTING=OFF
```

`warpkep.propagate_kepler(states, durations, mu=...)` takes CuPy arrays of shape `(6, N)` and `(N,)`. It returns the propagated states, statuses and iteration counts on the GPU, ready when the call returns. The [Python/CuPy guide](docs/python_cuda.md) covers array layout, streams and error handling.

## Build the documentation

The documentation needs Python 3.11+ and can be built without compiling the library:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -n -W --keep-going -b html docs build/docs
python -m http.server 8000 --directory build/docs
```

Open [localhost:8000](http://localhost:8000). A separate CI workflow builds the documentation and runs the C++ Lambert example on each push and pull request.

## Project direction

The next steps are broader Lambert validation and a public API, followed by GPU batching and trajectory search. See the [roadmap](docs/roadmap.md) for planned work and the [benchmark guide](docs/benchmarks.md) for performance measurements.

## License

Copyright 2026 Benjamin von Plehn. Licensed under the [Apache License, Version 2.0](LICENSE).
