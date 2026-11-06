# Validation

The tests compare results with analytical orbits and independent Kepler propagators. They also check CPU/GPU agreement, conserved quantities and failure handling. The table below shows which checks run in CI and which need a local GPU.

## Coverage

| Area | Evidence in this repository | Execution |
| --- | --- | --- |
| Stumpff and universal equation | `test_stumpff.cpp`, `test_universal_kepler.cpp`, `test_kepler_solver.cpp` | CPU CTest |
| Kepler state propagation | Analytical motion, backward propagation and failure cases in `test_kepler_propagation.cpp` | CPU CTest |
| FP32 / FP64 Kepler | Circular and elliptic analytical cases in `test_kepler_precision.cpp` | CPU CTest |
| Independent Kepler references | Shared cases and JSON comparisons with pykep, heyoka and Barker's equation | Reference CI job |
| Zero-revolution Lambert | Known conics, time residual and position/velocity Kepler closure | CPU CTest `lambert` |
| Multi-revolution Lambert | Circular short/long arcs at `M = 0, 1, 2`, polar `M = 1`, ellipse `M = 2`, threshold and failure paths | CPU CTest `lambert_multi_rev` |
| GPU propagation | Shared conics, block boundaries, invariants, round trips and failures in `test_kepler_gpu.cpp` | Explicit GPU run required |
| Python/CuPy | Analytical mixed batch, empty batch and invalid `mu` in `test_python_cuda.py` | Explicit GPU run required |
| Documentation example | Compiled C++ example with position/velocity closure and fixed-M enumeration | Documentation CI job |

## CPU tests

```bash
cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DWARPKEP_ENABLE_CUDA=OFF
cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure

# Only Lambert tests:
ctest --test-dir build/cpu -R lambert --output-on-failure
```

CTest runs seven CPU test executables. Each contains several numerical cases or assertions.

### Lambert tests

The zero-revolution tests cover circular short, long and polar transfers, plus elliptic, hyperbolic and parabolic orbits between true anomalies ±π/3. In these normalized cases, analytical velocity errors, arrival position and velocity errors, and relative time residuals must stay within `1e-12`. The suite also checks both sides of the parabolic point and the main failure cases.

The multi-revolution suite additionally checks:

- Left and right roots lie on opposite sides of the minimum bracket.
- Exactly one of the two solutions recovers the known reference orbit.
- Relative time residuals, scaled arrival errors and revolution counts reconstructed from orbital energy and phase stay within `1e-12` for the tested cases.
- The minimum agrees with a stored high-precision reference. Flight times below, within and just above the threshold band produce the expected statuses and number of solutions.
- Invalid inputs, unsupported geometry and short iteration budgets return the expected failures. Partial results retain successful branches and fill failed velocities with NaN.
- Direct and prepared-geometry calls agree for `M = 0` through 3.

Independent Lambert solver comparisons are still planned, along with broader tests of revolution counts, input scales and nearly degenerate geometries. The reference CI job currently checks Kepler propagation only.

## Independent Kepler references

`validation/kepler_cases.hpp` defines the inputs, scales and tolerances. `warpkep_cpu_samples` writes the C++ results to JSON, and `validation/compare_cpu_references.py` compares them with the reference propagators.

```bash
cmake --build build/cpu --target warpkep_cpu_samples
./build/cpu/warpkep_cpu_samples > build/cpu/propagation_samples.json

docker build --platform linux/amd64 \
  -f docker/Dockerfile.references -t warpkep-references:dev docker
docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace:ro" -w /workspace warpkep-references:dev \
  python validation/compare_cpu_references.py build/cpu/propagation_samples.json
```

The reference container pins pykep 3.0.1 and heyoka.py 7.10.1, using a shared native heyoka installation.

The comparison script lists known reference divergences by engine version and case name. Six pykep 3.0.1 cases near or at the parabolic limit, with `dt = ±1`, show errors consistent with cancellation at the elliptic/hyperbolic transition. Their errors are printed in the report. heyoka must pass every case; exact parabolic cases are also checked with Barker's equation.

## GPU execution

Build with CUDA and `WARPKEP_RUN_GPU_TESTS=ON`, as described in [installation](installation.md). On a compatible NVIDIA GPU:

```bash
ctest --test-dir build/gpu -L gpu --output-on-failure
./build/gpu/warpkep_test_kepler_gpu --samples build/gpu/gpu_samples.json
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
  ./build/gpu/warpkep_test_kepler_gpu
```

The GPU suite uses batch sizes 1, 255, 256, 257 and 1025 to test block boundaries. It also checks empty batches, mixed valid and invalid inputs, energy and angular momentum drift, and forward/backward propagation. Invariant and round-trip tolerances are `1e-11`; state comparisons use each case's scaled tolerance. Exported GPU samples can be checked with the same independent reference script.

The Python tests require a CUDA-enabled package and CuPy:

```bash
python -m pytest tests/test_python_cuda.py -q
```

The Python tests include a non-blocking stream and a mixed batch of 257 trajectories, with analytical state checks at `1e-12`. If CuPy is missing, pytest skips the file. These tests need a GPU and do not run on the standard hosted CI runners.

## What hosted CI checks

`.github/workflows/ci.yml` runs on pushes, pull requests and manual dispatch. The `cpu-python` job builds with `BUILD_TESTING=ON` and runs `ctest --test-dir build/cpu --output-on-failure`, which includes all seven CPU tests: the Kepler tests, `lambert` and `lambert_multi_rev`. It also checks the Python package import and version, and runs a small parallel CPU batch.

The `references` job compares CPU Kepler results with independent propagators. The `cuda-compile` job compiles and links the CUDA targets, including the GPU test executable, but does not execute kernels. Neither GPU CTest tests nor `tests/test_python_cuda.py` run on these hosted runners.

`.github/workflows/docs.yml` builds HTML with warnings treated as errors, compiles and runs the Lambert example, and uploads the HTML as a downloadable artifact.

When reporting results, include the commit, platform, compiler, precision, inputs, units, solver options, reference method and maximum observed errors. Use the [benchmark procedure](benchmarks.md) for timing results.
