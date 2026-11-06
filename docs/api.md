# API reference

Functions and types in `warpkep::detail` are internal and may change during development. `Vector3<T>` is warpkep's fixed-size vector type; access its three components with `operator[]`.

## CPU Lambert

### Zero-revolution solve

Header: `<warpkep/detail/lambert_cpu.hpp>`. Namespace: `warpkep::detail`.

```cpp
template <typename T>
LambertResult<T> solve_lambert_cpu(
    const Vector3<T>& departure,
    const Vector3<T>& arrival,
    T dt, T mu, LambertArc arc,
    const LambertOptions& options = {}
);
```

This function returns the departure and arrival velocities for `M = 0`. `T` must be `double`; other types fail at compile time. See [arc and geometry conventions](lambert.md#geometry-arcs-and-revolutions) for the input requirements.

| `LambertResult<T>` field | Type | Meaning |
| --- | --- | --- |
| `departure_velocity` | `Vector3<T>` | Initial velocity, L/T |
| `arrival_velocity` | `Vector3<T>` | Final velocity, L/T |
| `status` | `warpkep::LambertStatus` | Numerical outcome |
| `iterations` | `int` | Root refinement iterations; excludes bracketing/probes |

The defaults are `max_iterations = 64` and `relative_tolerance = 1e-12`. The tolerance applies to the dimensionless time of flight. On failure, both velocity vectors contain NaN.

### Fixed-M solve

Header: `<warpkep/detail/lambert_multi_rev_cpu.hpp>`. This includes the zero-revolution header. Namespace: `warpkep::detail`.

```cpp
template <typename T>
LambertMultiRevResult<T> solve_lambert_multi_rev_cpu(
    const Vector3<T>& departure,
    const Vector3<T>& arrival,
    T dt, T mu, LambertArc arc, int revolutions,
    const LambertMultiRevOptions& options = {}
);
```

Pass a non-negative revolution count and use `T = double`. For `M = 0`, the function calls the zero-revolution solver. For `M > 0`, it finds the minimum flight time and attempts both branches if the target time is above the threshold band.

```cpp
struct LambertMultiRevOptions {
    LambertOptions roots{};
    int minimum_max_iterations{64};
    double minimum_x_tolerance{1e-12};
    double threshold_relative_tolerance{1e-12};
};
```

The [numerical contract](numerical_contract.md#precision-and-termination) defines these tolerances and their allowed ranges.

### Multi-revolution result

| `LambertMultiRevResult<T>` field | Type | Meaning |
| --- | --- | --- |
| `solutions` | `LambertMultiRevSolution<T>[2]` | Solution storage; read slots `[0, slot_count)` |
| `slot_count` | `int` | One `single` slot for `M = 0`; `left`, then `right` for `M > 0` |
| `solution_count` | `int` | Number of successful slots; their positions in the array are unchanged |
| `complete` | `bool` | All solutions resolved, or `no_solution` below the threshold band |
| `status` | `warpkep::LambertStatus` | Overall outcome; individual slots may still succeed if this reports a failure |
| `minimum` | `LambertMinimumResult<T>` | Minimum diagnostics for `M > 0`; inapplicable at `M = 0` |
| `threshold` | `T` | Half-width of the minimum classification band in dimensionless time |

For `M = 0`, no minimum search is needed: `minimum.status` is `unsupported_case`, its floating fields and `threshold` are NaN, and `minimum.iterations` is zero. The transfer itself can still succeed. For any request, check `minimum.status` before reading its diagnostics.

| `LambertMultiRevSolution<T>` field | Type | Meaning |
| --- | --- | --- |
| `revolutions` | `int` | Requested complete revolution count |
| `arc` | `LambertArc` | Requested short or long arc |
| `branch` | `LambertBranch` | `single`, `left` or `right` |
| `x` | `T` | Scalar Lambert root; NaN for a failed slot |
| `result` | `LambertResult<T>` | Velocities, individual status and refinement count |

Use a slot only when its `result.status` is `success`. A `near_minimum` result has no usable velocities. The [Lambert guide](lambert.md#minimum-time-classification) explains `no_solution`, `near_minimum` and `complete`.

### Prepared geometry

```cpp
template <typename T>
LambertGeometryResult<T> prepare_lambert_geometry(
    const Vector3<T>& departure, const Vector3<T>& arrival,
    T dt, T mu, LambertArc arc
);

template <typename T>
LambertMultiRevResult<T> solve_lambert_multi_rev_prepared(
    const LambertGeometry<T>& g, T mu, LambertArc arc, int revolutions,
    const LambertMultiRevOptions& options = {}
);
```

`LambertGeometryResult<T>` contains `geometry` and `status`. After a successful preparation, reuse the geometry with the original `mu` and `arc`. Prepare it again if the endpoints, flight time, `mu` or arc change. The prepared solver cannot check whether the supplied geometry matches those inputs.

### Minimum diagnostics and scalar helpers

```cpp
template <typename T>
LambertMinimumResult<T> find_lambert_minimum(
    T lambda, T q, int revolutions,
    const LambertMultiRevOptions& options = {}
);
```

This helper requires `M > 0`. Take `lambda` and `q` from the same successful geometry preparation.

| `LambertMinimumResult<T>` field | Meaning |
| --- | --- |
| `x` | Midpoint of the final derivative bracket |
| `time` | Dimensionless `T_M(x)` at this point |
| `lower`, `upper` | Signed bracket: derivative ≤ 0 at lower, ≥ 0 at upper |
| `curvature` | Second derivative of dimensionless time with respect to `x` |
| `time_uncertainty` | Numerical diagnostic used in classification; not a proved bound |
| `status` | Minimum-search outcome |
| `iterations` | Refinement count, excluding expansion and boundary probes |
| `precision_limited` | Search terminated at adjacent representable endpoints |

Other internal helpers are `evaluate_lambert_time`, `evaluate_lambert_derivatives`, `solve_lambert_root`, `solve_lambert_branch_root` and `reconstruct_lambert_velocities`. The multi-revolution header adds time and derivative overloads with an explicit `M` argument. When using these helpers, keep geometry parameters and minimum brackets consistent with the requested geometry and revolution count. For ordinary use, call one of the full solvers above.

## Scalar Kepler propagation

Header: `<warpkep/detail/kepler_cpu.hpp>`. Namespace: `warpkep::detail`.

```cpp
template <typename T>
KeplerPropagationResult<T> propagate_kepler_cpu(
    const CartesianState<T>& initial, T dt, T mu,
    const KeplerSolveOptions<T>& options = {}
);
```

`CartesianState<T>` contains `position` and `velocity`, both `Vector3<T>`. The result contains `state`, `warpkep::KeplerPropagationStatus status` and `int iterations`. The template accepts `float` and `double` and compiles for both CPU and CUDA, including when called from a GPU kernel.

`KeplerSolveOptions<T>` defaults to 64 iterations and a relative equation residual tolerance of `8 * epsilon(T)`. The CUDA launcher uses these defaults and has no argument for changing them.

## C++ CUDA propagation

Header: `<warpkep/cuda/kepler.hpp>`. CMake target: `warpkep::cuda`.

```cpp
cudaError_t warpkep::launch_kepler_soa_double(
    CartesianSoAConstView<double> input,
    const double* durations,
    double mu,
    CartesianSoAView<double> output,
    KeplerPropagationStatus* statuses,
    int* iterations,
    std::size_t n,
    cudaStream_t stream
);
```

Each SoA view holds six borrowed pointers, in order `r_x, r_y, r_z, v_x, v_y, v_z`. Every array must have at least `n` elements accessible on the selected device. The caller allocates the buffers, transfers data and waits for completion. Keep the buffers and stream alive until the work finishes. An empty batch accepts null pointers.

The returned `cudaError_t` reports launch errors. Check for asynchronous execution errors when waiting for completion, then read `statuses` for each trajectory's numerical result. See the [memory and execution contract](numerical_contract.md#cuda-batch-memory-and-execution).

## Python interface

### `warpkep.version()`

Returns the compiled library version as a string, currently `0.1.0.dev0`. This function is available in the default CPU-only Python installation.

### `warpkep.propagate_kepler(states, durations, *, mu=1.0)`

Returns `(out, statuses, iterations)` as CuPy arrays, ready when the call returns. It requires the CUDA extension and compatible CuPy. Input arrays must be `float64`, C-contiguous and on the same GPU, with shapes `(6, N)` and `(N,)`.

The wrapper allocates its outputs and waits for execution. It accepts GPU inputs without converting their dtype or layout. See the [Python/CuPy guide](python_cuda.md) for streams and exceptions.

## Status types

`<warpkep/kepler_status.hpp>` defines `warpkep::KeplerPropagationStatus` and `warpkep::propagation_status_name`. `<warpkep/lambert_status.hpp>` defines `warpkep::LambertStatus`, including `no_solution` and `near_minimum`. The Lambert example includes a local helper for printing status names.

The Python GPU extension exports `SUCCESS`, `INVALID_INPUT`, `UNSUPPORTED_CASE`, `NOT_CONVERGED` and `NUMERICAL_FAILURE` in `warpkep._cuda`. Compare against these constants rather than hard-coding integers. See the [status contract](numerical_contract.md#status-meanings).
