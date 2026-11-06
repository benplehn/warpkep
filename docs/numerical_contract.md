# Numerical contract

The two-body routines use the conventions below. Scalar Kepler and CPU Lambert interfaces are internal; the C++ CUDA and Python/CuPy interfaces are experimental.

## Units, frames and time

Use a consistent system of units throughout each request:

| Quantity | Unit |
| --- | --- |
| Position | L |
| Velocity | L/T |
| Gravitational parameter `mu` | L³/T² |
| Relative duration `dt` | T |

Express positions and velocities in the same inertial frame, centred on the attracting body. warpkep does not convert units, time scales or reference frames, and cannot detect a mismatch from the numbers alone.

Convert absolute epochs to a relative duration before calling a solver. Use compatible time scales and subtract the epochs before reducing precision.

- **Kepler:** finite positive, zero and negative durations are accepted.
- **Lambert:** the duration must be finite and strictly positive.

## Input validation and supported geometry

### Kepler propagation

Position and velocity components must be finite, the initial position non-zero, and `mu` finite and strictly positive. Solver iteration limits must be positive; the relative tolerance must be finite and positive.

Inputs are validated before the `dt == 0` check. A valid zero-duration request returns the initial state with zero iterations. For non-zero durations, radial motion and zero velocity return `unsupported_case`. The radial check tests whether the cross product of normalized position and velocity is zero.

The universal-variable formulation handles non-radial elliptic, parabolic and hyperbolic motion. See [validation](validation.md) for the analytical cases and independent CPU comparisons.

### Lambert transfers

`r1` and `r2` must be finite and non-zero, `dt > 0`, and `mu > 0`, with finite scalars. Arc values must be `short_arc` or `long_arc`. A multi-revolution request also requires a non-negative integer `M`.

Coincident, collinear and nearly collinear positions are unsupported. The geometry check rejects `|u1 × u2| <= 64 ε`, where `u1` and `u2` are unit position vectors and ε is machine epsilon. This includes transfer angles of 0 and π.

`short_arc` follows the normal `r1 × r2` with final arc angle in (0, π); `long_arc` follows the opposite normal with angle in (π, 2π). For `M > 0`, complete revolutions precede the final arc. The solver does not check periapsis altitude or collisions with the central body.

`M = 0` covers elliptic, parabolic and hyperbolic solutions. `M > 0` uses elliptic roots in `−1 < x < 1`. The analytical multi-revolution tests cover `M = 1, 2`; large counts and extreme scales remain uncharacterised.

## Precision and termination

CPU Kepler primitives accept FP32 and FP64; independent reference comparisons use FP64. Both Lambert entry points require FP64 at compile time. The CUDA propagation API and Python/CuPy wrapper also use FP64, with no implicit downcasting.

| Solver setting | Default | Acceptance meaning |
| --- | --- | --- |
| `KeplerSolveOptions<T>::max_iterations` | 64 | Refinement iteration budget |
| `KeplerSolveOptions<T>::relative_tolerance` | `8 * epsilon(T)` | Universal equation residual relative to `sqrt(mu) * abs(dt)` |
| `LambertOptions::max_iterations` | 64 | Root refinement iteration budget |
| `LambertOptions::relative_tolerance` | `1e-12` | Relative dimensionless time residual |
| `LambertMultiRevOptions::minimum_max_iterations` | 64 | Minimum search refinement budget |
| `LambertMultiRevOptions::minimum_x_tolerance` | `1e-12` | Absolute width of a signed derivative bracket in `x` |
| `LambertMultiRevOptions::threshold_relative_tolerance` | `1e-12` | Relative contribution to the minimum-time classification band |

All Lambert tolerances must be finite and strictly between 0 and 1. The minimum search can also stop at adjacent floating point endpoints, reported by `precision_limited`.

For Lambert, with semiperimeter `s`, success requires

$$
T = dt \sqrt{\frac{2\mu}{s^3}}, \qquad
\frac{|T_M(x)-T|}{T} \leq \tau_{root}.
$$

For `M > 0`, each accepted root must also lie strictly outside the minimum bracket on the requested side. Iteration counts exclude initial bracketing and endpoint probes, so a root accepted at an endpoint can have zero iterations.

The residual tolerance does not give a uniform bound on position or velocity errors. Extreme input scales, nearly collinear positions and limited root precision can reduce accuracy. If the iteration budget is exhausted or the root bracket collapses before the tolerance is met, the solver returns `not_converged`.

The baseline build leaves fast-math disabled. Results may differ slightly between compilers, architectures and CPU/GPU execution; compare them using numerical tolerances.

## Status meanings

The two public status enums share the first five names; the last two belong only to `LambertStatus`.

| Status | Meaning |
| --- | --- |
| `success` | The routine met its documented termination criteria |
| `invalid_input` | Numerical inputs or solver options failed validation |
| `unsupported_case` | An explicitly excluded geometry or domain |
| `not_converged` | The numerical termination criteria could not be met |
| `numerical_failure` | Non-finite arithmetic or another unusable intermediate result |
| `no_solution` | For `M > 0`, the target lies below the minimum-time threshold band |
| `near_minimum` | The number of distinct solutions near the minimum cannot be resolved at the working precision |

Success means the numerical criteria were met. The caller must still check units, frames and mission constraints. A convergence failure does not mean that no physical solution exists.

### Failed outputs and partial Lambert results

- Failed Kepler propagation fills the six state components with NaN.
- Failed scalar Lambert solves fill the six velocity components with NaN.
- A failed multi-revolution slot has NaN velocities and NaN root `x`.
- If one multi-revolution branch fails, the other may still succeed. Iterate over `slot_count` and check each slot's status.
- A `near_minimum` classification returns no usable velocities.
- Minimum diagnostics and `threshold` may remain available when a request fails; check their status before reading them.

`complete` is true when the solver returns the single solution for `M = 0`, both branches for `M > 0`, or `no_solution` below the threshold band. It is false for `near_minimum` and numerical failures, including when only one branch succeeds. The [Lambert guide](lambert.md) gives the threshold formula.

## CUDA batch memory and execution

`launch_kepler_soa_double` borrows caller-owned device storage. For `n > 0`:

- State components are six contiguous arrays in order `r_x, r_y, r_z, v_x, v_y, v_z`.
- Each trajectory has `durations[i]`; a single `mu` applies to the batch.
- All input/output arrays and the required status/iteration arrays contain at least `n` elements accessible on the selected CUDA device.
- Outputs overlap neither inputs nor one another.
- Keep buffers and the stream alive until completion, and establish stream dependencies before accessing shared data.

The launcher submits work to the supplied stream without allocating, copying or synchronizing. After a successful launch, wait for the stream and check execution errors before using the outputs.

For `n == 0`, it returns `cudaSuccess` without submitting work or touching pointers; null pointers are accepted. One numerical failure does not abort other trajectories in a completed batch.

The CUDA kernel uses the shared scalar Kepler implementation. Its dedicated GPU test covers conic cases, invariants and round trips; the default hosted CI only compiles CUDA and does not execute these checks.

## Python/CuPy execution

The Python wrapper requires CuPy `float64` inputs in C-contiguous layout, on the same GPU. It allocates three output arrays and returns only after the outputs are ready. The binding additionally allocates a temporary typed status buffer.

The wrapper waits for the current CuPy stream on the input device before calling the binding. If another stream produced the inputs, wait for that producer first. Sharing an array between libraries does not by itself synchronize their streams.

The [Python/CuPy guide](python_cuda.md) lists array shapes, exceptions and status constants. An invalid `mu` rejects the whole Python call. Invalid individual states are reported in the returned statuses.

## Structural and execution errors

Array shapes, dtypes, memory locations and capacity must satisfy the interface requirements. In C++, the caller is responsible for supplying valid pointers and enough storage.

A CUDA execution error can leave the batch incomplete. Discard its outputs, even if some values look valid. Failed trajectories are filled with NaN only when the kernel and its failure handling finish executing.

## Validation and evolution

[Validation](validation.md) describes the analytical tests, independent comparisons and GPU checks. CPU/GPU agreement checks the shared implementation's consistency; an independent reference is needed to assess errors in the algorithm itself.

Update this contract, the relevant guide and the API reference when an interface or tolerance changes. Record the corresponding validation results. Planned work is listed in the [roadmap](roadmap.md).
