# Lambert transfers

Given two positions and a flight time, Lambert's problem asks for the departure and arrival velocities of a two-body orbit. warpkep solves it on the CPU in double precision using [Izzo's formulation](https://arxiv.org/abs/1403.2705).

The C++ routines are in `warpkep::detail` and may change during development. They allocate no memory and throw no exceptions. Python bindings and a CUDA solver are not available yet.

## Geometry, arcs and revolutions

The inputs are two finite, non-zero positions `r1` and `r2`, a positive flight time `dt`, and a positive gravitational parameter `mu`. Both scalars must be finite. Use consistent units and an inertial frame centred on the attracting body.

| Choice | Meaning |
| --- | --- |
| `LambertArc::short_arc` | Transfer angle between 0 and π, around the normal `r1 × r2` |
| `LambertArc::long_arc` | Transfer angle between π and 2π, around the opposite normal |
| `revolutions = M` | Number of additional complete revolutions before the final arc |

The chosen arc sets the direction of motion; there is no separate prograde/retrograde argument. For `M > 0`, the total angular sweep is `2πM` plus the final arc. Coincident, collinear and nearly collinear endpoints are unsupported. The solver does not check periapsis altitude or collisions with the central body.

## Zero revolutions

Include `<warpkep/detail/lambert_cpu.hpp>` and call:

```cpp
const auto result = wd::solve_lambert_cpu<double>(
    r1, r2, dt, mu, wd::LambertArc::short_arc
);
if (result.status == warpkep::LambertStatus::success) {
    // result.departure_velocity, result.arrival_velocity
}
```

A supported zero-revolution problem has one solution in exact arithmetic. The tests cover elliptic, parabolic and hyperbolic transfers. If the numerical solver fails to converge, that does not mean the physical solution is absent.

You can also pass `M = 0` to the multi-revolution solver. It follows the same numerical path and returns one solution slot labelled `single`, without searching for a minimum.

## Request a fixed revolution count

Include `<warpkep/detail/lambert_multi_rev_cpu.hpp>`. A transfer with `M > 0` is elliptic. Its time-of-flight curve has a minimum, with two roots above it:

| Slot | Branch | Location |
| --- | --- | --- |
| 0 | `LambertBranch::left` | `x < minimum.lower` |
| 1 | `LambertBranch::right` | `x > minimum.upper` |

Left and right refer to the root variable `x`; the short or long arc is chosen separately. Store `(revolutions, arc, branch)` with each solution so you can identify it later.

For a unit circular reference with one additional revolution:

```{literalinclude} ../examples/lambert_multi_rev.cpp
:language: cpp
:start-after: // [solve-begin]
:end-before: // [solve-end]
:dedent: 4
```

## Reading results

`LambertMultiRevResult<double>` stores up to two solutions in fixed slots. `slot_count` is one for `M = 0` and two for `M > 0`. `solution_count` tells you how many succeeded; successful solutions stay in their original slots.

```{warning}
Loop over `[0, slot_count)` and check each `result.status`. If only the right branch succeeds, it remains in slot 1, so a loop ending at `solution_count` would miss it. Check individual slots even when the overall request reports a failure.
```

The example propagates each successful solution from departure and checks that it recovers the arrival position and velocity:

```{literalinclude} ../examples/lambert_multi_rev.cpp
:language: cpp
:start-after: // [slots-begin]
:end-before: // [slots-end]
:dedent: 4
```

`Status` is an alias for `warpkep::LambertStatus`. The error tolerance used here applies to this normalized example.

## Minimum-time classification

Let `s = (|r1| + |r2| + |r2 − r1|) / 2`. The solver uses the dimensionless time

$$
T = dt\sqrt{\frac{2\mu}{s^3}}.
$$

For a fixed geometry and `M > 0`, it locates the minimum of `T_M(x)` on `−1 < x < 1`. With `T_min = minimum.time` and `δ = threshold`:

| Condition | Global status | `solution_count` | `complete` |
| --- | --- | --- | --- |
| `T − T_min < −δ` | `no_solution` | 0 | `true` |
| `abs(T − T_min) <= δ` | `near_minimum` | 0 | `false` |
| `T − T_min > δ`, both branches solved | `success` | 2 | `true` |
| Root/minimum computation fails | Failure status | 0 or 1 | `false` |

The two roots merge at the minimum. Within the threshold band, floating point precision is insufficient to distinguish zero, one or two solutions reliably. The solver returns `near_minimum` with no usable velocities.

`no_solution` applies to the requested arc and revolution count. Other counts may still have solutions. The threshold includes an estimate of numerical uncertainty; it is not a mathematically proved bound.

## Tolerances and diagnostics

```cpp
wd::LambertMultiRevOptions options;
options.roots.max_iterations = 64;
options.roots.relative_tolerance = 1e-12;
options.minimum_max_iterations = 64;
options.minimum_x_tolerance = 1e-12;
options.threshold_relative_tolerance = 1e-12;

const auto transfer = wd::solve_lambert_multi_rev_cpu<double>(
    r1, r2, dt, mu, arc, revolutions, options
);
```

These are the default settings. A root succeeds when `abs(T_M(x) − T) / T <= roots.relative_tolerance` and, for `M > 0`, it lies on the correct side of the minimum bracket. The minimum search stops when its signed derivative bracket is no wider than `minimum_x_tolerance`. It can also stop at adjacent floating point values, reported by `minimum.precision_limited`.

The time threshold is computed as

$$
\delta = \max(\tau_{threshold},\tau_{root},64\epsilon)
         \max(T,T_{min}) + u_T,
$$

where `u_T = minimum.time_uncertainty` estimates the uncertainty in the minimum time and ε is FP64 machine epsilon. All three tolerances must be finite and strictly between 0 and 1. Both iteration limits must be positive.

A small time residual does not guarantee equally small velocity errors, especially near the minimum or a degenerate geometry. For a difficult case, inspect the minimum bracket, `threshold`, individual statuses and iteration counts before tightening the tolerances. Failed slots have NaN velocities and root `x`; minimum diagnostics may still be available.

## Enumerate several revolution counts

Each call solves one value of `M`. To try every count from zero to `M_max`, prepare the geometry once and reuse it in a loop:

```{literalinclude} ../examples/lambert_multi_rev.cpp
:language: cpp
:start-after: // [enumerate-begin]
:end-before: // [enumerate-end]
:dedent: 4
```

Check that preparation succeeded, then keep the same `mu` and `arc` in subsequent calls. The geometry also contains the flight time, so prepare it again whenever an input changes. Collect successful slots using `slot_count` and their individual statuses. The example stops on an incomplete result; an application can instead keep whichever branch succeeded.

Choose `M_max` in the calling code; the solver does not select it automatically. Revolution counts are non-negative C++ integers. Analytical tests currently cover `M = 1, 2`; large counts and extreme input scales need further validation.

## Relationship to pykep

If you are coming from [pykep's Lambert class](https://esa.github.io/pykep/lambert.html), note that warpkep solves one revolution count per call. Its arguments, arc convention and result layout differ from pykep's Python interface.

The [API reference](api.md) lists the signatures and result fields. See [validation](validation.md) for the cases covered by the tests.
