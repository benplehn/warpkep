# Numerical contract

# Numerical contract

**Status:** design contract, partially implemented.

Implemented so far:

* a C++ library skeleton and Python bindings that only expose the version;
* generic host/device Kepler primitives in float and double;
* an internal CPU Kepler propagator in FP64;
* an experimental FP64 CUDA Kepler batch interface (section 9);
* an internal CPU Lambert solver in FP64, zero revolution (section 10).

The other numerical APIs below, including the Python numerical interface, are not implemented yet. This contract defines their intended behaviour; solver-specific domains, tolerances and defaults are documented as each implementation is validated.

## 1. Units

Any consistent system of units is accepted. For a length unit L and a time unit T:


| Quantity                   | Unit    |
| -------------------------- | ------- |
| Position                   | L       |
| Velocity                   | L/T     |
| Gravitational parameter mu | L^3/T^2 |
| Duration                   | T       |

Kilometres and seconds, SI units and dimensionless formulations are all valid. The caller is responsible for consistency: warpkep never infers or converts units, and a positive finite mu does not prove its units are correct.

## 2. Reference frames

For two-body problems, positions and velocities are expressed in the same inertial frame, centred on the attracting body. warpkep performs no frame transformation and cannot detect a frame mismatch from raw arrays.

Other dynamical models will document their own conventions; this requirement does not automatically apply to rotating frames.

## 3. Time

Two-body routines take a relative duration** **`dt` in the time unit of the state and mu. Absolute dates are handled by the caller, who must use compatible time scales and subtract the reference epoch before reducing precision.

For Kepler propagation, after input validation:

* `dt = 0` returns the initial state unchanged;
* `dt < 0` propagates backwards.

Lambert requires** **`dt > 0` (section 10). Time-dependent models will document their reference epoch, time scale and treatment of control discontinuities.

## 4. Input validation

For each two-body propagation request:

* position and velocity components are finite;
* mu is finite and strictly positive;
* dt is finite;
* the initial position is non-zero.

A zero velocity is not invalid by itself; radial trajectories require an explicit solver-domain policy. Validation happens before the zero-duration shortcut:** **`dt = 0` does not make a NaN state valid.

These checks do not verify physical plausibility, units or frames.

## 5. Supported domain

Each numerical routine documents:

* the dynamical model it solves;
* the branches and geometries it supports;
* the singular or degenerate cases it excludes;
* the meaning of its tolerances;
* its iteration limits and termination criteria.

The universal-variable propagator is intended to cover elliptic, parabolic and hyperbolic motion; each domain is advertised only once validated.

A known unsupported case is distinct from a failed solve. Non-convergence is not evidence that no physical solution exists.

## 6. Precision and scaling

The first implementations and their validation use double precision, and the initial Python numerical API will require float64 inputs. No silent float64 → float32 conversion is allowed; any convenience conversion is explicit and documented.

Internal scaling may improve conditioning or avoid overflow and underflow. Its definition and effect on tolerances are documented. Scaling cannot recover information already lost to rounding.

Float32 filtering and mixed precision remain experimental until their errors are measured. Candidate-selection experiments must also measure false negatives: reevaluating retained candidates in float64 cannot recover good candidates discarded by a float32 filter.

Fast-math is disabled in the validation baseline. Any other arithmetic mode is identified in benchmark results.

## 7. Result statuses

Each trajectory gets an explicit status.


| Status              | Meaning                                                                    |
| ------------------- | -------------------------------------------------------------------------- |
| `success`           | The computation completed and met the documented termination criteria.     |
| `invalid_input`     | An input failed the validation rules.                                      |
| `unsupported_case`  | The request is explicitly outside the implemented domain.                  |
| `not_converged`     | The termination criteria were not met within the allowed limits.           |
| `numerical_failure` | An intermediate operation produced a non-finite value or no usable result. |

`success` reports a numerical outcome only; it does not certify units, frames or mission constraints.

On any failure status, all floating-point result fields are NaN; the status remains authoritative. One failed trajectory never aborts the others in a batch.

## 8. Structural and execution errors

Structural errors concern the whole call: incompatible array lengths, unsupported dtype, wrong memory location, insufficient output capacity. They are rejected before computation where detectable. The C++ caller remains responsible for valid pointers and sufficient storage.

The Python interface raises** **`TypeError` for dtype mismatches and** **`ValueError` for shape or layout violations. Device and CUDA failures are reported as execution errors.

An execution failure may leave the batch incomplete; outputs must then not be used, even if they hold values from an earlier operation. The per-trajectory NaN rule applies only when computation and failure handling actually complete.

## 9. CUDA Kepler batch interface (experimental, FP64)

Header** **`<warpkep/cuda/kepler.hpp>`, function** **`warpkep::launch_kepler_soa_double`, CMake target** **`warpkep::cuda`.

The interface works on data already resident on the selected device and performs no implicit host-device transfer.

**Data layout**

* n trajectories, stored as six contiguous arrays:** **`r_x, r_y, r_z, v_x, v_y, v_z`.
* Each trajectory has its own** **`durations[i]`; mu is shared by the batch.
* `statuses` and** **`iterations` are required.
* Every array holds at least n elements and is accessible on the device.
* Outputs overlap neither the inputs nor each other.

Component ordering and any future augmented state (mass, state-transition matrix) will be documented explicitly.

**Memory and streams**

* No allocation, copy or synchronization; work is submitted to the caller's stream.
* All buffers stay valid until execution completes.
* `n == 0` is a no-op: returns** **`cudaSuccess` without touching pointers or submitting work; null pointers are accepted.

**Errors**

* The returned** **`cudaError_t` reports CUDA runtime errors only.
* `cudaSuccess` means submitted, not finished; execution errors are checked at completion.
* Numerical failures are reported in** **`statuses[i]`, with the six output components set to NaN.

**Current domain**

* Radial trajectories and zero velocity are rejected for a non-zero duration.
* GPU coverage of parabolic and hyperbolic cases is not established yet.

## 10. CPU Lambert solver (internal, FP64, zero revolution)

Header** **`<warpkep/detail/lambert_cpu.hpp>`, entry point** **`warpkep::detail::solve_lambert_cpu<double>(r1, r2, dt, mu, arc, options)`, statuses** **`warpkep::LambertStatus` from** **`<warpkep/lambert_status.hpp>`.

**Model.** Two-body Lambert problem, zero revolution (M = 0). Only FP64 is accepted (compile-time check). No allocation, no exception.

**Inputs**

* `r1`,** **`r2`: finite, non-zero positions in L.
* `dt`: finite and strictly positive, in T.
* `mu`: finite and strictly positive, in L^3/T^2.
* `options.max_iterations > 0` (default 64).
* `options.relative_tolerance` finite in (0, 1) (default 1e-12).

**Arc convention**

* `short_arc`: motion about the normal r1 × r2, transfer angle in (0, π).
* `long_arc`: motion about the opposite normal, transfer angle in (π, 2π).

**Excluded geometries** (`unsupported_case`)

* r1 = r2 (zero chord).
* Collinear or nearly collinear positions, including transfer angles of 0 and π: |u1 × u2| ≤ 64 ε.

No periapsis or collision constraint is imposed.

**Tolerance.** Success means the relative residual on the dimensionless time of flight T = dt · sqrt(2 mu / s^3) satisfies |T(x) − T\_target| / T\_target ≤** **`relative_tolerance`. This does not bound velocity errors uniformly. Near-collinear rounding and the quantization of x near −1 can limit attainable accuracy; an unattainable residual returns** **`not_converged`, never a false success.

**Statuses.** For M = 0 a solution always exists in exact arithmetic, so there is no "no solution" status.** **`not_converged`means the iteration budget ran out or the bracket collapsed with no representable interior point. On failure, all six velocity components are NaN.** **`iterations` counts refinement-loop evaluations only.

**Method.** Izzo (2015) formulation, with Battin's series for |x − 1| ≤ 0.1, the logarithmic starter, and a bracketed order-4 Householder iteration with bisection fallback. Velocities follow Izzo's algorithm 1. Reference: D. Izzo,** ***Revisiting Lambert's Problem*, arXiv:1403.2705, DOI 10.1007/s10569-014-9587-y.

The scalar helpers (`evaluate_lambert_time`,** **`solve_lambert_root`, …) require λ and q from the same prepared geometry.

**Validation.** Analytical cases with mu = p = 1 (circular short, long and 3D polar arcs; ellipse, hyperbola and parabola between true anomalies ±π/3): velocity errors, closure through the CPU Kepler propagator and time residual all ≤ 1e-12. Additional checks cover both sides of the parabolic point x = 1 and the main failure paths.

## 11. Memory ownership

The caller owns input and output arrays; the library borrows them for the duration of the operation. Outputs can be preallocated and reused. Required sizes, alignment and temporary workspace are documented per routine.

Inputs and outputs must not overlap, and output component arrays must be mutually disjoint. In-place operation is unsupported unless a routine explicitly allows it.

CPU routines complete before returning. The CUDA interface may return before device execution finishes; the caller keeps borrowed storage alive and avoids conflicting access until completion.

## 12. CUDA streams and interoperability

Work is submitted to the caller's stream, without implicit device-wide synchronization. Any operation that waits for completion documents it.

The caller orders dependencies (events or explicit synchronization) before reading outputs or reusing their storage from another stream. Sharing memory through DLPack does not by itself order execution; each Python adapter documents how it handles producer and consumer streams.

Submission errors and asynchronous execution errors are distinct; the latter may only appear at a synchronization or completion check.

## 13. Tolerances and determinism

Each tolerance states the quantity it constrains and whether it is absolute, relative or applied to a normalized residual. Position and velocity tolerances use the caller's units; solver-residual tolerances are defined explicitly.

Defaults are chosen from validation cases before a routine is released; no unmeasured tolerance is assigned in advance.

For a fixed implementation, arithmetic configuration and environment, results are reproducible. Bitwise equality is not promised across CPU and GPU, architectures or compiler versions. Comparisons use documented tolerances. Future sequence-search routines will use deterministic tie-breaking with a stable candidate identifier.

## 14. Validation and performance reporting

Correctness is checked against independent references, analytical solutions and published cases. Compiling the same code for CPU and GPU helps debugging but is not independent validation.

Property checks complement these comparisons: conservation of two-body invariants, forward-backward propagation, and propagation of a Lambert solution to its target.

Validation records the case, units, frame, solver settings, reference method and observed errors.

Benchmarks compare methods at equal accuracy and domain coverage, and report:

* batch size, precision, tolerances and success rate;
* CPU hardware, thread count and parallel baseline;
* GPU model, driver and CUDA toolkit;
* compiler versions, build flags and source commit;
* kernel time separately from total time including transfers;
* warm-up and repeated-measurement procedure.

Asynchronous GPU work is timed up to completion; timing only the submission is not an execution benchmark.

Known reference divergences are listed in** **`validation/compare_cpu_references.py`, pinned to an engine version. With pykep 3.0.1, six near-parabolic cases (|e − 1| ≤ 1e-8, dt = ±1) diverge from warpkep and heyoka, consistent with cancellation in pykep's separate elliptic and hyperbolic branches. Their errors are still computed and reported. heyoka is required on every case, and exact parabolic cases are also checked against Barker's equation.

## 15. Changes to this contract

This document describes intended behaviour while the numerical prototypes mature. API changes they reveal are reflected here, with validation and migration notes when they affect users.

Implemented behaviour, demonstrated support and experimental features remain clearly distinguished.

Status: design contract, partially implemented.

warpkep currently provides a C++ library skeleton, Python bindings that only expose the version, generic host/device Kepler primitives in float and double, an internal CPU Kepler propagator in FP64, and an experimental FP64 CUDA batch interface (section 9). The other public numerical APIs described below, including the Python numerical interface, are not implemented yet.

This contract defines their intended behaviour. Solver-specific domains, tolerances and defaults will be documented and validated as the implementations become available.

## 1. Units

The API accepts any consistent system of units.

For a length unit L and a time unit T:


| Quantity                   | Unit    |
| -------------------------- | ------- |
| Position                   | L       |
| Velocity                   | L/T     |
| Gravitational parameter mu | L^3/T^2 |
| Propagation duration       | T       |

Kilometres and seconds, SI units, and consistent dimensionless formulations are valid choices.

The caller is responsible for unit consistency. warpkep does not infer units from numerical values or convert units implicitly. In particular, a positive and finite value of mu does not prove that it uses the correct units.

## 2. Reference frames

For two-body propagation, position and velocity are expressed in the same inertial frame, relative to the attracting body's centre.

The caller supplies states in the required frame. warpkep does not perform implicit frame transformations and cannot detect a frame mismatch from raw arrays alone.

Other dynamical models will specify their own frame conventions. The inertial-frame requirement above does not automatically apply to models defined in rotating frames.

## 3. Time

Two-body propagation takes a relative duration dt, using the time unit associated with the state and mu.

Absolute dates are handled outside this primitive. When converting absolute epochs to relative times, the caller must use compatible time scales and subtract the reference epoch before reducing numerical precision.

After validating the inputs:

* dt = 0 returns the initial state unchanged
* dt < 0 requests backward propagation

Time-dependent models will additionally document their reference epoch, time scale and treatment of control discontinuities.

## 4. Numerical input validation

For each two-body propagation request:

* position and velocity components must be finite
* mu must be finite and strictly positive
* dt must be finite
* the initial position must be non-zero

A zero velocity is not, by itself, an invalid input. Radial trajectories and trajectories approaching the central singularity require an explicit solver-domain policy. Input validation precedes the zero-duration shortcut. For example, dt = 0 does not make a state containing NaN valid.

These checks do not verify physical plausibility, units or reference frames.

## 5. Supported domain

Each numerical routine must state:

* the dynamical model it solves
* the branches and geometries it supports
* any excluded singular or degenerate cases
* the meaning of its tolerances
* its iteration limits and termination criteria

The universal-variable two-body propagator is intended to cover elliptic, parabolic and hyperbolic motion. Support for each domain will only be advertised once it has been validated.

A known unsupported case must be distinguished from a failed solve. Non-convergence is not evidence that a physical solution does not exist.

## 6. Precision and scaling

The first numerical implementation and its validation use double precision. The initial Python numerical API will require float64 inputs.

No silent conversion from float64 to float32 is allowed. Any convenience conversion must be explicit and documented.

Internal scaling may improve conditioning and avoid overflow or underflow. Its definition and effect on tolerances must be documented. Scaling does not recover information already lost through rounding.

Float32 filtering and mixed precision remain experimental until their errors have been measured. Candidate-selection experiments must also measure false negatives: reevaluating retained candidates in float64 cannot recover good candidates discarded by a float32 filter.

Fast-math transformations are disabled in the validation baseline. Any alternative arithmetic mode must be identified in benchmark results.

## 7. Result statuses

Numerical routines return an explicit status for each trajectory.


| Status             | Meaning                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| success            | The computation completed and met the documented numerical termination criteria.                        |
| invalid\_input     | An input failed the numerical validation rules.                                                         |
| unsupported\_case  | The request is explicitly outside the implemented domain.                                               |
| not\_converged     | The numerical termination criteria were not met within the allowed limits.                              |
| numerical\_failure | An intermediate numerical operation produced a non-finite value or otherwise prevented a usable result. |

A success status reports the routine's numerical outcome. It does not certify the caller's units, frame conventions or mission constraints.

For every completed trajectory with a failure status, floating-point result fields are filled with NaN. The status remains the authoritative indication of failure.

One failed trajectory does not abort the other trajectories in a batch.

## 8. Structural and execution errors

Structural errors apply to the whole call and are separate from per-trajectory numerical statuses.

Examples include incompatible array lengths, an unsupported dtype, an incompatible memory location or insufficient output capacity.

The Python interface reports dtype mismatches with TypeError and shape or layout violations with ValueError. Device and CUDA execution failures are reported as execution errors.

Where detectable, structural errors are rejected before computation. The C++ caller remains responsible for supplying valid pointers and sufficient storage.

An execution failure may prevent the batch from completing. In that case, outputs must not be treated as valid, even if they contain values from an earlier operation. The per-trajectory NaN rule only applies when the numerical computation and failure handling actually complete.

## 9. CUDA batch interface (experimental, FP64)

Header `<warpkep/cuda/kepler.hpp>`, function `warpkep::launch_kepler_soa_double`, CMake target `warpkep::cuda`.

The interface operates on data already resident on the selected device. It performs no implicit host-device transfers.

### Data layout

- For n trajectories, input and output states use six contiguous one-dimensional arrays:
  `r_x, r_y, r_z, v_x, v_y, v_z`.
- Each trajectory `i` has its own duration `durations[i]`; `mu` is shared by the batch.
- The `statuses` and `iterations` arrays are required.
- Every array holds at least `n` elements and is accessible on the selected device.
- Outputs must not overlap inputs or each other.

Component ordering and any future representation of augmented states, such as mass or a state-transition matrix, must be documented explicitly.

### Memory and streams

- No allocation, no copy, no synchronization.
- Work is submitted to the caller's stream.
- All buffers must stay valid until execution completes.
- `n == 0` is a no-op: it returns `cudaSuccess` without touching pointers or submitting work; null pointers are accepted.

### Errors

- The returned `cudaError_t` reports CUDA runtime errors only.
  `cudaSuccess` means "submitted", not "finished";
  execution errors must also be checked at completion.
- Numerical failures are reported per trajectory in `statuses[i]`;
  the six output components are then NaN.

### Current domain

- Radial trajectories and zero velocity are rejected for a non-zero duration.
- GPU coverage of parabolic and hyperbolic cases is not established yet.

## 10. Memory ownership

The caller owns input and output arrays. The library borrows their storage for the duration of the operation.

Outputs can be preallocated and reused. Required sizes, alignment constraints and temporary workspace requirements must be documented.

Input and output storage must not overlap. Output component arrays must also be mutually disjoint. In-place operation is unsupported unless a particular routine explicitly documents it.

For asynchronous operations, the caller must keep all borrowed storage alive and avoid conflicting access until completion.

The CPU interface completes its computation before returning. The CUDA batch interface may return before device execution finishes.

## 11. CUDA streams and interoperability

The CUDA batch interface accepts a stream supplied by the caller. Work is submitted to that stream.

The interface does not implicitly synchronize the whole device. Any operation that waits for completion must document that behaviour.

The caller must establish dependencies before reading outputs or reusing their storage from another stream. Events or explicit synchronization may be used to order these operations.

Sharing memory through an interoperability protocol such as DLPack does not, on its own, establish execution ordering. Each Python adapter must document how it handles producer and consumer streams.

Immediate submission errors and asynchronous execution errors are distinct. The latter may only become visible at a synchronization or completion check.

## 12. Tolerances and determinism

Tolerances must identify the quantity they constrain and whether they are absolute, relative or applied to a normalized residual.

Position and velocity acceptance tolerances are expressed in the caller's corresponding units. Solver-residual tolerances require their own explicit definition.

Default values will be selected using validation cases before a numerical routine is released. This initial contract deliberately does not assign unmeasured tolerance values.

For a fixed implementation, arithmetic configuration and supported execution environment, results must be reproducible. Bitwise equality is not promised across CPU and GPU, different architectures or different compiler versions.

Numerical comparisons use documented tolerances. Future sequence-search routines must define deterministic tie-breaking and use a stable candidate identifier when scores are equal.

## 13. Validation and performance reporting

Numerical correctness is checked against independent references, analytical solutions where available, and published cases. Compiling the same implementation for CPU and GPU is useful for debugging but is not independent validation.

Property checks complement these comparisons. Examples include conservation of two-body invariants, forward-backward propagation, and propagating a Lambert solution to its target within a stated tolerance.

Validation records the case, units, frame, solver settings, reference method and observed errors.

Benchmarks compare methods at equivalent accuracy and supported-domain coverage. They report:

* batch size, precision, tolerances and success rate
* CPU hardware, thread count and parallel baseline
* GPU model, driver and CUDA toolkit
* compiler versions, build flags and source commit
* kernel execution time separately from total time including transfers
* warm-up and repeated-measurement procedure

Asynchronous GPU work is timed using a procedure that accounts for completion. Timing only the submission of work is not an execution benchmark.

Known reference divergences are listed in `validation/compare_cpu_references.py`, pinned to an engine version. With pykep 3.0.1, six near-parabolic cases (|e − 1| ≤ 1e-8, dt = ±1) diverge from warpkep and heyoka, consistent with cancellation in pykep's separate elliptic and hyperbolic branches. Their errors are still computed and reported. heyoka is required on every case, and exact parabolic cases are also checked against Barker's equation.

## 14. Changes to this contract

This document describes intended behaviour until the first numerical prototype is available.

The prototype may reveal necessary API changes. Such changes must be reflected here, with corresponding validation and migration notes where they affect users.

Implemented behaviour, demonstrated support and experimental features must remain distinguishable in the documentation.
