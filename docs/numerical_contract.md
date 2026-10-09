# Numerical contract

**Status:** initial design contract.

warpkep currently provides a C++ library skeleton, Python bindings, a CUDA compilation example and an internal CPU prototype of Kepler propagation in FP64. The public numerical APIs described below, including the GPU batch interface, are not implemented yet.

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

## 9. Batch layout

The initial GPU propagation interface is designed around a structure-of-arrays layout.

For N trajectories, Cartesian states use six contiguous one-dimensional arrays:

* r\_x, r\_y, r\_z
* v\_x, v\_y, v\_z

The initial design uses one shared mu and an array of N relative durations. Output states use the same layout, with N result statuses.

Component ordering and any future representation of augmented states, such as mass or a state-transition matrix, must be documented explicitly.

The GPU interface operates on data already resident on the selected device. It performs no implicit host-device transfers.

A structurally valid empty batch is a no-op and launches no computation.

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

## 14. Changes to this contract

This document describes intended behaviour until the first numerical prototype is available.

The prototype may reveal necessary API changes. Such changes must be reflected here, with corresponding validation and migration notes where they affect users.

Implemented behaviour, demonstrated support and experimental features must remain distinguishable in the documentation.
