# Status and roadmap

The current version is `0.1.0.dev0`. This page lists what is implemented and the work planned next.

## Implemented

- Universal-variable Kepler propagation using the same numerical code on CPU and GPU.
- FP32/FP64 precision tests and independent FP64 CPU comparisons with pykep and heyoka.
- Experimental FP64 CUDA propagation with preallocated SoA buffers and caller-supplied streams.
- Synchronous Python/CuPy propagation returning GPU arrays and a status for each trajectory.
- CPU FP64 Lambert for short and long arcs at zero revolutions.
- CPU FP64 multi-revolution Lambert with a requested `M`, minimum search, two branches and reusable geometry.
- FP64 Lambert batches through C++ CUDA and Python/CuPy.
- [Trajectory examples](trajectory_examples.md) for rendezvous date grids, flyby refinement and beam pruning.
- CPU, GPU and Python test programs and CPU/GPU benchmarks.

## Next numerical work

- Compare Lambert results with independent solvers across more geometries, scales and revolution counts, especially near the minimum and degenerate cases.
- Measure numerical uncertainty and examine cases where only one branch converges.
- Stabilise the Lambert interfaces and broaden CPU/GPU validation.
- Record reproducible GPU validation runs and benchmark full application costs, including Python allocation and synchronization.

## Longer-term applications

- Perturbed dynamics and thrust models.
- Trajectory optimisation over larger rendezvous and flyby windows.
- Sequence search over multiple encounters.
- GTOC applications with accuracy and performance measurements.

The current two-body routines do not provide ephemerides, planetary models or absolute epoch handling. Applications must supply those inputs and check mission constraints themselves.
