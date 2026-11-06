# Status and roadmap

The current version is `0.1.0.dev0`. This page lists what is implemented and the work planned next.

## Implemented

- Universal-variable Kepler propagation using the same numerical code on CPU and GPU.
- FP32/FP64 precision tests and independent FP64 CPU comparisons with pykep and heyoka.
- Experimental FP64 CUDA propagation with preallocated SoA buffers and caller-supplied streams.
- Synchronous Python/CuPy propagation returning GPU arrays and a status for each trajectory.
- CPU FP64 Lambert for short and long arcs at zero revolutions.
- CPU FP64 multi-revolution Lambert with a requested `M`, minimum search, two branches and reusable geometry.
- CPU, GPU and Python test programs, CPU/GPU benchmarks and reference documentation.

## Next numerical work

- Compare Lambert results with independent solvers across more geometries, scales and revolution counts, especially near the minimum and degenerate cases.
- Measure numerical uncertainty and examine cases where only one branch converges.
- Define a public Lambert API, then add Python bindings and GPU batches.
- Record reproducible GPU validation runs and benchmark full application costs, including Python allocation and synchronization.

## Longer-term applications

- Perturbed dynamics and thrust models.
- Rendezvous cost evaluation and trajectory optimisation.
- Beam search and sequence search for mission design.
- GTOC applications with accuracy and performance measurements.

The current two-body routines do not provide ephemerides, planetary models or absolute epoch handling. Applications must supply those inputs and check mission constraints themselves.
