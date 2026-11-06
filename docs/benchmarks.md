# Benchmarks

Two programs measure Kepler propagation: a serial/parallel CPU baseline and a CPU/GPU comparison. Both check the numerical results as well as timing the computation.

## CPU baseline

```bash
cmake -S . -B build/bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DWARPKEP_BUILD_BENCHMARKS=ON \
  -DWARPKEP_ENABLE_CUDA=OFF
cmake --build build/bench --target warpkep_cpu_baseline --parallel 2
./build/bench/warpkep_cpu_baseline 200000 4 5
```

Arguments are `[count] [threads] [repeats]`, all positive integers. This example runs 200,000 propagations with four threads and five repeats. The program reports median timings and checks that serial and parallel results agree. There is no pass/fail threshold on speed.

## CPU/GPU comparison

On a CUDA machine:

```bash
cmake -S . -B build/bench-gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWARPKEP_BUILD_BENCHMARKS=ON \
  -DWARPKEP_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/bench-gpu --target warpkep_kepler_benchmark --parallel 2
./build/bench-gpu/warpkep_kepler_benchmark \
  > build/bench-gpu/results.csv 2> build/bench-gpu/metadata.log
```

The program tests fixed batch sizes from 1 to 1,048,576. Results go to stdout as CSV; environment information and individual timing samples go to stderr.

| CSV column | Meaning |
| --- | --- |
| `n`, `threads` | Batch size and CPU parallelism |
| `cpu_serial_ms` | Median serial CPU propagation time |
| `cpu_parallel_ms` | Median parallel CPU propagation time |
| `gpu_kernel_ms` | Per-launch kernel time measured with CUDA events on resident data |
| `gpu_total_ms` | Input transfer, one launch, output/status/iteration transfers and synchronization |
| `max_error_r`, `max_error_v` | Maximum position/velocity disagreement checked by the harness |

GPU memory and pinned host buffers are allocated before timing. After a one-second warm-up, the program takes nine samples, with 32 launches per kernel timing sample. The measurement including transfers excludes allocation and CPU conversion from AoS to SoA. It does not include the Python/CuPy wrapper's allocation and binding costs.

The CPU and GPU use the same propagation code. Their agreement helps catch execution errors; the separate [reference comparisons](validation.md#independent-kepler-references) check numerical accuracy independently.

## Reporting a result

Publish the CSV and metadata together, including:

- Source commit, compiler, build flags and arithmetic mode.
- CPU model, thread count, GPU model, driver and CUDA toolkit.
- Batch size, precision, input distribution and solver settings.
- Warm-up, number of repeats, use of medians, and success/failure counts.
- Accuracy checks and observed errors, using comparable inputs and tolerances for each method.
- Kernel time with data already on the GPU, time including transfers, and full application time where available.

Wait for asynchronous work to finish before stopping the timer. Timing the host launch call alone measures submission overhead. State which costs are included so readers can judge whether a measured speedup applies to their workload.
