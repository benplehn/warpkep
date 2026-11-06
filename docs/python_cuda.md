# Python/CuPy propagation

`warpkep.propagate_kepler(states, durations, *, mu=1.0)` propagates a batch of two-body states on the GPU. It returns `(out, statuses, iterations)` as CuPy arrays and waits until the outputs are ready.

Install warpkep with CUDA enabled and a compatible CuPy package. The default CPU-only installation provides version information; it cannot run propagation. Lambert has no Python binding yet.

## Arrays and units

| Argument / result | Shape and dtype | Meaning |
| --- | --- | --- |
| `states` | `(6, N)`, `float64`, C-contiguous | Rows `rx, ry, rz, vx, vy, vz`, in CuPy GPU memory |
| `durations` | `(N,)`, `float64`, contiguous | Relative duration per trajectory; same GPU as `states` |
| `mu` | Finite, positive scalar | Gravitational parameter shared by the batch |
| `out` | `(6, N)`, `float64`, C-contiguous | Propagated states on the input GPU |
| `statuses` | `(N,)`, `int32` | Numerical result per trajectory |
| `iterations` | `(N,)`, `int32` | Solver refinement counts |

Use consistent units: positions in L, velocities in L/T, durations in T and `mu` in L³/T². Durations can be positive, zero or negative; see the [numerical contract](numerical_contract.md) for the supported motion. The function leaves inputs unchanged and allocates all three outputs. An empty batch returns three empty arrays.

## Example

```python
import math
import cupy as cp
import warpkep
from warpkep import _cuda

with cp.cuda.Device(0):
    states = cp.zeros((6, 257), dtype=cp.float64)
    states[0, :] = 1.0
    states[4, :] = 1.0
    durations = cp.full(257, math.pi / 2.0, dtype=cp.float64)

    out, statuses, iterations = warpkep.propagate_kepler(
        states, durations, mu=1.0
    )
    valid = statuses == _cuda.SUCCESS
    if not bool(cp.all(valid).item()):
        raise RuntimeError("Inspect the statuses of failed trajectories")
    print(cp.asnumpy(out[:, 0]))
```

After a quarter period on a unit circular orbit, the state should be `(0, 1, 0, -1, 0, 0)` to numerical precision. `cp.asnumpy` copies the first result to the CPU for printing. The propagation itself keeps the arrays on the GPU.

Convert host data explicitly with `cp.asarray(host_states, dtype=cp.float64, order="C")`. The function does not convert input units, dtypes or layouts. For a non-contiguous slice such as `states[:, ::2]`, make a contiguous copy before calling it.

## Statuses and errors

Check each trajectory against the constants in `warpkep._cuda`:

| Constant | Meaning |
| --- | --- |
| `SUCCESS` | Numerical termination criteria met |
| `INVALID_INPUT` | Invalid state components, duration or other numerical input |
| `UNSUPPORTED_CASE` | Excluded motion, such as radial propagation with non-zero duration |
| `NOT_CONVERGED` | Solver could not meet its termination criteria |
| `NUMERICAL_FAILURE` | Unusable intermediate arithmetic |

When a trajectory fails, its output column contains six NaN values. Other trajectories continue normally. Check statuses before using the states.

Errors affecting the whole call raise an exception:

| Situation | Result |
| --- | --- |
| Non-CuPy input, wrong dtype or incompatible contiguity | `TypeError` |
| Wrong `states` shape, mismatched batch lengths, different GPUs or invalid `mu` | `ValueError` |
| Wrong `durations` rank rejected by nanobind | `TypeError` |
| Checked byte-size overflow | `OverflowError` |
| CUDA runtime failure | `RuntimeError`; outputs may be incomplete |

An invalid `mu` rejects the whole call, even for an empty batch. Invalid individual states are reported in `statuses`. If CUDA execution fails, discard the batch outputs.

## Streams and memory

The wrapper selects the GPU containing `states` and uses its current CuPy stream. It waits for that stream before passing arrays to nanobind, then waits for the kernel and status copy before returning. The binding releases the GIL during execution, and the wrapper restores the previously selected device afterward.

Prepare inputs on the same stream. If another stream produces them, wait for it first, using an event or explicit synchronization. The wrapper cannot infer that dependency. Avoid concurrent writes to the buffers while the call is running.

CuPy owns the input and output arrays. The binding also allocates a temporary GPU status buffer and copies it to the `int32` status output. Each call therefore includes allocation, a device-to-device status copy and synchronization.

The [C++ CUDA API](api.md#c-cuda-propagation) accepts preallocated buffers and returns before execution finishes. Use it when you need to manage storage and synchronization yourself.

## Installation and checks

Follow the [CUDA installation instructions](installation.md#pythoncupy-with-cuda), then run the tests on a machine with an NVIDIA GPU:

```bash
python -m pytest tests/test_python_cuda.py -q
```

The tests use a non-blocking stream and a mixed batch of 257 trajectories. They compare states with analytical results and check individual failures, zero duration, empty batches and invalid `mu`. The file is skipped if CuPy is missing. See [validation](validation.md) for the full test coverage.
