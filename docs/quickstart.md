# Quickstart

## Propagate a state on CPU

For a unit circular orbit with `mu = 1`, propagating by a quarter period takes the state from `(1, 0, 0; 0, 1, 0)` to `(0, 1, 0; -1, 0, 0)`. This example uses the internal C++ propagator and needs no CUDA installation:

```cpp
#include <warpkep/detail/kepler_cpu.hpp>
#include <cmath>

int main() {
    namespace wd = warpkep::detail;
    const wd::CartesianState<double> initial{{1, 0, 0}, {0, 1, 0}};
    const auto result = wd::propagate_kepler_cpu(
        initial, std::acos(-1.0) / 2.0, 1.0
    );
    if (result.status != warpkep::KeplerPropagationStatus::success) {
        return 1;
    }
    // result.state.position and result.state.velocity contain the final state.
}
```

Check `status` before reading the state. If propagation fails, its six components are NaN. The [numerical contract](numerical_contract.md) covers backward propagation, radial motion and tolerances.

## Solve a Lambert transfer

The Lambert solver finds the velocities needed to connect two positions in a given flight time. Here the transfer makes one complete revolution before the final short arc:

```{literalinclude} ../examples/lambert_multi_rev.cpp
:language: cpp
:start-after: // [solve-begin]
:end-before: // [solve-end]
:dedent: 4
```

Here `wd` is an alias for `warpkep::detail`. There are two solutions for this flight time. One follows the unit circular orbit; both reach the same endpoint at the requested time.

Build and run the complete example from the repository root:

```bash
mkdir -p build/examples
c++ -std=c++17 -O2 -Wall -Wextra -pedantic -I include \
  examples/lambert_multi_rev.cpp -o build/examples/lambert_multi_rev
./build/examples/lambert_multi_rev
```

The program propagates both solutions to check their arrival position and velocity, then tries `M = 0, 1, 2, 3` using the same geometry. It exits with code zero if the checks pass. The [Lambert guide](lambert.md) explains the branches and what happens near the minimum flight time.

## Propagate a batch from Python

After a CUDA-enabled [installation](installation.md), create a CuPy array with one row per state component:

```python
import math
import cupy as cp
import warpkep
from warpkep import _cuda

states = cp.zeros((6, 257), dtype=cp.float64)
states[0] = 1.0
states[4] = 1.0
durations = cp.full(257, math.pi / 2.0, dtype=cp.float64)

out, statuses, iterations = warpkep.propagate_kepler(states, durations, mu=1.0)
if not bool(cp.all(statuses == _cuda.SUCCESS).item()):
    raise RuntimeError("Inspect the statuses of failed trajectories")
print(cp.asnumpy(out[:, 0]))
```

Expected first column, within numerical tolerance:

```text
[0, 1, 0, -1, 0, 0]
```

The results stay on the GPU and are ready when the call returns. `cp.asnumpy` copies the first state to the CPU for display. See the [Python/CuPy guide](python_cuda.md) for input requirements, streams and errors.
