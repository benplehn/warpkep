# warpkep

**Kepler propagation and Lambert transfers for orbital computation.**

warpkep provides Kepler propagation on CPU and GPU, together with CPU Lambert solvers for zero and multiple revolutions. The library is written in C++17 and CUDA; Python users can propagate batches through CuPy.

Start with [installation](installation.md) and the [quickstart](quickstart.md). The [Lambert guide](lambert.md) explains how to request a revolution count and work with the two solution branches.

## What is available

| Component | Runs on | Interface | Status |
| --- | --- | --- | --- |
| Kepler propagation | CPU; shared primitives compile for CUDA | Internal C++ templates, FP32 / FP64 | Implemented; FP64 checked against independent references |
| Kepler batches | NVIDIA GPU | C++ CUDA, FP64 | Experimental |
| Kepler batches | NVIDIA GPU | Python/CuPy, `float64` | Experimental, synchronous |
| Lambert, `M = 0` | CPU | Internal C++, FP64 | Short and long arcs; elliptic, parabolic and hyperbolic test cases |
| Lambert, `M > 0` | CPU | Internal C++, FP64 | Left/right branch solving; analytical coverage at `M = 1, 2` |

```{note}
Version **0.1.0.dev0** is under development. Interfaces under `warpkep/detail/` may change. Lambert currently runs through C++ on the CPU; Python bindings and GPU batching are planned.
```

```{toctree}
:caption: Get started
:maxdepth: 1

installation
quickstart
```

```{toctree}
:caption: Guides
:maxdepth: 1

lambert
python_cuda
```

```{toctree}
:caption: Reference
:maxdepth: 1

api
numerical_contract
```

```{toctree}
:caption: Development
:maxdepth: 1

architecture
validation
benchmarks
contributing
roadmap
references
```
