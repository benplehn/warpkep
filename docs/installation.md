# Installation

Run these commands from the repository root. Choose the CPU build for Lambert and scalar propagation, or enable CUDA for GPU batches. The Python package and documentation have separate installation steps below.

## Requirements

| Component | Requirements |
| --- | --- |
| CPU library and tests | C++17 compiler, CMake 3.24+ |
| CUDA library | CPU requirements, CUDA toolkit, a supported CUDA host compiler |
| GPU execution | Compatible NVIDIA GPU and driver |
| Python package | Python 3.10+, pip; scikit-build-core and nanobind installed by build isolation |
| Python GPU interface | CUDA-enabled package build and compatible CuPy installed separately |
| Documentation | Python 3.11+ and `docs/requirements.txt`; no compiled extension or GPU |

## CPU library

```bash
cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DWARPKEP_BUILD_PYTHON=OFF \
  -DWARPKEP_ENABLE_CUDA=OFF
cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure
./build/cpu/warpkep_smoke
```

This builds both Lambert solvers and the CPU propagation tests. It works on macOS and Linux without a CUDA toolkit.

To use warpkep in another CMake project:

```cmake
add_subdirectory(path/to/warpkep)
target_link_libraries(my_app PRIVATE warpkep)
```

Use `add_subdirectory` for now; an installed package for `find_package(warpkep)` is not available yet. The CPU numerical routines are header-only, while the core static library provides version information.

## Python package

In a virtual environment:

```bash
python -m pip install .
python -c 'import warpkep; print(warpkep.version())'
```

The default installation builds `_core` and provides `warpkep.version()`. Calling `warpkep.propagate_kepler` requires the CUDA extension and CuPy. CPU propagation and Lambert are currently available only through C++.

## CUDA library and tests

On a machine with a compatible CUDA toolkit and GPU:

```bash
cmake -S . -B build/gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DWARPKEP_ENABLE_CUDA=ON \
  -DWARPKEP_RUN_GPU_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/gpu --parallel 2
ctest --test-dir build/gpu --output-on-failure
```

Link your program to `warpkep::cuda` to get the headers, compiled kernel and CUDA runtime. The `native` architecture setting detects the GPU during configuration. If the build machine has no GPU, set the target architecture explicitly.

### Compile CUDA without running a GPU

The development container lets you check CUDA compilation and linking from macOS or Linux without a local GPU. Running the resulting programs still requires a compatible NVIDIA GPU.

```bash
docker build --platform linux/amd64 \
  -f docker/Dockerfile.cuda -t warpkep-cuda:dev docker
docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace" -w /workspace warpkep-cuda:dev \
  bash -c '
    set -e
    cmake -S . -B build/cuda -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DWARPKEP_ENABLE_CUDA=ON \
      -DWARPKEP_RUN_GPU_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=80
    cmake --build build/cuda --parallel 2 \
      --target warpkep_cuda warpkep_cuda_compile_smoke \
               warpkep_gpu_smoke warpkep_test_kepler_gpu
  '
```

### Python/CuPy with CUDA

Install a CuPy distribution compatible with your CUDA environment, following the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html). Then:

```bash
python -m pip install . \
  -Ccmake.define.WARPKEP_ENABLE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native \
  -Ccmake.define.BUILD_TESTING=OFF \
  -Cbuild.tool-args=-j2
python -m pip install pytest
python -m pytest tests/test_python_cuda.py -q
```

Compilation needs the CUDA toolkit; the tests also need a compatible GPU. If CuPy is missing, pytest skips the file. See the [Python/CuPy guide](python_cuda.md) for usage and stream handling.

## Build options

| CMake option | Default | Effect |
| --- | --- | --- |
| `BUILD_TESTING` | `ON` through CTest | Build/register CPU tests and build the CPU sample generator |
| `WARPKEP_BUILD_PYTHON` | `OFF` in CMake; `ON` through pip | Build the nanobind `_core` extension |
| `WARPKEP_ENABLE_CUDA` | `OFF` | Build the CUDA library and consumers; also `_cuda` when Python is enabled |
| `WARPKEP_RUN_GPU_TESTS` | `OFF` | Register two GPU CTest tests; requires CUDA and `BUILD_TESTING=ON` |
| `WARPKEP_BUILD_BENCHMARKS` | `OFF` | Build the CPU baseline and, with CUDA, the CPU/GPU benchmark |
| `CMAKE_CUDA_ARCHITECTURES` | Toolchain-dependent | Select CUDA architectures explicitly for reproducible builds |

## Documentation

Use Python 3.11 or newer. Documentation builds without compiling or importing warpkep; the version is read from `pyproject.toml`.

```bash
python -m venv build/docs-env
build/docs-env/bin/python -m pip install -r docs/requirements.txt
build/docs-env/bin/python -m sphinx -n -W --keep-going -b html docs build/docs
python -m http.server 8000 --directory build/docs
```

Open [localhost:8000](http://localhost:8000). The build fails on warnings, including broken internal references.
