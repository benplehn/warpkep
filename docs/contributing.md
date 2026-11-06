# Contributing

## Development workflow

Use the [CPU build](installation.md#cpu-library) unless your change needs CUDA. Put binaries, virtual environments, generated documentation and benchmark logs under the ignored `build/` directory.

```bash
cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DWARPKEP_ENABLE_CUDA=OFF
cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure
```

For CUDA changes, run the GPU tests on a compatible device. Hosted CI checks compilation only. See [validation](validation.md) for GPU commands and the independent reference environment.

## Numerical changes

Document supported inputs, units, conventions, stopping criteria and failure outputs when changing a solver. For a regression, add a test that reproduces the problem and checks an analytical result, an independent reference or a relevant invariant.

For Lambert changes, keep `(revolutions, arc, branch)` with each solution and check the propagated arrival position and velocity. Test the distinction between a numerical failure and an infeasible flight time. Near the minimum, check the number of returned roots and their separation. If only one branch succeeds, its result must remain available.

Update the [API reference](api.md), [numerical contract](numerical_contract.md) and relevant guide together. Describe accuracy and performance using measured results and the cases actually tested.

## Documentation workflow

Documentation uses Sphinx, MyST Markdown and Furo. It needs Python 3.11+ and can be built without compiling the library or installing CUDA.

Write direct sentences about what the code currently does. Keep each paragraph or list item on one source line and let the editor wrap it for display. Start a new paragraph when the subject changes. Preserve the formatting of code, equations and Sphinx directives.

```bash
python -m venv build/docs-env
build/docs-env/bin/python -m pip install -r docs/requirements.txt
build/docs-env/bin/python -m sphinx -n -W --keep-going -b html docs build/docs
python -m http.server 8000 --directory build/docs
```

Open [localhost:8000](http://localhost:8000). Check navigation, tables, equations and code blocks. When changing the layout, check narrow and wide windows too. The build fails on warnings and broken internal references. Add new pages to the table of contents in `docs/index.md`.

The quickstart and Lambert guide read their C++ snippets from `examples/lambert_multi_rev.cpp`. Compile and run that example after editing them:

```bash
mkdir -p build/examples
c++ -std=c++17 -O2 -Wall -Wextra -pedantic -I include \
  examples/lambert_multi_rev.cpp -o build/examples/lambert_multi_rev
./build/examples/lambert_multi_rev
```

To check external links separately, with network access:

```bash
build/docs-env/bin/python -m sphinx -b linkcheck docs build/docs-links
```

The HTML CI build checks internal references. Run the separate link check when editing external URLs, and inspect redirects or temporary network errors before changing a link.

## Submitting a change

Explain the problem, what changed and how you checked it. For numerical work, include tolerances, observed errors, inputs and environment details so someone else can reproduce the result. Follow the [benchmark procedure](benchmarks.md) for performance changes.
