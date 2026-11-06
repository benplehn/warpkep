# References

## Lambert formulation

Dario Izzo, *Revisiting Lambert's Problem*, Celestial Mechanics and Dynamical Astronomy **121**, 1–15 (2015). [Paper](https://arxiv.org/abs/1403.2705) · [DOI](https://doi.org/10.1007/s10569-014-9587-y).

The CPU solver uses this paper's time-of-flight formulation, initial guesses and velocity reconstruction. Its bracketing, residual stopping criteria and threshold around the minimum are specific to warpkep; see the [Lambert guide](lambert.md).

## Related astrodynamics tools

- [pykep](https://esa.github.io/pykep/): Kepler propagation reference used in validation. Its [Lambert class](https://esa.github.io/pykep/lambert.html) provides another interface to multi-revolution transfers.
- [heyoka.py](https://bluescarni.github.io/heyoka.py/): independent numerical integration used for Kepler validation.

The [validation guide](validation.md) lists the reference versions and automated comparisons.

## Build and interoperability

- [CuPy installation](https://docs.cupy.dev/en/stable/install.html).
- [Sphinx](https://www.sphinx-doc.org/en/master/), [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/) and [Furo](https://pradyunsg.me/furo/): documentation tooling.

## Project

[Source repository](https://github.com/benplehn/warpkep) · [Issue tracker](https://github.com/benplehn/warpkep/issues) · [Apache-2.0 license](https://github.com/benplehn/warpkep/blob/main/LICENSE).
