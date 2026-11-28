# Rendezvous and flyby examples

The examples combine Kepler propagation and Lambert transfers in a two-body model. Each case supplies reference states and dates relative to a common origin.

## Rendezvous on a date grid

Evaluate a rendezvous from orbit A to orbit B by pairing departure and arrival dates.

1. Propagate A to each departure date and B to each arrival date. For each state, use the offset from its reference date.
2. Keep pairs with a positive flight time and solve the corresponding short arcs in a single Lambert batch with zero revolutions.
3. For each successful arc, add the norm of the departure velocity change to the norm of the arrival velocity change. Select the lowest total Δv and repeat on a finer grid around that pair.

`examples/rendezvous.py` checks the reference states and flight time. The full calculation is in `tests/test_python_rendezvous.py`. Its circular case checks the arrival position and velocity by propagating the transfer and compares targets moving in the same and opposite directions. The grid case checks that refinement retains or improves the best cost.

Run from the repository root with pytest and a [CUDA-enabled Python installation](installation.md#pythoncupy-with-cuda).

```bash
python -m pytest -q tests/test_python_rendezvous.py
```

## Unpowered flyby

The two Lambert arcs meet at body B. Subtract B's velocity from the incoming and outgoing spacecraft velocities to obtain the two hyperbolic excess velocities. Their magnitudes must agree within the selected tolerance. The turn angle then gives the periapsis radius, which must meet the minimum radius set for B.

`examples/flyby.py` reconstructs two arcs from a known encounter. It recovers a periapsis radius of about 4,142.14 km for a minimum of 4,000 km.

`examples/flyby_search.py` varies the encounter date. For each trial date, it propagates B and solves the two arcs. Brent's method refines a bracketed zero of the outgoing excess speed minus the incoming excess speed. The script then checks periapsis and compares the departure and arrival Δv across three arrival dates.

The search example also compares exhaustive matching with beam search. Matching requires the incoming and outgoing arcs to reach B at the same date within tolerance. A width-one beam can discard the only feasible incoming arc when it ranks arcs by departure cost before checking the flyby.

Run both scripts with the CUDA-enabled package, CuPy and NumPy. The search script also requires SciPy.

```bash
python examples/flyby.py
python examples/flyby_search.py
```
