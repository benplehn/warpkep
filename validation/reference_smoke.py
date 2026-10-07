"""Check reference propagators in separate Python processes."""

import argparse
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def normalize_inputs(r0, v0, dt, mu):
    """Give native bindings lists of Python floats, including JSON inputs."""
    vectors = np.asarray([r0, v0], dtype=np.float64)
    if vectors.shape != (2, 3):
        raise ValueError("Expected two vectors of three components")
    dt = float(dt)
    mu = float(mu)
    if not np.all(np.isfinite(vectors)) or not math.isfinite(dt):
        raise ValueError("State and duration must be finite")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    return vectors[0].tolist(), vectors[1].tolist(), dt, mu


def propagate_pykep(r0, v0, dt, mu):
    r0, v0, dt, mu = normalize_inputs(r0, v0, dt, mu)
    import pykep as pk

    r, v = pk.propagate_lagrangian(rv=[r0, v0], tof=dt, mu=mu)
    return (
        np.array(r, dtype=np.float64, copy=True),
        np.array(v, dtype=np.float64, copy=True),
    )


def propagate_heyoka(r0, v0, dt, mu):
    r0, v0, dt, mu = normalize_inputs(r0, v0, dt, mu)
    import heyoka as hy

    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    radius_squared = x * x + y * y + z * z
    factor = -mu / radius_squared**1.5
    integrator = hy.taylor_adaptive(
        sys=[
            (x, vx), (y, vy), (z, vz),
            (vx, factor * x), (vy, factor * y), (vz, factor * z),
        ],
        state=r0 + v0,
        tol=1e-15,
        high_accuracy=True,
    )
    outcome = integrator.propagate_until(dt)[0]
    if outcome != hy.taylor_outcome.time_limit:
        raise RuntimeError(f"Heyoka propagation failed: {outcome}")
    return integrator.state[:3].copy(), integrator.state[3:].copy()


def check_reference(engine):
    propagate = propagate_pykep if engine == "pykep" else propagate_heyoka
    # Integer inputs also verify the explicit float conversion.
    r, v = propagate([1, 0, 0], [0, 1, 0], math.pi / 2, 1)
    errors = np.array([
        np.linalg.norm(r - np.array([0.0, 1.0, 0.0])),
        np.linalg.norm(v - np.array([-1.0, 0.0, 0.0])),
    ])
    print(f"{engine}: {version(engine)}", flush=True)
    print(f"  Position error: {errors[0]:.3e}", flush=True)
    print(f"  Velocity error: {errors[1]:.3e}", flush=True)
    if not np.all(np.isfinite(errors)) or np.any(errors > 1e-12):
        raise AssertionError(f"{engine} failed the circular orbit check")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("pykep", "heyoka"))
    args = parser.parse_args()
    if args.engine is not None:
        check_reference(args.engine)
        return
    script = str(Path(__file__).resolve())
    for engine in ("pykep", "heyoka"):
        subprocess.run(
            [sys.executable, "-X", "faulthandler", script, "--engine", engine],
            check=True,
        )
    print("Circular orbit reference check passed.")


if __name__ == "__main__":
    main()
