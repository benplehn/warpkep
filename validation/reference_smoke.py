"""Check reference propagators in separate Python processes."""

import argparse
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def propagate_pykep(r0, v0, dt, mu):
    import pykep as pk

    r, v = pk.propagate_lagrangian(rv=[r0, v0], tof=dt, mu=mu)
    return np.asarray(r), np.asarray(v)


def propagate_heyoka(r0, v0, dt, mu):
    import heyoka as hy

    x, y, z, vx, vy, vz = hy.make_vars(
        "x", "y", "z", "vx", "vy", "vz"
    )
    radius_squared = x * x + y * y + z * z
    factor = -mu / radius_squared**1.5

    integrator = hy.taylor_adaptive(
        sys=[
            (x, vx),
            (y, vy),
            (z, vz),
            (vx, factor * x),
            (vy, factor * y),
            (vz, factor * z),
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
    r0 = [1.0, 0.0, 0.0]
    v0 = [0.0, 1.0, 0.0]
    dt = math.pi / 2.0
    mu = 1.0

    expected_r = np.array([0.0, 1.0, 0.0])
    expected_v = np.array([-1.0, 0.0, 0.0])

    # Threshold for this normalized smoke test only.
    error_limit = 1e-12

    propagator = (
        propagate_pykep if engine == "pykep" else propagate_heyoka
    )
    r, v = propagator(r0, v0, dt, mu)

    position_error = np.linalg.norm(r - expected_r)
    velocity_error = np.linalg.norm(v - expected_v)

    print(f"{engine}: {version(engine)}")
    print(f"  Position error: {position_error:.3e}")
    print(f"  Velocity error: {velocity_error:.3e}")

    errors = np.array([position_error, velocity_error])
    if not np.all(np.isfinite(errors)) or np.any(errors > error_limit):
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
            [sys.executable, script, "--engine", engine],
            check=True,
        )

    print("Circular orbit reference check passed.")


if __name__ == "__main__":
    main()