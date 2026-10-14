"""Compare warpkep propagation results with independent references.

References:
- heyoka: mandatory on every case.
- pykep: always compared; documented divergences are reported, not hidden.
- Barker: closed-form parabolic solution for the exact e = 1 cases.
"""

import argparse
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

from reference_smoke import propagate_heyoka, propagate_pykep

# Defaults for older sample files without per-case fields.
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_VELOCITY_SCALE = 1.0
DEFAULT_TOLERANCE = 1e-12

# Known reference divergences, pinned to an engine version.
# pykep separates elliptic and hyperbolic branches; its expressions lose
# precision near e = 1 (cancellation). warpkep agrees with heyoka to ~2e-16
# on these cases. Errors are still computed and printed.
KNOWN_DIVERGENCES = {
    ("pykep", "3.0.1"): {
        "periapsis_e1-1e-8_dt-1",
        "periapsis_e1-1e-8_dt+1",
        "periapsis_e1_dt-1",
        "periapsis_e1_dt+1",
        "periapsis_e1+1e-8_dt-1",
        "periapsis_e1+1e-8_dt+1",
    },
}

# Exact parabolic cases checked against Barker's equation.
BARKER_PREFIX = "periapsis_e1_"


def read_case(sample):
    """Extract the inputs, scales and tolerance of one sample."""
    name = sample["name"]
    if sample["status"] != "success":
        raise RuntimeError(f"warpkep propagation failed: {name}")

    # JSON can decode components such as 0 and 1 as integers.
    case = {
        "name": name,
        "r0": np.asarray([float(x) for x in sample["r0"]]),
        "v0": np.asarray([float(x) for x in sample["v0"]]),
        "r1": np.asarray(sample["r1"], dtype=np.float64),
        "v1": np.asarray(sample["v1"], dtype=np.float64),
        "dt": float(sample["dt"]),
        "mu": float(sample["mu"]),
        "L": float(sample.get("length_scale", DEFAULT_LENGTH_SCALE)),
        "V": float(sample.get("velocity_scale", DEFAULT_VELOCITY_SCALE)),
        "tol": float(sample.get("tolerance", DEFAULT_TOLERANCE)),
    }

    for key in ("L", "V", "tol"):
        value = case[key]
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"Invalid {key}={value!r}: {name}")
    for key in ("r0", "v0", "r1", "v1"):
        if case[key].shape != (3,):
            raise ValueError(f"Invalid state shape ({key}): {name}")
    return case


def scaled_errors(case, ref_r, ref_v):
    """e_r = |r - r_ref| / L, e_v = |v - v_ref| / V."""
    ref_r = np.asarray(ref_r, dtype=np.float64)
    ref_v = np.asarray(ref_v, dtype=np.float64)
    if ref_r.shape != (3,) or ref_v.shape != (3,):
        raise ValueError(f"Invalid reference shape: {case['name']}")
    error_r = float(np.linalg.norm(case["r1"] - ref_r)) / case["L"]
    error_v = float(np.linalg.norm(case["v1"] - ref_v)) / case["V"]
    return error_r, error_v


def within(case, error_r, error_v):
    return (
        np.isfinite(error_r)
        and np.isfinite(error_v)
        and error_r <= case["tol"]
        and error_v <= case["tol"]
    )


def compare(path, engine):
    """Compare every sample with one engine. Returns unexpected failures."""
    samples = json.loads(path.read_text())
    if not samples:
        raise ValueError("No propagation samples provided")

    propagate = propagate_pykep if engine == "pykep" else propagate_heyoka
    engine_version = import_module(engine).__version__
    known = KNOWN_DIVERGENCES.get((engine, engine_version), set())
    print(f"Reference: {engine} {engine_version}", flush=True)

    failures = []
    known_hits = []
    max_error_r = 0.0
    max_error_v = 0.0

    for sample in samples:
        case = read_case(sample)
        name = case["name"]

        # A failing reference is reported, never replaced by warpkep output.
        try:
            ref_r, ref_v = propagate(
                list(case["r0"]), list(case["v0"]),
                case["dt"], case["mu"]
            )
            error_r, error_v = scaled_errors(case, ref_r, ref_v)
        except Exception as exc:
            failures.append(name)
            print(
                f"  {name}: REFERENCE FAILED ({engine}): {exc!r}",
                flush=True,
            )
            continue

        detail = (
            f"error_r={error_r:.3e}, error_v={error_v:.3e}, "
            f"tol={case['tol']:.1e}"
        )
        finite = np.isfinite(error_r) and np.isfinite(error_v)

        if within(case, error_r, error_v):
            max_error_r = max(max_error_r, error_r)
            max_error_v = max(max_error_v, error_v)
            note = " (listed as known divergence)" if name in known else ""
            print(f"  {name}: {detail} ok{note}", flush=True)
        elif finite and name in known:
            known_hits.append(name)
            print(
                f"  {name}: {detail} "
                f"KNOWN DIVERGENCE ({engine} {engine_version})",
                flush=True,
            )
        else:
            failures.append(name)
            print(f"  {name}: {detail} FAILED", flush=True)

    print(f"  max (checked cases): error_r={max_error_r:.3e}, error_v={max_error_v:.3e}", flush=True)
    if known_hits:
        print(f"  {len(known_hits)} known divergence(s), documented, not counted as failures", flush=True)
    return failures


def barker_reference(dt, mu, q):
    """Parabolic orbit from periapsis at t = 0, periapsis on +x, motion towards +y.

    Barker: D + D^3 / 3 = M, with D = tan(nu / 2), M = dt * sqrt(mu / (2 q^3)).
    Closed form: D = A - 1 / A, A^3 = 3M/2 + sqrt(1 + (3M/2)^2).
    """
    m = dt * np.sqrt(mu / (2.0 * q**3))
    a = np.cbrt(1.5 * m + np.sqrt(1.0 + (1.5 * m) ** 2))
    d = a - 1.0 / a
    nu = 2.0 * np.arctan(d)
    r = q * (1.0 + d * d)
    position = r * np.array([np.cos(nu), np.sin(nu), 0.0])
    velocity = np.sqrt(mu / (2.0 * q)) * np.array([-np.sin(nu), 1.0 + np.cos(nu), 0.0])
    return position, velocity


def compare_barker(path):
    """Check exact parabolic cases. Returns failures."""
    samples = json.loads(path.read_text())
    print("Reference: Barker (analytic parabola)", flush=True)

    failures = []
    checked = 0
    for sample in samples:
        if not sample["name"].startswith(BARKER_PREFIX):
            continue
        case = read_case(sample)
        name = case["name"]

        # Barker reference assumes the periapsis geometry built by periapsis_case.
        q = float(np.linalg.norm(case["r0"]))
        expected_v0 = np.array([0.0, np.sqrt(2.0 * case["mu"] / q), 0.0])
        if not (
            np.allclose(case["r0"], [q, 0.0, 0.0], rtol=0, atol=1e-15 * q)
            and np.allclose(case["v0"], expected_v0, rtol=1e-15, atol=0)
        ):
            raise ValueError(f"Not a periapsis parabola on +x/+y: {name}")

        ref_r, ref_v = barker_reference(case["dt"], case["mu"], q)
        error_r, error_v = scaled_errors(case, ref_r, ref_v)
        checked += 1

        verdict = "ok" if within(case, error_r, error_v) else "FAILED"
        print(
            f"  {name}: error_r={error_r:.3e}, error_v={error_v:.3e}, "
            f"tol={case['tol']:.1e} {verdict}",
            flush=True,
        )
        if verdict != "ok":
            failures.append(name)

    if checked == 0:
        print("  no exact parabolic case in this file", flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--engine", choices=("pykep", "heyoka"))
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be positive")

    # Child process: one engine.
    if args.engine is not None:
        for _ in range(args.repeat):
            failures = compare(args.samples, args.engine)
            if failures:
                print(f"{args.engine}: {len(failures)} case(s) failed: " + ", ".join(failures), flush=True)
                sys.exit(1)
        return

    # Parent process: each engine in its own process; all checks always run.
    script = str(Path(__file__).resolve())
    samples = str(args.samples.resolve())
    failed = []

    for run in range(args.repeat):
        print(f"Validation run {run + 1}/{args.repeat}", flush=True)
        for engine in ("pykep", "heyoka"):
            result = subprocess.run(
                [sys.executable, "-X", "faulthandler", script, samples, "--engine", engine],
                check=False,
            )
            if result.returncode != 0:
                print(f"{engine}: reference check failed (exit code {result.returncode})", flush=True)
                failed.append(engine)

    if compare_barker(args.samples):
        failed.append("barker")

    if failed:
        print("Reference checks FAILED: " + ", ".join(failed), flush=True)
        sys.exit(1)

    print("Propagation reference checks passed.", flush=True)


if __name__ == "__main__":
    main()