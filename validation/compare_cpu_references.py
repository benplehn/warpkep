"""Compare C++ propagation results with independent references."""

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

from reference_smoke import propagate_heyoka, propagate_pykep


def compare(path, engine):
    samples = json.loads(path.read_text())
    if not samples:
        raise ValueError("No propagation samples provided")

    propagate = propagate_pykep if engine == "pykep" else propagate_heyoka

    # Absolute tolerance for these normalized test cases.
    tolerance = 1e-12
    print(f"Reference: {engine} {version(engine)}", flush=True)

    for sample in samples:
        if sample["status"] != "success":
            raise RuntimeError(f"C++ propagation failed: {sample['name']}")

        # JSON can decode components such as 0 and 1 as integers.
        r0 = [float(x) for x in sample["r0"]]
        v0 = [float(x) for x in sample["v0"]]
        dt = float(sample["dt"])
        mu = float(sample["mu"])

        ref_r, ref_v = propagate(r0, v0, dt, mu)
        errors = []
        for key, reference in (("r1", ref_r), ("v1", ref_v)):
            actual = np.asarray(sample[key], dtype=np.float64)
            reference = np.asarray(reference, dtype=np.float64)
            if actual.shape != (3,) or reference.shape != (3,):
                raise ValueError(f"Invalid state shape: {sample['name']}")
            errors.append(float(np.linalg.norm(actual - reference)))

        error_r, error_v = errors
        print(f"  {sample['name']}: error_r={error_r:.3e}, error_v={error_v:.3e}", flush=True)
        if not np.all(np.isfinite(errors)) or error_r > tolerance or error_v > tolerance:
            raise AssertionError(f"Reference comparison failed: {sample['name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--engine", choices=("pykep", "heyoka"))
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be positive")

    if args.engine is not None:
        for _ in range(args.repeat):
            compare(args.samples, args.engine)
        return

    script = str(Path(__file__).resolve())
    samples = str(args.samples.resolve())

    for run in range(args.repeat):
        print(f"Validation run {run + 1}/{args.repeat}", flush=True)
        for engine in ("pykep", "heyoka"):
            subprocess.run(
                [sys.executable, "-X", "faulthandler", script, samples, "--engine", engine],
                check=True,
            )

    print("C++ propagation reference checks passed.", flush=True)


if __name__ == "__main__":
    main()
