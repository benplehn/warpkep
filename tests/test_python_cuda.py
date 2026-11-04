import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import warpkep
from warpkep import _cuda


def test_kepler_mixed_batch():
    """Vérifie les états analytiques, les statuts et les sorties invalides."""
    n = 257
    tolerance = 1e-12
    sqrt3 = np.sqrt(3.0)

    # Chaque ligne décrit un cas ; la transposition donnera le format SoA.
    initial_cases = np.array([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, sqrt3, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    duration_cases = np.array([
        np.pi / 2.0,
        -np.pi / 2.0,
        np.pi,
        np.pi / 2.0,
    ], dtype=np.float64)

    expected_cases = np.array([
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        [-1.5, 0.0, 0.0, 0.0, -1.0 / sqrt3, 0.0],
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    ], dtype=np.float64)

    # Répéter les quatre cas et préparer des tableaux (6, N) contigus en ordre C.
    indices = np.arange(n) % 4
    states_cpu = initial_cases[indices].T.copy(order="C")
    durations_cpu = duration_cases[indices].copy()
    expected = expected_cases[indices].T.copy(order="C")
    expected_statuses = np.full(n, _cuda.SUCCESS, dtype=np.int32)

    # Mélanger succès et échecs pour vérifier leur indépendance dans un même lot.
    states_cpu[:3, 1] = 0.0
    expected_statuses[1] = _cuda.INVALID_INPUT

    states_cpu[4, 2] = np.nan
    expected_statuses[2] = _cuda.INVALID_INPUT

    durations_cpu[3] = 0.0
    expected[:, 3] = states_cpu[:, 3]

    states_cpu[:, 256] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    durations_cpu[256] = 1.0
    expected_statuses[256] = _cuda.UNSUPPORTED_CASE

    # Produire les entrées et appeler l'API sur le même stream non bloquant.
    with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True):
        states = cp.asarray(states_cpu)
        durations = cp.asarray(durations_cpu)

        out, statuses, iterations = warpkep.propagate_kepler(states, durations, mu=1.0)

        # Vérifier le contrat des sorties avant leur rapatriement pour comparaison.
        assert isinstance(out, cp.ndarray)
        assert isinstance(statuses, cp.ndarray)
        assert isinstance(iterations, cp.ndarray)
        assert out.shape == (6, n) and out.dtype == np.dtype("float64")
        assert statuses.shape == (n,) and statuses.dtype == np.dtype("int32")
        assert iterations.shape == (n,) and iterations.dtype == np.dtype("int32")
        assert out.flags.c_contiguous

        actual = cp.asnumpy(out)
        actual_statuses = cp.asnumpy(statuses)

    np.testing.assert_array_equal(actual_statuses, expected_statuses)

    # Comparer séparément les normes des erreurs de position et de vitesse.
    success = expected_statuses == _cuda.SUCCESS
    assert np.isfinite(actual[:, success]).all()

    position_errors = np.linalg.norm(actual[:3, success] - expected[:3, success], axis=0)
    velocity_errors = np.linalg.norm(actual[3:, success] - expected[3:, success], axis=0)

    assert np.all(position_errors <= tolerance), f"Erreur position maximale : {position_errors.max():.3e}"
    assert np.all(velocity_errors <= tolerance), f"Erreur vitesse maximale : {velocity_errors.max():.3e}"

    # Chaque trajectoire en échec doit produire six NaN.
    assert np.isnan(actual[:, ~success]).all()