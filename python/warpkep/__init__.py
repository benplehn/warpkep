from ._core import version

__all__ = ["version", "propagate_kepler", "solve_lambert"]


def propagate_kepler(states, durations, *, mu=1.0):
    """Renvoie (out, statuses, iterations) sur GPU, prêts au retour."""
    from .cuda import propagate_kepler as _propagate_kepler

    return _propagate_kepler(states, durations, mu=mu)


def solve_lambert(departures, arrivals, durations, *, mu=1.0, revolutions=0, long_arc=False):
    """Solve Lambert transfers on the GPU. See warpkep.cuda.solve_lambert for output shapes and order."""
    from .cuda import solve_lambert as _solve_lambert

    return _solve_lambert(departures, arrivals, durations, mu=mu, revolutions=revolutions, long_arc=long_arc)