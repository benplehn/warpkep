from ._core import version

__all__ = ["version", "propagate_kepler"]


def propagate_kepler(states, durations, *, mu=1.0):
    """Renvoie (out, statuses, iterations) sur GPU, prêts au retour."""
    # Charger les dépendances CUDA seulement lors de l'appel.
    from .cuda import propagate_kepler as _propagate_kepler

    return _propagate_kepler(states, durations, mu=mu)