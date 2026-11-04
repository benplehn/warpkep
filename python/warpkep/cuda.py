import cupy as cp  # pyright: ignore[reportMissingImports]
from . import _cuda


def propagate_kepler(states, durations, *, mu=1.0):
    """Propage un lot et renvoie (out, statuses, iterations) sur GPU.

    Entrées : states (6, N), durations (N,), float64, contigus en ordre C.
    Unités cohérentes : positions en L, durées en T, mu en L³/T².
    Préparer les entrées sur le stream courant du GPU de states ;
    sinon, attendre leurs streams producteurs avant cet appel.
    Les sorties sont prêtes au retour.
    """

    # Exiger des tableaux GPU : aucune conversion ou copie implicite des entrées.
    if not isinstance(states, cp.ndarray):
        raise TypeError(f"states doit être un cupy.ndarray (reçu : {type(states).__name__}). Utilise cp.asarray(states) pour le placer sur le GPU.")

    if not isinstance(durations, cp.ndarray):
        raise TypeError(f"durations doit être un cupy.ndarray (reçu : {type(durations).__name__}). Utilise cp.asarray(durations) pour le placer sur le GPU.")

    # Vérifier le rang avant shape[0], puis extraire N sans risque.
    if states.ndim != 2 or states.shape[0] != 6:
        raise ValueError(f"states doit avoir la forme (6, N), lignes (rx, ry, rz, vx, vy, vz), reçu : {states.shape}.")

    n = states.shape[1]

    # Allouer sur le GPU des états ; le GPU précédent sera restauré à la sortie.
    with states.device:
        # CuPy possède ces buffers,  empty évite une initialisation inutile, car le kernel les remplira.
        out = cp.empty((6, n), dtype=cp.float64)
        statuses = cp.empty((n,), dtype=cp.int32)
        iterations = cp.empty((n,), dtype=cp.int32)

        # Terminer la préparation sur ce stream avant l'import des tableaux par nanobind.
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        stream_ptr = stream.ptr

        # Le binding vérifie les autres contraintes, lance le kernel et attend sa fin.
        _cuda.propagate_kepler(states, durations, out, statuses, iterations, mu=mu, stream_ptr=stream_ptr)

        # Renvoyer les tableaux CuPy directement, sans transfert vers la RAM.
        return out, statuses, iterations