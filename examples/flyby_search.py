"""
Une sonde tourne autour d'un corps central et frôle un corps B à t = 0. Sa trajectoire se compose de deux arcs képlériens, c'est-à-dire de
deux morceaux d'orbite autour du corps central

    A ──(arc d'approche)──► B ──(arc de départ)──► C
   t_a                     t_b                    t_c

  - arc d'approche : le véhicule part du point A à t_a et arrive en B 
  - arc de départ  : le véhicule repart de B et arrive au point C à t_c

Au moment du survol, le véhicule est à la position de B, mais l'assistance gravitationnelle change brusquement sa vitesse. 
On a donc 2 vitesses en B à t = 0: une juste avant le survol et une juste après.


1. On calcule les points A et C à partir de l'état du véhicule au survol:
     - pour A, il fait remonter le temps au VÉHICULE sur son arc d'approche, de t = 0 jusqu'à t_a
     - pour C, il fait avancer le VÉHICULE sur son arc de départ, de t = 0 jusqu'à t_c.
   A et C restent ensuite fixes.
2. On essaie plusieurs dates de survol t_b autour de 0. Pour chacune, on calcule la position du CORPS B à cette date (B se déplace sur son orbite).
3. Pour chaque t_b, on résout deux problèmes de Lambert. Un problème de Lambert pose la question : « quelle orbite relie deux positions données en une durée donnée ? »
     - A → B(t_b) en (t_b − t_a)  →  vitesse d'arrivée du véhicule en B 
     - B(t_b) → C en (t_c − t_b)  →  vitesse de départ du véhicule de B
4. Contrôle : pour t_b = 0, on doit retrouver les deux vitesses de l'énoncé
"""

import cupy as cp

import warpkep


MU_CENTRAL = 1e7  # paramètre gravitationnel du corps central (km³/s²)
T_A = -1000.0     # date à laquelle le véhicule est en A (s)
T_C = 1000.0      # date à laquelle le véhicule est en C (s)

# Position du survol à t = 0 (c'est aussi la position de B à t = 0).
R_FLYBY = [100_000.0, 0.0, 0.0]        # km
# Vitesse du véhicule juste avant le survol (fin de l'arc d'approche).
V_BEFORE_FLYBY = [3.0, 10.0, 0.0]      # km/s
# Vitesse du véhicule juste après le survol (début de l'arc de départ).
V_AFTER_FLYBY = [0.0, 13.0, 0.0]       # km/s

# État du corps B à t = 0 : même position que le véhicule, vitesse propre de B.
STATE_B_T0 = R_FLYBY + [0.0, 10.0, 0.0]

# Dates de survol testées (s). t_b = 0 correspond au cas de l'énoncé.
T_B_CANDIDATES = [-200.0, -100.0, 0.0, 100.0, 200.0]


def as_columns(*vectors):
    """Range des vecteurs en colonnes dans un tableau GPU contigu."""
    return cp.ascontiguousarray(cp.array(vectors, dtype=cp.float64).T)


def tile_column(vector, n):
    """Répète un vecteur colonne n fois : (k,) → (k, n)."""
    return cp.ascontiguousarray(cp.repeat(vector[:, None], n, axis=1))


def propagate(states, dates):
    """Propage chaque état (colonne) de t = 0 jusqu'à sa date et vérifie le succès."""
    dates = cp.asarray(dates, dtype=cp.float64)
    propagated, statuses, _ = warpkep.propagate_kepler(states, dates, mu=MU_CENTRAL)
    assert cp.all(statuses == 0).item(), f"échec de propagation : {statuses}"
    return propagated


def fmt(v):
    """Affiche un vecteur 3D de façon lisible."""
    return "[" + " ".join(f"{x:9.4f}" for x in v) + "]"


# ---------------------------------------------------------------------------
# Étapes du calcul
# ---------------------------------------------------------------------------

def compute_points_a_and_c():
    """Étape 1 : retrouver A et C à partir de l'état du véhicule au survol."""
    # Deux états du véhicule au même endroit (le survol) à t = 0 :
    #   colonne 0 : sur l'arc d'approche (vitesse avant le survol) 
    #   colonne 1 : sur l'arc de départ  (vitesse après le survol).
    vehicle_at_flyby = as_columns(
        R_FLYBY + V_BEFORE_FLYBY,
        R_FLYBY + V_AFTER_FLYBY,
    )

    # Colonne 0 : on remonte le temps jusqu'à T_A, ce qui donne le point A
    # Colonne 1 : on avance jusqu'à T_C, ce qui donne le point C
    vehicle = propagate(vehicle_at_flyby, [T_A, T_C])

    r_a = vehicle[:3, 0]
    r_c = vehicle[:3, 1]
    return r_a, r_c


def compute_positions_of_b(t_b):
    """Étape 2 : position du corps B à chaque date de survol testée."""
    # Le même état de départ de B (à t = 0), propagé jusqu'à chaque date t_b
    b_at_t0 = tile_column(cp.asarray(STATE_B_T0, dtype=cp.float64), t_b.size)
    b = propagate(b_at_t0, t_b)
    return b[:3, :]


def solve_both_arcs(r_a, r_c, r_b, t_b):
    """Étape 3 : résoudre A → B et B → C par Lambert pour chaque date t_b.

    Retourne :
        v_in  (3, n) : vitesse du véhicule en arrivant en B (fin de A → B)
        v_out (3, n) : vitesse du véhicule en quittant B   (début de B → C)
    """
    n = t_b.size

    # Durées de vol de chaque arc ; elles doivent être strictement positives.
    tof_in = t_b - T_A    # durée de A → B
    tof_out = T_C - t_b   # durée de B → C
    assert cp.all((tof_in > 0.0) & (tof_out > 0.0)).item()

    # Les 2n problèmes de Lambert sont rangés dans un seul lot GPU :
    #   colonnes 0 … n-1   : A → B(t_b)
    #   colonnes n … 2n-1  : B(t_b) → C
    departures = cp.ascontiguousarray(cp.concatenate((tile_column(r_a, n), r_b), axis=1))
    arrivals = cp.ascontiguousarray(cp.concatenate((r_b, tile_column(r_c, n)), axis=1))
    durations = cp.concatenate((tof_in, tof_out))

    # revolutions=0 et long_arc=False : on cherche l'orbite directe (arc court, moins d'un tour complet). Il y a alors une seule solution par problème.
    v_depart, v_arrive, _, branch_statuses, _, lambert_statuses, _, solution_count, complete = warpkep.solve_lambert(departures, arrivals, durations, mu=MU_CENTRAL, revolutions=0, long_arc=False,)
    assert cp.all(lambert_statuses == 0).item(), lambert_statuses
    assert cp.all(solution_count == 1).item(), solution_count
    assert cp.all(complete == 1).item(), complete
    assert cp.all(branch_statuses[0] == 0).item(), branch_statuses[0]

    # Le premier indice choisit la solution (ici la seule, l'indice 0).
    v_in = v_arrive[0, :, :n]    # vitesse d'arrivée en B sur les arcs A → B
    v_out = v_depart[0, :, n:]   # vitesse de départ de B sur les arcs B → C
    return v_in, v_out

def main():
    t_b = cp.asarray(T_B_CANDIDATES, dtype=cp.float64)

    r_a, r_c = compute_points_a_and_c()
    r_b = compute_positions_of_b(t_b)
    v_in, v_out = solve_both_arcs(r_a, r_c, r_b, t_b)

    # Étape 4 : contrôle sur la date de l'énoncé (t_b = 0).
    i0 = T_B_CANDIDATES.index(0.0)
    cp.testing.assert_array_equal(r_b[:, i0], cp.asarray(R_FLYBY))
    cp.testing.assert_allclose(v_in[:, i0], cp.asarray(V_BEFORE_FLYBY), atol=1e-8, rtol=0)
    cp.testing.assert_allclose(v_out[:, i0], cp.asarray(V_AFTER_FLYBY), atol=1e-8, rtol=0)

    print(f"A = {fmt(r_a.get())} km")
    print(f"C = {fmt(r_c.get())} km\n")
    print(f"{'t_b (s)':>8} | {'vitesse en arrivant en B (km/s)':^31} | {'vitesse en quittant B (km/s)':^31}")
    for t, vi, vo in zip(t_b.get(), v_in.T.get(), v_out.T.get()):
        print(f"{t:8.0f} | {fmt(vi)} | {fmt(vo)}")
    print("\nContrôle t_b = 0 : on retrouve bien les vitesses de l'énoncé.")


if __name__ == "__main__":
    main()