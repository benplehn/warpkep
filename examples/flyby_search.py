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
from scipy.optimize import brentq
import warpkep
from flyby import check_flyby

MU_CENTRAL = 1e7  # paramètre gravitationnel du corps central (km³/s²)
T_A = -1000.0     # date à laquelle le véhicule est en A (s)
T_C = 1000.0      # date à laquelle le véhicule est en C (s)

# Corps B
MU_B = 9.0e4      # paramètre gravitationnel de B (km³/s²)
R_MIN = 4000.0    # rayon de périastre minimal autorisé (km)

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
T_C_CANDIDATES = [990.0, 1000.0, 1010.0]   # dates d'arrivée testées (s)
T_B_BRACKET = (-100.0, 100.0)              # intervalle de recherche de Brent (s)

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


def compute_points_a_and_c():
    """Retrouver A et C à partir de l'état du véhicule au survol."""
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


def compute_states_of_b(t_b):
    """Calcule l'état du corps B à chaque date de survol."""
    state_b_ref = cp.asarray(STATE_B_T0, dtype=cp.float64)
    b_at_t0 = tile_column(state_b_ref, t_b.size)
    return propagate(b_at_t0, t_b)


def solve_both_arcs(r_a, r_c, r_b, t_b, t_c=T_C):
    """Résoudre A → B et B → C par Lambert pour chaque date t_b.

    t_c : date d'arrivée en C (s), T_C par défaut.
    """
    n = t_b.size

    # Durées de vol de chaque arc, elles doivent être strictement positives
    tof_in = t_b - T_A    # durée de A → B
    tof_out = t_c - t_b   # durée de B → C   ← T_C remplacé par t_c
    assert cp.all((tof_in > 0.0) & (tof_out > 0.0)).item()

    # Les 2n problèmes de Lambert sont rangés dans un seul lot GPU :
    #   colonnes 0 … n-1   : A → B(t_b)
    #   colonnes n … 2n-1  : B(t_b) → C
    departures = cp.ascontiguousarray(cp.concatenate((tile_column(r_a, n), r_b), axis=1))
    arrivals = cp.ascontiguousarray(cp.concatenate((r_b, tile_column(r_c, n)), axis=1))
    durations = cp.concatenate((tof_in, tof_out))

    # revolutions=0 et long_arc=False : on cherche l'orbite directe (arc court, moins d'un tour complet). Il y a alors une seule solution par problème
    v_depart, v_arrive, _, branch_statuses, _, lambert_statuses, _, solution_count, complete = warpkep.solve_lambert(departures, arrivals, durations, mu=MU_CENTRAL, revolutions=0, long_arc=False,)
    assert cp.all(lambert_statuses == 0).item(), lambert_statuses
    assert cp.all(solution_count == 1).item(), solution_count
    assert cp.all(complete == 1).item(), complete
    assert cp.all(branch_statuses[0] == 0).item(), branch_statuses[0]

    # Le premier indice choisit la solution (ici la seule, l'indice 0)
    v_dep_a = v_depart[0, :, :n]  # départ de A
    v_in = v_arrive[0, :, :n]     # arrivée en B
    v_out = v_depart[0, :, n:]    # départ de B
    v_arr_c = v_arrive[0, :, n:]  # arrivée en C
    return v_dep_a, v_in, v_out, v_arr_c


def flyby_velocities_at(t, r_a, r_c, t_c=T_C):
    dates = cp.asarray([t], dtype=cp.float64)
    state_b = compute_states_of_b(dates)
    v_dep_a, v_in, v_out, v_arr_c = solve_both_arcs(
        r_a, r_c, state_b[:3], dates, t_c
    )
    return v_dep_a[:, 0], v_in[:, 0], v_out[:, 0], v_arr_c[:, 0], state_b[3:, 0]


def flyby_residual_at(t, r_a, r_c, t_c=T_C):
    _, v_in, v_out, _, v_b = flyby_velocities_at(t, r_a, r_c, t_c)
    speed_in = cp.linalg.norm(v_in - v_b)
    speed_out = cp.linalg.norm(v_out - v_b)
    return (speed_out - speed_in).item()


def circular_velocity_xy(position):
    """Vitesse circulaire prograde au point donné dans le plan xy."""
    x, y, _ = position
    radius = cp.linalg.norm(position)
    tangent = cp.stack((-y, x, cp.zeros_like(x))) / radius
    return cp.sqrt(MU_CENTRAL / radius) * tangent


def state_of_c_at(t_c, state_c_ref):
    """Position et vitesse du corps C à la date t_c.

    state_c_ref est l'état de C à T_C ; on le propage de (t_c − T_C).
    """
    state = propagate(tile_column(state_c_ref, 1), [t_c - T_C])
    return state[:3, 0], state[3:, 0]


def evaluate_arrival(t_c, r_a, state_c_ref):
    """Un trajet complet A → B → C pour une date d'arrivée t_c.

    Retourne un dict, ou None si Brent n'a pas de racine dans T_B_BRACKET.
    """
    r_c_tc, v_c_tc = state_of_c_at(t_c, state_c_ref)

    # Brent n'a le droit de démarrer que si f change de signe entre les bornes.
    lo, hi = T_B_BRACKET
    f_lo = flyby_residual_at(lo, r_a, r_c_tc, t_c)
    f_hi = flyby_residual_at(hi, r_a, r_c_tc, t_c)
    if f_lo * f_hi > 0.0:
        print(f"t_c = {t_c:.0f} s : f({lo:+.0f}) = {f_lo:+.3e}, f({hi:+.0f}) = {f_hi:+.3e}" " → même signe, pas de racine encadrée")
        return None

    t_root, info = brentq(flyby_residual_at, lo, hi, args=(r_a, r_c_tc, t_c), full_output=True)
    assert info.converged

    # Vitesses à la date de survol trouvée
    v_dep_a, v_in, v_out, v_arr_c, v_b = flyby_velocities_at(t_root, r_a, r_c_tc, t_c)

    # Périastre : le survol est-il admissible ?
    feasible, r_p = check_flyby(v_b.get(), v_in.get(), v_out.get(), mu_b=MU_B, r_min=R_MIN, speed_atol=1e-8)

    # Coût : départ depuis l'orbite circulaire en A, arrivée sur le mouvement de C
    dv_a = cp.linalg.norm(v_dep_a - circular_velocity_xy(r_a)).item()
    dv_c = cp.linalg.norm(v_c_tc - v_arr_c).item()

    return {"t_c": t_c, "t_b": t_root, "r_p": r_p, "feasible": feasible, "dv_a": dv_a, "dv_c": dv_c, "dv_total": dv_a + dv_c}


def compare_arrivals(r_a, r_c):
    """Compare les trajets pour plusieurs dates d'arrivée en C."""
    # État de référence de C à T_C : position r_c, vitesse circulaire.
    state_c_ref = cp.ascontiguousarray(cp.concatenate((r_c, circular_velocity_xy(r_c))))

    results = [evaluate_arrival(t_c, r_a, state_c_ref) for t_c in T_C_CANDIDATES]
    results = [r for r in results if r is not None]

    print(f"\n{'t_c (s)':>8} | {'t_b (s)':>12} | {'r_p (km)':>10} | " f"{'Δv total (km/s)':>16} | admissible")
    for r in results:
        rp_text = "—" if r["r_p"] is None else f"{r['r_p']:.2f}"
        verdict = "oui" if r["feasible"] else "non"
        print(f"{r['t_c']:8.0f} | {r['t_b']:+12.6f} | {rp_text:>10} | " f"{r['dv_total']:16.6f} | {verdict}")
    
    # Contrôle : la ligne t_c = 1000 s doit retrouver le trajet de référence.
    ref = next(r for r in results if r["t_c"] == T_C)
    assert abs(ref["dv_total"] - 6.038943) < 1e-6, ref["dv_total"]

    # Le meilleur trajet, parmi les seuls survols admissibles
    admissible = [r for r in results if r["feasible"]]
    assert admissible, "aucun trajet admissible"
    best = min(admissible, key=lambda r: r["dv_total"])
    print(f"\nMeilleur trajet : t_c = {best['t_c']:.0f} s, t_b = {best['t_b']:+.6f} s, " f"Δv total = {best['dv_total']:.6f} km/s")
    return best










def main():
    t_b = cp.asarray(T_B_CANDIDATES, dtype=cp.float64)

    r_a, r_c = compute_points_a_and_c()


    # Brent cherche la date t_B où f(t_B) = 0 dans [-100, 100]. Il ne reçoit que les deux bornes, jamais la date témoin t_b = 0.
    t_root, info = brentq(flyby_residual_at,  -100.0, 100.0, args=(r_a, r_c), full_output=True)
    v_dep_a, v_in_r, v_out_r, v_arr_c, v_b_r = flyby_velocities_at(t_root, r_a, r_c)

    # On calcule la vitesse du véhicule par rapport au corps B juste avant et juste après le survol, et on vérifie que la différence est nulle.
    speed_in_root = cp.linalg.norm(v_in_r - v_b_r)
    speed_out_root = cp.linalg.norm(v_out_r - v_b_r)
    f_root = (speed_out_root - speed_in_root).item()
    print(f"Brent : t_B = {t_root:+.3e} s, f(t_B) = {f_root:+.3e} km/s, " f"{info.iterations} itérations, convergé = {info.converged}")
    assert info.converged
    assert abs(t_root) < 1e-5
    assert abs(f_root) < 1e-8

    # On vérifie que le survol est bien faisable, et on récupère le rayon de périastre r_p.
    feasible, r_p = check_flyby(v_b_r.get(), v_in_r.get(), v_out_r.get(), mu_b=MU_B, r_min=R_MIN, speed_atol=1e-8,)
    assert feasible, f"Survol refusé à t_B = {t_root:.6g} s, r_p = {r_p}"
    assert abs(r_p - 4142.14) < 0.01
    print(f"Survol accepté, r_p = {r_p:.2f} km")

    # On calcule les vitesses circulaires au point A et C, puis les Δv de départ et d'arrivée.
    v_body_a = circular_velocity_xy(r_a)
    v_body_c = circular_velocity_xy(r_c)
    dv_a = cp.linalg.norm(v_dep_a - v_body_a).item()
    dv_c = cp.linalg.norm(v_body_c - v_arr_c).item()

    print(f"Δv départ A  = {dv_a:.6f} km/s")
    print(f"Δv arrivée C = {dv_c:.6f} km/s")
    print(f"Δv total     = {dv_a + dv_c:.6f} km/s")

    # On calcule les vitesses du véhicule par rapport au corps B juste avant et juste après le survol pour toutes les dates t_b testées.
    states_b = compute_states_of_b(t_b)
    r_b = states_b[:3, :]
    v_b = states_b[3:, :]
    _, v_in, v_out, _ = solve_both_arcs(r_a, r_c, r_b, t_b)

    # Soustraire la vitesse du corps B et calculer une norme par date
    speed_in = cp.linalg.norm(v_in - v_b, axis=0)
    speed_out = cp.linalg.norm(v_out - v_b, axis=0)
    residual = speed_out - speed_in

    # Contrôle sur la date de l'énoncé (t_b = 0)
    i0 = T_B_CANDIDATES.index(0.0)
    cp.testing.assert_array_equal(r_b[:, i0], cp.asarray(R_FLYBY))
    cp.testing.assert_allclose(v_in[:, i0], cp.asarray(V_BEFORE_FLYBY), atol=1e-8, rtol=0)
    cp.testing.assert_allclose(v_out[:, i0], cp.asarray(V_AFTER_FLYBY), atol=1e-8, rtol=0)

    print(f"A = {fmt(r_a.get())} km")
    print(f"C = {fmt(r_c.get())} km\n")
    assert abs(residual[i0].item()) < 1e-8
    print(f"{'t_b (s)':>8} | {'v_in (km/s)':^31} | " f"{'v_out (km/s)':^31} | {'f(t_b) (km/s)':>15}")

    # On affiche les vitesses du véhicule par rapport au corps B juste avant et juste après le survol pour toutes les dates t_b testées.
    for t, vi, vo, f in zip(t_b.get(), v_in.T.get(), v_out.T.get(), residual.get()):
        print(f"{t:8.0f} | {fmt(vi)} | {fmt(vo)} | {f:+15.8e}")
    
    print("\nContrôle t_b = 0 : on retrouve bien les vitesses de l'énoncé.")


    compare_arrivals(r_a, r_c)


    
if __name__ == "__main__":
    main()