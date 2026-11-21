import pytest
from math import pi
import warpkep
cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("sens_b", [1, -1])
def test_ephemerides_cas_circulaire(sens_b):
    # sens_b = 1  : B tourne dans le même sens que A (même orbite)
    # sens_b = -1 : B tourne en sens inverse (orbite rétrograde)

    # Define states
    state_A = (1, 0, 0, 0, 1, 0)
    state_B = (-1, 0, 0, 0, -sens_b, 0)

    # Empiler les deux états : une ligne par corps (--> shape (2, 6))
    states = cp.array([state_A, state_B], dtype = cp.float64)

    # On transpose pour voir la shape (6, 2)
    # .T ne bouge pas la mémoire : c'est une vue avec les strides inversés
    states = states.T

    # Rendre contigu en mémoire (vraie copie dans l'ordre C attendu par le kernel)
    states = cp.ascontiguousarray(states)
    assert states.shape == (6, 2)
    assert states.dtype == cp.float64
    assert states.flags.c_contiguous

    # après 1 : (2, 6)                    après 2-3 : (6, 2)
#          rx ry rz vx vy vz                   A    B
#   A  [   1  0  0  0  1  0 ]            rx [  1   -1 ]
#   B  [  -1  0  0  0 -1  0 ]            ry [  0    0 ]
#                                        rz [  0    0 ]
#                                        vx [  0    0 ]
#                                        vy [  1   -1 ]
#                                        vz [  0    0 ]

    # Definition des variables de propagations
    # durée à propager = date cible - date de référence de chaque corps
    t_ref_A = 0.0
    t_ref_B = pi
    t_depart = 3 * pi / 2
    t_arrivee = 2 * pi
    durations = cp.array([t_depart - t_ref_A, t_arrivee - t_ref_B], dtype=cp.float64)
    assert durations.shape == (2,)          # [3pi/2, pi] : une durée par corps
    print(durations)

    # Appel du propagateur
    # out : (6, 2) -> A à t_depart en colonne 0, B à t_arrivee en colonne 1
    out, statuses, _ = warpkep.propagate_kepler(states, durations, mu=1.0)
    print("Forme de la sortie :", out.shape)
    print("Statuts :", statuses)

    # Verification des résultats
    # A à 3pi/2 : r = (0, -1, 0), v = (1, 0, 0)
    # B à 2pi   : r = (1, 0, 0),  v = (0, sens_b, 0)
    expected = cp.array([(0, -1, 0, 1, 0, 0), (1, 0, 0, 0, sens_b, 0)], dtype=cp.float64).T
    assert out.shape == expected.shape
    cp.testing.assert_array_equal(statuses, cp.zeros(2, dtype=cp.int32))   # 0 = OK
    cp.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)

    # Prepare lambert entries
    # :3  -> positions seulement (rx, ry, rz)
    # 0:1 -> colonne 0 en gardant la dimension : shape (3, 1) et pas (3,)
    # slice = vue non contiguë (une case sur deux) -> ascontiguousarray
    departures = cp.ascontiguousarray(out[:3, 0:1])
    arrivals = cp.ascontiguousarray(out[:3, 1:2])
    flight_times = cp.array([t_arrivee - t_depart], dtype=cp.float64)   # (1,) : un arc

    # Solve lambert for one arc
    # 9 sorties : on garde les vitesses, le nombre de solutions et le flag complete
    # v_depart, v_arrivee : (branche, composante, arc) = (2, 3, 1)
    v_depart, v_arrivee, _, _, _, _, _, solution_count, complete = warpkep.solve_lambert(departures, arrivals, flight_times, mu=1.0, revolutions=0, long_arc=False)
    assert solution_count.item() == 1       # .item() : scalaire Python, vérifie aussi la taille 1
    assert complete.item() == 1

    # Compute deltaV
    # départ  : vitesse sur l'arc - vitesse de A
    # arrivée : vitesse de B - vitesse en fin d'arc
    dv_depart = cp.linalg.norm(v_depart[0, :, 0] - out[3:, 0])
    dv_arrivee = cp.linalg.norm(out[3:, 1] - v_arrivee[0, :, 0])
    dv_total = dv_depart + dv_arrivee       # les deux poussées sont obligatoires : on additionne

    # Check results
    # l'arc est un quart de l'orbite de A -> aucun dV au départ
    # arrivée : 0 si B va dans le même sens, 2 s'il faut inverser la vitesse
    print(dv_depart.item(), dv_arrivee.item(), dv_total.item())
    assert dv_depart.item() < 1e-10
    assert abs(dv_arrivee.item() - (1 - sens_b)) < 1e-10

    # Check that arc closes
    # on repropage (position de départ + vitesse Lambert) et on doit tomber sur l'arrivée
    transfer_state = cp.concatenate((departures, v_depart[0]), axis=0)  # (6, 1)
    transfer_end, transfer_status, _ = warpkep.propagate_kepler(transfer_state, flight_times, mu=1.0)
    assert transfer_status.item() == 0
    cp.testing.assert_allclose(transfer_end[:3], arrivals, rtol=0, atol=1e-10)
    cp.testing.assert_allclose(transfer_end[3:], v_arrivee[0], rtol=0, atol=1e-10)



def test_grille_rendezvous():
    # But : trouver le meilleur couple (date de départ de A, date d'arrivée sur B)
    # 1. grille grossière 3 x 3  2. on vérifie le meilleur arc  3. on raffine autour

    # Create departure grid
    departure_dates = cp.array([0.0, 0.5, 1.0], dtype=cp.float64)
    arrival_dates = cp.array([0.75, 1.5, 2.5], dtype=cp.float64)

    # A : orbite circulaire r = 1 ; B : orbite circulaire r = 1.5, même sens
    # les deux états sont donnés à t = 0 (t_ref = 0 implicite)
    state_a = cp.array([1, 0, 0, 0, 1, 0], dtype=cp.float64)
    state_b = cp.array([0, 1.5, 0, -(1 / 1.5) ** 0.5, 0, 0], dtype=cp.float64)
    candidates = evaluate_grid(state_a, state_b, departure_dates, arrival_dates)
    assert candidates["dv_total"].shape == (8,)     # 9 couples - 1 temps de vol négatif

    # Find best candidat
    costs = candidates["dv_total"]
    eligible = candidates["solution_ok"] & cp.isfinite(costs)
    assert eligible.all().item()                    # les 8 arcs doivent être résolus

    # where : coût -> +inf pour les non éligibles, puis argmin (évite les NaN)
    best_idx = cp.argmin(cp.where(eligible, costs, cp.inf)).item()
    best_departure_date = candidates["departure_dates"][best_idx].item()
    best_arrival_date = candidates["arrival_dates"][best_idx].item()
    best_cost = costs[best_idx].item()

    print(best_departure_date, best_arrival_date, best_cost)

    # Recompute best arc
    # contrôle indépendant : on refait le calcul à la main pour ce seul arc
    # stack axis=1 : A en colonne 0, B en colonne 1 -> (6, 2)
    initial = cp.ascontiguousarray(cp.stack((state_a, state_b), axis=1))
    durations = cp.array([best_departure_date, best_arrival_date], dtype=cp.float64)
    propagated, statuses, _ = warpkep.propagate_kepler(initial, durations, mu=1.0)
    assert (statuses == 0).all().item()

    flight_time = cp.array([best_arrival_date - best_departure_date], dtype=cp.float64)
    v0, v1, _, _, _, _, _, count, complete = warpkep.solve_lambert(cp.ascontiguousarray(propagated[:3, 0:1]), cp.ascontiguousarray(propagated[:3, 1:2]), flight_time, mu=1.0, revolutions=0, long_arc=False)
    assert count.item() == 1 and complete.item() == 1

    dv0 = cp.linalg.norm(v0[0, :, 0] - propagated[3:, 0])
    dv1 = cp.linalg.norm(propagated[3:, 1] - v1[0, :, 0])
    assert abs((dv0 + dv1).item() - best_cost) < 1e-10     # même coût que la grille

    # Refine around best candidate
    # petite grille 3 x 3 centrée sur le meilleur couple, pas de 0.1
    offsets = cp.array([-0.1, 0.0, 0.1], dtype=cp.float64)
    fine_departures = best_departure_date + offsets
    fine_arrivals = best_arrival_date + offsets

    # on reste dans la fenêtre de la grille grossière (masque booléen)
    fine_departures = fine_departures[(fine_departures >= departure_dates[0]) & (fine_departures <= departure_dates[-1])]
    fine_arrivals = fine_arrivals[(fine_arrivals >= arrival_dates[0]) & (fine_arrivals <= arrival_dates[-1])]

    fine = evaluate_grid(state_a, state_b, fine_departures, fine_arrivals)
    costs = fine["dv_total"]
    eligible = fine["solution_ok"] & cp.isfinite(costs)
    assert eligible.any().item()

    idx = cp.argmin(cp.where(eligible, costs, cp.inf)).item()
    best_cost_fine = costs[idx].item()
    print(fine["departure_dates"][idx].item(), fine["arrival_dates"][idx].item(), best_cost_fine)
    # offset 0.0 -> le meilleur point grossier est dans la grille fine : on ne peut pas faire pire
    assert best_cost_fine <= best_cost + 1e-10


def evaluate_grid(state_a, state_b, departure_dates, arrival_dates):
    # Évalue tous les couples (départ de A, arrivée sur B) en un seul lot GPU
    # Retourne un dict de colonnes : une case par arc valide

    # Flight times grid
    # broadcasting : (1, n_arr) - (n_dep, 1) -> (n_dep, n_arr)
    # tof[i, j] = arrival_dates[j] - departure_dates[i]
    flight_times_grid = arrival_dates[None, :] - departure_dates[:, None]
    valid = flight_times_grid > 0                   # on ne peut pas arriver avant de partir

#                  arr 0.75   arr 1.5   arr 2.5
#   dep 0.0     [   0.75       1.5       2.5  ]
#   dep 0.5     [   0.25       1.0       2.0  ]
#   dep 1.0     [  -0.25 x     0.5       1.5  ]

    # Propagate with one call
    # A recopié n_dep fois, B recopié n_arr fois, côte à côte -> (6, n_dep + n_arr)
    # durée = date car t_ref = 0 pour les deux corps
    n_departures = departure_dates.size
    states_a = cp.repeat(state_a[:, None], n_departures, axis=1)
    states_b = cp.repeat(state_b[:, None], arrival_dates.size, axis=1)
    states = cp.concatenate((states_a, states_b), axis=1)
    durations = cp.concatenate((departure_dates, arrival_dates))
    out, statuses, _ = warpkep.propagate_kepler(states, durations, mu=1.0)
    assert (statuses == 0).all().item()

#   colonne :   0    1    2  |  3     4     5
#   corps   :   A    A    A  |  B     B     B
#   durée   :  0.0  0.5  1.0 | 0.75  1.5   2.5

    departure_states = out[:, :n_departures]        # A à chaque date de départ
    arrival_states = out[:, n_departures:]          # B à chaque date d'arrivée

    # Prepare lambert
    # nonzero : coordonnées (i, j) des cases valides -> grille aplatie en liste d'arcs
    #   dep (i) : [0, 0, 0, 1, 1, 1, 2, 2]
    #   arr (j) : [0, 1, 2, 0, 1, 2, 1, 2]
    departure_indices, arrival_indices = cp.nonzero(valid)
    # fancy indexing : on va chercher la bonne colonne pour chaque arc -> (3, n_arcs)
    departures = cp.ascontiguousarray(departure_states[:3, departure_indices])
    arrivals = cp.ascontiguousarray(arrival_states[:3, arrival_indices])
    flight_times = flight_times_grid[departure_indices, arrival_indices]   # (n_arcs,)

    # Solve all arcs in one call
    # v_depart, v_arrivee : (2, 3, n_arcs) = (branche, composante, arc)
    # revolutions = 0 -> seule la branche 0 sert
    v_depart, v_arrivee, _, branch_statuses, _, lambert_statuses, _, solution_count, complete, = warpkep.solve_lambert(departures, arrivals, flight_times, mu=1.0, revolutions=0, long_arc=False)
    print("Solutions par arc :", solution_count)

    # Filter solutions
    # masques booléens (n_arcs,) ; & = ET élément par élément (parenthèses obligatoires)
    # codes déduits des noms : 0 = OK, 5 = pas de solution, 3/4 = échec numérique
    solution_ok = (lambert_statuses == 0) & (branch_statuses[0] == 0) & (solution_count == 1) & (complete == 1)
    no_solution = lambert_statuses == 5
    numerical_failure = (lambert_statuses == 3) | (lambert_statuses == 4)

    # Compute cost of each valid arc
    # vitesses de A au départ et de B à l'arrivée, seulement pour les arcs OK
    v_a = departure_states[3:, departure_indices[solution_ok]]
    v_b = arrival_states[3:, arrival_indices[solution_ok]]

    # NaN par défaut : un arc raté ne peut jamais sortir comme meilleur candidat
    # norm(..., axis=0) : une norme par arc (sur les 3 composantes)
    dv_depart = cp.full_like(flight_times, cp.nan)
    dv_arrivee = cp.full_like(flight_times, cp.nan)
    dv_depart[solution_ok] = cp.linalg.norm(v_depart[0][:, solution_ok] - v_a, axis=0)
    dv_arrivee[solution_ok] = cp.linalg.norm(v_b - v_arrivee[0][:, solution_ok], axis=0)
    dv_total = dv_depart + dv_arrivee

    # Store results of all candidates table en colonnes : la case k de chaque clé décrit l'arc k
    candidates = {
        "departure_body": "A",
        "arrival_body": "B",
        "long_arc": False,
        "revolutions": 0,
        "branch": 0,
        "departure_dates": departure_dates[departure_indices],
        "arrival_dates": arrival_dates[arrival_indices],
        "departure_statuses": statuses[departure_indices],
        "arrival_statuses": statuses[n_departures + arrival_indices],
        "lambert_statuses": lambert_statuses,
        "branch_statuses": branch_statuses[0],
        "solution_ok": solution_ok,
        "no_solution": no_solution,
        "numerical_failure": numerical_failure,
        "dv_depart": dv_depart,
        "dv_arrivee": dv_arrivee,
        "dv_total": dv_total,


    }

    return candidates