import cupy as cp
import numpy as np

import warpkep


def flyby_vinf(v_body, v_in, v_out):
    return v_in - v_body, v_out - v_body


def check_flyby(v_body, v_in, v_out, mu_b, r_min, *, speed_atol=1e-12):
    vinf_in, vinf_out = flyby_vinf(v_body, v_in, v_out)
    speed_in = np.linalg.norm(vinf_in)
    speed_out = np.linalg.norm(vinf_out)

    if speed_in == 0.0 or speed_out == 0.0:
        return False, None
    if not np.isclose(speed_in, speed_out, rtol=0.0, atol=speed_atol):
        return False, None

    cross_norm = np.linalg.norm(np.cross(vinf_in, vinf_out))
    dot = np.dot(vinf_in, vinf_out)
    delta = np.arctan2(cross_norm, dot)
    if delta == 0.0:
        return False, None

    eccentricity = 1.0 / np.sin(delta / 2.0)
    rp = mu_b * (eccentricity - 1.0) / speed_in**2
    return rp >= r_min, rp


def main():
    mu_central = 1e7  # km³/s², pour Kepler et Lambert
    mu_b = 90_000.0   # km³/s², pour le survol
    tof = 1000.0      # s

    r_b = np.array([100_000.0, 0.0, 0.0])
    v_body = np.array([0.0, 10.0, 0.0])
    v_in_ref = np.array([3.0, 10.0, 0.0])
    v_out_ref = np.array([0.0, 13.0, 0.0])

    # Deux états au même point B, un pour chaque arc, WarpKEP attend (6, N), avec un état par colonne.
    state_in = np.concatenate((r_b, v_in_ref))
    state_out = np.concatenate((r_b, v_out_ref))
    states = cp.ascontiguousarray(cp.asarray(np.column_stack((state_in, state_out))))
    signed_times = cp.array([-tof, tof], dtype=cp.float64)
    endpoints, kepler_statuses, _ = warpkep.propagate_kepler(states, signed_times, mu=mu_central)
    assert cp.all(kepler_statuses == 0).item(), kepler_statuses

    # Colonne 0 : A → B. Colonne 1 : B → C.
    r_a = endpoints[:3, 0]
    r_c = endpoints[:3, 1]
    r_b_gpu = cp.asarray(r_b)
    departures = cp.ascontiguousarray(cp.stack((r_a, r_b_gpu), axis=1))
    arrivals = cp.ascontiguousarray(cp.stack((r_b_gpu, r_c), axis=1))
    flight_times = cp.full(2, tof, dtype=cp.float64)

    v_depart, v_arrive, _, branch_statuses, _, lambert_statuses, _, solution_count, complete = warpkep.solve_lambert(departures, arrivals, flight_times, mu=mu_central, revolutions=0, long_arc=False)
    assert cp.all(lambert_statuses == 0).item(), lambert_statuses
    assert cp.all(solution_count == 1).item(), solution_count
    assert cp.all(complete == 1).item(), complete
    assert cp.all(branch_statuses[0] == 0).item(), branch_statuses[0]

    # Indices des vitesses Lambert : [branche, composante, arc].
    v_in = cp.asnumpy(v_arrive[0, :, 0])   # arrivée de A → B
    v_out = cp.asnumpy(v_depart[0, :, 1])  # départ de B → C

    speed_atol = 1e-8  # km/s, pour le test aller-retour Kepler–Lambert
    np.testing.assert_allclose(v_in, v_in_ref, rtol=0.0, atol=speed_atol)
    np.testing.assert_allclose(v_out, v_out_ref, rtol=0.0, atol=speed_atol)

    feasible, rp = check_flyby(v_body, v_in, v_out, mu_b, r_min=4000.0, speed_atol=speed_atol)
    assert feasible, f"Survol refusé, rp={rp}"
    np.testing.assert_allclose(rp, 4142.1356237, rtol=0.0, atol=1e-3)
    print(f"Survol faisable, rayon de périastre = {rp:.2f} km")


if __name__ == "__main__":
    main()