from math import hypot, isclose, pi, sqrt


# Repère inertiel centré sur l'attracteur, orbite dans le plan xy
# Positions en UL, vitesses en UL/UT, dates en UT depuis t = 0.

mu = 1.0 # UL³/UT²
catalogue = {
    "A" : {"tref" : 0.0, "state_ref" : (1, 0, 0, 0, 1, 0)},
    "B": {"t_ref": pi, "state_ref": (-1, 0, 0, 0, -1, 0)},
}

for body_id, body in catalogue.items():
    rx, ry, rz, vx, vy, vz = body["state_ref"]
    radius = hypot(rx, ry, rz)
    speed = hypot(vx, vy, vz)
    dot_rv = rx * vx + ry * vy + rz * vz
    h = (ry * vz - rz * vy, rz * vx - rx * vz, rx * vy - ry * vx)
    assert(isclose(radius, 1.0))
    assert isclose(speed, sqrt(mu / radius))
    assert isclose(dot_rv, 0.0, abs_tol=1e-12)
    assert h == (0, 0, 1)
    print(f"{body_id} : rayon={radius}, vitesse={speed}, r×v={h}")


# Definition des variables de test
t_depart = 3 * pi / 2
t_arrivee = 2 * pi
duree_vol = t_arrivee - t_depart
assert isclose(duree_vol, pi / 2)
print(f"Durée de vol : {duree_vol:.6f} UT")



