#include <warpkep/detail/lambert_multi_rev_cpu.hpp>
#include <warpkep/detail/kepler_cpu.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>

namespace wd = warpkep::detail;
using Status = warpkep::LambertStatus;
using Vector = wd::Vector3<double>;
using Arc = wd::LambertArc;
using Branch = wd::LambertBranch;

namespace {

constexpr double tolerance = 1e-12; // Cas normalisés bien conditionnés ci-dessous
int checks = 0;
int failures = 0;

bool expect(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
    return condition;
}

// Fonctions utilitaires pour les tests, indépendantes de l'implémentation
double norm(const Vector& v) { return wd::norm3(v[0], v[1], v[2]); }
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
double distance(const Vector& a, const Vector& b) {
    return norm(Vector{a[0] - b[0], a[1] - b[1], a[2] - b[2]});
}

// Contrôle indépendant des tours, mu=1 : énergie -> période, phases -> arc de base
// Retourne le nombre de tours fractionnaire reconstruit à partir des vitesses
double inferred_revolutions(
    const Vector& r1, const Vector& r2, const Vector& v1, const Vector& v2, double dt
) {
    const double R1 = norm(r1), R2 = norm(r2);
    const double a = 1.0 / (2.0 / R1 - dot(v1, v1));
    if (!(a > 0.0) || !std::isfinite(a)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double pi = std::acos(-1.0);
    const double time_scale = a * std::sqrt(a);
    const Vector h = cross(r1, v1);
    Vector eccentricity = cross(v1, h);
    for (std::size_t k = 0; k < 3; ++k) { eccentricity[k] -= r1[k] / R1; }
    const double e = norm(eccentricity);
    double phase;
    if (e < 1e-10) { // Éviter de diviser par une excentricité presque nulle
        phase = std::atan2(dot(h, cross(r1, r2)) / (norm(h) * R1 * R2), dot(r1, r2) / (R1 * R2));
    } else {
        const double E1 = std::atan2(dot(r1, v1) / (e * std::sqrt(a)), (1.0 - R1 / a) / e);
        const double E2 = std::atan2(dot(r2, v2) / (e * std::sqrt(a)), (1.0 - R2 / a) / e);
        phase = (E2 - e * std::sin(E2)) - (E1 - e * std::sin(E1));
    }
    if (phase <= 0.0) { phase += 2.0 * pi; }
    return (dt - phase * time_scale) / (2.0 * pi * time_scale);
}

void check_conic(
    const char* name, const Vector& r1, const Vector& r2,
    const Vector& expected_v1, const Vector& expected_v2, double dt, Arc arc, int M
) {
    const int before = failures;
    const auto result = wd::solve_lambert_multi_rev_cpu(r1, r2, dt, 1.0, arc, M);
    const int expected_count = M == 0 ? 1 : 2;
    if (!expect(result.status == Status::success && result.complete, "requête conique complète")
        || !expect(result.solution_count == expected_count && result.slot_count == expected_count,
                   "nombre de solutions")) {
        std::cerr << "Cas concerné : " << name << '\n';
        return;
    }
    const auto prepared = wd::prepare_lambert_geometry(r1, r2, dt, 1.0, arc);
    if (!expect(prepared.status == Status::success, "préparation géométrique")) { return; }
    const auto& g = prepared.geometry;
    const double velocity_scale = std::sqrt(1.0 / g.semiperimeter);
    int matching_reference = 0;
    double max_residual = 0.0;

    for (int i = 0; i < result.slot_count; ++i) {
        const auto& s = result.solutions[i];
        expect(s.result.status == Status::success && s.revolutions == M && s.arc == arc,
               "statut et métadonnées de solution");
        expect(s.branch == (M == 0 ? Branch::single : i == 0 ? Branch::left : Branch::right),
               "ordre single / left / right");
        const double analytic_error = std::max(
            distance(s.result.departure_velocity, expected_v1),
            distance(s.result.arrival_velocity, expected_v2)
        ) / velocity_scale;
        if (analytic_error <= tolerance) { ++matching_reference; }

        const auto time = wd::evaluate_lambert_time(s.x, g.lambda, g.q, M);
        const double residual = std::abs(time.time - g.target_time) / g.target_time;
        expect(time.status == Status::success && residual <= tolerance, "résidu avec le M demandé");
        max_residual = std::max(max_residual, residual);
        const auto closure = wd::propagate_kepler_cpu<double>(
            {r1, s.result.departure_velocity}, dt, 1.0
        );
        expect(closure.status == warpkep::KeplerPropagationStatus::success
               && distance(closure.state.position, r2) / g.semiperimeter <= tolerance
               && distance(closure.state.velocity, s.result.arrival_velocity) / velocity_scale <= tolerance,
               "fermeture Kepler en position ET vitesse");
        expect(std::abs(inferred_revolutions(r1, r2, s.result.departure_velocity,
                   s.result.arrival_velocity, dt) - M) <= tolerance, "nombre de tours reconstruit");
    }
    // Une seule des deux branches doit retrouver la conique de référence
    expect(matching_reference == 1, "une et une seule conique analytique retrouvée");
    if (M > 0) {
        expect(result.solutions[0].x < result.minimum.lower
               && result.minimum.upper < result.solutions[1].x, "branches séparées par le minimum");
        expect(distance(result.solutions[0].result.departure_velocity,
                        result.solutions[1].result.departure_velocity) > 1e-5, "vitesses distinctes");
    }
    std::cout << name << ": " << (failures == before ? "OK" : "FAILED")
              << " résidu_T_max=" << max_residual << '\n';
}

void check_failure(const wd::LambertMultiRevResult<double>& result, Status expected) {
    expect(result.status == expected && result.solution_count == 0, "statut d'échec attendu");
    expect(result.complete == (expected == Status::no_solution), "complétude distincte du succès");
    for (int i = 0; i < result.slot_count; ++i) {
        const auto& s = result.solutions[i];
        expect(s.result.status == expected && std::isnan(s.x), "échec individuel et x NaN");
        for (std::size_t k = 0; k < 3; ++k) {
            expect(std::isnan(s.result.departure_velocity[k]) && std::isnan(s.result.arrival_velocity[k]),
                   "vitesses d'échec NaN");
        }
    }
}

} // namespace


// Cas de test principal : six coniques connues, puis un échec numérique et un échec de multiplicité
int main() {
    std::cout << std::scientific << std::setprecision(3);
    const double pi = std::acos(-1.0);
    const Vector r1{1, 0, 0}, r2{0, 1, 0};

    // Six demandes circulaires : compatibilité M=0, puis deux branches pour M=1,2
    for (const auto arc : std::array<Arc, 2>{Arc::short_arc, Arc::long_arc}) {
        const bool short_arc = arc == Arc::short_arc;
        for (const int M : std::array<int, 3>{0, 1, 2}) {
            const auto name = short_arc ? "cercle court" : "cercle long";
            std::cout << "M=" << M << ' ';
            check_conic(name, r1, r2, Vector{0, short_arc ? 1.0 : -1.0, 0},
                Vector{short_arc ? -1.0 : 1.0, 0, 0},
                (short_arc ? pi / 2.0 : 3.0 * pi / 2.0) + 2.0 * pi * M, arc, M);
        }
    }

    // Cas de test de la contribution des tours aux dérivées en x=0 : cercle polaire M=1
    check_conic("cercle polaire M=1", Vector{0, 1, 0}, Vector{0, 0, 1},
                Vector{0, 0, 1}, Vector{0, -1, 0}, pi / 2.0 + 2.0 * pi, Arc::short_arc, 1);

    // Ellipse connue, p=mu=1, e=1/2, f1=-pi/3 et f2=+pi/3, puis deux périodes
    const double a = 4.0 / 3.0, E = 2.0 * std::atan(1.0 / 3.0), sqrt3 = std::sqrt(3.0);
    const double ellipse_dt = 2.0 * (E - 0.5 * std::sin(E)) * a * std::sqrt(a)
        + 2.0 * (2.0 * pi * a * std::sqrt(a));
    check_conic("ellipse M=2", Vector{0.4, -0.4 * sqrt3, 0}, Vector{0.4, 0.4 * sqrt3, 0},
                Vector{sqrt3 / 2.0, 1, 0}, Vector{-sqrt3 / 2.0, 1, 0}, ellipse_dt, Arc::short_arc, 2);
    

    auto g = wd::prepare_lambert_geometry(r1, r2, 1.0, 1.0, Arc::short_arc).geometry;
    const auto base = wd::evaluate_lambert_derivatives(0.0, g.lambda, g.q);
    const auto multi = wd::evaluate_lambert_derivatives(0.0, g.lambda, g.q, 1);

    // Vérification de la contribution des tours aux dérivées en x=0 : la dérivée première est négative, la seconde est augmentée de 3*pi
    expect(multi.status == Status::success && std::abs(multi.time - base.time - pi) < 1e-13
           && std::abs(multi.first + 2.0) < 1e-13
           && std::abs(multi.second - base.second - 3.0 * pi) < 1e-13
           && multi.third == base.third, "contribution des tours et dérivées en x=0");

    
    // Vérification de la recherche du minimum et de la résolution juste au-dessus du minimum
    constexpr double reference_x = 0.14592373858333223393180152154972178548;
    constexpr double reference_time = 4.5166567516743618262023768799437710312;
    const auto minimum = wd::find_lambert_minimum(g.lambda, g.q, 1);
    if (expect(minimum.status == Status::success, "recherche du minimum")) {
        expect(std::abs(minimum.x - reference_x) < 5e-13
               && std::abs(minimum.time - reference_time) < 5e-14, "minimum haute précision");
        expect(minimum.lower <= reference_x && reference_x <= minimum.upper
               && minimum.upper - minimum.lower <= 1e-12, "encadrement du minimum");
        expect(wd::evaluate_lambert_derivatives(minimum.lower, g.lambda, g.q, 1).first <= 0.0
               && wd::evaluate_lambert_derivatives(minimum.upper, g.lambda, g.q, 1).first >= 0.0,
               "signes aux bornes du minimum");
    }
    g.target_time = reference_time * (1.0 - 1e-6);
    check_failure(wd::solve_lambert_multi_rev_prepared(g, 1.0, Arc::short_arc, 1), Status::no_solution);
    g.target_time = reference_time;
    check_failure(wd::solve_lambert_multi_rev_prepared(g, 1.0, Arc::short_arc, 1), Status::near_minimum);
    g.target_time = reference_time * (1.0 + 1e-10); // Au-dessus, hors de la bande.
    const auto near = wd::solve_lambert_multi_rev_prepared(g, 1.0, Arc::short_arc, 1);
    if (expect(near.status == Status::success && near.solution_count == 2 && near.complete,
               "deux solutions juste au-dessus du seuil")) {
        expect(near.solutions[0].x < reference_x && reference_x < near.solutions[1].x,
               "séparation près du seuil");
        for (const auto& s : near.solutions) {
            const auto time = wd::evaluate_lambert_time(s.x, g.lambda, g.q, 1);
            expect(time.status == Status::success
                   && std::abs(time.time - g.target_time) / g.target_time <= tolerance,
                   "résidu près du seuil");
        }
    }

    // Échecs principaux; les erreurs M=0 restent couvertes par test_lambert.cpp.
    expect(wd::evaluate_lambert_time(1.0, g.lambda, g.q, 1).status == Status::invalid_input
           && wd::evaluate_lambert_time(-1.0, g.lambda, g.q, 1).status == Status::invalid_input,
           "frontières elliptiques exclues pour M>0");
    check_failure(wd::solve_lambert_multi_rev_cpu(r1, r2, 8.0, 1.0, Arc::short_arc, -1), Status::invalid_input);
    check_failure(wd::solve_lambert_multi_rev_cpu(r1, Vector{-1, 0, 0}, 8.0, 1.0, Arc::short_arc, 1), Status::unsupported_case);
    wd::LambertMultiRevOptions options;
    options.threshold_relative_tolerance = 0.0;
    check_failure(wd::solve_lambert_multi_rev_cpu(r1, r2, 8.0, 1.0, Arc::short_arc, 1, options), Status::invalid_input);
    options = {};
    options.minimum_max_iterations = 1;
    check_failure(wd::solve_lambert_multi_rev_cpu(r1, r2, 8.0, 1.0, Arc::short_arc, 1, options), Status::not_converged);
    options = {};
    options.roots.max_iterations = 1;
    check_failure(wd::solve_lambert_multi_rev_cpu(r1, r2, 8.0, 1.0, Arc::short_arc, 1, options), Status::not_converged);

    // Budget court : vérifier la cohérence si une seule branche a convergé
    // Ne pas imposer un nombre exact d'itérations dépendant de la plateforme
    options.roots.max_iterations = 3;
    const auto partial = wd::solve_lambert_multi_rev_cpu(r1, r2, 9.0, 1.0, Arc::short_arc, 1, options);
    int successful_slots = 0;
    for (int i = 0; i < partial.slot_count; ++i) {
        const auto& s = partial.solutions[i];
        if (s.result.status == Status::success) {
            ++successful_slots;
            expect(std::isfinite(s.x) && std::isfinite(norm(s.result.departure_velocity))
                   && std::isfinite(norm(s.result.arrival_velocity)), "succès individuel conservé");
        } else {
            for (std::size_t k = 0; k < 3; ++k) {
                expect(std::isnan(s.result.departure_velocity[k]) && std::isnan(s.result.arrival_velocity[k]),
                       "échec individuel sans vitesse exploitable");
            }
        }
    }
    expect(partial.solution_count == successful_slots
           && partial.complete == (successful_slots == 2)
           && (partial.status == Status::success) == partial.complete, "compteurs et complétude");

    // Petite énumération : même résultat qu'un appel explicite pour chaque M
    const double dt = pi / 2.0 + 2.0 * pi;
    g = wd::prepare_lambert_geometry(r1, r2, dt, 1.0, Arc::short_arc).geometry;
    for (int M = 0; M <= 3; ++M) {
        const auto shared = wd::solve_lambert_multi_rev_prepared(g, 1.0, Arc::short_arc, M);
        const auto direct = wd::solve_lambert_multi_rev_cpu(r1, r2, dt, 1.0, Arc::short_arc, M);
        expect(shared.status == direct.status && shared.solution_count == direct.solution_count
               && shared.slot_count == direct.slot_count && shared.complete == direct.complete,
               "énumération et appel explicite cohérents");
        for (int i = 0; i < shared.slot_count; ++i) {
            expect(shared.solutions[i].revolutions == M && shared.solutions[i].arc == Arc::short_arc
                   && shared.solutions[i].branch == direct.solutions[i].branch,
                   "identifiants dans l'énumération");
            if (shared.solutions[i].result.status == Status::success) {
                expect(distance(shared.solutions[i].result.departure_velocity,
                                direct.solutions[i].result.departure_velocity) == 0.0
                       && distance(shared.solutions[i].result.arrival_velocity,
                                   direct.solutions[i].result.arrival_velocity) == 0.0,
                       "vitesses dans l'énumération");
            }
        }
    }
    std::cout << "Lambert multirévolution : " << checks << " contrôles, " << failures << " échec(s).\n";
    return failures == 0 ? 0 : 1;
}
