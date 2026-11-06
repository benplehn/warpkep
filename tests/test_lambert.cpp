#include <warpkep/detail/lambert_cpu.hpp>
#include <warpkep/detail/kepler_cpu.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>

namespace wd = warpkep::detail;
using Status = warpkep::LambertStatus;
using Vector = wd::Vector3<double>;
using State = wd::CartesianState<double>;

namespace {

    // Tolérance absolue, valable uniquement pour les cas normalisés (mu = p = 1)
    constexpr double tolerance = 1e-12;

    struct AnalyticalCase {
        const char* name;
        State departure;
        State arrival;
        double dt;
        wd::LambertArc arc;
    };

    double distance(const Vector& a, const Vector& b) {
        return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    }

    // Conique connue (mu = p = 1) parcourue entre les anomalies vraies -pi/3 et +pi/3
    // Référence analytique indépendante de Lambert et du propagateur Kepler
    AnalyticalCase symmetric_conic(const char* name, double e) {
        // r = p / (1 + e * cos(f)), avec cos(pi/3) = 1/2
        const double sqrt3 = std::sqrt(3.0);
        const double radius = 1.0 / (1.0 + e / 2.0);

        // Temps de vol de -f à +f, avec tan(f/2) = 1/sqrt(3)
        double dt;
        if (e < 1.0) {
            // Ellipse : équation de Kepler, tan(E/2) = sqrt((1 - e) / (1 + e)) * tan(f/2)
            const double a = 1.0 / (1.0 - e * e);
            const double E = 2.0 * std::atan(std::sqrt((1.0 - e) / (1.0 + e)) / sqrt3);
            dt = 2.0 * (E - e * std::sin(E)) * a * std::sqrt(a);
        } else if (e > 1.0) {
            // Hyperbole : équation de Kepler hyperbolique, tanh(H/2) = sqrt((e - 1) / (e + 1)) * tan(f/2)
            const double abs_a = 1.0 / (e * e - 1.0);
            const double H = 2.0 * std::atanh(std::sqrt((e - 1.0) / (e + 1.0)) / sqrt3);
            dt = 2.0 * (e * std::sinh(H) - H) * abs_a * std::sqrt(abs_a);
        } else {
            // Parabole : équation de Barker, D = tan(f/2)
            const double D = 1.0 / sqrt3;
            dt = D + D * D * D / 3.0;
        }

        // Vitesse périfocale : v = sqrt(mu / p) * (-sin(f), e + cos(f))
        return {
            name,
            {{radius / 2.0, -sqrt3 * radius / 2.0, 0.0}, {sqrt3 / 2.0, e + 0.5, 0.0}},
            {{radius / 2.0, sqrt3 * radius / 2.0, 0.0}, {-sqrt3 / 2.0, e + 0.5, 0.0}},
            dt,
            wd::LambertArc::short_arc,
        };
    }

    bool check_case(const AnalyticalCase& reference) {
        const Vector& r1 = reference.departure.position;
        const Vector& r2 = reference.arrival.position;

        // 1. Solve the Lambert problem
        const auto result = wd::solve_lambert_cpu(r1, r2, reference.dt, 1.0, reference.arc);
        if (result.status != Status::success) {
            std::cerr << reference.name << ": échec Lambert, statut=" << static_cast<int>(result.status) << '\n';
            return false;
        }

        // 2. Compare with the analytical velocities
        // Les vitesses de référence détectent aussi un mauvais sens de parcours
        const double ev1 = distance(result.departure_velocity, reference.departure.velocity);
        const double ev2 = distance(result.arrival_velocity, reference.arrival.velocity);

        // 3. Check closure with the Kepler propagator
        // Propager (r1, v1) pendant dt doit redonner (r2, v2)
        const auto closure = wd::propagate_kepler_cpu<double>({r1, result.departure_velocity}, reference.dt, 1.0);
        const double er = distance(closure.state.position, r2);
        const double ev = distance(closure.state.velocity, result.arrival_velocity);

        // 4. Check the scalar root-solver contract
        // On réévalue T à la racine retournée et on contrôle le résidu relatif
        const auto prepared = wd::prepare_lambert_geometry(r1, r2, reference.dt, 1.0, reference.arc);
        const auto& g = prepared.geometry;
        const auto root = wd::solve_lambert_root(g.target_time, g.lambda, g.q);
        const auto time = wd::evaluate_lambert_time(root.x, g.lambda, g.q);
        const double residual = std::abs(time.time - g.target_time) / g.target_time;

        const bool passed = prepared.status == Status::success
            && root.status == Status::success
            && time.status == Status::success
            && closure.status == warpkep::KeplerPropagationStatus::success
            && ev1 <= tolerance && ev2 <= tolerance
            && er <= tolerance && ev <= tolerance
            && residual <= tolerance;

        std::cout << reference.name << ": " << (passed ? "OK" : "FAILED")
                  << " dv1=" << ev1 << " dv2=" << ev2
                  << " fermeture_r=" << er << " fermeture_v=" << ev
                  << " résidu_T=" << residual << '\n';
        return passed;
    }

}  // namespace

int main() {
    const double pi = std::acos(-1.0);
    bool ok = true;

    std::cout << std::scientific << std::setprecision(3);

    // 1. Analytical reference cases
    // Cercles (mu = r = 1) : la vitesse vaut 1 et le temps de vol est égal à l'angle parcouru
    const std::array<AnalyticalCase, 6> cases{{
        {"cercle court", {{1, 0, 0}, {0, 1, 0}}, {{0, 1, 0}, {-1, 0, 0}}, pi / 2.0, wd::LambertArc::short_arc},
        {"cercle long", {{1, 0, 0}, {0, -1, 0}}, {{0, 1, 0}, {1, 0, 0}}, 3.0 * pi / 2.0, wd::LambertArc::long_arc},
        {"cercle polaire 3D", {{0, 1, 0}, {0, 0, 1}}, {{0, 0, 1}, {0, -1, 0}}, pi / 2.0, wd::LambertArc::short_arc},
        symmetric_conic("ellipse", 0.5),
        symmetric_conic("hyperbole", 1.5),
        symmetric_conic("parabole", 1.0),
    }};

    // On exécute tous les cas, même après un échec
    for (const auto& reference : cases) {
        ok = check_case(reference) && ok;
    }

    // 2. Parabolic neighbourhood
    // La formule générale est fragile près de x = 1 : on contrôle les deux côtés
    // T(1) = (2/3) * (1 - lambda³) et T'(1) = (2/5) * (lambda⁵ - 1)
    const double lambda = std::sqrt(2.0) - 1.0;
    const double q = (1.0 - lambda) * (1.0 + lambda);
    const double T1 = (2.0 / 3.0) * (1.0 - lambda * lambda * lambda);
    const double slope = (2.0 / 5.0) * (std::pow(lambda, 5) - 1.0);

    for (const double h : std::array<double, 2>{-1e-8, 1e-8}) {
        // Développement au premier ordre : le terme négligé en O(h²) reste sous 1e-14
        const auto result = wd::evaluate_lambert_time(1.0 + h, lambda, q);
        const bool passed = result.status == Status::success
            && std::abs(result.time - (T1 + slope * h)) <= 1e-14;

        std::cout << "voisinage parabolique h=" << h << ": " << (passed ? "OK" : "FAILED") << '\n';
        ok = passed && ok;
    }

    // 3. Failure paths
    // Chaque échec doit donner le statut attendu et six composantes de vitesse à NaN
    const auto expect_failure = [&ok](const char* name, const wd::LambertResult<double>& result, Status expected) {
        bool passed = result.status == expected;
        for (std::size_t i = 0; i < 3; ++i) {
            passed = passed && std::isnan(result.departure_velocity[i]) && std::isnan(result.arrival_velocity[i]);
        }

        std::cout << name << ": " << (passed ? "OK" : "FAILED") << '\n';
        ok = passed && ok;
    };

    const Vector r1{1, 0, 0};
    const Vector r2{0, 1, 0};
    const auto arc = wd::LambertArc::short_arc;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Entrées invalides
    expect_failure("mu invalide", wd::solve_lambert_cpu(r1, r2, 1.0, 0.0, arc), Status::invalid_input);
    expect_failure("dt nul", wd::solve_lambert_cpu(r1, r2, 0.0, 1.0, arc), Status::invalid_input);
    expect_failure("position NaN", wd::solve_lambert_cpu(Vector{nan, 0, 0}, r2, 1.0, 1.0, arc), Status::invalid_input);

    // Géométrie colinéaire hors domaine
    expect_failure("antipode exclu", wd::solve_lambert_cpu(r1, Vector{-1, 0, 0}, 1.0, 1.0, arc), Status::unsupported_case);

    // Budget d'une seule itération : insuffisant pour atteindre la tolérance
    wd::LambertOptions limited;
    limited.max_iterations = 1;
    expect_failure("budget épuisé", wd::solve_lambert_cpu(r1, r2, 1.1, 1.0, arc, limited), Status::not_converged);

    std::cout << (ok ? "Lambert : tous les contrôles passent.\n" : "Lambert : échec.\n");
    return ok ? 0 : 1;
}