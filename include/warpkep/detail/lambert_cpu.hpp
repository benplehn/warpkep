#pragma once

// Entry point: solve_lambert_cpu<double>(r1, r2, dt, mu, arc, options).
//
// - Short arc follows r1 x r2; long arc follows the opposite normal.
// - Exact and nearly collinear geometries are explicitly unsupported.
// - No periapsis/collision constraint is imposed. No allocation or exception.
// - Numeric limits can still prevent evaluation for extreme input scales.
// - Scalar helpers require lambda and q from the SAME prepared geometry.
// - Success controls the time residual, not a uniform error bound on velocities.
// - Near-collinear input rounding and x quantization near -1 can limit accuracy; an unattainable time residual returns not_converged, never a false success.
//
// Reference: D. Izzo, Revisiting Lambert's Problem, equations 17--23, section 4.2.1 (logarithmic starter) and algorithm 1 (velocities).
// https://arxiv.org/abs/1403.2705 ; DOI 10.1007/s10569-014-9587-y
// Bracketing, bisection fallback and residual-based success are our own choices.

#include <warpkep/lambert_status.hpp>
#include <warpkep/detail/vector3.hpp>
#include <warpkep/detail/numeric_limits.hpp>
#include <warpkep/detail/math.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace warpkep::detail {

    enum class LambertArc {
        short_arc,
        long_arc,
    };

    struct LambertOptions {

        // Nombre max d'itérations du solveur de racine (borne le coût de calcul, à valider sur les cas tests)
        int max_iterations{64};

        // Tolérance relative sur le temps adimensionné : |T_calc - T_cible| <= tol * T_cible
        // Cible de précision sur le temps, pas une garantie sur l'erreur en vitesse
        double relative_tolerance{1e-12};
    };

    template <typename T>
    struct LambertResult {
        Vector3<T> departure_velocity;
        Vector3<T> arrival_velocity;
        LambertStatus status;
        int iterations;
    };

    template <typename T>
    struct LambertGeometry {
        T radius_departure;
        T radius_arrival;
        T chord;
        T semiperimeter;
        T q;            // c / s
        T lambda;       // paramètre de Lambert, signé selon l'arc
        T target_time;  // temps de vol adimensionné
        Vector3<T> radial_departure;
        Vector3<T> radial_arrival;
        Vector3<T> normal;
        Vector3<T> tangent_departure;
        Vector3<T> tangent_arrival;
    };

    template <typename T>
    struct LambertGeometryResult {
        LambertGeometry<T> geometry;
        LambertStatus status;
    };

    // Résultat d'échec : tous les champs flottants à NaN pour qu'aucune valeur ne soit exploitée par erreur
    template <typename T>
    inline LambertGeometryResult<T> lambert_geometry_failure(LambertStatus status) {
        const T nan = NumericLimits<T>::quiet_NaN();

        LambertGeometryResult<T> result{};
        result.status = status;

        auto& g = result.geometry;
        g.radius_departure = g.radius_arrival = g.chord = g.semiperimeter = nan;
        g.q = g.lambda = g.target_time = nan;

        for (std::size_t i = 0; i < 3; ++i) {
            g.radial_departure[i] = g.radial_arrival[i] = g.normal[i] = nan;
            g.tangent_departure[i] = g.tangent_arrival[i] = nan;
        }

        return result;
    }

    template <typename T>
    inline LambertGeometryResult<T> prepare_lambert_geometry(
        const Vector3<T>& departure, const Vector3<T>& arrival, T dt, T mu, LambertArc arc) {
        using Status = LambertStatus;
        const auto fail = [](Status status) { return lambert_geometry_failure<T>(status); };

        // 1. Validate inputs
        // mu et dt finis et strictement positifs, arc dans l'énumération
        const bool valid_arc = arc == LambertArc::short_arc || arc == LambertArc::long_arc;
        if (!std::isfinite(mu) || mu <= T{0} || !std::isfinite(dt) || dt <= T{0} || !valid_arc) {
            return fail(Status::invalid_input);
        }

        // Positions finies et non nulles
        bool departure_nonzero = false;
        bool arrival_nonzero = false;
        for (std::size_t i = 0; i < 3; ++i) {
            if (!std::isfinite(departure[i]) || !std::isfinite(arrival[i])) {
                return fail(Status::invalid_input);
            }
            departure_nonzero = departure_nonzero || departure[i] != T{0};
            arrival_nonzero = arrival_nonzero || arrival[i] != T{0};
        }
        if (!departure_nonzero || !arrival_nonzero) {
            return fail(Status::invalid_input);
        }

        // 2. Compute the triangle geometry
        LambertGeometry<T> g{};

        // Normes robustes au dépassement : norm3 ne forme jamais x² + y² + z² directement
        g.radius_departure = norm3(departure[0], departure[1], departure[2]);
        g.radius_arrival = norm3(arrival[0], arrival[1], arrival[2]);
        g.chord = norm3(arrival[0] - departure[0], arrival[1] - departure[1], arrival[2] - departure[2]);

        if (!std::isfinite(g.radius_departure) || g.radius_departure <= T{0}
            || !std::isfinite(g.radius_arrival) || g.radius_arrival <= T{0}
            || !std::isfinite(g.chord)) {
            return fail(Status::numerical_failure);
        }

        // Départ et arrivée confondus : hors domaine de ce prototype
        if (g.chord == T{0}) {
            return fail(Status::unsupported_case);
        }

        // Demi-périmètre : on divise avant de sommer pour éviter un dépassement intermédiaire
        g.semiperimeter = g.radius_departure / T{2} + g.radius_arrival / T{2} + g.chord / T{2};
        if (!std::isfinite(g.semiperimeter) || g.semiperimeter <= T{0}) {
            return fail(Status::numerical_failure);
        }

        // q = c / s est dans ]0, 1] en exact : on tolère un léger dépassement d'arrondi, puis on borne
        const T rounding_tolerance = T{64} * NumericLimits<T>::epsilon();
        g.q = g.chord / g.semiperimeter;
        if (!std::isfinite(g.q) || g.q <= T{0} || g.q > T{1} + rounding_tolerance) {
            return fail(Status::numerical_failure);
        }
        if (g.q > T{1}) { g.q = T{1}; }

        // 3. Build the transfer-plane basis
        // Vecteurs radiaux unitaires (normalisés avant le produit vectoriel)
        for (std::size_t i = 0; i < 3; ++i) {
            g.radial_departure[i] = departure[i] / g.radius_departure;
            g.radial_arrival[i] = arrival[i] / g.radius_arrival;
        }
        const auto& u1 = g.radial_departure;
        const auto& u2 = g.radial_arrival;

        const auto cross = [](const Vector3<T>& a, const Vector3<T>& b) {
            return Vector3<T>{
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            };
        };

        // w = u1 x u2 est normal au plan de transfert, avec ||w|| = |sin(theta)|
        const Vector3<T> w = cross(u1, u2);
        const T cross_norm = norm3(w[0], w[1], w[2]);
        if (!std::isfinite(cross_norm)) {
            return fail(Status::numerical_failure);
        }

        // Colinéarité : plan indéterminé ; quasi-colinéarité : plan mal conditionné
        // Ces géométries sont exclues du prototype
        const T collinearity_threshold = T{64} * NumericLimits<T>::epsilon();
        if (cross_norm <= collinearity_threshold) {
            return fail(Status::unsupported_case);
        }

        // Normale unitaire, inversée pour l'arc long
        const T orientation = arc == LambertArc::short_arc ? T{1} : T{-1};
        for (std::size_t i = 0; i < 3; ++i) {
            g.normal[i] = orientation * (w[i] / cross_norm);
        }

        // Tangentes aux extrémités, dans le sens du mouvement
        g.tangent_departure = cross(g.normal, u1);
        g.tangent_arrival = cross(g.normal, u2);

        // 4. Compute lambda
        // |lambda| = sqrt(r1 * r2) / s * cos(theta_court / 2)
        // cos(theta / 2) = ||u1 + u2|| / 2 : pas de soustraction de termes voisins près de theta = pi
        const T sqrt_s = std::sqrt(g.semiperimeter);
        const T half_angle_cosine = norm3(u1[0] + u2[0], u1[1] + u2[1], u1[2] + u2[2]) / T{2};

        // Racines séparées pour ne jamais former r1 * r2
        T lambda_magnitude = (std::sqrt(g.radius_departure) / sqrt_s)
                           * (std::sqrt(g.radius_arrival) / sqrt_s)
                           * half_angle_cosine;

        // |lambda| doit être dans [0, 1] à l'arrondi près
        if (!std::isfinite(lambda_magnitude) || lambda_magnitude < T{0}
            || lambda_magnitude > T{1} + rounding_tolerance) {
            return fail(Status::numerical_failure);
        }
        if (lambda_magnitude > T{1}) { lambda_magnitude = T{1}; }

        // Signe : positif pour l'arc court, négatif pour l'arc long
        g.lambda = orientation * lambda_magnitude;

        // 5. Compute the dimensionless time of flight
        // T = dt * sqrt(2 * mu / s^3), sans former mu / s ni s^3
        const T scaled_speed = std::sqrt(mu) / sqrt_s;
        const T scaled_duration = dt / g.semiperimeter;
        if (!std::isfinite(scaled_speed) || scaled_speed <= T{0}
            || !std::isfinite(scaled_duration) || scaled_duration <= T{0}) {
            return fail(Status::numerical_failure);
        }

        g.target_time = scaled_duration * scaled_speed * std::sqrt(T{2});
        if (!std::isfinite(g.target_time) || g.target_time <= T{0}) {
            return fail(Status::numerical_failure);
        }

        return {g, Status::success};
    }

    template <typename T>
    struct LambertTimeResult {
        T time;
        T y;
        LambertStatus status;
    };

    template <typename T>
    inline LambertTimeResult<T> evaluate_lambert_time(T x, T lambda, T q) {
        using Status = LambertStatus;
        const T nan = NumericLimits<T>::quiet_NaN();
        const auto fail = [nan](Status status) { return LambertTimeResult<T>{nan, nan, status}; };

        // 1. Validate the scalar domain (zero-revolution transfers)
        // x > -1, |lambda| <= 1, q dans ]0, 1]
        if (!std::isfinite(x) || x <= T{-1}
            || !std::isfinite(lambda) || std::abs(lambda) > T{1}
            || !std::isfinite(q) || q <= T{0} || q > T{1}) {
            return fail(Status::invalid_input);
        }

        // 2. Compute the auxiliary variables
        // y² = q + (lambda * x)², via hypot pour conserver q quand |lambda| -> 1
        const T lambda_x = lambda * x;
        const T y = std::hypot(std::sqrt(q), lambda_x);
        if (!std::isfinite(lambda_x) || !std::isfinite(y) || y <= T{0}) {
            return fail(Status::numerical_failure);
        }

        // eta = y - lambda * x, rationalisé en q / (y + lambda * x) quand la soustraction perdrait des chiffres
        T eta;
        if (lambda_x > T{0}) {
            const T denominator = y + lambda_x;
            if (!std::isfinite(denominator) || denominator <= T{0}) {
                return fail(Status::numerical_failure);
            }
            eta = q / denominator;
        } else {
            eta = y - lambda_x;
        }

        // delta = 1 - lambda, écrit q / (1 + lambda) pour éviter la cancellation près de lambda = 1
        const T delta = lambda >= T{0} ? q / (T{1} + lambda) : T{1} - lambda;

        if (!std::isfinite(eta) || eta <= T{0} || !std::isfinite(delta) || delta <= T{0}) {
            return fail(Status::numerical_failure);
        }

        T time;

        if (x == T{1}) {
            // 3. Exact parabolic limit
            // T = (2/3) * (1 - lambda³) = (2/3) * (1 - lambda) * (1 + lambda + lambda²)
            time = (T{2} / T{3}) * delta * (T{1} + lambda + lambda * lambda);

        } else if (std::abs(x - T{1}) <= T{1} / T{10}) {
            // 4. Battin's series near x = 1
            // Évite la cancellation de la formule générale ; largeur 0.1 à valider
            const T series_argument = (delta - x * eta) / T{2};
            if (!std::isfinite(series_argument) || std::abs(series_argument) >= T{1}) {
                return fail(Status::numerical_failure);
            }

            // F = somme des a_n, avec a_0 = 1 et a_(n+1) = a_n * S * (n + 3) / (n + 5/2)
            // Budget propre à la série, indépendant de celui du solveur
            constexpr int max_series_iterations = 64;
            const T epsilon = NumericLimits<T>::epsilon();

            T term = T{1};
            T sum = T{1};
            bool converged = false;

            for (int n = 0; n < max_series_iterations; ++n) {
                const T index = static_cast<T>(n);
                term *= series_argument * ((index + T{3}) / (index + T{5} / T{2}));
                sum += term;

                if (!std::isfinite(term) || !std::isfinite(sum)) {
                    return fail(Status::numerical_failure);
                }

                // Critère de troncature (ne borne pas l'erreur d'arrondi totale)
                if (std::abs(term) <= epsilon * std::abs(sum)) {
                    converged = true;
                    break;
                }
            }

            if (!converged) {
                return fail(Status::not_converged);
            }

            // T = (eta³ * Q + 4 * lambda * eta) / 2, avec Q = (4/3) * F
            const T Q = (T{4} / T{3}) * sum;
            time = (eta * eta * eta * Q + T{4} * lambda * eta) / T{2};

        } else {
            // 5. General elliptic / hyperbolic expression
            // u = 1 - x², factorisé pour garder la précision près de x = -1
            const T u = (T{1} - x) * (T{1} + x);
            const T z = std::sqrt(std::abs(u));
            if (!std::isfinite(u) || u == T{0} || !std::isfinite(z) || z <= T{0}) {
                return fail(Status::numerical_failure);
            }

            const T sine_argument = eta * z;
            if (!std::isfinite(sine_argument)) {
                return fail(Status::numerical_failure);
            }

            T psi;
            if (x < T{1}) {
                // Elliptique : sin(psi) = eta * z, cos(psi) = lambda + x * eta
                // atan2 garde le bon quadrant sans clamp d'acos
                const T cosine_argument = lambda + x * eta;
                if (!std::isfinite(cosine_argument)) {
                    return fail(Status::numerical_failure);
                }
                psi = std::atan2(sine_argument, cosine_argument);
            } else {
                // Hyperbolique : sinh(psi) = eta * z
                psi = std::asinh(sine_argument);
            }

            if (!std::isfinite(psi)) {
                return fail(Status::numerical_failure);
            }

            // T = (psi / z - x + lambda * y) / u, avec lambda * y - x = lambda * eta - q * x (préserve les petits q)
            time = (psi / z + lambda * eta - q * x) / u;
        }

        if (!std::isfinite(time) || time <= T{0}) {
            return fail(Status::numerical_failure);
        }

        return {time, y, Status::success};
    }

    template <typename T>
    struct LambertGuessResult {
        T x;
        LambertStatus status;
    };

    template <typename T>
    inline LambertGuessResult<T> initial_lambert_guess(T target_time, T lambda, T q) {
        using Status = LambertStatus;
        const T nan = NumericLimits<T>::quiet_NaN();
        const auto fail = [nan](Status status) { return LambertGuessResult<T>{nan, status}; };

        // 1. Validate inputs
        // Le temps cible doit être fini et strictement positif
        if (!std::isfinite(target_time) || target_time <= T{0}) {
            return fail(Status::invalid_input);
        }

        // 2. Evaluate the reference times T(0) and T(1)
        // On réutilise l'évaluateur de temps, qui valide aussi lambda et q
        const auto at_zero = evaluate_lambert_time(T{0}, lambda, q);
        if (at_zero.status != Status::success) {
            return fail(at_zero.status);
        }

        const auto at_one = evaluate_lambert_time(T{1}, lambda, q);
        if (at_one.status != Status::success) {
            return fail(at_one.status);
        }

        const T T0 = at_zero.time;
        const T T1 = at_one.time;

        // Pour M = 0, T est décroissant en x : on doit avoir T(0) > T(1) > 0
        if (!(T0 > T1) || T1 <= T{0}) {
            return fail(Status::numerical_failure);
        }

        T x0;

        if (target_time >= T0) {
            // 3. Long-duration elliptic starter
            // x0 = (T0 / T)^(2/3) - 1, via les logarithmes pour ne pas former un rapport extrême
            // expm1 garde la précision quand x0 est proche de 0
            const T exponent = (T{2} / T{3}) * (std::log(T0) - std::log(target_time));
            x0 = std::expm1(exponent);

            // Pour des durées très longues, l'arrondi peut donner x0 = -1 (exclu) : on décale juste vers l'intérieur
            // Ce n'est qu'un point de départ, pas une garantie de solution
            if (x0 == T{-1}) {
                x0 = std::nextafter(T{-1}, T{0});
            }

        } else if (target_time <= T1) {
            // 4. Hyperbolic starter, continuous at the parabolic time
            // x0 = 1 + A * (T1 - T) / T, avec A = (5/2) * T1 / (1 - lambda^5)
            // Le facteur (1 - lambda) commun à T1 et à 1 - lambda^5 est simplifié analytiquement :
            // A = (5/3) * (1 + lambda + lambda²) / (1 + lambda + lambda² + lambda³ + lambda⁴)
            const T lambda2 = lambda * lambda;
            const T polynomial2 = T{1} + lambda + lambda2;
            const T polynomial4 = (T{1} + lambda) * (T{1} + lambda2) + lambda2 * lambda2;

            const T coefficient = (T{5} / T{3}) * (polynomial2 / polynomial4);
            x0 = T{1} + coefficient * ((T1 - target_time) / target_time);

        } else {
            // 5. Intermediate elliptic starter
            // Interpolation linéaire dans le plan (log T, log(1 + x)) entre (T0, x = 0) et (T1, x = 1)
            const T log_T0 = std::log(T0);
            const T exponent = std::log(T{2}) * ((std::log(target_time) - log_T0) / (std::log(T1) - log_T0));
            x0 = std::expm1(exponent);
        }

        // Le point de départ doit rester dans le domaine x > -1
        if (!std::isfinite(x0) || x0 <= T{-1}) {
            return fail(Status::numerical_failure);
        }

        return {x0, Status::success};
    }

    template <typename T>
    struct LambertDerivativesResult {
        T time;
        T y;
        T first;
        T second;
        T third;
        LambertStatus status;
    };

    template <typename T>
    inline LambertDerivativesResult<T> evaluate_lambert_derivatives(T x, T lambda, T q) {
        using Status = LambertStatus;
        const T nan = NumericLimits<T>::quiet_NaN();
        const auto fail = [nan](Status status) {
            return LambertDerivativesResult<T>{nan, nan, nan, nan, nan, status};
        };

        // 1. Evaluate the time of flight
        // On réutilise l'évaluateur de temps : validation des entrées et calcul stable de T et y
        const auto evaluation = evaluate_lambert_time(x, lambda, q);
        if (evaluation.status != Status::success) {
            return fail(evaluation.status);
        }

        const T time = evaluation.time;
        const T y = evaluation.y;
        const T lambda2 = lambda * lambda;
        const T lambda_x = lambda * x;

        // eta = y - lambda * x, rationalisé quand la soustraction perdrait des chiffres
        const T eta = lambda_x > T{0} ? q / (y + lambda_x) : y - lambda_x;
        if (!std::isfinite(eta) || eta <= T{0}) {
            return fail(Status::numerical_failure);
        }

        T first, second, third;

        if (std::abs(x - T{1}) <= T{1} / T{10}) {
            // 2. Differentiate Battin's expression near x = 1
            // delta = 1 - lambda, sans cancellation près de lambda = 1
            const T delta = lambda >= T{0} ? q / (T{1} + lambda) : T{1} - lambda;

            // Dérivées de eta par rapport à x
            const T eta1 = -lambda * (eta / y);
            const T eta2 = lambda2 * (q / y / y) / y;
            const T eta3 = -T{3} * eta2 * (lambda / y) * (lambda_x / y);

            // S = (delta - x * eta) / 2 et ses dérivées par rapport à x
            const T S = (delta - x * eta) / T{2};
            const T S1 = -(eta + x * eta1) / T{2};
            const T S2 = -(T{2} * eta1 + x * eta2) / T{2};
            const T S3 = -(T{3} * eta2 + x * eta3) / T{2};

            if (!std::isfinite(S) || std::abs(S) >= T{1}
                || !std::isfinite(S1) || !std::isfinite(S2) || !std::isfinite(S3)) {
                return fail(Status::numerical_failure);
            }

            // F(S), F'(S), F''(S), F'''(S) accumulés simultanément (indice k = ordre de dérivation en S)
            T term[4]{T{1}, T{0}, T{0}, T{0}};
            T sum[4]{T{1}, T{0}, T{0}, T{0}};

            constexpr int max_series_iterations = 64;
            const T epsilon = NumericLimits<T>::epsilon();
            bool converged = false;

            for (int n = 0; n < max_series_iterations; ++n) {
                const T index = static_cast<T>(n);
                const T factor = (index + T{3}) / (index + T{5} / T{2});

                // Récurrence de Leibniz, sans division par S (sûre quand S -> 0)
                // Ordre décroissant pour utiliser les termes de l'itération précédente
                for (int k = 3; k >= 1; --k) {
                    term[k] = factor * (S * term[k] + static_cast<T>(k) * term[k - 1]);
                }
                term[0] *= factor * S;

                // On parcourt au moins les termes qui contribuent à la dérivée troisième
                bool small_terms = n >= 3;

                for (int k = 0; k < 4; ++k) {
                    sum[k] += term[k];

                    if (!std::isfinite(term[k]) || !std::isfinite(sum[k])) {
                        return fail(Status::numerical_failure);
                    }

                    small_terms = small_terms && std::abs(term[k]) <= epsilon * std::abs(sum[k]);
                }

                if (small_terms) {
                    converged = true;
                    break;
                }
            }

            if (!converged) {
                return fail(Status::not_converged);
            }

            // Q = (4/3) * F(S(x)) : dérivées par rapport à x via la règle de chaîne (Faà di Bruno)
            const T scale = T{4} / T{3};
            const T Q = scale * sum[0];
            const T Q1 = scale * sum[1] * S1;
            const T Q2 = scale * (sum[2] * S1 * S1 + sum[1] * S2);
            const T Q3 = scale * (sum[3] * S1 * S1 * S1 + T{3} * sum[2] * S1 * S2 + sum[1] * S3);

            // A = eta³ et ses dérivées par rapport à x
            const T A = eta * eta * eta;
            const T A1 = T{3} * eta * eta * eta1;
            const T A2 = T{6} * eta * eta1 * eta1 + T{3} * eta * eta * eta2;
            const T A3 = T{6} * eta1 * eta1 * eta1 + T{18} * eta * eta1 * eta2 + T{3} * eta * eta * eta3;

            // T(x) = A * Q / 2 + 2 * lambda * eta, dérivé par Leibniz
            first = (A1 * Q + A * Q1) / T{2} + T{2} * lambda * eta1;
            second = (A2 * Q + T{2} * A1 * Q1 + A * Q2) / T{2} + T{2} * lambda * eta2;
            third = (A3 * Q + T{3} * A2 * Q1 + T{3} * A1 * Q2 + A * Q3) / T{2} + T{2} * lambda * eta3;

        } else {
            // 3. Izzo's derivative recurrences away from the parabolic region
            // u = 1 - x², factorisé pour la précision près de x = -1
            const T u = (T{1} - x) * (T{1} + x);
            const T q_over_y2 = q / y / y;
            const T ratio = lambda / y;

            // Termes correctifs : 2 * q * lambda³ / y³ et 6 * q * lambda⁵ * x / y⁵
            const T correction2 = T{2} * q_over_y2 * lambda2 * ratio;
            const T correction3 = T{6} * q_over_y2 * lambda2 * ratio * ratio * (lambda_x / y);

            // Identité stable : 1 - lambda³ * x / y = q + lambda² * eta / y
            first = (T{3} * time * x - T{2} * (q + lambda2 * (eta / y))) / u;
            second = (T{3} * time + T{5} * x * first + correction2) / u;
            third = (T{7} * x * second + T{8} * first - correction3) / u;
        }

        // 4. Check the results
        // Pour M = 0, T doit être strictement décroissant en x
        if (!std::isfinite(first) || !std::isfinite(second) || !std::isfinite(third) || first >= T{0}) {
            return fail(Status::numerical_failure);
        }

        return {time, y, first, second, third, Status::success};
    }

    template <typename T>
    struct LambertBracketResult {
        T lower;
        T upper;
        LambertStatus status;
    };

    // En cas de succès : bornes finies et évaluables, -1 < lower <= upper et T(lower) >= T_cible >= T(upper)
    // lower == upper signifie que la racine tombe exactement sur un point échantillonné
    // Les évaluations d'expansion ne comptent pas comme itérations du solveur
    template <typename T>
    inline LambertBracketResult<T> bracket_lambert_root(T target_time, T lambda, T q) {
        using Status = LambertStatus;
        const T nan = NumericLimits<T>::quiet_NaN();
        const auto fail = [nan](Status status) { return LambertBracketResult<T>{nan, nan, status}; };

        constexpr int max_bracket_expansions = 64;

        // 1. Validate inputs
        // Le temps cible doit être fini et strictement positif
        if (!std::isfinite(target_time) || target_time <= T{0}) {
            return fail(Status::invalid_input);
        }

        // 2. Compare the target with T(0)
        const auto at_zero = evaluate_lambert_time(T{0}, lambda, q);
        if (at_zero.status != Status::success) {
            return fail(at_zero.status);
        }

        // Racine exacte en x = 0
        if (target_time == at_zero.time) {
            return {T{0}, T{0}, Status::success};
        }

        if (target_time > at_zero.time) {
            // 3. Expand towards x = -1 for long durations
            // T diverge quand x -> -1+, mais x = -1 n'est jamais évalué : on s'arrête au dernier flottant au-dessus
            T lower = -T{1} / T{2};
            T upper = T{0};

            for (int expansion = 0; expansion < max_bracket_expansions; ++expansion) {
                const auto evaluation = evaluate_lambert_time(lower, lambda, q);
                if (evaluation.status != Status::success) {
                    return fail(evaluation.status);
                }

                // T(lower) >= T_cible : l'encadrement est établi
                if (evaluation.time >= target_time) {
                    return {lower, upper, Status::success};
                }

                // Sinon on divise par deux la distance entre lower et -1
                upper = lower;
                T candidate = (lower - T{1}) / T{2};
                if (candidate <= T{-1}) {
                    candidate = std::nextafter(T{-1}, T{0});
                }

                // Plus aucun flottant entre -1 et lower : temps cible inatteignable en FP64
                if (!(candidate < lower)) {
                    return fail(Status::not_converged);
                }
                lower = candidate;
            }

            return fail(Status::not_converged);
        }

        // 4. Compare the target with T(1)
        const auto at_one = evaluate_lambert_time(T{1}, lambda, q);
        if (at_one.status != Status::success) {
            return fail(at_one.status);
        }

        // Racine exacte en x = 1 (transfert parabolique)
        if (target_time == at_one.time) {
            return {T{1}, T{1}, Status::success};
        }

        // T(1) < T_cible < T(0) : la racine est dans ]0, 1[
        if (target_time > at_one.time) {
            return {T{0}, T{1}, Status::success};
        }

        // 5. Expand towards +infinity for short (hyperbolic) durations
        // T décroît vers 0 quand x -> +inf ; le starter accélère la recherche,
        // mais seules des évaluations finies établissent l'encadrement
        T lower = T{1};
        T upper = T{2};
        const auto guess = initial_lambert_guess(target_time, lambda, q);
        if (guess.status == Status::success && guess.x > upper) {
            upper = guess.x;
        }

        for (int expansion = 0; expansion < max_bracket_expansions; ++expansion) {
            const auto evaluation = evaluate_lambert_time(upper, lambda, q);
            if (evaluation.status != Status::success) {
                return fail(evaluation.status);
            }

            // T(upper) <= T_cible : l'encadrement est établi
            if (evaluation.time <= target_time) {
                return {lower, upper, Status::success};
            }

            // Sinon on double la borne supérieure
            lower = upper;
            const T candidate = upper * T{2};
            if (!std::isfinite(candidate)) {
                return fail(Status::numerical_failure);
            }
            if (!(candidate > upper)) {
                return fail(Status::not_converged);
            }
            upper = candidate;
        }

        return fail(Status::not_converged);
    }

    template <typename T>
    struct LambertRootResult {
        T x;
        T y;
        T time;
        LambertStatus status;
        int iterations;
    };

    // Succès uniquement si |T(x) - T_cible| / T_cible <= tolérance
    // iterations compte les évaluations de la boucle de raffinement, hors encadrement,
    // vérification des bornes et termes de la série hypergéométrique
    template <typename T>
    inline LambertRootResult<T> solve_lambert_root(T target_time, T lambda, T q, const LambertOptions& options = {}) {
        using Status = LambertStatus;
        const T nan = NumericLimits<T>::quiet_NaN();
        const auto fail = [nan](Status status, int iterations = 0) {
            return LambertRootResult<T>{nan, nan, nan, status, iterations};
        };

        // 1. Validate options
        // Budget d'itérations strictement positif, tolérance finie dans ]0, 1[
        const T tolerance = static_cast<T>(options.relative_tolerance);
        if (options.max_iterations <= 0 || !std::isfinite(tolerance) || tolerance <= T{0} || tolerance >= T{1}) {
            return fail(Status::invalid_input);
        }

        // 2. Bracket the root
        const auto bracket = bracket_lambert_root(target_time, lambda, q);
        if (bracket.status != Status::success) {
            return fail(bracket.status);
        }
        T lower = bracket.lower;
        T upper = bracket.upper;

        // Test de résidu : soustraction avant division pour garder un petit résidu
        // et éviter le sous-dépassement de tolerance * target_time pour les vols très courts
        const auto accepted = [target_time, tolerance](T time) {
            return std::abs(time - target_time) / target_time <= tolerance;
        };

        // 3. Check the bracket endpoints
        // Une borne peut déjà satisfaire la tolérance
        const T endpoints[2]{lower, upper};
        for (const T endpoint : endpoints) {
            const auto evaluation = evaluate_lambert_time(endpoint, lambda, q);
            if (evaluation.status != Status::success) {
                return fail(evaluation.status);
            }
            if (accepted(evaluation.time)) {
                return {endpoint, evaluation.y, evaluation.time, Status::success, 0};
            }
        }

        // Bornes confondues sans résidu acceptable : rien à raffiner
        if (!(lower < upper)) {
            return fail(Status::not_converged);
        }

        // 4. Choose the starting point
        // Starter d'Izzo s'il tombe strictement dans l'encadrement, sinon milieu
        T x = lower + (upper - lower) / T{2};
        const auto guess = initial_lambert_guess(target_time, lambda, q);
        if (guess.status == Status::success && guess.x > lower && guess.x < upper) {
            x = guess.x;
        }

        // 5. Safeguarded Householder iterations
        for (int index = 0; index < options.max_iterations; ++index) {
            const int iteration = index + 1;
            const auto derivatives = evaluate_lambert_derivatives(x, lambda, q);

            T time;
            T y;
            if (derivatives.status == Status::success) {
                time = derivatives.time;
                y = derivatives.y;
            } else {
                // Échec des dérivées : on évalue seulement T pour pouvoir faire une bissection
                const auto evaluation = evaluate_lambert_time(x, lambda, q);
                if (evaluation.status != Status::success) {
                    return fail(evaluation.status, iteration);
                }
                time = evaluation.time;
                y = evaluation.y;
            }

            if (accepted(time)) {
                return {x, y, time, Status::success, iteration};
            }

            // Mise à jour de l'encadrement (T décroissant en x)
            if (time > target_time) {
                lower = x;
            } else {
                upper = x;
            }

            // Pas de Householder d'ordre 4, normalisé par T'³ pour ne pas former T'² ou T'³
            // x_new = x - h * (1 - a / 2) / (1 - a + b / 6), avec h = f / T', a = h * T'' / T', b = h² * T''' / T'
            T candidate = nan;
            if (derivatives.status == Status::success) {
                const T residual = time - target_time;
                const T h = residual / derivatives.first;
                const T a = h * (derivatives.second / derivatives.first);
                const T b = h * (h * (derivatives.third / derivatives.first));
                const T denominator = T{1} - a + b / T{6};

                if (std::isfinite(h) && std::isfinite(a) && std::isfinite(b)
                    && std::isfinite(denominator) && denominator != T{0}) {
                    candidate = x - h * ((T{1} - a / T{2}) / denominator);
                }
            }

            // Repli sur la bissection si le pas sort de l'encadrement ou stagne
            if (!std::isfinite(candidate) || candidate <= lower || candidate >= upper || candidate == x) {
                candidate = lower + (upper - lower) / T{2};
            }

            // Bornes adjacentes : plus aucun flottant intérieur, on n'accepte jamais un petit pas à la place du résidu
            if (!(candidate > lower && candidate < upper) || candidate == x) {
                return fail(Status::not_converged, iteration);
            }
            x = candidate;
        }

        return fail(Status::not_converged, options.max_iterations);
    }

    // Résultat d'échec : six composantes de vitesse à NaN
    template <typename T>
    inline LambertResult<T> lambert_failure(LambertStatus status, int iterations = 0) {
        const T nan = NumericLimits<T>::quiet_NaN();
        return {{nan, nan, nan}, {nan, nan, nan}, status, iterations};
    }

    // Préconditions : g vient d'une géométrie préparée avec succès, (x, y) d'une résolution réussie
    // pour cette même géométrie, et mu est le même que celui utilisé pour la préparer
    template <typename T>
    inline LambertResult<T> reconstruct_lambert_velocities(
        const LambertGeometry<T>& g, T x, T y, T mu, int iterations = 0) {
        using Status = LambertStatus;
        const auto fail = [iterations](Status status) { return lambert_failure<T>(status, iterations); };

        // 1. Validate inputs
        // mu > 0, x > -1, y > 0, tous finis
        if (!std::isfinite(mu) || mu <= T{0} || !std::isfinite(x) || x <= T{-1} || !std::isfinite(y) || y <= T{0}) {
            return fail(Status::invalid_input);
        }

        // 2. Compute rho and sigma
        // rho = (r1 - r2) / c, dans [-1, 1] à l'arrondi près
        const T rounding_tolerance = T{64} * NumericLimits<T>::epsilon();
        T rho = (g.radius_departure - g.radius_arrival) / g.chord;
        if (!std::isfinite(rho) || std::abs(rho) > T{1} + rounding_tolerance) {
            return fail(Status::numerical_failure);
        }
        if (rho > T{1}) { rho = T{1}; }
        if (rho < T{-1}) { rho = T{-1}; }

        // sigma = sqrt(1 - rho²) = sqrt(r1 * r2) * ||u2 - u1|| / c
        // Calculé depuis la géométrie pour garder une petite composante tangentielle quand |rho| -> 1
        const T direction_difference = norm3(
            g.radial_arrival[0] - g.radial_departure[0],
            g.radial_arrival[1] - g.radial_departure[1],
            g.radial_arrival[2] - g.radial_departure[2]);
        const T sqrt_chord = std::sqrt(g.chord);
        const T sqrt_r1 = std::sqrt(g.radius_departure);
        const T sqrt_r2 = std::sqrt(g.radius_arrival);

        T sigma = (sqrt_r1 / sqrt_chord) * (sqrt_r2 / sqrt_chord) * direction_difference;
        if (!std::isfinite(sigma) || sigma <= T{0} || sigma > T{1} + rounding_tolerance) {
            return fail(Status::numerical_failure);
        }
        if (sigma > T{1}) { sigma = T{1}; }

        // 3. Compute the velocity scales
        // gamma / r_i = sqrt(mu * s / 2) / r_i, sans former mu * s ni gamma
        const T sqrt_mu = std::sqrt(mu);
        const T sqrt_s = std::sqrt(g.semiperimeter);
        const T sqrt_two = std::sqrt(T{2});
        const T scale1 = (sqrt_mu / sqrt_r1) * (sqrt_s / sqrt_r1) / sqrt_two;
        const T scale2 = (sqrt_mu / sqrt_r2) * (sqrt_s / sqrt_r2) / sqrt_two;

        // 4. Compute the radial and tangential components (Izzo, algorithm 1)
        // eta = y - lambda * x et zeta = y + lambda * x, chacun rationalisé du côté où il y aurait cancellation
        const T lambda_x = g.lambda * x;
        const T eta = lambda_x > T{0} ? g.q / (y + lambda_x) : y - lambda_x;
        const T zeta = lambda_x < T{0} ? g.q / (y - lambda_x) : y + lambda_x;

        // lambda * y - x et lambda * y + x, réécrits pour préserver les petits q
        const T difference = g.lambda * eta - g.q * x;
        const T sum = g.lambda * zeta + g.q * x;

        const T radial1 = scale1 * (difference - rho * sum);
        const T radial2 = -scale2 * (difference + rho * sum);
        const T tangential1 = scale1 * sigma * zeta;
        const T tangential2 = scale2 * sigma * zeta;

        // Composantes tangentielles strictement positives : mouvement dans le sens de la normale choisie
        if (!std::isfinite(scale1) || scale1 <= T{0} || !std::isfinite(scale2) || scale2 <= T{0}
            || !std::isfinite(radial1) || !std::isfinite(radial2)
            || !std::isfinite(tangential1) || tangential1 <= T{0}
            || !std::isfinite(tangential2) || tangential2 <= T{0}) {
            return fail(Status::numerical_failure);
        }

        // 5. Assemble the velocity vectors
        // v = v_r * u_r + v_t * u_t aux deux extrémités
        Vector3<T> departure_velocity{};
        Vector3<T> arrival_velocity{};
        for (std::size_t i = 0; i < 3; ++i) {
            departure_velocity[i] = radial1 * g.radial_departure[i] + tangential1 * g.tangent_departure[i];
            arrival_velocity[i] = radial2 * g.radial_arrival[i] + tangential2 * g.tangent_arrival[i];

            if (!std::isfinite(departure_velocity[i]) || !std::isfinite(arrival_velocity[i])) {
                return fail(Status::numerical_failure);
            }
        }

        return {departure_velocity, arrival_velocity, Status::success, iterations};
    }

    // Point d'entrée CPU scalaire, FP64 uniquement (le template garde les conventions de type)
    // Unités cohérentes : positions en L, dt en temps, mu en L³/temps², vitesses retournées en L/temps
    // dt > 0, même repère inertiel centré sur le corps attracteur, M = 0
    // relative_tolerance finie dans ]0, 1[, max_iterations > 0
    // Tout échec renvoie six NaN ; pas de statut « pas de solution physique » pour M = 0
    template <typename T>
    inline LambertResult<T> solve_lambert_cpu(
        const Vector3<T>& departure, const Vector3<T>& arrival, T dt, T mu, LambertArc arc,
        const LambertOptions& options = {}) {
        static_assert(std::is_same_v<T, double>, "Lambert CPU supports FP64 only");
        using Status = LambertStatus;

        // 1. Validate options
        if (options.max_iterations <= 0 || !std::isfinite(options.relative_tolerance)
            || options.relative_tolerance <= 0.0 || options.relative_tolerance >= 1.0) {
            return lambert_failure<T>(Status::invalid_input);
        }

        // 2. Prepare the geometry
        const auto preparation = prepare_lambert_geometry(departure, arrival, dt, mu, arc);
        if (preparation.status != Status::success) {
            return lambert_failure<T>(preparation.status);
        }
        const auto& g = preparation.geometry;

        // 3. Solve T(x) = target_time
        const auto root = solve_lambert_root(g.target_time, g.lambda, g.q, options);
        if (root.status != Status::success) {
            return lambert_failure<T>(root.status, root.iterations);
        }

        // 4. Reconstruct the velocities
        return reconstruct_lambert_velocities(g, root.x, root.y, mu, root.iterations);
    }

}  // namespace warpkep::detail