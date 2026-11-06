#pragma once

#include <warpkep/detail/lambert_cpu.hpp>

// Lambert CPU FP64, M >= 0, sans allocation ni exception
// Entrée principale : solve_lambert_multi_rev_cpu<double>(r1,r2,dt,mu,arc,M)

// Référence : D. Izzo, https://arxiv.org/abs/1403.2705, équations 18,22,31

namespace warpkep::detail {

enum class LambertBranch { single, left, right };

struct LambertMultiRevOptions {
    LambertOptions roots{};
    int minimum_max_iterations{64};
    double minimum_x_tolerance{1e-12}; // Largeur absolue de l'encadrement en x
    double threshold_relative_tolerance{1e-12}; // Échelle temporelle adimensionnée
};

template <typename T>
struct LambertMinimumResult {
    T x;
    T time;
    T lower;
    T upper;
    T curvature;
    T time_uncertainty; // Diagnostic numérique, pas une borne formelle démontrée
    LambertStatus status;
    int iterations;
    bool precision_limited; // Encadrement réduit à deux flottants adjacents
};

template <typename T>
struct LambertMultiRevSolution {
    int revolutions;
    LambertArc arc;
    LambertBranch branch;
    T x;
    LambertResult<T> result;
};

template <typename T>
struct LambertMultiRevResult {
    LambertMultiRevSolution<T> solutions[2];
    int slot_count;     // M=0 : un slot single ; M>0 : slots left puis right
    int solution_count; // Nombre de slots dont result.status == success
    bool complete;     // false si échec numérique ou multiplicité non résolue
    LambertStatus status;
    LambertMinimumResult<T> minimum; // Non applicable à M=0 (unsupported_case)
    T threshold; // Demi-largeur de la bande du seuil en temps adimensionné
};

namespace lambert_multi_detail {

inline bool valid_options(const LambertMultiRevOptions& options) {
    return options.roots.max_iterations > 0
        && std::isfinite(options.roots.relative_tolerance)
        && options.roots.relative_tolerance > 0.0
        && options.roots.relative_tolerance < 1.0
        && options.minimum_max_iterations > 0
        && std::isfinite(options.minimum_x_tolerance)
        && options.minimum_x_tolerance > 0.0 && options.minimum_x_tolerance < 1.0
        && std::isfinite(options.threshold_relative_tolerance)
        && options.threshold_relative_tolerance > 0.0
        && options.threshold_relative_tolerance < 1.0;
}

template <typename T>
inline LambertMinimumResult<T> minimum_failure(LambertStatus status, int iterations = 0) {
    const T nan = NumericLimits<T>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, status, iterations, false};
}

template <typename T>
inline LambertRootResult<T> root_failure(LambertStatus status, int iterations = 0) {
    const T nan = NumericLimits<T>::quiet_NaN();
    return {nan, nan, nan, status, iterations};
}

template <typename T>
inline LambertMultiRevResult<T> request_failure(
    int revolutions, LambertArc arc, LambertStatus status
) {
    const T nan = NumericLimits<T>::quiet_NaN();
    LambertMultiRevResult<T> out{};
    out.slot_count = revolutions == 0 ? 1 : 2;
    out.solution_count = 0;
    out.complete = status == LambertStatus::no_solution;
    out.status = status;
    out.minimum = minimum_failure<T>(LambertStatus::unsupported_case);
    out.threshold = nan;
    for (int i = 0; i < 2; ++i) {
        const auto branch = revolutions == 0 ? LambertBranch::single
            : (i == 0 ? LambertBranch::left : LambertBranch::right);
        out.solutions[i] = {revolutions, arc, branch, nan, lambert_failure<T>(status)};
    }
    return out;
}

// M*pi/u^(3/2), sans pow ; la géométrie scalaire doit être cohérente.
template <typename T>
inline T revolution_time(T x, int revolutions) {
    const T u = (T{1} - x) * (T{1} + x);
    return (static_cast<T>(revolutions) * std::acos(T{-1}) / u) / std::sqrt(u);
}

} // namespace lambert_multi_detail

// Surcharge sans valeur par défaut : aucun changement des appels lambert mono
// lambda et q doivent provenir de la même préparation géométrique
template <typename T>
inline LambertTimeResult<T> evaluate_lambert_time(T x, T lambda, T q, int revolutions) {
    const T nan = NumericLimits<T>::quiet_NaN();
    if (revolutions < 0 || (revolutions > 0 && (!std::isfinite(x) || x <= T{-1} || x >= T{1}))) {
        return {nan, nan, LambertStatus::invalid_input};
    }
    const auto base = evaluate_lambert_time(x, lambda, q);
    if (revolutions == 0 || base.status != LambertStatus::success) {
        return base;
    }
    const T extra = lambert_multi_detail::revolution_time(x, revolutions);
    const T time = base.time + extra;
    if (!std::isfinite(extra) || extra <= T{0} || !std::isfinite(time) || time <= T{0}) {
        return {nan, nan, LambertStatus::numerical_failure};
    }
    return {time, base.y, LambertStatus::success};
}

template <typename T>
inline LambertDerivativesResult<T> evaluate_lambert_derivatives(
    T x, T lambda, T q, int revolutions
) {
    const T nan = NumericLimits<T>::quiet_NaN();
    const auto fail = [nan](LambertStatus status) {
        return LambertDerivativesResult<T>{nan, nan, nan, nan, nan, status};
    };
    if (revolutions < 0 || (revolutions > 0 && (!std::isfinite(x) || x <= T{-1} || x >= T{1}))) {
        return fail(LambertStatus::invalid_input);
    }
    const auto base = evaluate_lambert_derivatives(x, lambda, q);
    if (revolutions == 0 || base.status != LambertStatus::success) {
        return base;
    }

    const T u = (T{1} - x) * (T{1} + x);
    const T extra = lambert_multi_detail::revolution_time(x, revolutions);
    const T x2 = x * x;
    const T first = base.first + T{3} * x * (extra / u);
    const T second = base.second + T{3} * (T{1} + T{4} * x2) * (extra / u / u);
    const T third = base.third + T{15} * x * (T{3} + T{4} * x2) * (extra / u / u / u);
    const T time = base.time + extra;

    if (!std::isfinite(extra) || extra <= T{0} || !std::isfinite(time)
        || !std::isfinite(first) || !std::isfinite(second) || !std::isfinite(third)) {
        return fail(LambertStatus::numerical_failure);
    }
    // Ici, first peut être négatif, nul OU positif.
    return {time, base.y, first, second, third, LambertStatus::success};
}

namespace lambert_multi_detail {

template <typename T>
inline LambertMinimumResult<T> finish_minimum(
    T lambda, T q, int revolutions, T lower, T upper,
    int iterations, bool precision_limited
) {
    const T x = lower + (upper - lower) / T{2};
    const auto a = evaluate_lambert_derivatives(lower, lambda, q, revolutions);
    const auto b = evaluate_lambert_derivatives(upper, lambda, q, revolutions);
    const auto center = evaluate_lambert_derivatives(x, lambda, q, revolutions);
    if (a.status != LambertStatus::success || b.status != LambertStatus::success
        || center.status != LambertStatus::success || a.first > T{0} || b.first < T{0}
        || center.second <= T{0}) {
        return minimum_failure<T>(LambertStatus::numerical_failure, iterations);
    }
    const T gradient = std::abs(a.first) > std::abs(b.first) ? std::abs(a.first) : std::abs(b.first);
    const T endpoint_spread = std::abs(a.time - center.time) > std::abs(b.time - center.time)
        ? std::abs(a.time - center.time) : std::abs(b.time - center.time);
    // Diagnostic conservateur dans nos cas, sans prétention de borne rigoureuse
    // pour toute géométrie. La classification ajoute encore sa propre marge
    const T uncertainty = endpoint_spread + gradient * (upper - lower)
        + T{64} * NumericLimits<T>::epsilon() * center.time;
    if (!std::isfinite(uncertainty)) {
        return minimum_failure<T>(LambertStatus::numerical_failure, iterations);
    }
    return {x, center.time, lower, upper, center.second, uncertainty,
            LambertStatus::success, iterations, precision_limited};
}

} // namespace lambert_multi_detail

// Minimum de T_M : Halley sur g=T'_M, avec g'=T''_M et g''=T'''_M
// Succès : encadrement signé de largeur <= minimum_x_tolerance, ou flottants
// adjacents (precision_limited=true). Le seul petit pas de Halley ne suffit pas
// iterations compte les raffinements, hors expansion et sondages des bornes
template <typename T>
inline LambertMinimumResult<T> find_lambert_minimum(
    T lambda, T q, int revolutions, const LambertMultiRevOptions& options = {}
) {
    using Status = LambertStatus;
    namespace md = lambert_multi_detail;
    if (revolutions <= 0 || !md::valid_options(options)) {
        return md::minimum_failure<T>(Status::invalid_input);
    }
    const auto at_zero = evaluate_lambert_derivatives(T{0}, lambda, q, revolutions);
    if (at_zero.status != Status::success) {
        return md::minimum_failure<T>(at_zero.status);
    }
    if (!(at_zero.first < T{0})) {
        return md::minimum_failure<T>(Status::numerical_failure);
    }

    T lower = T{0};
    T upper = T{1} / T{2};
    bool bracket_found = false;
    for (int expansion = 0; expansion < 64; ++expansion) {
        const auto evaluation = evaluate_lambert_derivatives(upper, lambda, q, revolutions);
        if (evaluation.status != Status::success) {
            return md::minimum_failure<T>(evaluation.status);
        }
        if (evaluation.first >= T{0}) {
            bracket_found = true;
            break;
        }
        lower = upper;
        T next = upper + (T{1} - upper) / T{2};
        if (next >= T{1}) { next = std::nextafter(T{1}, T{0}); }
        if (!(next > upper)) { break; }
        upper = next;
    }
    if (!bracket_found) {
        return md::minimum_failure<T>(Status::not_converged);
    }

    const T tolerance = static_cast<T>(options.minimum_x_tolerance);
    T x = lower + (upper - lower) / T{2};
    for (int index = 0; index < options.minimum_max_iterations; ++index) {
        const int iteration = index + 1;
        const auto evaluation = evaluate_lambert_derivatives(x, lambda, q, revolutions);
        if (evaluation.status != Status::success || evaluation.second <= T{0}) {
            return md::minimum_failure<T>(Status::numerical_failure, iteration);
        }

        // Quand Halley est déjà près du point stationnaire, établir réellement
        // un petit encadrement signé autour de lui, au lieu de certifier un pas
        const T correction = evaluation.first / evaluation.second;
        if (std::isfinite(correction) && std::abs(correction) <= tolerance / T{8}) {
            T a = x - tolerance / T{4};
            T b = x + tolerance / T{4};
            if (a < lower) { a = lower; }
            if (b > upper) { b = upper; }
            if (a < b && a <= x && x <= b) {
                const auto left = evaluate_lambert_derivatives(a, lambda, q, revolutions);
                const auto right = evaluate_lambert_derivatives(b, lambda, q, revolutions);
                if (left.status == Status::success && right.status == Status::success
                    && left.first <= T{0} && right.first >= T{0}
                    && b - a <= tolerance) {
                    return md::finish_minimum(lambda, q, revolutions, a, b, iteration, false);
                }
            }
        }

        if (evaluation.first < T{0}) { lower = x; }
        else if (evaluation.first > T{0}) { upper = x; }

        const bool adjacent = std::nextafter(lower, upper) >= upper;
        if (upper - lower <= tolerance || adjacent) {
            return md::finish_minimum(lambda, q, revolutions, lower, upper, iteration, adjacent);
        }

        T candidate = NumericLimits<T>::quiet_NaN();
        // Halley normalisé : h / (1 - h*g''/(2*g')).
        const T h = evaluation.first / evaluation.second;
        const T denominator = T{1} - (h / T{2}) * (evaluation.third / evaluation.second);
        if (std::isfinite(h) && std::isfinite(denominator) && denominator != T{0}) {
            candidate = x - h / denominator;
        }
        if (!std::isfinite(candidate) || candidate <= lower || candidate >= upper || candidate == x) {
            candidate = lower + (upper - lower) / T{2};
        }
        // g évalué exactement à zéro mais localisation encore trop large :
        // sonder un côté du point stationnaire pour continuer à resserrer
        if (candidate == x) {
            candidate = lower + (x - lower) / T{2};
            if (!(candidate > lower && candidate < upper) || candidate == x) {
                candidate = x + (upper - x) / T{2};
            }
        }
        if (!(candidate > lower && candidate < upper) || candidate == x) {
            return md::minimum_failure<T>(Status::not_converged, iteration);
        }
        x = candidate;
    }
    return md::minimum_failure<T>(Status::not_converged, options.minimum_max_iterations);
}

namespace lambert_multi_detail {

template <typename T>
inline LambertBracketResult<T> branch_bracket(
    T target, T lambda, T q, int revolutions, LambertBranch branch,
    const LambertMinimumResult<T>& minimum
) {
    const T nan = NumericLimits<T>::quiet_NaN();
    const auto fail = [nan](LambertStatus status) {
        return LambertBracketResult<T>{nan, nan, status};
    };
    // Les bords du minimum sont des bornes INTERIEURES différentes
    T inner = branch == LambertBranch::left ? minimum.lower : minimum.upper;
    const auto at_inner = evaluate_lambert_time(inner, lambda, q, revolutions);
    if (at_inner.status != LambertStatus::success) { return fail(at_inner.status); }
    if (!(at_inner.time < target)) { return fail(LambertStatus::near_minimum); }

    T outer = branch == LambertBranch::left ? -T{1} / T{2} : (T{1} + inner) / T{2};
    for (int expansion = 0; expansion < 64; ++expansion) {
        const auto evaluation = evaluate_lambert_time(outer, lambda, q, revolutions);
        if (evaluation.status != LambertStatus::success) { return fail(evaluation.status); }
        if (evaluation.time >= target) {
            return branch == LambertBranch::left
                ? LambertBracketResult<T>{outer, inner, LambertStatus::success}
                : LambertBracketResult<T>{inner, outer, LambertStatus::success};
        }
        inner = outer;
        T next;
        if (branch == LambertBranch::left) {
            next = (outer - T{1}) / T{2};
            if (next <= T{-1}) { next = std::nextafter(T{-1}, T{0}); }
            if (!(next < outer)) { return fail(LambertStatus::not_converged); }
        } else {
            next = outer + (T{1} - outer) / T{2};
            if (next >= T{1}) { next = std::nextafter(T{1}, T{0}); }
            if (!(next > outer)) { return fail(LambertStatus::not_converged); }
        }
        outer = next;
    }
    return fail(LambertStatus::not_converged);
}

template <typename T>
inline T branch_starter(T target, int revolutions, LambertBranch branch) {
    // Équation 31 d'Izzo, via tanh(log(ratio)/3) pour éviter les rapports
    // extrêmes et les puissances. Le starter ne décide jamais de l'existence
    const T m = static_cast<T>(revolutions);
    const T log_pi = std::log(std::acos(T{-1}));
    const T log_eight = std::log(T{8});
    const T logarithm = branch == LambertBranch::left
        ? std::log(m + T{1}) + log_pi - log_eight - std::log(target)
        : log_eight + std::log(target) - std::log(m) - log_pi;
    return std::tanh(logarithm / T{3});
}

} // namespace lambert_multi_detail

// Racine d'une branche déjà classée au-dessus du seuil. Le minimum et ses
// bornes doivent provenir de la même géométrie et du même M
template <typename T>
inline LambertRootResult<T> solve_lambert_branch_root(
    T target, T lambda, T q, int revolutions, LambertBranch branch,
    const LambertMinimumResult<T>& minimum, const LambertOptions& options = {}
) {
    using Status = LambertStatus;
    namespace md = lambert_multi_detail;
    if (revolutions <= 0 || (branch != LambertBranch::left && branch != LambertBranch::right)
        || minimum.status != Status::success || !std::isfinite(target) || target <= T{0}
        || !std::isfinite(options.relative_tolerance) || options.relative_tolerance <= 0.0
        || options.relative_tolerance >= 1.0 || options.max_iterations <= 0) {
        return md::root_failure<T>(Status::invalid_input);
    }
    const auto bracket = md::branch_bracket(target, lambda, q, revolutions, branch, minimum);
    if (bracket.status != Status::success) { return md::root_failure<T>(bracket.status); }
    T lower = bracket.lower;
    T upper = bracket.upper;
    const T tolerance = static_cast<T>(options.relative_tolerance);
    const auto accepted = [target, tolerance, branch, &minimum](T x, T time) {
        const bool correct_side = branch == LambertBranch::left
            ? x < minimum.lower : x > minimum.upper;
        return correct_side && std::abs(time - target) / target <= tolerance;
    };

    const T endpoints[2]{lower, upper};
    for (const T endpoint : endpoints) {
        const auto evaluation = evaluate_lambert_time(endpoint, lambda, q, revolutions);
        if (evaluation.status != Status::success) { return md::root_failure<T>(evaluation.status); }
        if (accepted(endpoint, evaluation.time)) {
            return {endpoint, evaluation.y, evaluation.time, Status::success, 0};
        }
    }

    T x = md::branch_starter(target, revolutions, branch);
    if (!std::isfinite(x) || x <= lower || x >= upper) {
        x = lower + (upper - lower) / T{2};
    }
    for (int index = 0; index < options.max_iterations; ++index) {
        const int iteration = index + 1;
        const auto derivatives = evaluate_lambert_derivatives(x, lambda, q, revolutions);
        T time, y;
        if (derivatives.status == Status::success) {
            time = derivatives.time;
            y = derivatives.y;
        } else {
            const auto evaluation = evaluate_lambert_time(x, lambda, q, revolutions);
            if (evaluation.status != Status::success) {
                return md::root_failure<T>(evaluation.status, iteration);
            }
            time = evaluation.time;
            y = evaluation.y;
        }
        if (accepted(x, time)) { return {x, y, time, Status::success, iteration}; }

        // La courbe décroît à gauche et CROÎT à droite
        const bool move_lower = branch == LambertBranch::left ? time > target : time < target;
        if (move_lower) { lower = x; }
        else { upper = x; }

        T candidate = NumericLimits<T>::quiet_NaN();
        const bool derivative_sign = derivatives.status == Status::success
            && (branch == LambertBranch::left ? derivatives.first < T{0} : derivatives.first > T{0});
        if (derivative_sign) {
            const T h = (time - target) / derivatives.first;
            const T a = h * (derivatives.second / derivatives.first);
            const T b = h * (h * (derivatives.third / derivatives.first));
            const T denominator = T{1} - a + b / T{6};
            if (std::isfinite(h) && std::isfinite(a) && std::isfinite(b)
                && std::isfinite(denominator) && denominator != T{0}) {
                candidate = x - h * ((T{1} - a / T{2}) / denominator);
            }
        }
        if (!std::isfinite(candidate) || candidate <= lower || candidate >= upper || candidate == x) {
            candidate = lower + (upper - lower) / T{2};
        }
        if (!(candidate > lower && candidate < upper) || candidate == x) {
            return md::root_failure<T>(Status::not_converged, iteration);
        }
        x = candidate;
    }
    return md::root_failure<T>(Status::not_converged, options.max_iterations);
}

// Variante préparée : permet à une boucle hôte sur M de réutiliser la géométrie
// Précondition : g provient d'une préparation réussie, avec les mêmes mu et arc
template <typename T>
inline LambertMultiRevResult<T> solve_lambert_multi_rev_prepared(
    const LambertGeometry<T>& g, T mu, LambertArc arc, int revolutions,
    const LambertMultiRevOptions& options = {}
) {
    static_assert(std::is_same_v<T, double>, "Lambert CPU supports FP64 only");
    using Status = LambertStatus;
    namespace md = lambert_multi_detail;
    auto out = md::request_failure<T>(revolutions, arc, Status::numerical_failure);
    const auto fail = [&out](Status status) {
        out.status = status;
        out.solution_count = 0;
        out.complete = status == Status::no_solution;
        for (int i = 0; i < out.slot_count; ++i) {
            out.solutions[i].x = NumericLimits<T>::quiet_NaN();
            out.solutions[i].result = lambert_failure<T>(status);
        }
        return out;
    };
    if (revolutions < 0 || !md::valid_options(options)
        || !std::isfinite(mu) || mu <= T{0}
        || (arc != LambertArc::short_arc && arc != LambertArc::long_arc)) {
        return fail(Status::invalid_input);
    }

    if (revolutions == 0) {
        // Chemin numérique M=0, sans rechercher un minimum
        const auto root = solve_lambert_root(g.target_time, g.lambda, g.q, options.roots);
        if (root.status != Status::success) {
            out = fail(root.status);
            out.solutions[0].result.iterations = root.iterations;
            return out;
        }
        out.solutions[0].x = root.x;
        out.solutions[0].result = reconstruct_lambert_velocities(g, root.x, root.y, mu, root.iterations);
        out.status = out.solutions[0].result.status;
        out.solution_count = out.status == Status::success ? 1 : 0;
        out.complete = out.status == Status::success;
        if (!out.complete) { out.solutions[0].x = NumericLimits<T>::quiet_NaN(); }
        return out;
    }

    if (!std::isfinite(g.target_time) || g.target_time <= T{0}) { return fail(Status::invalid_input); }
    out.minimum = find_lambert_minimum(g.lambda, g.q, revolutions, options);
    if (out.minimum.status != Status::success) { return fail(out.minimum.status); }

    // La bande inclut le résidu accepté par les racines : sinon deux points
    // confondus au minimum pourraient être annoncés à tort comme deux succès
    T relative_band = static_cast<T>(options.threshold_relative_tolerance);
    if (relative_band < static_cast<T>(options.roots.relative_tolerance)) {
        relative_band = static_cast<T>(options.roots.relative_tolerance);
    }
    const T roundoff = T{64} * NumericLimits<T>::epsilon();
    if (relative_band < roundoff) { relative_band = roundoff; }
    const T scale = g.target_time > out.minimum.time ? g.target_time : out.minimum.time;
    out.threshold = relative_band * scale + out.minimum.time_uncertainty;
    if (!std::isfinite(out.threshold)) { return fail(Status::numerical_failure); }

    const T difference = g.target_time - out.minimum.time;
    if (difference < -out.threshold) { return fail(Status::no_solution); }
    if (std::abs(difference) <= out.threshold) { return fail(Status::near_minimum); }

    for (int i = 0; i < 2; ++i) {
        auto& solution = out.solutions[i];
        const auto root = solve_lambert_branch_root(
            g.target_time, g.lambda, g.q, revolutions, solution.branch, out.minimum, options.roots
        );
        if (root.status != Status::success) {
            solution.result = lambert_failure<T>(root.status, root.iterations);
            continue;
        }
        solution.result = reconstruct_lambert_velocities(g, root.x, root.y, mu, root.iterations);
        if (solution.result.status == Status::success) {
            solution.x = root.x;
            ++out.solution_count;
        }
    }

    // Une multiplicité non résolue ne fournit aucune vitesse exploitable
    if (out.solutions[0].result.status == Status::near_minimum
        || out.solutions[1].result.status == Status::near_minimum) {
        return fail(Status::near_minimum);
    }
    out.complete = out.solution_count == 2;
    if (out.complete) {
        // Vérification explicite de la séparation des deux branches.
        if (!(out.solutions[0].x < out.minimum.lower
              && out.solutions[1].x > out.minimum.upper)) {
            return fail(Status::near_minimum);
        }
        out.status = Status::success;
    } else {
        // Un succès partiel reste exploitable par son statut individuel
        // Ne jamais parcourir seulement [0, solution_count) : utiliser slot_count
        out.status = out.solutions[0].result.status != Status::success
            ? out.solutions[0].result.status : out.solutions[1].result.status;
    }
    return out;
}

// Point d'entrée CPU complet. Préparation UNE fois, minimum UNE fois pour les
// deux branches. En cas d'échec, les vitesses du slot concerné sont NaN
template <typename T>
inline LambertMultiRevResult<T> solve_lambert_multi_rev_cpu(
    const Vector3<T>& departure, const Vector3<T>& arrival, T dt, T mu,
    LambertArc arc, int revolutions, const LambertMultiRevOptions& options = {}
) {
    static_assert(std::is_same_v<T, double>, "Lambert CPU supports FP64 only");
    if (revolutions < 0 || !lambert_multi_detail::valid_options(options)) {
        return lambert_multi_detail::request_failure<T>(revolutions, arc, LambertStatus::invalid_input);
    }
    const auto preparation = prepare_lambert_geometry(departure, arrival, dt, mu, arc);
    if (preparation.status != LambertStatus::success) {
        return lambert_multi_detail::request_failure<T>(revolutions, arc, preparation.status);
    }
    return solve_lambert_multi_rev_prepared(preparation.geometry, mu, arc, revolutions, options);
}

} // namespace warpkep::detail
