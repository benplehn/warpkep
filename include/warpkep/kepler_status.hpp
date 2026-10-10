#pragma once




namespace warpkep {
// Outcome of a propagation. enum class values must be written with their
// scope (KeplerPropagationStatus::success) and never convert silently to int.
enum class KeplerPropagationStatus {
    success,
    invalid_input,
    unsupported_case,
    not_converged,
    numerical_failure
};


// Human-readable status name, used by tests and diagnostics.
inline const char* propagation_status_name(KeplerPropagationStatus status) {
    switch (status) {
    case KeplerPropagationStatus::success:
        return "success";
    case KeplerPropagationStatus::invalid_input:
        return "invalid_input";
    case KeplerPropagationStatus::unsupported_case:
        return "unsupported_case";
    case KeplerPropagationStatus::not_converged:
        return "not_converged";
    case KeplerPropagationStatus::numerical_failure:
        return "numerical_failure";
    }
    return "unknown";
}


} // namespace warpkep