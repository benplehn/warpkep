// Liaisons Python pour la propagation Kepler et la résolution Lambert en FP64 sur GPU.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>

#include <warpkep/cuda/kepler.hpp>
#include <warpkep/cuda/lambert.hpp>
#include <warpkep/kepler_status.hpp>
#include <warpkep/lambert_status.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nb = nanobind;
namespace wk = warpkep;

namespace {

// Nanobind vérifie le type, la forme et l’ordre C des tableaux CUDA lors de l’appel Python.
// Dans shape, -1 laisse la taille du lot libre. const double donne une vue en lecture seule.
using StatesIn = nb::ndarray<const double, nb::shape<6, -1>, nb::c_contig, nb::device::cuda>;
using DurationsIn = nb::ndarray<const double, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using StatesOut = nb::ndarray<double, nb::shape<6, -1>, nb::c_contig, nb::device::cuda>;
using IntArray = nb::ndarray<int, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using Status = wk::KeplerPropagationStatus;

// Positions (composante, transfert) et vitesses (slot, composante, transfert).
using PositionsIn = nb::ndarray<const double, nb::shape<3, -1>, nb::c_contig, nb::device::cuda>;
using VelocitiesOut = nb::ndarray<double, nb::shape<2, 3, -1>, nb::c_contig, nb::device::cuda>;
using RootsOut = nb::ndarray<double, nb::shape<2, -1>, nb::c_contig, nb::device::cuda>;
using BranchIntArray = nb::ndarray<int, nb::shape<2, -1>, nb::c_contig, nb::device::cuda>;

using LambertStatus = wk::LambertStatus;

// Les tableaux CuPy de diagnostics utilisent des entiers signés de 32 bits.
static_assert(sizeof(int) == sizeof(std::int32_t), "CuPy exige des entiers 32 bits");

// Nanobind traduit les erreurs du runtime CUDA en RuntimeError côté Python.
// Les échecs numériques du solveur sont signalés dans les tableaux de statuts.
void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

// Ce tampon possède les statuts typés écrits par le kernel avant leur copie vers les tableaux int32 de CuPy.
// Son destructeur libère la mémoire GPU. Les copies de cet objet sont interdites pour éviter une double libération.
template <typename StatusT>
class DeviceStatuses {
    static_assert(std::is_same_v<std::underlying_type_t<StatusT>, int>, "Status storage must match CuPy int32");

public:
    explicit DeviceStatuses(std::size_t count) {
        if (count == 0) {
            return;
        }

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(StatusT)), "cudaMalloc(statuses)");
    }

    ~DeviceStatuses() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceStatuses(const DeviceStatuses&) = delete;
    DeviceStatuses& operator=(const DeviceStatuses&) = delete;

    // Prête l’adresse GPU. Le tampon reste propriétaire de la mémoire.
    StatusT* data() const noexcept {
        return data_;
    }

private:
    StatusT* data_ = nullptr;
};

// Deux vues CuPy peuvent référencer la même allocation. Chaque plage décrit les octets utilisés par une vue.
struct MemoryRange {
    const void* data;
    std::size_t bytes;
};

bool overlaps(MemoryRange lhs, MemoryRange rhs) {
    if (lhs.bytes == 0 || rhs.bytes == 0) {
        return false;
    }

    // Comparer l’écart entre les adresses évite de calculer une adresse de fin qui pourrait déborder.
    const auto a = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto b = reinterpret_cast<std::uintptr_t>(rhs.data);
    return a <= b ? b - a < lhs.bytes : a - b < rhs.bytes;
}

// Refuse tout chevauchement entre les plages contiguës d’un même appel.
template <std::size_t Count>
void require_disjoint(const std::array<MemoryRange, Count>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (overlaps(ranges[i], ranges[j])) {
                throw std::invalid_argument("Les tableaux CUDA ne doivent pas se chevaucher");
            }
        }
    }
}

// Chaque état contient (rx, ry, rz, vx, vy, vz). La durée relative peut être négative.
// Le paramètre mu est commun au lot et les unités doivent être cohérentes.
// Les descripteurs ndarray conservent une référence aux buffers CuPy pendant l’appel.
void propagate_kepler_cuda(StatesIn states, DurationsIn durations, StatesOut out, IntArray statuses, IntArray iterations, double mu, std::uintptr_t stream_ptr) {
    
    // Compare la taille libre N entre les tableaux à partir de leurs métadonnées.
    const std::size_t n = states.shape(1);
    if (durations.shape(0) != n || out.shape(1) != n) {
        throw std::invalid_argument("states, durations et out doivent avoir le même N");
    }
    if (statuses.shape(0) != n || iterations.shape(0) != n) {
        throw std::invalid_argument("statuses et iterations doivent contenir N éléments");
    }

    // Compare les identifiants des GPU après la vérification du type de mémoire par nanobind.
    const int device_id = states.device_id();
    if (durations.device_id() != device_id || out.device_id() != device_id || statuses.device_id() != device_id || iterations.device_id() != device_id) {
        throw std::invalid_argument("Tous les tableaux doivent résider sur le même GPU CUDA");
    }
    if (!(mu > 0.0) || !std::isfinite(mu)) {
        throw std::invalid_argument("mu doit être fini et strictement positif");
    }
    if (n == 0) {
        return;
    }

    // Vérifie que la taille du plus grand tableau peut s’exprimer en octets sans débordement.
    if (n > std::numeric_limits<std::size_t>::max() / (6 * sizeof(double))) {
        throw std::overflow_error("Lot trop grand pour l'espace d'adressage");
    }

    // data() fournit les adresses GPU utilisées pour vérifier le chevauchement.
    const std::array<MemoryRange, 5> ranges{{
        {states.data(), 6 * n * sizeof(double)},
        {durations.data(), n * sizeof(double)},
        {out.data(), 6 * n * sizeof(double)},
        {statuses.data(), n * sizeof(int)},
        {iterations.data(), n * sizeof(int)}
    }};
    require_disjoint(ranges);

    // Le GPU courant doit correspondre à celui des buffers avant l’allocation et le lancement.
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
    if (current_device != device_id) {
        throw std::invalid_argument("Le GPU courant ne correspond pas au GPU des tableaux");
    }

    // L’appelant fournit un stream valide sur ce GPU et assure la disponibilité des entrées avant leur import par nanobind.
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // En ordre C, chaque composante occupe un bloc contigu de N doubles. Les vues empruntent ces adresses.
    const double* in = states.data();
    double* dst = out.data();
    const wk::CartesianSoAConstView<double> input{in + 0 * n, in + 1 * n, in + 2 * n, in + 3 * n, in + 4 * n, in + 5 * n};
    const wk::CartesianSoAView<double> output{dst + 0 * n, dst + 1 * n, dst + 2 * n, dst + 3 * n, dst + 4 * n, dst + 5 * n};

    // Le kernel écrit ses codes typés dans ce tampon temporaire.
    DeviceStatuses<Status> gpu_statuses(n);

    // Le lancement soumet le kernel. La synchronisation vérifiera ensuite son exécution.
    const cudaError_t launch_error = wk::launch_kepler_soa_double(input, durations.data(), mu, output, gpu_statuses.data(), iterations.data(), n, stream);

    // La copie des statuts vers CuPy est soumise au même stream pour s’exécuter après le kernel.
    cudaError_t copy_error = cudaSuccess;
    if (launch_error == cudaSuccess) {
        copy_error = cudaMemcpyAsync(statuses.data(), gpu_statuses.data(), n * sizeof(Status), cudaMemcpyDeviceToDevice, stream);
    }

    // Attend le travail soumis avant de vérifier les erreurs susceptibles de détruire le tampon temporaire.
    const cudaError_t completion_error = cudaStreamSynchronize(stream);
    check_cuda(launch_error, "launch_kepler_soa_double");
    check_cuda(copy_error, "cudaMemcpyAsync(statuses)");
    check_cuda(completion_error, "cudaStreamSynchronize");
}

// Chaque transfert associe deux positions et une durée. Les slots de solution partagent la même disposition en mémoire.
void solve_lambert_cuda(PositionsIn departures, PositionsIn arrivals, DurationsIn durations, VelocitiesOut departure_velocities, VelocitiesOut arrival_velocities, RootsOut roots, BranchIntArray branch_statuses, BranchIntArray iterations, IntArray statuses, IntArray slot_count, IntArray solution_count, IntArray complete, double mu, int revolutions, bool long_arc, std::uintptr_t stream_ptr) {
    const std::size_t n = departures.shape(1);

    // Le dernier axe de chaque tableau correspond aux transferts.
    const bool positions_and_durations_match = arrivals.shape(1) == n && durations.shape(0) == n;
    const bool velocities_match = departure_velocities.shape(2) == n && arrival_velocities.shape(2) == n;
    const bool slot_outputs_match = roots.shape(1) == n && branch_statuses.shape(1) == n && iterations.shape(1) == n;
    const bool global_outputs_match = statuses.shape(0) == n && slot_count.shape(0) == n && solution_count.shape(0) == n && complete.shape(0) == n;
    if (!positions_and_durations_match || !velocities_match || !slot_outputs_match || !global_outputs_match) {
        throw std::invalid_argument("Lambert arrays must have the same batch size");
    }

    // mu et le nombre de révolutions s’appliquent à tout le lot.
    if (mu <= 0.0 || !std::isfinite(mu)) {
        throw std::invalid_argument("mu must be finite and strictly positive");
    }

    if (revolutions < 0) {
        throw std::invalid_argument("Lambert revolutions must be non-negative");
    }

    // Les entrées et les sorties doivent résider sur le GPU des positions de départ.
    const int device_id = departures.device_id();
    const bool inputs_on_device = arrivals.device_id() == device_id && durations.device_id() == device_id;
    const bool velocities_on_device = departure_velocities.device_id() == device_id && arrival_velocities.device_id() == device_id;
    const bool slot_outputs_on_device = roots.device_id() == device_id && branch_statuses.device_id() == device_id && iterations.device_id() == device_id;
    const bool global_outputs_on_device = statuses.device_id() == device_id && slot_count.device_id() == device_id && solution_count.device_id() == device_id && complete.device_id() == device_id;
    if (!inputs_on_device || !velocities_on_device || !slot_outputs_on_device || !global_outputs_on_device) {
        throw std::invalid_argument("All arrays must be stored on the same GPU");
    }

    // Un lot vide valide se termine avant tout appel au runtime CUDA.
    if (n == 0) {
        return;
    }

    // Les vitesses occupent 6 * N doubles. Vérifie la borne avant de calculer les tailles en octets.
    if (n > std::numeric_limits<std::size_t>::max() / (6 * sizeof(double))) {
        throw std::overflow_error("Lot trop grand pour l'espace d'adressage");
    }

    // Vérifie les plages complètes, y compris les deux slots de chaque tableau de sortie.
    const std::array<MemoryRange, 12> ranges{{
        {departures.data(), 3 * n * sizeof(double)},
        {arrivals.data(), 3 * n * sizeof(double)},
        {durations.data(), n * sizeof(double)},
        {departure_velocities.data(), 6 * n * sizeof(double)},
        {arrival_velocities.data(), 6 * n * sizeof(double)},
        {roots.data(), 2 * n * sizeof(double)},
        {branch_statuses.data(), 2 * n * sizeof(int)},
        {iterations.data(), 2 * n * sizeof(int)},
        {statuses.data(), n * sizeof(int)},
        {slot_count.data(), n * sizeof(int)},
        {solution_count.data(), n * sizeof(int)},
        {complete.data(), n * sizeof(int)}
    }};

    require_disjoint(ranges);

    // Le GPU courant doit être celui des tableaux avant l’allocation et le lancement.
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
    if (current_device != device_id) {
        throw std::invalid_argument("Le GPU courant ne correspond pas au GPU des tableaux");
    }

    // Les vues SoA regroupent les adresses GPU des composantes dans les buffers CuPy existants.
    const double* departure_positions = departures.data();
    const double* arrival_positions = arrivals.data();
    const double* duration_data = durations.data();

    double* departure_velocities_data = departure_velocities.data();
    double* arrival_velocities_data = arrival_velocities.data();
    double* roots_data = roots.data();
    int* iterations_data = iterations.data();

    const wk::Vector3SoAConstView<double> departure_view{departure_positions, departure_positions + n, departure_positions + 2 * n};
    const wk::Vector3SoAConstView<double> arrival_view{arrival_positions, arrival_positions + n, arrival_positions + 2 * n};

    // Chaque slot de vitesse contient trois blocs de N doubles. Le slot 1 commence à 3 * N.
    // Les racines et les itérations ont un seul bloc de N éléments par slot.
    wk::LambertMultiRevSoAView<double> output{};
    output.solutions[0].departure_velocity = {departure_velocities_data, departure_velocities_data + n, departure_velocities_data + 2 * n};
    output.solutions[0].arrival_velocity = {arrival_velocities_data, arrival_velocities_data + n, arrival_velocities_data + 2 * n};
    output.solutions[0].x = roots_data;
    output.solutions[0].iterations = iterations_data;

    output.solutions[1].departure_velocity = {departure_velocities_data + 3 * n, departure_velocities_data + 4 * n, departure_velocities_data + 5 * n};
    output.solutions[1].arrival_velocity = {arrival_velocities_data + 3 * n, arrival_velocities_data + 4 * n, arrival_velocities_data + 5 * n};
    output.solutions[1].x = roots_data + n;
    output.solutions[1].iterations = iterations_data + n;

    // Le tampon de statuts contient trois blocs de N éléments, un global puis un par slot.
    DeviceStatuses<LambertStatus> gpu_statuses(3 * n);
    LambertStatus* status_data = gpu_statuses.data();

    output.statuses = status_data;
    output.solutions[0].statuses = status_data + n;
    output.solutions[1].statuses = status_data + 2 * n;

    output.slot_count = slot_count.data();
    output.solution_count = solution_count.data();
    output.complete = complete.data();

    // Le kernel et les copies de statuts partagent le stream fourni par CuPy.
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const wk::detail::LambertArc arc = long_arc ? wk::detail::LambertArc::long_arc : wk::detail::LambertArc::short_arc;
    const wk::detail::LambertMultiRevOptions options{};

    const cudaError_t launch_error = wk::launch_lambert_multi_rev_soa_double(departure_view, arrival_view, duration_data, mu, arc, revolutions, options, output, n, stream);

    cudaError_t status_copy_error = cudaSuccess;
    cudaError_t branch_copy_error = cudaSuccess;

    if (launch_error == cudaSuccess) {
        status_copy_error = cudaMemcpyAsync(statuses.data(), status_data, n * sizeof(LambertStatus), cudaMemcpyDeviceToDevice, stream);

        if (status_copy_error == cudaSuccess) {
            branch_copy_error = cudaMemcpyAsync(branch_statuses.data(), status_data + n, 2 * n * sizeof(LambertStatus), cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Attend le kernel et les copies déjà soumises avant de signaler une erreur et de libérer le tampon.
    const cudaError_t completion_error = cudaStreamSynchronize(stream);

    check_cuda(launch_error, "launch_lambert_multi_rev_soa_double");
    check_cuda(status_copy_error, "cudaMemcpyAsync(statuses)");
    check_cuda(branch_copy_error, "cudaMemcpyAsync(branch_statuses)");
    check_cuda(completion_error, "cudaStreamSynchronize");
}
} // namespace

NB_MODULE(_cuda, m) {
    m.doc() = "Propagation Kepler et résolution Lambert par lots en FP64 sur GPU CUDA.";

    // noconvert refuse les conversions implicites de type ou de disposition des tableaux.
    // Le GIL est relâché pendant l’appel C++. Les sorties sont prêtes au retour.
    m.def("propagate_kepler", &propagate_kepler_cuda, nb::arg("states").noconvert(), nb::arg("durations").noconvert(), nb::arg("out").noconvert(), nb::arg("statuses").noconvert(), nb::arg("iterations").noconvert(), nb::arg("mu"), nb::arg("stream_ptr"), nb::call_guard<nb::gil_scoped_release>(), "Remplit out, statuses et iterations. Les entrées doivent être prêtes avant leur import. Attend le stream avant de retourner.");
    m.def("solve_lambert", &solve_lambert_cuda, nb::arg("departures").noconvert(), nb::arg("arrivals").noconvert(), nb::arg("durations").noconvert(), nb::arg("departure_velocities").noconvert(), nb::arg("arrival_velocities").noconvert(), nb::arg("roots").noconvert(), nb::arg("branch_statuses").noconvert(), nb::arg("iterations").noconvert(), nb::arg("statuses").noconvert(), nb::arg("slot_count").noconvert(), nb::arg("solution_count").noconvert(), nb::arg("complete").noconvert(), nb::arg("mu"), nb::arg("revolutions"), nb::arg("long_arc"), nb::arg("stream_ptr"), nb::call_guard<nb::gil_scoped_release>(), "Résout un lot de transferts Lambert et remplit les tableaux de sortie. Les entrées doivent être prêtes avant leur import. Attend le stream avant de retourner.");

    // Exporte les valeurs de l’énumération Kepler utilisées par l’interface Python.
    m.attr("SUCCESS") = static_cast<int>(Status::success);
    m.attr("INVALID_INPUT") = static_cast<int>(Status::invalid_input);
    m.attr("UNSUPPORTED_CASE") = static_cast<int>(Status::unsupported_case);
    m.attr("NOT_CONVERGED") = static_cast<int>(Status::not_converged);
    m.attr("NUMERICAL_FAILURE") = static_cast<int>(Status::numerical_failure);
}
