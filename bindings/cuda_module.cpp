// Adaptateur nanobind/CuPy du propagateur Kepler CUDA FP64.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>

#include <warpkep/cuda/kepler.hpp>
#include <warpkep/kepler_status.hpp>

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

// Fonctions et types internes à ce fichier.
namespace {
// Vues des tableaux GPU de CuPy : aucune allocation ni copie de leurs éléments.
// shape fixe le rang et les dimensions ; -1 laisse N libre ; c_contig impose l’ordre C.
// const double interdit les écritures via la vue C++ ; device::cuda impose la mémoire CUDA.
using StatesIn = nb::ndarray<const double, nb::shape<6, -1>, nb::c_contig, nb::device::cuda>;
using DurationsIn = nb::ndarray<const double, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using StatesOut = nb::ndarray<double, nb::shape<6, -1>, nb::c_contig, nb::device::cuda>;
using IntArray = nb::ndarray<int, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using Status = wk::KeplerPropagationStatus;

// Le tampon CUDA contient des Status ; la sortie CuPy contient des int32.
// Ces assertions vérifient la représentation nécessaire à leur copie en octets.
static_assert(sizeof(int) == sizeof(std::int32_t), "CuPy exige des entiers 32 bits");
static_assert(std::is_same_v<std::underlying_type_t<Status>, int>, "L'enum doit reposer sur int");

// Transforme les erreurs du runtime CUDA en exceptions C++, traduites en RuntimeError par nanobind.
// Les échecs numériques individuels restent dans le tableau statuses.
void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

// CuPy possède les cinq tableaux reçus ; seul ce tampon temporaire appartient au wrapper.
// Il adapte la sortie Status* du lanceur à la sortie int32 de CuPy sans réinterpréter un int*.
// RAII libère cette allocation même si une exception remonte ; les copies sont interdites pour éviter un double cudaFree.
class DeviceStatuses {
public:
    explicit DeviceStatuses(std::size_t count) {
        check_cuda(cudaMalloc(&data_, count * sizeof(Status)), "cudaMalloc(statuses)");
    }

    ~DeviceStatuses() { cudaFree(data_); }
    DeviceStatuses(const DeviceStatuses&) = delete;
    DeviceStatuses& operator=(const DeviceStatuses&) = delete;

    // Adresse GPU empruntée : ce getter ne copie pas les éléments et ne transfère pas la propriété.
    Status* data() const noexcept { return data_; }

private:
    Status* data_ = nullptr;
};

// Des vues CuPy distinctes peuvent partager une allocation ; on vérifie leurs intervalles contigus.
// Le calcul suppose des adresses CUDA UVA représentables par uintptr_t.
struct MemoryRange {
    const void* data;
    std::size_t bytes;
};

bool overlaps(MemoryRange lhs, MemoryRange rhs) {
    if (lhs.bytes == 0 || rhs.bytes == 0) return false;

    // Comparer les distances évite de calculer une borne adresse + taille susceptible de déborder.
    const auto a = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto b = reinterpret_cast<std::uintptr_t>(rhs.data);
    return a <= b ? b - a < lhs.bytes : a - b < rhs.bytes;
}

void require_disjoint(const std::array<MemoryRange, 5>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (overlaps(ranges[i], ranges[j])) {
                throw std::invalid_argument("Les tableaux CUDA ne doivent pas se chevaucher");
            }
        }
    }
}

// states[:, i] = (rx, ry, rz, vx, vy, vz) ; durations[i] est une durée relative, éventuellement négative.
// Unités cohérentes : r en L, dt en T, v en L/T, mu en L^3/T^2 ; mu est commun au lot.
// Les ndarray sont passés par valeur : seuls leurs descripteurs sont copiés et gardent les données vivantes durant cet appel.
// L’appel remplit out/statuses/iterations et attend leur production ; void devient None en Python.
void propagate_kepler_cuda(StatesIn states, DurationsIn durations, StatesOut out, IntArray statuses, IntArray iterations, double mu, std::uintptr_t stream_ptr) {
    // shape() lit les métadonnées ; les types ndarray ont déjà imposé dtype, rang, forme fixe et contiguïté.
    const std::size_t n = states.shape(1);
    if (durations.shape(0) != n || out.shape(1) != n) {
        throw std::invalid_argument("states, durations et out doivent avoir le même N");
    }
    if (statuses.shape(0) != n || iterations.shape(0) != n) {
        throw std::invalid_argument("statuses et iterations doivent contenir N éléments");
    }

    // device_id() identifie le GPU ; le tag device::cuda ne garantit pas que tous les tableaux partagent le même GPU.
    const int device_id = states.device_id();
    if (durations.device_id() != device_id || out.device_id() != device_id || statuses.device_id() != device_id || iterations.device_id() != device_id) {
        throw std::invalid_argument("Tous les tableaux doivent résider sur le même GPU CUDA");
    }
    if (!(mu > 0.0) || !std::isfinite(mu)) {
        throw std::invalid_argument("mu doit être fini et strictement positif");
    }
    if (n == 0) return; // Formes/GPU commun/mu vérifiés ; aucun appel runtime CUDA ici.

    // Vérifie les tailles en octets avant les multiplications ; ne garantit pas la VRAM disponible.
    if (n > std::numeric_limits<std::size_t>::max() / (6 * sizeof(double))) {
        throw std::overflow_error("Lot trop grand pour l'espace d'adressage");
    }

    // data() renvoie une adresse GPU sans transfert ; ces pointeurs ne doivent pas être déréférencés par le CPU.
    const std::array<MemoryRange, 5> ranges{{
        {states.data(), 6 * n * sizeof(double)},
        {durations.data(), n * sizeof(double)},
        {out.data(), 6 * n * sizeof(double)},
        {statuses.data(), n * sizeof(int)},
        {iterations.data(), n * sizeof(int)}
    }};
    require_disjoint(ranges); // Interdit aussi out == states (calcul en place non pris en charge).

    // cudaGetDevice écrit le GPU courant à l’adresse CPU &current_device.
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
    if (current_device != device_id) {
        throw std::invalid_argument("Le GPU courant ne correspond pas au GPU des tableaux");
    }

    // CuPy fournit get_current_stream().ptr ; le stream doit être valide, sur ce GPU et vivant pendant l’appel.
    // Les entrées doivent être prêtes avant l’import nanobind, ou une dépendance correcte doit être établie pour l’import et le lancement.
    // Le handle entier seul ne vérifie ni ces propriétés ni l’ordre entre streams.
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // SoA contigu (6, N) : la composante k commence à data() + k*N, en éléments double.
    // Les deux vues agrègent six pointeurs ; elles n’allouent et ne réorganisent rien.
    const double* in = states.data();
    double* dst = out.data();
    const wk::CartesianSoAConstView<double> input{in + 0 * n, in + 1 * n, in + 2 * n, in + 3 * n, in + 4 * n, in + 5 * n};
    const wk::CartesianSoAView<double> output{dst + 0 * n, dst + 1 * n, dst + 2 * n, dst + 3 * n, dst + 4 * n, dst + 5 * n};

    // Seule allocation GPU effectuée par le wrapper ; coût par appel : cudaMalloc, copie GPU/GPU puis cudaFree.
    DeviceStatuses gpu_statuses(n);

    // Le lancement est asynchrone : cudaSuccess ne signifie pas que le calcul est terminé.
    const cudaError_t launch_error = wk::launch_kepler_soa_double(input, durations.data(), mu, output, gpu_statuses.data(), iterations.data(), n, stream);

    // Même stream : cette copie en octets suit le kernel et conserve la représentation entière des statuts.
    cudaError_t copy_error = cudaSuccess;
    if (launch_error == cudaSuccess) {
        copy_error = cudaMemcpyAsync(statuses.data(), gpu_statuses.data(), n * sizeof(Status), cudaMemcpyDeviceToDevice, stream);
    }

    // Attendre avant toute exception et avant la destruction du tampon encore utilisé par le GPU.
    // Un retour normal garantit les sorties prêtes ; en cas d’erreur CUDA, leur contenu peut être incomplet.
    const cudaError_t completion_error = cudaStreamSynchronize(stream);
    check_cuda(launch_error, "launch_kepler_soa_double");
    check_cuda(copy_error, "cudaMemcpyAsync(statuses)");
    check_cuda(completion_error, "cudaStreamSynchronize");
}

} // namespace

// _cuda est le nom de l’extension ; m représente le module créé à l’import Python.
NB_MODULE(_cuda, m) {
    m.doc() = "Propagation Kepler FP64 par lots sur GPU CUDA, interface synchrone.";

    // m.def expose la fonction ; nb::arg nomme les paramètres ; noconvert refuse les copies implicites de dtype/disposition.
    // call_guard libère le GIL pendant le wrapper, puis le reprend ; l’appel reste synchrone pour son appelant.
    m.def("propagate_kepler", &propagate_kepler_cuda, nb::arg("states").noconvert(), nb::arg("durations").noconvert(), nb::arg("out").noconvert(), nb::arg("statuses").noconvert(), nb::arg("iterations").noconvert(), nb::arg("mu"), nb::arg("stream_ptr"), nb::call_guard<nb::gil_scoped_release>(), "Remplit out, statuses et iterations ; attend la fin du stream. Les entrees doivent etre pretes avant leur import.");

    // Publier les codes depuis l’enum C++ évite de recopier leurs valeurs numériques en Python.
    m.attr("SUCCESS") = static_cast<int>(Status::success);
    m.attr("INVALID_INPUT") = static_cast<int>(Status::invalid_input);
    m.attr("UNSUPPORTED_CASE") = static_cast<int>(Status::unsupported_case);
    m.attr("NOT_CONVERGED") = static_cast<int>(Status::not_converged);
    m.attr("NUMERICAL_FAILURE") = static_cast<int>(Status::numerical_failure);
}
