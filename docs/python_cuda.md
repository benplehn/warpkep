# Interface Python/CuPy — Kepler FP64

`warpkep.propagate_kepler(states, durations, *, mu=1.0)` propage un lot à deux corps et renvoie `(out, statuses, iterations)`, trois tableaux CuPy prêts au retour.

## Contrat

| Argument ou sortie | Forme / type | Signification |
| --- | --- | --- |
| `states` | `(6, N)`, `float64`, ordre C | Lignes `rx, ry, rz, vx, vy, vz` ; mémoire GPU CuPy |
| `durations` | `(N,)`, `float64`, contigu | Durée relative par trajectoire, éventuellement négative ; même GPU que `states` |
| `mu` | Scalaire fini, strictement positif | Paramètre gravitationnel commun au lot |
| `out` | `(6, N)`, `float64`, ordre C | États propagés sur le GPU de `states` |
| `statuses` | `(N,)`, `int32` | Code de résultat par trajectoire |
| `iterations` | `(N,)`, `int32` | Nombre d'itérations par trajectoire |

Les unités doivent être cohérentes : positions en L, vitesses en L/T, durées en T, `mu` en L³/T². Aucune conversion d'unité, de dtype ou de disposition des entrées n'est implicite. Les entrées restent inchangées ; CuPy alloue les trois sorties. Un lot vide renvoie trois sorties vides.

## Exemple

```python
import numpy as np
import cupy as cp
import warpkep
from warpkep import _cuda

with cp.cuda.Device(0):
    states = cp.zeros((6, 257), dtype=cp.float64)
    states[0, :] = 1.0
    states[4, :] = 1.0
    durations = cp.full(257, np.pi / 2, dtype=cp.float64)
    out, statuses, iterations = warpkep.propagate_kepler(states, durations, mu=1.0)
    if not bool(cp.all(statuses == _cuda.SUCCESS).item()):
        raise RuntimeError("Examiner les statuts des trajectoires en échec")
    print(cp.asnumpy(out[:, 0]))
```

Référence analytique en unités normalisées : après un quart de période, l'état attendu est `(0, 1, 0, -1, 0, 0)`. Les lectures `.item()` et `cp.asnumpy` servent ici à l'inspection sur CPU ; le passage des états au binding ne transfère pas leurs éléments vers la RAM.

## Streams et mémoire

L'API utilise le stream CuPy courant du GPU de `states`. Elle attend ce stream avant l'import nanobind, puis le binding attend la fin du kernel et de la copie des statuts. L'appel est synchrone pour son appelant, même si le binding libère le GIL.

Préparer les entrées sur ce stream. Si un autre stream les produit, l'appelant doit attendre ce producteur avant l'appel. L'attente du seul stream courant ne suffit pas dans ce cas. Exclure tout accès concurrent incompatible aux buffers pendant l'appel.

CuPy possède les cinq tableaux visibles depuis Python. Le binding alloue en plus un tampon GPU `Status`, le copie vers la sortie `int32`, puis le libère par RAII. Ce tampon adapte l'API CUDA typée existante ; il ajoute allocation, copie GPU/GPU et libération à chaque appel. Il ne s'agit pas d'une interface asynchrone ni sans allocation.

## Erreurs

| Situation | Résultat |
| --- | --- |
| Entrée non CuPy, mauvais dtype ou contiguïté incompatible | `TypeError` |
| Mauvaise forme de `states`, tailles de lots incohérentes, GPU différents ou `mu` invalide | `ValueError` |
| Rang ou forme fixe de `durations` refusés par nanobind | `TypeError` |
| Débordement du calcul des tailles en octets | `OverflowError` |
| Échec du runtime CUDA | `RuntimeError` ; sorties potentiellement incomplètes |
| Échec orbital individuel | Code dans `statuses`, six NaN dans la colonne de sortie concernée |

Les codes sont exposés dans `warpkep._cuda` : `SUCCESS`, `INVALID_INPUT`, `UNSUPPORTED_CASE`, `NOT_CONVERGED`, `NUMERICAL_FAILURE`. Vérifier les statuts avant d'exploiter les états. Un `mu` invalide fait rejeter tout le lot, contrairement à un état individuel invalide.

## Qui appelle qui ?

| Emplacement | Rôle / lieu d'exécution |
| --- | --- |
| `python/warpkep/__init__.py` | Façade publique ; charge les dépendances GPU seulement à l'appel |
| `python/warpkep/cuda.py` | Python : sélection du GPU, allocations CuPy, choix et attente du stream |
| `bindings/cuda_module.cpp` | CPU : vues nanobind, contrôles, vues SoA, tampon de statuts et attente finale |
| `src/cuda/kepler.cu` : `launch_kepler_soa_double` | CPU : soumission du kernel dans le stream fourni |
| `src/cuda/kepler.cu` : `kepler_propagation_soa_kernel<double>` | GPU : un thread par trajectoire appelle le calcul orbital existant |

## Installation et validation du prototype

Depuis la racine du dépôt, dans un environnement Python avec CuPy compatible et un GPU NVIDIA disponible :

```bash
python -m pip install . -Ccmake.define.WARPKEP_ENABLE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native -Ccmake.define.BUILD_TESTING=OFF -Cbuild.tool-args=-j2
python -m pytest tests/test_python_cuda.py -q
```

Le toolkit CUDA est nécessaire pour compiler ; CuPy et pytest sont à fournir séparément dans cet environnement. Par défaut, `python -m pip install .` construit la version CPU sans activer CUDA. L'import de `warpkep` charge `_core` ; CuPy et `_cuda` sont chargés lors du premier appel à `propagate_kepler`.

Résultat rapporté sur Kaggle pour ce TD : **6 passed** — lot mixte de 257 trajectoires comparées aux références analytiques en FP64 (normes des erreurs position/vitesse ≤ 1e-12 et contrôle des statuts/NaN), lot vide et quatre valeurs invalides de `mu`. Le test numérique utilise un stream non bloquant. Sans CuPy, le fichier pytest est ignoré ; ce skip ne valide pas le GPU.

Ces résultats valident le périmètre du prototype TD 10 ; ils ne constituent ni une mesure de performance de l'interface ni l'achèvement de la phase 1 du projet.
