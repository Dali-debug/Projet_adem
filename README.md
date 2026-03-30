# Projet PFE — NILM par Modèle de Markov Caché (HMM)
### Adam BEN RHAIEM — HMM-based Energy Disaggregation

---

## Table des matières

1. [Présentation du projet](#1-présentation-du-projet)
2. [Arborescence complète du projet](#2-arborescence-complète-du-projet)
3. [Description et rôle de chaque fichier](#3-description-et-rôle-de-chaque-fichier)
   - [Datasets](#31-datasets)
   - [Preprocessing (Pré-traitement)](#32-preprocessing-pré-traitement)
   - [Event Detection (Détection d'événements)](#33-event-detection-détection-dévénements)
   - [Clustering (Regroupement)](#34-clustering-regroupement)
   - [HMM-based Load Disaggregation (Désagrégation)](#35-hmm-based-load-disaggregation-désagrégation)
   - [Documents PDF de référence](#36-documents-pdf-de-référence)
4. [Flux de travail global (workflow end-to-end)](#4-flux-de-travail-global-workflow-end-to-end)

---

## 1. Présentation du projet

Ce projet de fin d'études (PFE) réalisé chez **ACTIA** porte sur le **NILM** (*Non-Intrusive Load Monitoring* — Surveillance Non-Intrusive de la Charge).

L'objectif est de **désagréger la consommation électrique totale** d'un bâtiment ou d'un foyer (mesurée sur un seul compteur global) afin d'**identifier et d'estimer la consommation de chaque appareil électrique** (réfrigérateur, lave-vaisselle, machine à laver, etc.) **sans capteur individuel** sur chaque appareil.

La méthode principale retenue est le **Super-State Hidden Markov Model (SSHMM)**, combiné à l'algorithme de **Viterbi** pour l'inférence. La chaîne de traitement complète comprend :

1. **Chargement des données** (jeux de données réels REFIT / AMPds)
2. **Pré-traitement** (filtrage des valeurs aberrantes, interpolation des données manquantes)
3. **Détection d'événements** (identification des changements d'état des appareils)
4. **Clustering** (apprentissage non supervisé des états de puissance de chaque appareil)
5. **Désagrégation** (inférence par HMM + Viterbi pour retrouver la consommation individuelle)
6. **Évaluation** (métriques de performance)

---

## 2. Arborescence complète du projet

```
Projet_adem/
│
├── README.md                                           ← Ce fichier de documentation
├── PFE_Actia (3).pdf                                   ← Rapport PFE complet (PDF)
│
└── Adam BEN RHAIEM-HMM-based Energy Disaggregation/
    │
    ├── Datasets/
    │   ├── dataset.txt                                 ← Liens vers les jeux de données
    │   └── References/
    │       ├── EPEC_2013.pdf                           ← Article de référence EPEC 2013
    │       └── Towards_trustworthy_Energy_Disaggregation_A_review.pdf  ← Revue NILM
    │
    ├── Preprocessing/
    │   ├── Algorithms/
    │   │   ├── hampel_filter.py                        ← Filtre Hampel (suppression outliers)
    │   │   └── interpolation.py                        ← Interpolation des données manquantes
    │   └── References/
    │       ├── On the Relationship Between Sampling Rate and Hidden Markov Model.pdf
    │       └── Research_on_Interpolation_Method_for_Missing_Elect.pdf
    │
    ├── Event Detection/
    │   ├── Algorithms/
    │   │   ├── steady_states.py                        ← Détection des états stables (Hart 1985)
    │   │   ├── cumsum.py                               ← Algorithme CUSUM
    │   │   ├── ca_cfar.py                              ← Algorithme CA-CFAR
    │   │   ├── local_threshold__based.py               ← Seuillage local adaptatif
    │   │   └── binary segmentation.py                  ← Segmentation binaire (ruptures)
    │   ├── References/
    │   │   ├── barsim_nilmworkshop2014.pdf             ← Article workshop NILM 2014
    │   │   └── DeKoning_BA_EEMCS.pdf                   ← Thèse sur la détection d'événements
    │   └── Summary/
    │       └── Event Detection Review-Step 1-NILM (1).pdf  ← Synthèse Step 1
    │
    ├── Clustering/
    │   ├── Coding/
    │   │   ├── Appliance.py                            ← Classe Appliance (clustering en ligne)
    │   │   └── cluster.py                              ← Fonctions K-Means / MeanShift
    │   └── Summary/
    │       └── Clustering.pdf                          ← Synthèse du clustering
    │
    └── HMM-based Load Disaggregation/
        └── Super-State Hidden Markov Model/
            ├── Viterbi/
            │   ├── algo_Viterbi.py                     ← Algorithme de Viterbi classique
            │   └── algo_SparseViterbi.py               ← Viterbi creux (Sparse Viterbi)
            └── Testing/
                └── test_Algorithm.py                   ← Script de test et d'évaluation
```

---

## 3. Description et rôle de chaque fichier

### 3.1 Datasets

#### `Datasets/dataset.txt`
**Rôle :** Fichier texte référençant les deux jeux de données publics utilisés dans le projet.

- **REFIT** (*Real world Energy Flexible Intelligent metering dataset*) — Université de Strathclyde : mesures de consommation électrique réelles dans 20 maisons britanniques sur plusieurs années. Fréquence d'échantillonnage : 1 mesure toutes les 8 secondes.
- **AMPds** (*Almanac of Minutely Power dataset*) — Harvard Dataverse : mesures minutieuses de consommation dans une maison canadienne sur 2 ans.

Ces deux jeux de données fournissent à la fois le **signal agrégé** (consommation totale du foyer) et les **signaux individuels** par appareil (utilisés comme vérité terrain pour l'évaluation).

---

### 3.2 Preprocessing (Pré-traitement)

Le pré-traitement a pour but de **nettoyer et compléter les données brutes** avant de les utiliser dans la chaîne NILM. Deux problèmes classiques sont traités : les **valeurs aberrantes** (*outliers*) et les **données manquantes**.

---

#### `Preprocessing/Algorithms/hampel_filter.py`
**Rôle :** Implémente le **filtre de Hampel** pour détecter et corriger les valeurs aberrantes dans une série temporelle de puissance.

**Fonctionnement :**
1. Calcul de la **médiane glissante** sur une fenêtre de taille `2 × window_size` (centrée).
2. Calcul de la **MAD** (*Median Absolute Deviation* — déviation absolue médiane), pondérée par le facteur d'échelle `k = 1.4826` (équivalent gaussien).
3. Tout point dont l'écart à la médiane dépasse `n_sigmas × MAD` est considéré comme aberrant et **remplacé par la médiane locale**.

**Utilisation dans le projet :**
- Appliqué sur la colonne `'Aggregate'` du DataFrame sur un extrait d'un mois (96 échantillons/jour × 30 jours).
- Affiche le pourcentage d'outliers détectés.
- Produit une figure avec vue zoomée sur une zone corrigée, permettant de valider visuellement la qualité du filtre.

**Nécessité :** Les signaux de puissance électrique peuvent contenir des pics parasites dus à des erreurs de capteurs, des transmissions corrompues ou des transitoires courts. Ces valeurs aberrantes fausseraient les statistiques utilisées plus loin (moyenne, variance, transitions) et doivent être éliminées avant tout traitement.

---

#### `Preprocessing/Algorithms/interpolation.py`
**Rôle :** Compare différentes méthodes d'**interpolation pour les données manquantes** dans la série temporelle de puissance agrégée.

**Fonctionnement :**
1. Extrait un segment de 200 points de la colonne `'Aggregate'`.
2. Simule des données manquantes (positions 18 à 22) en les remplaçant par `NaN`.
3. Compare visuellement quatre méthodes d'interpolation :
   - **Linéaire** : interpolation par droite entre les deux voisins connus.
   - **Polynomiale d'ordre 3** : ajustement par polynôme du 3e degré.
   - **Polynomiale d'ordre 5** : ajustement par polynôme du 5e degré.
   - **Spline d'ordre 3** : spline cubique (courbe lisse par morceaux).
4. Calcule l'**erreur absolue moyenne (MAE)** de chaque méthode sur 2 % de données masquées aléatoirement.
5. Compare également au remplissage naïf par zéro.

**Nécessité :** Les capteurs de consommation peuvent présenter des lacunes (coupures de réseau, problèmes de transmission). Les données manquantes doivent être reconstituées pour maintenir la continuité temporelle nécessaire à l'algorithme CUSUM et aux HMM, qui supposent des séries complètes.

---

### 3.3 Event Detection (Détection d'événements)

La détection d'événements est l'**étape clé du NILM event-based** : elle consiste à identifier les instants où un appareil **change d'état** (allumage, extinction, changement de mode). Cinq algorithmes différents ont été étudiés et implémentés.

---

#### `Event Detection/Algorithms/steady_states.py`
**Rôle :** Implémente la détection d'états stables et de transitions basée sur l'**algorithme de Hart (1985)**, la méthode fondatrice du NILM.

**Fonctionnement :**
- Parcourt la série temporelle point par point.
- À chaque instant, calcule la différence entre la mesure courante et la précédente.
- Si cette différence dépasse le seuil `state_threshold`, une **transition** (changement d'état) est détectée.
- Entre deux transitions, le signal est dans un **état stable** ; sa puissance moyenne est estimée de façon incrémentale.
- Les transitions dont la magnitude est supérieure au `noise_level` sont conservées.
- Retourne deux DataFrames : `steady_states` (puissance moyenne dans chaque état stable) et `transitions` (changements de puissance entre états).
- Contient également une fonction `cluster()` (basée sur K-Means avec score de silhouette pour choisir automatiquement le nombre de clusters optimal).

**Nécessité :** C'est l'approche de référence historique pour le NILM. Elle permet d'identifier les fronts montants (allumage) et descendants (extinction) qui correspondent aux activations d'appareils.

---

#### `Event Detection/Algorithms/cumsum.py`
**Rôle :** Implémente deux variantes de l'**algorithme CUSUM** (*Cumulative Sum*) pour la détection de points de changement dans une série temporelle.

**Classe `CUSUM_Detector` :**
- Maintient deux sommes cumulées : `S_pos` (déviation positive) et `S_neg` (déviation négative).
- Normalise chaque observation par rapport à la moyenne et l'écart-type estimés pendant une période d'échauffement (`warmup_period`).
- Détecte un changement dès que `S_pos` ou `S_neg` dépasse le seuil (`threshold`).
- Se réinitialise après chaque détection.

**Classe `ProbCUSUM_Detector` :**
- Variante probabiliste : calcule une p-valeur (via la loi normale) pour chaque observation.
- Détecte un changement si la p-valeur est inférieure à `threshold_probability`.
- Permet d'exprimer la confiance dans chaque détection.

**Nécessité :** CUSUM est particulièrement adapté à la détection de **changements abruptes de niveau** dans une série stationnaire, ce qui correspond exactement aux allumages et extinctions d'appareils électriques. Sa sensibilité paramétrable (via `delta` et `threshold`) permet d'équilibrer le taux de fausses alarmes et la détection réelle.

---

#### `Event Detection/Algorithms/ca_cfar.py`
**Rôle :** Implémente l'algorithme **CA-CFAR** (*Cell-Averaging Constant False Alarm Rate*), emprunté au traitement radar, pour détecter les événements de puissance.

**Fonctionnement :**
1. Pour chaque point `i` du signal, définit :
   - Des **cellules de garde** (`num_guard_cells`) de chaque côté de `i` (ignorées dans le calcul du bruit).
   - Des **cellules de référence** (`num_ref_cells`) de part et d'autre des cellules de garde.
2. Estime le **niveau de bruit** comme la moyenne des cellules de référence (avant + après).
3. Calcule un **seuil adaptatif** = `alpha × noise_level`.
4. Marque le point `i` comme événement si `signal[i] > threshold`.
5. Appliqué sur la colonne `'Dishwasher'` avec paramètres : `num_guard_cells=10`, `num_ref_cells=20`, `alpha=2`.

**Nécessité :** La méthode CA-CFAR présente l'avantage d'adapter dynamiquement son seuil au niveau de bruit local, évitant ainsi les fausses alarmes en présence d'un bruit non stationnaire. Elle est robuste aux variations lentes de la consommation de fond.

---

#### `Event Detection/Algorithms/local_threshold__based.py`
**Rôle :** Implémente une **détection d'événements par seuillage local adaptatif** combinée à la détection de pics.

**Fonctionnement :**
1. **Filtre médian glissant** (fenêtre = 60) sur la série de puissance pour lisser le signal.
2. Calcule les **différences absolues** (`delta_p = |diff(p_med)|`).
3. Fait glisser une fenêtre de taille `window_size` et calcule pour chaque position la **moyenne** (`mu_w`) et l'**écart-type** (`sigma_w`).
4. Définit un seuil global : `s_p = mean(sigma_w) + mean(mu_w)` ; seuil d'activité : `s_a = s_p / 2`.
5. Ne traite que les fenêtres où `sigma_w > s_a` (zones d'activité significative).
6. Dans ces fenêtres actives, détecte les **pics** (`find_peaks`) dont la hauteur dépasse `a × mu_w + b × sigma_w`.
7. Retourne les indices des événements détectés.

**Nécessité :** Cette approche locale s'adapte aux variations globales du signal. En ne cherchant des événements que dans les zones de forte variabilité, elle réduit le temps de calcul et le nombre de fausses détections dans les périodes calmes.

---

#### `Event Detection/Algorithms/binary segmentation.py`
**Rôle :** Implémente la **segmentation binaire** (via la librairie `ruptures`) pour détecter les points de changement, puis regroupe les segments détectés par K-Means pour identifier les cycles ON/OFF du lave-vaisselle.

**Fonctionnement :**
1. **Étape 1 — Détection d'événements :** Applique `rpt.Binseg` (segmentation binaire récursive, modèle de coût L2) sur la série `'Dishwasher'`. Le nombre de points de changement est déterminé automatiquement par la pénalité BIC : `pen = log(n) × dim × σ²`.
2. **Étape 2 — Clustering des signatures :** Pour chaque segment, calcule la puissance moyenne. Regroupe ces signatures par K-Means en 2 clusters (états ON et OFF).
3. **Étape 3 — Appariement ON/OFF :** Associe chaque événement ON à l'événement OFF suivant le plus proche, identifiant ainsi les **cycles complets** de fonctionnement du lave-vaisselle.
4. **Visualisations :** Produit trois graphiques (signal avec événements, clustering, cycles identifiés).

**Nécessité :** La segmentation binaire est une méthode robuste pour détecter un nombre inconnu de changements dans une série temporelle. En la combinant avec un clustering, on obtient non seulement les instants de transition, mais aussi la **classification** de chaque transition (allumage ou extinction), ce qui est essentiel pour l'étape suivante de modélisation HMM.

---

### 3.4 Clustering (Regroupement)

Le clustering vise à **apprendre les états de puissance caractéristiques** de chaque appareil à partir des données historiques. Ces états (centroïdes) seront utilisés comme paramètres `means` du HMM.

---

#### `Clustering/Coding/cluster.py`
**Rôle :** Fournit les **fonctions utilitaires de clustering** utilisables sur n'importe quelle série de puissance d'appareil.

**Fonctions principales :**

- **`cluster(X, max_num_clusters, exact_num_clusters)`** : Fonction principale. Filtre les données au-dessus d'un seuil (10 W), applique K-Means, ajoute l'état `OFF` (0 W), et retourne les centroïdes triés (puissances caractéristiques de chaque état).

- **`_transform_data(data)`** : Pré-traitement avant clustering :
  - Supprime les valeurs ≤ 10 W (état éteint).
  - Sous-échantillonne aléatoirement si > 2000 points (pour limiter le temps de calcul).
  - Rejette si < 20 points (pas assez de données).

- **`_apply_clustering(X, max_num_clusters)`** : Sélectionne automatiquement le nombre optimal de clusters (entre 1 et `max_num_clusters`) en utilisant le **score de silhouette** (mesure la compacité et la séparation des clusters).

- **`_apply_clustering_n_clusters(X, n_clusters)`** : Applique K-Means avec initialisation `k-means++` pour un nombre fixe de clusters.

- **`hart85_means_shift_cluster(pair_buffer_df, columns)`** : Variante utilisant l'algorithme **MeanShift** (pas besoin de spécifier le nombre de clusters a priori) sur des paires de transitions (puissances active, réactive, apparente), inspirée de la méthode Hart 1985.

**Nécessité :** Ce module est la brique de base pour apprendre les paramètres des HMM. Sans clustering préalable, il est impossible de déterminer le nombre d'états et les niveaux de puissance associés à chaque appareil.

---

#### `Clustering/Coding/Appliance.py`
**Rôle :** Définit la **classe `Appliance`** qui encapsule toute la logique de clustering *en ligne* (incrémental) pour un appareil, et génère directement les **paramètres du HMM** correspondant.

**Attributs principaux :**
- `name` : nom de l'appareil.
- `series` : série temporelle de puissance brute.
- `means`, `covs` : paramètres gaussiens par cluster (moyennes et variances).
- `transitionMatrix` : matrice de transition entre états (paramètre `A` du HMM).
- `stateTransitions` : liste des transitions d'état observées.

**Méthodes principales :**

- **`findCluster(val)`** : Assigne une valeur au cluster le plus proche (distance minimale aux moyennes actuelles).

- **`calculateClusterMean(assignedCluster, val)`** : Met à jour la moyenne du cluster de façon **incrémentale** (sans stocker toutes les données).

- **`calculateClustersMeans()`** : Parcourt toute la série et met à jour toutes les moyennes.

- **`classifyPoints()`** : Assigne chaque point de la série à son cluster (reconstruit `seriesPerCluster`).

- **`updateParameters()`** : Recalcule les statistiques finales (min, max, moyenne, variance) de chaque cluster, puis appelle `getClusteredSeries()`, `getStateTransitions()` et `calculateTransitionMatrix()`.

- **`getStateTransitions()`** : Extrait les transitions d'état depuis la série clusterisée, en filtrant les transitions transitoires courtes (≤ 3 points) qui pourraient être du bruit de commutation.

- **`calculateTransitionMatrix()`** : Construit la **matrice de transition normalisée** du HMM à partir des transitions observées.

- **`__str__()`** : Génère directement le code Python formaté pour initialiser les paramètres HMM (`pi`, `a`, `mean`, `cov`) — prêt à être copié dans le script de modélisation.

**Nécessité :** Cette classe automatise l'entraînement non supervisé du modèle pour chaque appareil. Elle rend le processus de génération des paramètres HMM reproductible et facilement extensible à de nouveaux appareils.

---

### 3.5 HMM-based Load Disaggregation (Désagrégation)

C'est le **cœur du projet** : une fois les paramètres du HMM appris, l'algorithme de Viterbi est utilisé pour désagréger le signal agrégé.

---

#### `HMM-based Load Disaggregation/Super-State Hidden Markov Model/Viterbi/algo_Viterbi.py`
**Rôle :** Implémente l'**algorithme de Viterbi classique** pour la désagrégation de charge.

**Fonctionnement :**
- Prend en entrée le modèle HMM (`hmm`) et deux observations consécutives `[y0, y1]`.
- Calcule la probabilité de chaque état `j` à `t=0` : `Pt[0][j] = P0[j] × B[j, y0]` (probabilité initiale × émission).
- Pour `t=1`, calcule : `Pt[1][j] = max_i(Pt[0][i] × A[i,j]) × B[j, y1]` (Viterbi à 2 pas).
- Retourne l'état le plus probable `k` et sa probabilité `p`.

**Complexité :** O(K²) où K est le nombre total d'états combinés (super-états), car on teste toutes les transitions possibles.

**Nécessité :** L'algorithme de Viterbi est la méthode standard pour trouver la **séquence d'états cachés la plus probable** dans un HMM. Il permet de retrouver quel appareil est dans quel état à chaque instant, à partir de la seule mesure agrégée.

---

#### `HMM-based Load Disaggregation/Super-State Hidden Markov Model/Viterbi/algo_SparseViterbi.py`
**Rôle :** Implémente une version **optimisée (creuse) de l'algorithme de Viterbi** qui évite les calculs inutiles.

**Différence avec `algo_Viterbi.py` :**
- Utilise des **dictionnaires creux** au lieu de listes denses : seuls les états avec probabilité non nulle sont stockés.
- Pour `t=0` : ne traite que les états `j` pour lesquels `B[y0]` est non nulle **et** `P0[j] ≠ 0`.
- Pour `t=1` : ne traite que les transitions `(i → j)` pour lesquelles `i` existe dans `Pt[0]`.
- Retourne également `cdone` (nombre de calculs réellement effectués) vs `ctotal` (nombre théorique maximal), permettant de mesurer le gain.

**Nécessité :** Dans un Super-State HMM, le nombre d'états K peut être très grand (produit cartésien des états de tous les appareils). La matrice de transition A est souvent **creuse** (peu de transitions sont possibles). Cette optimisation peut réduire le temps de calcul de plusieurs ordres de grandeur, rendant la désagrégation en temps réel possible.

---

#### `HMM-based Load Disaggregation/Super-State Hidden Markov Model/Testing/test_Algorithm.py`
**Rôle :** Script principal de **test et d'évaluation** de l'algorithme de désagrégation. C'est le point d'entrée de la pipeline de validation.

**Fonctionnement :**
1. **Chargement des paramètres en ligne de commande :** identifiant de test, fichier modèle, dataset, précision, mesure, débruitage, limite, nom de l'algorithme.
2. **Chargement du modèle :** lit le fichier JSON contenant les SSHMMs pré-entraînés (un par fold de validation croisée).
3. **Validation croisée :** pour chaque fold, charge les données de test, appelle `disagg_algo()` (depuis `algo_Viterbi.py` ou `algo_SparseViterbi.py` selon le paramètre), compare les prédictions à la vérité terrain.
4. **Métriques d'évaluation :**
   - **Classification** : comparaison des états estimés vs réels.
   - **Mesure** : comparaison des puissances estimées vs réelles.
   - Comptage des événements inattendus, multi-commutations, adaptations.
5. **Visualisation :** pour le dernier fold, génère un graphique `Actual vs Predicted` pour chaque appareil (sauvegardé en PNG).
6. **Rapport CSV :** produit un rapport complet avec tous les paramètres et résultats, prêt à être importé dans un tableur.

**Dépendances externes (non incluses dans le repo) :**
- `libDataLoaders` : chargement des datasets CSV.
- `libFolding` : gestion de la validation croisée k-fold.
- `libSSHMM` : implémentation du Super-State HMM.
- `libAccuracy` : calcul des métriques de précision.

**Nécessité :** Ce script constitue l'**évaluation quantitative finale** du projet. Sans lui, il est impossible de mesurer objectivement les performances de l'algorithme de désagrégation. Il centralise toute la logique de test et génère les résultats reproductibles présentés dans le rapport PFE.

---

### 3.6 Documents PDF de référence

| Fichier | Contenu |
|---|---|
| `PFE_Actia (3).pdf` | Rapport PFE complet d'Adam BEN RHAIEM |
| `Datasets/References/EPEC_2013.pdf` | Article fondateur NILM (EPEC 2013) |
| `Datasets/References/Towards_trustworthy_Energy_Disaggregation_A_review.pdf` | Revue de littérature NILM |
| `Event Detection/References/barsim_nilmworkshop2014.pdf` | Méthodes de détection d'événements (NILM Workshop 2014) |
| `Event Detection/References/DeKoning_BA_EEMCS.pdf` | Thèse sur la détection d'événements pour NILM |
| `Event Detection/Summary/Event Detection Review-Step 1-NILM (1).pdf` | Synthèse personnelle : revue des méthodes de détection |
| `Clustering/Summary/Clustering.pdf` | Synthèse personnelle : revue des méthodes de clustering |
| `Preprocessing/References/On the Relationship Between Sampling Rate and Hidden Markov Model.pdf` | Impact du taux d'échantillonnage sur le HMM |
| `Preprocessing/References/Research_on_Interpolation_Method_for_Missing_Elect.pdf` | Méthodes d'interpolation pour données électriques manquantes |

---

## 4. Flux de travail global (workflow end-to-end)

Le projet suit un pipeline en **5 grandes étapes** :

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DONNÉES BRUTES (REFIT / AMPds)                      │
│           Signal agrégé global + signaux individuels (vérité terrain)   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ÉTAPE 1 : PRÉ-TRAITEMENT                         │
│                                                                         │
│  ┌─────────────────────┐    ┌────────────────────────────────────────┐  │
│  │  hampel_filter.py   │    │          interpolation.py              │  │
│  │ Suppression des     │    │  Reconstruction des données manquantes │  │
│  │ valeurs aberrantes  │    │  (linéaire, polynomiale, spline)       │  │
│  └─────────────────────┘    └────────────────────────────────────────┘  │
│                                                                         │
│  → Sortie : série temporelle propre et complète                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ÉTAPE 2 : DÉTECTION D'ÉVÉNEMENTS                     │
│                    (5 algorithmes comparés et étudiés)                  │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ steady_states.py │  │  cumsum.py   │  │      ca_cfar.py         │   │
│  │ Hart 1985 :      │  │ CUSUM :      │  │ CA-CFAR :               │   │
│  │ seuil fixe sur   │  │ sommes       │  │ seuil adaptatif         │   │
│  │ les variations   │  │ cumulées     │  │ basé sur bruit local    │   │
│  └──────────────────┘  └──────────────┘  └─────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │  local_threshold__based.py  │  │    binary segmentation.py        │  │
│  │  Seuillage local adaptatif  │  │    Segmentation binaire          │  │
│  │  + détection de pics        │  │    (ruptures) + K-Means ON/OFF   │  │
│  └─────────────────────────────┘  └──────────────────────────────────┘  │
│                                                                         │
│  → Sortie : liste des instants de transition (ON/OFF par appareil)      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ÉTAPE 3 : CLUSTERING                            │
│           Apprentissage des états de puissance par appareil             │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         Appliance.py                               │ │
│  │  Classe complète gérant le clustering incrémental d'un appareil    │ │
│  │  → calcule : means[], covs[], transitionMatrix[]                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                          cluster.py                                │ │
│  │  Fonctions K-Means (silhouette score) et MeanShift                 │ │
│  │  → retourne les centroïdes = niveaux de puissance des états        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  → Sortie : paramètres HMM par appareil (π, A, μ, Σ)                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ÉTAPE 4 : DÉSAGRÉGATION PAR SSHMM + VITERBI               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Super-State Hidden Markov Model (SSHMM)             │    │
│  │  Combine les HMM individuels de chaque appareil en un seul HMM   │    │
│  │  global dont les états sont le produit cartésien des états       │    │
│  │  individuels → K = ∏ Kᵢ états totaux                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │       algo_Viterbi.py       │  │    algo_SparseViterbi.py         │  │
│  │  Viterbi classique O(K²)    │  │  Viterbi creux O(|transitions|)  │  │
│  │  → séquence d'états optimale│  │  → même résultat, plus rapide   │  │
│  └─────────────────────────────┘  └──────────────────────────────────┘  │
│                                                                         │
│  → Sortie : état de chaque appareil à chaque instant                    │
│             + puissance estimée par appareil                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       ÉTAPE 5 : ÉVALUATION                              │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      test_Algorithm.py                             │ │
│  │  • Validation croisée k-fold                                       │ │
│  │  • Métriques : classification (états) + mesure (puissance W)       │ │
│  │  • Graphiques Actual vs Predicted par appareil                     │ │
│  │  • Rapport CSV complet                                             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  → Sortie : rapport de performance (précision, MAE, F1-score, etc.)    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Résumé du flux de données

| Étape | Entrée | Traitement | Sortie |
|-------|--------|-----------|--------|
| **Pré-traitement** | Série brute (W, horodatée) | Filtre Hampel + Interpolation | Série propre et complète |
| **Détection d'événements** | Série propre | CUSUM / CA-CFAR / Binseg / Hart / Seuillage local | Instants de transition |
| **Clustering** | Séries par appareil | K-Means / MeanShift | États de puissance (centroïdes) + matrice de transition |
| **Modélisation HMM** | Paramètres (π, A, μ, Σ) | Construction du SSHMM | Modèle entraîné (.json) |
| **Désagrégation** | Signal agrégé + modèle | Viterbi / Sparse Viterbi | Puissance estimée par appareil |
| **Évaluation** | Prédictions + vérité terrain | Métriques k-fold | Rapport de performance |

### Principe fondamental du NILM (SSHMM)

Le **Super-State HMM** modélise simultanément tous les appareils d'un foyer. Chaque **super-état** représente une combinaison d'états individuels (ex : réfrigérateur=ON, lave-vaisselle=OFF, machine à laver=cycle1). La **puissance observée** est la somme des puissances individuelles, à laquelle s'ajoute un bruit de mesure. L'algorithme de Viterbi trouve, à chaque instant, le super-état le plus probable compte tenu de la puissance agrégée mesurée, permettant ainsi de **remonter** à la consommation de chaque appareil.

---

## 5. Pipeline Projet_NILM — Guide d'utilisation

### 5.1 Installation des dépendances

```bash
pip install -r Projet_NILM/requirements.txt
```

### 5.2 Préparer les données

Télécharger les fichiers REFIT et les placer dans `Processed_Data_CSV/` (voir `Processed_Data_CSV/README.md`).

### 5.3 Commandes principales (`run_nilm.py`)

#### Pipeline complet sur une seule maison (train + test sur House 3)

```bash
cd Projet_NILM
python run_nilm.py --house ../Processed_Data_CSV/House_3.csv
```

#### **Cross-house : entraîner sur House 9, tester sur House 3** *(recommandé)*

```bash
cd Projet_NILM
python run_nilm.py \
  --train-house ../Processed_Data_CSV/House_9.csv \
  --test-house  ../Processed_Data_CSV/House_3.csv
```

#### Entraînement seul sur House 9

```bash
python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv --mode train
```

#### Désagrégation seule sur House 3 (modèles de House 9 requis)

```bash
python run_nilm.py \
  --train-house ../Processed_Data_CSV/House_9.csv \
  --test-house  ../Processed_Data_CSV/House_3.csv \
  --mode disaggregate
```

#### Mode NILM pur (signal agrégé uniquement, sans sous-comptage)

```bash
python run_nilm.py \
  --train-house ../Processed_Data_CSV/House_9.csv \
  --test-house  ../Processed_Data_CSV/House_3.csv \
  --nilm
```

#### Test rapide (limiter le nombre d'échantillons)

```bash
python run_nilm.py \
  --train-house ../Processed_Data_CSV/House_9.csv \
  --test-house  ../Processed_Data_CSV/House_3.csv \
  --limit 5000
```

### 5.4 Correspondance appareils → colonnes (vérifiée)

Les colonnes CSV REFIT (`Appliance1` … `Appliance9`) correspondent aux appareils
suivants pour les maisons 3 et 9 :

| Colonne      | House 3           | House 9           |
|-------------|-------------------|-------------------|
| Appliance1  | Toaster           | Fridge-Freezer    |
| Appliance2  | Fridge-Freezer    | Washer Dryer      |
| Appliance3  | Freezer           | Washing Machine   |
| Appliance4  | Tumble Dryer      | Dishwasher        |
| Appliance5  | Dishwasher        | Television Site   |
| Appliance6  | Washing Machine   | Microwave         |
| Appliance7  | Television        | Kettle            |
| Appliance8  | Microwave         | Hi-Fi             |
| Appliance9  | Kettle            | Electric Heater   |

Le pipeline résout automatiquement les noms canoniques (`kettle`, `microwave`,
`fridge`, `tv`) vers la bonne colonne pour chaque maison grâce aux alias définis
dans `refit_metadata.py`. Les variantes gérées incluent par exemple :
`Fridge-Freezer` → `fridge`, `Television Site` → `tv`.

### 5.5 Exécuter les tests de mapping

```bash
cd Projet_NILM
python -m pytest tests/test_mapping.py -v
```
