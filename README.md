# Projet PFE - NILM avec HMM

Ce document est ecrit pour presenter clairement au professeur la logique du modele implemente dans ce depot, en particulier dans le dossier Projet_NILM.

## 1. Objectif du projet

Le probleme traite est le NILM (Non-Intrusive Load Monitoring):

- Entree: une puissance agregée (un seul compteur principal).
- Sortie: l'etat et la puissance estimee de chaque appareil (kettle, microwave, fridge, tv, etc.).

L'idee centrale est de modeliser chaque appareil par un HMM gaussien, puis de decoder les etats avec Viterbi (ou une approximation combinatoire en mode NILM aggregate-only).

## 2. Organisation utile pour la soutenance

- Projet_NILM: pipeline operationnel end-to-end (preprocessing, training, disaggregation, evaluation, plots).
- Processed_Data_CSV: jeux de donnees REFIT prets a etre utilises.
- Adam BEN RHAIEM-HMM-based Energy Disaggregation: code de recherche plus large (event detection, clustering, variantes Viterbi, references).

Pour expliquer ce qui a vraiment ete execute pendant les essais recents, il faut se concentrer sur Projet_NILM.

## 3. Logique du modele (etapes exactes)

## 3.1 Preprocessing

Objectif: stabiliser les series temporelles avant apprentissage/inference.

Traitements:

1. Chargement CSV REFIT + parsing temporel.
2. Resampling a 8 secondes.
3. Filtre Hampel pour supprimer les outliers.
4. Interpolation des trous.
5. Clip a 0 et fill final.

Sortie: un DataFrame propre et regulier pour toutes les colonnes (Aggregate + Appliance1..9).

Ce qui a ete ajoute:

- Plots de preprocessing raw vs preprocessed par colonne cible.
- Sauvegarde automatique dans Projet_NILM/plots.

## 3.2 Entrainement HMM par appareil

Pour chaque appareil cible:

1. Selection de la colonne correcte selon la maison (mapping via refit_metadata.py).
2. Extraction de la serie de puissance.
3. Entrainement d'un GaussianHMM (hmmlearn).
4. Sauvegarde JSON des parametres (startprob, transmat, means, covars).

Points importants a presenter:

- C'est un HMM gaussien univarie par appareil.
- L'initialisation des moyennes se fait par quantiles.
- Les modeles sont sauvegardes sous Projet_NILM/models/<house>.

Ce qui a ete ajoute:

- Override du nombre d'etats specifique fridge via --fridge-states.

## 3.3 Decodage des etats (Viterbi)

### Mode sub-metering (evaluation supervisee)

- Chaque serie appareil est decodee avec model.predict(...).
- Dans hmmlearn, ce predict applique Viterbi pour la sequence d'etats la plus probable.

### Mode NILM (aggregate-only)

- On n'utilise que la colonne Aggregate.
- A chaque instant, on teste les combinaisons d'etats des appareils et on choisit la somme de puissances la plus proche de l'agrege observe.

Important pour la soutenance:

- Le mode NILM actuel est une approche combinatoire instantanee.
- Ce n'est pas encore un FHMM Viterbi global complet dans le temps.

## 3.4 Correction critique appliquee: mapping semantique des etats

Probleme classique HMM:

- Les indices d'etat sont permutation-invariants.
- Sans correction, OFF/ON peut etre inverse entre runs/maisons.

Correction implementee:

1. Trier les etats par moyenne d'emission croissante.
2. Assigner les labels semantiques dans cet ordre (OFF, LOW, HIGH ou OFF/ON).

Impact observe:

- En cross-house, l'accuracy TV est remontee d'un niveau quasi nul a un niveau coherent apres ce remapping.

## 3.5 Evaluation

En mode sub-metering:

- Ground truth ON/OFF construit avec seuil > 10W sur la colonne reelle.
- Prediction ON/OFF = label != OFF.
- Accuracy ON/OFF affichee par appareil.

Ce choix est simple et lisible pour la presentation, meme si des metriques supplementaires sont possibles (F1, precision, recall, MAE energie).

## 3.6 Detection d'evenements et visualisation

Ce qui a ete ajoute:

1. Detection de transitions d'etat OFF->ON, ON->OFF, etc.
2. Affichage de plusieurs evenements par appareil (pas seulement un).
3. Generation de plots centres autour des evenements.

Optimisation realisee:

- Vectorisation de la detection d'evenements pour eviter le cout eleve sur des millions de lignes.

## 4. Scenario experimental execute

Scenario principal de soutenance:

- Train sur House 9.
- Test sur House 3.
- Fridge force a 2 etats.
- Plots preprocessing actives.
- Detection d'evenements activee.

Commande:

```bash
cd Projet_NILM
python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv --test-house ../Processed_Data_CSV/House_3.csv --fridge-states 2 --plot-preprocessing --detect-events --events-per-appliance 5 --event-window 120
```

Etape NILM aggregate-only:

```bash
cd Projet_NILM
python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv --test-house ../Processed_Data_CSV/House_3.csv --mode disaggregate --nilm --fridge-states 2 --plot-preprocessing --detect-events --events-per-appliance 5 --event-window 120
```

## 5. Comment expliquer les resultats au professeur

Message simple et defendable:

1. Le pipeline HMM/Viterbi fonctionne correctement en mode sub-metering.
2. Le cross-house est plus difficile (decalage de distribution entre maisons).
3. Le remapping semantique des etats etait indispensable pour des metriques fiables.
4. Le fridge a 2 etats est plus stable que 3 etats dans ce scenario.
5. Le mode NILM aggregate-only est plus realiste mais plus bruite (beaucoup de transitions), donc des filtres evenements sont des ameliorations naturelles.

## 6. Fichiers a citer pendant la presentation

- Projet_NILM/run_nilm.py: orchestration complete et interface CLI.
- Projet_NILM/preprocessing.py: nettoyage signal + plots preprocessing.
- Projet_NILM/train_hmm.py: entrainement et sauvegarde des HMM.
- Projet_NILM/disaggregate.py: decodage etats, mode NILM, evaluation, evenements.

## 7. Ce qui est "correct" scientifiquement et ce qui reste a faire

Correct dans l'etat:

- Modelisation HMM par appareil.
- Decodage Viterbi par appareil.
- Evaluation cross-house explicite.
- Visualisations interpretables (preprocessing, states, events).

Limites actuelles:

- Mode NILM = matching combinatoire instantane (pas FHMM Viterbi temporel complet).
- Evaluation principalement ON/OFF.

Extensions proposees:

1. Filtre de duree minimale des evenements.
2. Filtre de puissance minimale pour ignorer micro-transitions.
3. Metriques supplementaires: precision/recall/F1, MAE energie.
4. FHMM Viterbi global pour mode NILM avance.

## 8. Mini script oral (2-3 minutes)

"On part de donnees REFIT brutes, on nettoie avec Hampel + interpolation, puis on entraine un HMM gaussien par appareil. Pour l'inference, on decode les etats par Viterbi. En cross-house, on a ajoute un remapping semantique des etats base sur les moyennes d'emission pour eviter l'inversion OFF/ON. On a aussi force le fridge a 2 etats pour stabiliser le comportement. Enfin, on a ajoute des plots de preprocessing et une detection d'evenements multi-transitions par appareil, ce qui rend les resultats plus explicables en soutenance." 
