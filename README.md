# Nuclear Waste Canister — Temperature Prediction

**CIVIL-226 — Introduction to Machine Learning, EPFL 2026**
Final project based on the Mont-Terri FE experiment (2D FEM simulation).

## Objectif

Prédire la température des roches (Buffer + OPA) à des positions où l'on n'a **pas** de capteurs, sur une durée d'environ **250 ans**, à partir :
- des coordonnées `(x, y)` des capteurs train,
- de leurs mesures de température au cours du temps,
- de la puissance de chauffe de la canister (3 profils possibles).

Le test set contient des capteurs à des **positions différentes** (pas de recouvrement avec le train). La métrique finale **pénalise davantage les erreurs dans l'OPA** que dans le buffer.

## Données (`data_parquet_2026/`)

| Fichier            | Contenu                                              |
|--------------------|------------------------------------------------------|
| `train.parquet`    | 242 capteurs × 9128 pas de temps, `time / temperature / power / sensor` |
| `test.parquet`     | 80 capteurs à des positions nouvelles, sans température |
| `sensors.parquet`  | `sensor, coor_x, coor_y, region` (Buffer / OPA)      |

Domaine spatial ≈ `x ∈ [0, 50] m`, `y ∈ [0, 3.5] m`. Durée ≈ 250 ans (≈ 7.9 × 10⁹ s).

## Étapes du projet

### 1. Exploration (EDA)
- Carte spatiale train/test, séries temporelles, distribution de la puissance.
- Snapshots 2D de la température à différents instants (1, 10, 50, 200 ans).
- Diagnostic des NaN et des valeurs aberrantes.

### 2. Nettoyage des données
Le PDF insiste : *"Be careful with outliers — Failed sensors — Sensor drift"*.
- **Capteurs morts** : valeurs bloquées, NaN massifs → drop.
- **Outliers ponctuels** : IQR / box-plot par capteur.
- **Sensor drift** (bonus noté) : détection de dérive lente anormale via régression / test statistique sur la série.

### 3. Feature engineering
- Features spatiales : `x`, `y`, distance à la canister, région (Buffer / OPA).
- Features temporelles : `time`, `log(time)`, `power(t)`, intégrale de puissance.
- Features de voisinage : température des `k` plus proches capteurs train (spatial lag).
- Normalisation z-score des features continues.

### 4. Modélisation
Baselines puis modèles ML, comparés en **GroupKFold par capteur** (pas de fuite train→test sur un même capteur) :
1. Interpolation spatiale (IDW, RBF) — baseline physique.
2. Ridge / Régression linéaire régularisée.
3. Random Forest.
4. Gradient boosting (LightGBM, XGBoost).
5. (Optionnel) Réseau de neurones simple (MLP).

### 5. Évaluation
- RMSE / MAE globaux et **séparés Buffer vs OPA** (OPA pondéré plus fort).
- Analyse des résidus : spatialement (carte), temporellement (par année), par capteur.
- Diagnostic d'overfit (train vs CV).

### 6. Soumission
- **Kaggle leaderboard** : prédictions sur le test set (deadline 27 mai).
- **Code + poster** sur Moodle (deadline 27 mai).
- **Bonus** : rapport de détection de sensor drift.

## Architecture du code

```
Projet_ML/
├── main.py                  # point d'entrée : lance le pipeline
├── steps/
│   ├── pipeline.py          # orchestration des étapes 1→6
│   └── ...                  # un module par étape
├── data_parquet_2026/       # données brutes (non modifiées)
├── outputs/
│   └── figures/             # graphiques générés
└── Lectures/                # supports de cours + énoncé PDF
```

## Stack

- **Data** : `pandas`, `numpy`, `pyarrow`
- **Viz** : `matplotlib`, `seaborn`
- **ML** : `scikit-learn`, `lightgbm`, `xgboost`
- **Interpolation** : `scipy.interpolate` (RBFInterpolator)

## Usage

```bash
python main.py
```

## Livrables (barème)

- Poster — 25 %
- Performance Kaggle (passer le baseline) — 25 %
- Méthodes / résultats / code — 50 %
- **Bonus** : détection de capteurs en drift.
