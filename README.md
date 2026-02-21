# Energy Sights

## 1. Résumé Exécutif
`Energy Sights` est un projet de machine learning appliqué à la performance énergétique des bâtiments, avec deux objectifs de prédiction :
- la consommation énergétique (`SiteEnergyUse(kBtu)`)
- les émissions de gaz à effet de serre (`TotalGHGEmissions`)

Le projet suit une démarche data science complète : exploration, nettoyage, ingénierie de variables, entraînement, optimisation, évaluation et sauvegarde des modèles.

## 2. Contexte et Objectifs
Ce travail vise à construire des modèles de régression robustes, interprétables et exploitables pour :
- estimer la consommation énergétique d'un bâtiment à partir de ses caractéristiques structurelles et d'usage
- estimer son niveau d'émissions GES (Greenhouse Gases)
- identifier les leviers explicatifs les plus influents via l'importance des variables

Cibles modélisées (définies dans `energy_sights/config.py`) :
- `SiteEnergyUse(kBtu)`
- `TotalGHGEmissions`

## 3. Périmètre de Données
Source principale présente dans le dépôt :
- `data/raw/2016_Building_Energy_Benchmarking.csv`

Filtres analytiques appliqués dans la démarche notebook :
- conservation des enregistrements `ComplianceStatus == 'Compliant'`
- exclusion des lignes `DefaultData == True`
- nettoyage des observations non plausibles (ex. outliers manifestes)

Jeux intermédiaires et finaux produits :
- `data/interim/df_compliant_filtered.csv`
- `data/processed/processed_data.csv`

## 4. Démarche Méthodologique (de bout en bout)
### 4.1 Exploration et cadrage
Le notebook `notebooks/exploration_01.ipynb` structure la phase d'EDA :
- analyse des distributions
- contrôle de la qualité des valeurs
- analyse des corrélations
- validation des hypothèses métier (bâtiments non résidentiels, cohérence géographique, cohérence des niveaux de consommation)

### 4.2 Nettoyage et normalisation
Les étapes de nettoyage documentées dans le code et les notebooks incluent :
- suppression des colonnes constantes (`drop_constant_columns`)
- harmonisation des modalités de quartier (`clean_neighborhood`)
- détection de doublons textuels insensibles à la casse (`find_case_insensitive_duplicates`)
- traitement d'anomalies (ex. valeurs de planchers aberrantes)
- contrôle des valeurs manquantes (`nan_percent`)

### 4.3 Gestion du risque de fuite de données
La configuration centralise des listes explicitement dédiées à la gouvernance des variables :
- `Leaky_features` : variables à fuite ou fortement dérivées des cibles
- `useless_columns` : identifiants et champs non utiles à la généralisation
- `redundant_columns` : redondances techniques/sémantiques

Ces listes servent de référence méthodologique pour garder un pipeline cohérent avec un cadre de modélisation réaliste.

### 4.4 Feature Engineering
Le module `energy_sights/features.py` formalise la création de variables :
- `IsComplex` : binarisation du nombre de bâtiments
- `BuildingAge` : âge du bâtiment
- `AverageFloorArea` : surface moyenne par étage
- `DistToCenter` : distance au centre (Haversine)
- `ParkingRatio` : ratio de surface de stationnement

### 4.5 Modélisation
Le module `energy_sights/modeling/train.py` encapsule un entraîneur unique (`ModelTrainer`) avec :
- modèles supportés : `RandomForestRegressor`, `XGBRegressor`
- encodage catégoriel : `TargetEncoder` sur `PrimaryPropertyType` et `Neighborhood`
- transformation cible optionnelle : `TransformedTargetRegressor` (`log`/`exp`)

### 4.6 Optimisation des hyperparamètres
Le module `energy_sights/modeling/tuning.py` applique :
- `RandomizedSearchCV`
- validation croisée `KFold(n_splits=5, shuffle=True, random_state=42)`
- score d'optimisation `neg_mean_absolute_error`
- comparaison train/validation (gap de surapprentissage)

### 4.7 Évaluation et interprétabilité
Le module `energy_sights/modeling/evaluate.py` fournit :
- métriques : `R²`, `RMSE`, `MAE`, `MAPE`
- visualisation des résidus (`plot_residuals`)
- importance des variables (`plot_features_importance`, `print_feature_importances`)

## 5. Résultats Observés dans les Notebooks
Les notebooks `notebooks/modeling_energy.ipynb` et `notebooks/modeling_emissions.ipynb` montrent :
- des itérations successives de sélection de variables
- des comparaisons `XGBoost` vs `Random Forest`
- des améliorations après retrait de variables peu contributives et filtrage d'outliers

Ordres de grandeur observés dans les sorties notebook (selon variantes de split et de features) :
- énergie : `R²` autour de `0.63` a `0.75`
- émissions : `R²` autour de `0.45` a `0.79`

Modèles sauvegardés actuellement dans le dépôt :
- `models/energy_model.pkl`
- `models/co2_model.pkl`

## 6. Architecture du Projet
```text
C:\ML_projects\energy_sights
├── data/
│   ├── raw/                 # Données brutes
│   ├── interim/             # Données intermédiaires de travail
│   ├── processed/           # Données finales pour entraînement
│   └── external/            # Sources externes (placeholder)
├── energy_sights/
│   ├── config.py            # Chemins, variables centrales, listes de colonnes
│   ├── logging_config.py    # Configuration Loguru + logs par tâche
│   ├── preprocessing.py     # Fonctions de nettoyage
│   ├── features.py          # Ingénierie de variables
│   ├── plots.py             # Visualisations statiques / interactives
│   ├── dataset.py           # Point d'entrée data (squelette)
│   └── modeling/
│       ├── train.py         # Construction pipeline et entraînement
│       ├── tuning.py        # Recherche d'hyperparamètres
│       ├── evaluate.py      # Métriques et diagnostics
│       └── predict.py       # Point d'entrée inference (squelette)
├── notebooks/
│   ├── exploration_01.ipynb
│   ├── modeling_energy.ipynb
│   └── modeling_emissions.ipynb
├── models/                  # Modèles sérialisés (.pkl)
├── logs/                    # Logs techniques par tâche
├── reports/                 # Rapports et figures
├── references/              # Références projet (placeholder)
├── docs/                    # Documentation additionnelle (placeholder)
├── tests/                   # Tests unitaires
├── Makefile                 # Commandes d'exploitation
├── pyproject.toml           # Dépendances et configuration tooling
└── uv.lock                  # Verrouillage des versions
```

## 7. Exécution et Exploitation
### 7.1 Prérequis
- Python `3.14`
- `uv`

### 7.2 Commandes principales
- `make create_environment` : créer l'environnement virtuel
- `make requirements` : installer les dépendances (`uv sync`)
- `make lint` : contrôle qualité (`ruff format --check` puis `ruff check`)
- `make format` : formatage/corrections automatiques (`ruff`)
- `make test` : exécution des tests
- `make data` : point d'entrée de pipeline data (`energy_sights/dataset.py`)

## 8. Journalisation et Traçabilité
Le logging est centralisé via `loguru` dans `energy_sights/logging_config.py` :
- un logger par tâche (`preprocessing`, `feature_engineering`, `model_training`, `model_tuning`, `plots`)
- fichiers de logs dans `logs/`
- rotation des fichiers à `10 MB`

## 9. Qualité Logicielle et Reproductibilité
Forces actuelles :
- dépendances figées via `uv.lock`
- configuration de linting/formatting claire (`ruff`)
- séparation nette entre modules analytiques et modules de modélisation

Points de vigilance actuels :
- `tests/test_data.py` contient un test placeholder volontairement en échec (`assert False`)
- `energy_sights/dataset.py` et `energy_sights/modeling/predict.py` sont des squelettes d'orchestration
- certains workflows notebook utilisent un `train_test_split` sans `random_state` explicite, ce qui peut faire varier les scores d'une exécution à l'autre

## 10. Livrables Actuels
- pipeline analytique notebook complet (exploration -> modélisation)
- modules Python réutilisables pour preprocessing, features, training, tuning, evaluation
- modèles entraînés sauvegardés dans `models/`
- données traitées prêtes pour réexécution dans `data/processed/`


### Contact Projet
Auteur : `alhousseine_diallo` (déclaré dans `pyproject.toml`)
