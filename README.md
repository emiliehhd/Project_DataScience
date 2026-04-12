# Project Data Science

Problématique : 
Segmenter les joueurs de foot par profil et prédire leur valeur marchande pour optimiser le recrutement d'un club.

---

## Description

Ce projet de Data Science complet appliqué au dataset **FIFA 22** (19 239 joueurs, 110 variables) répond à la problématique  suivante :
*« Segmenter les joueurs de foot par profil et prédire leur valeur marchande pour optimiser le recrutement d'un club. »*


| Partie | Description |
|--------|-------------|
| **EDA & Nettoyage** | Exploration, nettoyage, feature engineering |
| **Modèle Supervisé** | Régression pour prédire la valeur marchande |
| **Modèle Non Supervisé** | Clustering pour segmenter les profils de joueurs |
| **API REST** | Déploiement des modèles via FastAPI |

---

## Structure du projet

```
projet_fifa/
│
├── project.ipynb                  # Notebook principal (EDA + modèles)
├── main.py                        # API FastAPI
├── requirements.txt               # Dépendances Python
├── players_22.csv                 # Dataset source (Kaggle)
│
├── supervised_model/
│   ├── model.pkl                  # Gradient Boosting Regressor
│   ├── scaler.pkl                 # StandardScaler (supervisé)
│   ├── columns.pkl                # 153 colonnes après encodage
│   ├── club_target_mean.pkl       # Target encoding — clubs
│   └── nation_target_mean.pkl     # Target encoding — nationalités
│
└── unsupervised_model/
    ├── kmeans_model.pkl           # KMeans k=5
    ├── scaler_km.pkl              # StandardScaler (clustering)
    ├── columns_km.pkl             # 151 colonnes de X_clean
    └── cluster_labels.pkl         # Noms des 5 clusters
```

---

## Dataset

- **Source** : [Kaggle — FIFA 22 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)
- **Joueurs** : 19 239
- **Variables initiales** : 110
- **Variables après nettoyage** : 62
- **Variable cible** : `value_eur` (valeur marchande en €)

---

## Pipeline

### 1. EDA & Nettoyage

- Suppression des colonnes inutiles (URLs, IDs, métadonnées)
- Gestion des valeurs manquantes :
  - `club_loaned_from` (94%) et `goalkeeping_speed` (89%) → suppression
  - `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic` → remplissage à 0 pour les gardiens
  - `release_clause_eur` → médiane
  - `value_eur` manquant → suppression des lignes (variable cible)
- Transformation logarithmique sur `value_eur`, `wage_eur`, `release_clause_eur`
- Création de `poste_principal` depuis `player_positions`
- Encodage : **Label Encoding**, **One-Hot Encoding**, **Target Encoding**

---

### 2. Modèle Supervisé — Régression

Prédiction de `value_eur_log` (reconvertie en € via `np.exp()`).

| Modèle | R² | MAE | RMSE |
|--------|----|-----|------|
| Régression Linéaire (baseline) | 0.9806 | 0.1184 | 0.1695 |
| Random Forest | 0.9972 | 0.0419 | 0.0647 |
| **Gradient Boosting ** | **0.9991** | **meilleur** | **meilleur** |

> Le Gradient Boosting*est le modèle retenu et déployé dans l'API.

---

### 3. Modèle Non Supervisé — Clustering

#### KMeans (k=5) — Segmentation des profils

| Cluster | Profil |
|---------|--------|
| 0 | Jeunes attaquants |
| 1 | Défenseurs physiques |
| 2 | Bons joueurs |
| 3 | Joueurs polyvalents |
| 4 | Gardiens |

> Le clustering par KMeans est le modèle non supervisé déployé dans l'API.


#### DBSCAN — Détection d'anomalies

- **26 clusters** détectés automatiquement
- **149 joueurs atypiques** identifiés (profils rares, joueurs sur/sous-évalués)

---

## API FastAPI

### Installation

```bash
pip install -r requirements.txt
```

### Lancement

```bash
uvicorn main:app --reload
```

L'API tourne sur `http://127.0.0.1:8000`
Documentation Swagger : `http://127.0.0.1:8000/docs`

---

### Endpoints

#### `GET /`
Vérifie que l'API est opérationnelle.

```json
{
  "status": "ok",
  "message": "FIFA 22 Value Predictor — opérationnel"
}
```

---

#### `POST /predict`
Prédit la valeur marchande d'un joueur.

**Exemple de requête :**
```json
{
  "overall": 82,
  "potential": 88,
  "age": 23,
  "height_cm": 181,
  "weight_kg": 75,
  "preferred_foot": "Right",
  "weak_foot": 3,
  "skill_moves": 4,
  "international_reputation": 2,
  "work_rate": "High/Medium",
  "pace": 85,
  "shooting": 78,
  "passing": 80,
  "dribbling": 83,
  "defending": 40,
  "physic": 72,
  "attacking_crossing": 75,
  "attacking_finishing": 76,
  "attacking_heading_accuracy": 65,
  "attacking_short_passing": 82,
  "attacking_volleys": 70,
  "skill_dribbling": 84,
  "skill_curve": 78,
  "skill_fk_accuracy": 72,
  "skill_long_passing": 78,
  "skill_ball_control": 85,
  "movement_acceleration": 86,
  "movement_sprint_speed": 84,
  "movement_agility": 88,
  "movement_reactions": 80,
  "movement_balance": 85,
  "power_shot_power": 77,
  "power_jumping": 70,
  "power_stamina": 82,
  "power_strength": 68,
  "power_long_shots": 74,
  "mentality_aggression": 65,
  "mentality_interceptions": 45,
  "mentality_positioning": 80,
  "mentality_vision": 82,
  "mentality_penalties": 75,
  "mentality_composure": 80,
  "defending_marking_awareness": 42,
  "defending_standing_tackle": 40,
  "defending_sliding_tackle": 38,
  "goalkeeping_diving": 10,
  "goalkeeping_handling": 10,
  "goalkeeping_kicking": 10,
  "goalkeeping_positioning": 10,
  "goalkeeping_reflexes": 10,
  "club_joined_year": 2021,
  "wage_eur_log": 10.1266,
  "release_clause_eur_log": 17.6213,
  "club_name": "Paris Saint-Germain",
  "nationality_name": "France",
  "league_name": "Ligue 1",
  "club_position": "SUB",
  "poste_principal": "ST"
}
```

**Réponse :**
```json
{
  "valeur_predite_eur": 52475992.14,
  "valeur_en_millions": 52.48,
  "log_prediction": 17.7759
}
```

---

#### `POST /cluster`
Assigne un joueur à un **profil KMeans**.

**Exemple de requête :**
```json
{
  "overall": 82, "potential": 88, "age": 23,
  "height_cm": 181, "weight_kg": 75,
  "weak_foot": 3, "skill_moves": 4,
  "international_reputation": 2,
  "pace": 85, "shooting": 78, "passing": 80,
  "dribbling": 83, "defending": 40, "physic": 72,
  "attacking_crossing": 75, "attacking_finishing": 76,
  "attacking_heading_accuracy": 65, "attacking_short_passing": 82,
  "attacking_volleys": 70, "skill_dribbling": 84,
  "skill_curve": 78, "skill_fk_accuracy": 72,
  "skill_long_passing": 78, "skill_ball_control": 85,
  "movement_acceleration": 86, "movement_sprint_speed": 84,
  "movement_agility": 88, "movement_reactions": 80,
  "movement_balance": 85, "power_shot_power": 77,
  "power_jumping": 70, "power_stamina": 82,
  "power_strength": 68, "power_long_shots": 74,
  "mentality_aggression": 65, "mentality_interceptions": 45,
  "mentality_positioning": 80, "mentality_vision": 82,
  "mentality_penalties": 75, "mentality_composure": 80,
  "defending_marking_awareness": 42, "defending_standing_tackle": 40,
  "defending_sliding_tackle": 38, "goalkeeping_diving": 10,
  "goalkeeping_handling": 10, "goalkeeping_kicking": 10,
  "goalkeeping_positioning": 10, "goalkeeping_reflexes": 10
}
```

**Réponse :**
```json
{
  "cluster_id": 0,
  "profil": "Jeunes attaquants"
}
```

---

## Dépendances

```txt
fastapi
uvicorn
scikit-learn
pandas
numpy
```

```bash
pip install fastapi uvicorn scikit-learn pandas numpy
```

---

## Notes

- **Compatibilité des `.pkl`** : les fichiers doivent être générés dans le **même environnement Python** que celui utilisé pour lancer l'API. En cas d'erreur `numpy._core not found` ou `BitGenerator`, réinstaller les dépendances et relancer tout le notebook.

  ```bash
  pip install "numpy==1.26.4" "scikit-learn==1.4.2"
  ```

- **Espace log** : le modèle prédit `value_eur_log`. La reconversion en € est automatique via `np.exp()` dans l'API.

- **Valeurs inconnues** : un `club_name` ou `nationality_name` absent du dataset d'entraînement sera remplacé par la **moyenne globale** du target encoding.

