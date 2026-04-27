from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(
    title="FIFA 22 — Player Value Predictor",
    description="Prédit la valeur marchande d'un joueur de foot (€)",
    version="1.0"
)

#  Chargement des artefacts du modèle supervisé au démarrage 
with open("supervised_model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("supervised_model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("supervised_model/columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

with open("supervised_model/club_target_mean.pkl", "rb") as f:
    club_target_mean = pickle.load(f)

with open("supervised_model/nation_target_mean.pkl", "rb") as f:
    nation_target_mean = pickle.load(f)


#  Chargement des artefacts du modèle non supervisé au démarrage 
with open("unsupervised_model/kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)
with open("unsupervised_model/scaler_km.pkl", "rb") as f:
    scaler_km = pickle.load(f)
with open("unsupervised_model/columns_km.pkl", "rb") as f:
    km_columns = pickle.load(f)
with open("unsupervised_model/cluster_labels.pkl", "rb") as f:
    cluster_labels = pickle.load(f)


#  Schéma d'entrée 
class PlayerInput(BaseModel):
    overall: int
    potential: int
    age: int
    height_cm: int
    weight_kg: int
    preferred_foot: str      
    weak_foot: int            
    skill_moves: int          
    international_reputation: int  
    work_rate: str             
    pace: float
    shooting: float
    passing: float
    dribbling: float
    defending: float
    physic: float
    attacking_crossing: float
    attacking_finishing: float
    attacking_heading_accuracy: float
    attacking_short_passing: float
    attacking_volleys: float
    skill_dribbling: float
    skill_curve: float
    skill_fk_accuracy: float
    skill_long_passing: float
    skill_ball_control: float
    movement_acceleration: float
    movement_sprint_speed: float
    movement_agility: float
    movement_reactions: float
    movement_balance: float
    power_shot_power: float
    power_jumping: float
    power_stamina: float
    power_strength: float
    power_long_shots: float
    mentality_aggression: float
    mentality_interceptions: float
    mentality_positioning: float
    mentality_vision: float
    mentality_penalties: float
    mentality_composure: float
    defending_marking_awareness: float
    defending_standing_tackle: float
    defending_sliding_tackle: float
    goalkeeping_diving: float
    goalkeeping_handling: float
    goalkeeping_kicking: float
    goalkeeping_positioning: float
    goalkeeping_reflexes: float
    club_joined_year: int     
    wage_eur_log: float        # np.log(salaire) 
    release_clause_eur_log: float  # np.log(clause)
    club_name: str             
    nationality_name: str      
    league_name: str           
    club_position: str        
    poste_principal: str      


## UNSUPERVISED CLUSTER
class PlayerClusterInput(BaseModel):
    overall: int
    potential: int
    age: int
    height_cm: int
    weight_kg: int
    weak_foot: int
    skill_moves: int
    international_reputation: int
    pace: float
    shooting: float
    passing: float
    dribbling: float
    defending: float
    physic: float
    attacking_crossing: float
    attacking_finishing: float
    attacking_heading_accuracy: float
    attacking_short_passing: float
    attacking_volleys: float
    skill_dribbling: float
    skill_curve: float
    skill_fk_accuracy: float
    skill_long_passing: float
    skill_ball_control: float
    movement_acceleration: float
    movement_sprint_speed: float
    movement_agility: float
    movement_reactions: float
    movement_balance: float
    power_shot_power: float
    power_jumping: float
    power_stamina: float
    power_strength: float
    power_long_shots: float
    mentality_aggression: float
    mentality_interceptions: float
    mentality_positioning: float
    mentality_vision: float
    mentality_penalties: float
    mentality_composure: float
    defending_marking_awareness: float
    defending_standing_tackle: float
    defending_sliding_tackle: float
    goalkeeping_diving: float
    goalkeeping_handling: float
    goalkeeping_kicking: float
    goalkeeping_positioning: float
    goalkeeping_reflexes: float

# ── Endpoint de santé ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "FIFA 22 Value Predictor — opérationnel"}


# ── Endpoint de prédiction ─────────────────────────────────────────────
@app.post("/predict")
def predict(player: PlayerInput):
    try:
        data = player.dict()

        # 1. Label encoding
        data['preferred_foot'] = 0 if data['preferred_foot'] == "Left" else 1
        work_rate_map = {v: i for i, v in enumerate(sorted([
            "High/High", "High/Low", "High/Medium",
            "Low/High", "Low/Low", "Low/Medium",
            "Medium/High", "Medium/Low", "Medium/Medium"
        ]))}
        data['work_rate'] = work_rate_map.get(data['work_rate'], 0)

        # 2. Target encoding
        data['club_name'] = club_target_mean.get(data['club_name'], 
                                                  np.mean(list(club_target_mean.values())))
        data['nationality_name'] = nation_target_mean.get(data['nationality_name'],
                                                           np.mean(list(nation_target_mean.values())))

        # 3.  DataFrame avec les colonnes de base
        df_input = pd.DataFrame([data])

        # 4. One-Hot Encoding
        df_input = pd.get_dummies(df_input, 
                                   columns=['poste_principal', 'league_name', 'club_position'])

        # 5. Aligner avec les colonnes du modèle (ajouter les colonnes manquantes à 0)
        df_input = df_input.reindex(columns=model_columns, fill_value=0)

        # 6. Convertir en int/float pour sklearn
        df_input = df_input.astype(float)

        # 7. Normalisation
        X_scaled = scaler.transform(df_input)

        # 8. Prédiction (en log) → retransformer en €
        log_pred = model.predict(X_scaled)[0]
        value_eur = np.exp(log_pred)

        return {
            "valeur_predite_eur": round(float(value_eur), 2),
            "valeur_en_millions": round(float(value_eur) / 1_000_000, 2),
            "log_prediction": round(float(log_pred), 4)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint de prédiction ─────────────────────────────────────────────

@app.post("/cluster")
def cluster(player: PlayerClusterInput):
    try:
        data = player.dict()

        # Créer DataFrame et aligner avec les colonnes du clustering
        df_input = pd.DataFrame([data])
        df_input = df_input.reindex(columns=km_columns, fill_value=0)
        df_input = df_input.astype(float)

        # Normalisation avec le scaler du clustering
        X_scaled = scaler_km.transform(df_input)

        # Prédiction du cluster
        cluster_id = int(kmeans_model.predict(X_scaled)[0])
        profil = cluster_labels.get(cluster_id, "Profil inconnu")

        return {
            "cluster_id": cluster_id,
            "profil": profil
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))