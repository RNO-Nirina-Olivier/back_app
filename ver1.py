from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import logging
import os

import pandas as pd

# -------------------------------------------------------------------
# Configuration logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Cycle de vie FastAPI
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage de l'application...")

    # Chargement optionnel du dataset
    df = None
    if os.path.exists("donnemald3.xlsx"):
        try:
            df = pd.read_excel("donnemald3.xlsx")
            logger.info(f"Dataset chargé avec {len(df)} lignes")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset: {e}")
    else:
        logger.warning("donnemald3.xlsx introuvable (pas bloquant pour la prédiction)")

    # Features finales (d’après ton notebook)
    selected_features = [
        "imp2",
        "imp3",
        "imp5",
        "imp6",
        "imp10",
        "nbre_elev_SDC",
        "mere_niv_ac",
        "Ran_TbB",
        "etab_prim_stat",
        "cour_supl",
        "elev_presco",
    ]

    app.state.df = df
    app.state.selected_features = selected_features

    yield

    logger.info("Arrêt de l'application...")
    app.state.df = None
    app.state.selected_features = None

# -------------------------------------------------------------------
# Création app FastAPI
# -------------------------------------------------------------------
app = FastAPI(
    title="API de Prédiction Étudiante",
    description="Prediction + recommandations pour améliorer la réussite",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # À restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Fonctions de scoring "light" (sans sklearn)
# -------------------------------------------------------------------
def simple_score(row: pd.Series) -> float:
    """
    Score simple à partir des features.
    Adapter les poids selon ton notebook si besoin.
    """
    score = 0.0

    # Variables d'importance (0/1)
    for col in ["imp2", "imp3", "imp5", "imp6", "imp10"]:
        val = float(row.get(col, 0) or 0)
        score += val * 2.0  # poids à ajuster si besoin

    # nbre_elev_SDC : effectif par classe
    nb_elev = float(row.get("nbre_elev_SDC", 0) or 0)
    score += max(0.0, (nb_elev - 30.0) * 0.1)

    # Ran_TbB : rang (plus grand => plus de risque)
    rang = float(row.get("Ran_TbB", 0) or 0)
    score += max(0.0, (rang - 10.0) * 0.1)

    # etab_prim_stat : statut établissement (déjà encodé 0/1 en général)
    etab = float(row.get("etab_prim_stat", 0) or 0)
    if etab == 0:
        score += 1.0

    # Effets protecteurs : cours supl, présco, niveau mère
    coursupl = float(row.get("cour_supl", 0) or 0)
    elevpresco = float(row.get("elev_presco", 0) or 0)
    mereniv = float(row.get("mere_niv_ac", 0) or 0)

    score -= coursupl * 1.0
    score -= elevpresco * 0.5
    score -= mereniv * 0.5

    # clamp
    return max(-10.0, min(10.0, score))


def score_to_class_and_proba(score: float) -> Dict[str, Any]:
    """
    Transforme le score en classe + probabilités + label lisible.
    Classe:
      0 = risque très élevé
      1 = risque élevé
      2 = risque moyen
      3 = bonne réussite
    """
    if score < -2:
        cls = 0
    elif score < 0:
        cls = 1
    elif score < 3:
        cls = 2
    else:
        cls = 3

    base = [0.1, 0.2, 0.3, 0.4]  # distribution de base
    base[cls] = 0.6              # classe prédite renforcée
    s = sum(base)
    probs = [p / s for p in base]

    labels = [
        "Risque très élevé",
        "Risque élevé",
        "Risque moyen",
        "Bonne réussite",
    ]

    return {
        "prediction": cls,
        "prediction_label": labels[cls],
        "probabilities": probs,
        "confidence": probs[cls],
        "score": score,
    }

# -------------------------------------------------------------------
# Génération de recommandations pédagogiques
# -------------------------------------------------------------------
def build_recommendations(row: pd.Series, prediction: int) -> List[str]:
    recos: List[str] = []

    # 1) Niveau global de risque
    if prediction == 0:
        recos.append(
            "Risque très élevé : mettre en place un suivi individualisé et des échanges réguliers avec la famille."
        )
    elif prediction == 1:
        recos.append(
            "Risque élevé : proposer un plan de remédiation avec soutien hebdomadaire."
        )
    elif prediction == 2:
        recos.append(
            "Risque moyen : consolider les acquis avec un suivi régulier des devoirs."
        )
    else:
        recos.append(
            "Bonne probabilité de réussite : maintenir les bonnes habitudes de travail."
        )

    # 2) Variables d'importance (imp2..imp10)
    for col in ["imp2", "imp3", "imp5", "imp6", "imp10"]:
        val = float(row.get(col, 0) or 0)
        if val == 0:
            recos.append(
                f"Travailler le facteur {col} (variable d'importance) : identifier les obstacles et proposer un accompagnement ciblé."
            )

    # 3) Effectif par classe
    nb_elev = float(row.get("nbre_elev_SDC", 0) or 0)
    if nb_elev >= 50:
        recos.append(
            "La classe est très chargée : privilégier des activités en petits groupes ou du tutorat entre élèves."
        )

    # 4) Rang dans la classe
    rang = float(row.get("Ran_TbB", 0) or 0)
    if rang > 15:
        recos.append(
            "Le rang de l'élève est bas : renforcer l'accompagnement en mathématiques et sur les devoirs à la maison."
        )

    # 5) Cours de soutien
    coursupl = float(row.get("cour_supl", 0) or 0)
    if coursupl == 0:
        recos.append(
            "Proposer des cours de soutien (cours supplémentaires) pour aider l'élève à rattraper ses difficultés."
        )

    # 6) Préscolarisation
    elevpresco = float(row.get("elev_presco", 0) or 0)
    if elevpresco == 0:
        recos.append(
            "Absence de préscolarisation : prévoir des activités pour renforcer les bases (lecture, calcul, langage)."
        )

    # 7) Niveau d'étude de la mère (selon ton encodage)
    mereniv_raw = str(row.get("mere_niv_ac", "") or "").upper()
    if mereniv_raw in ["0", "AUCUN", "PRIMAIRE"]:
        recos.append(
            "Informer et accompagner la famille sur la manière de suivre le travail scolaire de l'élève (réunions, fiches de suivi)."
        )

    # 8) Statut de l'établissement
    etab_raw = str(row.get("etab_prim_stat", "") or "").lower()
    if "public" in etab_raw:
        recos.append(
            "Utiliser les dispositifs d'appui disponibles dans l'établissement (clubs, remédiation, bibliothèque)."
        )

    # Supprimer doublons tout en gardant l'ordre
    recos_uniques = list(dict.fromkeys(recos))
    return recos_uniques
def score_to_binary_class_and_proba(score: float) -> dict:
    """
    0 = non admis
    1 = admis
    """
    if score >= 0:
        prediction = 1  # admis
    else:
        prediction = 0  # non admis

    # probas brutes entre 0 et 1
    proba_admis_raw = 1 / (1 + pow(2.71828, -score))
    proba_non_admis_raw = 1.0 - proba_admis_raw

    # passage en pourcentage + arrondi à 2 décimales
    proba_admis_pct = round(proba_admis_raw * 100, 2)
    proba_non_admis_pct = round(proba_non_admis_raw * 100, 2)
    score_rounded = round(score, 2)

    # si tu veux que confidence soit aussi en pourcentage
    confidence_pct = proba_admis_pct if prediction == 1 else proba_non_admis_pct

    return {
        "prediction": prediction,
        "prediction_label": "Admis" if prediction == 1 else "Non admis",
        "probabilities": {
            "admis": proba_admis_pct,      # ex: 78.35
            "non_admis": proba_non_admis_pct  # ex: 21.65
        },
        "confidence": confidence_pct,     # ex: 78.35
        "score": score_rounded           # score reste en valeur "brute"
    }

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dataset_loaded": app.state.df is not None,
        "features": app.state.selected_features,
    }

@app.post("/predict")
async def predict(data: Dict[str, Any]):
    try:
        # Vérifier les colonnes nécessaires
        missing = [c for c in app.state.selected_features if c not in data]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes: {missing}",
            )

        df_input = pd.DataFrame([data])

        if df_input.isna().any().any():
            raise HTTPException(
                status_code=400,
                detail="Les données contiennent des valeurs manquantes",
            )

        row = df_input.iloc[0]

        # 1) score (via simple_score ou ton vrai modèle)
        score = simple_score(row)   # ou: score = float(model.decision_function(...)) si tu as un modèle

        # 2) conversion score → 0/1 + probas admis / non admis
        res = score_to_binary_class_and_proba(score)

        # 3) recommandations (optionnel, si tu gardes la fonction build_recommendations)
        recos = build_recommendations(row, res["prediction"])

        return {
            "success": True,
            **res,
            "recommendations": recos,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

# -------------------------------------------------------------------
# Lancement local
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
