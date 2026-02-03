from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import pandas as pd
import joblib
import numpy as np
import logging
from typing import List, Dict, Any
import os
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Cycle de vie : chargement du dataset, du modèle, des features
# --------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage de l'application...")

    required_files = ["donnemald3.xlsx", "model_final.pkl", "selected_cols.pkl"]
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"Fichier requis manquant: {file}")
            raise FileNotFoundError(f"Fichier requis manquant: {file}")

    try:
        app.state.df = pd.read_excel("donnemald3.xlsx")
        logger.info(f"Dataset chargé avec {len(app.state.df)} lignes")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        raise

    try:
        app.state.model = joblib.load("model_final.pkl")
        logger.info("Modèle ML chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

    try:
        app.state.selected_features = joblib.load("selected_cols.pkl")
        logger.info(
            f"Colonnes sélectionnées chargées: {len(app.state.selected_features)} features"
        )
    except Exception as e:
        logger.error(f"Erreur lors du chargement des colonnes: {e}")
        raise

    yield

    logger.info("Arrêt de l'application...")
    app.state.df = None
    app.state.model = None
    app.state.selected_features = None


# --------------------------------------------------------
# Création FastAPI + CORS
# --------------------------------------------------------
app = FastAPI(
    title="API de Prédiction Étudiante",
    description="API pour prédire la performance des étudiants et proposer des recommandations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# Helpers : post-traitement binaire + recommandations
# --------------------------------------------------------
def proba_to_binary_output(
    proba: np.ndarray,
    row: pd.Series,
) -> Dict[str, Any]:
    """
    proba : array de shape (1, 2) avec [P(classe0), P(classe1)]
    On suppose 0 = non admis, 1 = admis.
    Retourne prediction 0/1 + proba en pourcentage + recommandations.
    """
    p_non_admis_raw = float(proba[0, 0])
    p_admis_raw = float(proba[0, 1])

    # Arrondi en pourcentage
    p_non_admis = round(p_non_admis_raw * 100, 2)
    p_admis = round(p_admis_raw * 100, 2)

    if p_admis_raw >= p_non_admis_raw:
        prediction = 1
        label = "Admis"
        confidence = p_admis
    else:
        prediction = 0
        label = "Non admis"
        confidence = p_non_admis

    # Générer les recommandations
    recos = build_recommendations(row, prediction)

    return {
        "prediction": prediction,
        "prediction_label": label,
        "probabilities": {
            "admis": p_admis,
            "non_admis": p_non_admis,
        },
        "confidence": confidence,
        "recommendations": recos,
    }


def build_recommendations(row: pd.Series, prediction: int) -> List[str]:
    """
    Génère des conseils pédagogiques.
    prediction : 0 = non admis, 1 = admis
    Variables non modifiables :
      - elev_presco
      - nbre_elev_SDC
      - mere_niv_ac
      - etab_prim_stat
    Elles servent seulement à comprendre le contexte, pas à proposer de les changer.
    """
    recos: List[str] = []

    # 1) Message global selon le résultat
    if prediction == 0:
        recos.append(
            "Le risque de non-admission est élevé. Il est recommandé de mettre en place un suivi individualisé et un plan de remédiation."
        )
    else:
        recos.append(
            "La probabilité d'admission est favorable. Il est recommandé de maintenir les bonnes habitudes de travail et le suivi actuel."
        )

    # 2) imp2 : vérification de devoir
    if "imp2" in row.index:
        try:
            v = float(row["imp2"])
        except Exception:
            v = 0.0
        if v == 0:
            recos.append(
                "Il est recommandé de renforcer la vérification des devoirs de l’élève, en s’assurant régulièrement que les exercices sont réalisés et compris."
            )

    # 3) imp3 : supervision de devoir
    if "imp3" in row.index:
        try:
            v = float(row["imp3"])
        except Exception:
            v = 0.0
        if v == 0:
            recos.append(
                "Il est recommandé de renforcer la supervision des devoirs à la maison, en accompagnant l’élève lors de la réalisation de ses travaux scolaires."
            )

    # 4) imp5 : attentes et aspirations en matière d’éducation
    if "imp5" in row.index:
        try:
            v = float(row["imp5"])
        except Exception:
            v = 0.0
        if v == 0:
            recos.append(
                "Il est recommandé de clarifier et de développer les attentes et les aspirations éducatives, en échangeant avec l’élève sur ses objectifs scolaires et futurs."
            )

    # 5) imp6 : lecture avec l’enfant
    if "imp6" in row.index:
        try:
            v = float(row["imp6"])
        except Exception:
            v = 0.0
        if v == 0:
            recos.append(
                "Il est recommandé d’augmenter les moments de lecture partagée avec l’élève, afin de renforcer sa compréhension et son intérêt pour les apprentissages."
            )

    # 6) imp10 : supervision à la maison
    if "imp10" in row.index:
        try:
            v = float(row["imp10"])
        except Exception:
            v = 0.0
        if v == 0:
            recos.append(
                "Il est recommandé de renforcer la supervision de l’élève à la maison (organisation du temps de travail, suivi des devoirs, encadrement des activités) pour améliorer ses conditions d’étude."
            )

    # 7) Rang de table-banc (Ran_TbB) : on agit sur le travail, pas sur la classe elle-même
    if "Ran_TbB" in row.index:
        try:
            rang = float(row["Ran_TbB"])
        except Exception:
            rang = 0.0
        if rang > 15:
            recos.append(
                "Le rang de table-banc de l’élève est défavorable : il peut être utile de rapprocher sa place de l’enseignant et d’organiser des séances de révision ciblées avec suivi des devoirs."
            )

    # 8) Cours de soutien (cour_supl) : modifiable
    if "cour_supl" in row.index:
        try:
            cs = float(row["cour_supl"])
        except Exception:
            cs = 0.0
        if cs == 0:
            recos.append(
                "Il est recommandé de proposer ou de renforcer la participation aux cours de soutien afin d’aider l’élève à rattraper ses difficultés."
            )
        else:
            recos.append(
                "Il est recommandé de valoriser les cours de soutien déjà suivis en fixant des objectifs précis pour chaque séance."
            )

    # 9) Préscolarisation (elev_presco) : contexte, non modifiable
    if "elev_presco" in row.index:
        try:
            ep = float(row["elev_presco"])
        except Exception:
            ep = 0.0
        if ep == 0:
            recos.append(
                "L’élève n’a pas bénéficié de préscolarisation : il est recommandé de prévoir davantage d’activités de base (lecture, écriture, calcul) en classe et à la maison."
            )

    # 10) Nombre d'élèves en classe (nbre_elev_SDC) : contexte, non modifiable
    if "nbre_elev_SDC" in row.index:
        try:
            nb = float(row["nbre_elev_SDC"])
        except Exception:
            nb = 0.0
        if nb >= 50:
            recos.append(
                "La classe est très chargée : il est recommandé de favoriser le travail en petits groupes et le tutorat entre élèves pour mieux accompagner l’élève."
            )

    # 11) Niveau académique de la mère (mere_niv_ac) : contexte, on adapte l’accompagnement
    if "mere_niv_ac" in row.index:
        m = str(row["mere_niv_ac"] or "").upper()
        if m in ["0", "AUCUN", "PRIMAIRE"]:
            recos.append(
                "Compte tenu du niveau académique de la mère, il est recommandé d’adapter la communication avec la famille (explications simples, supports visuels) et de proposer des outils faciles pour suivre les devoirs."
            )

    # 12) Statut de l’établissement (etab_prim_stat) : contexte, on mobilise les ressources
    if "etab_prim_stat" in row.index:
        e = str(row["etab_prim_stat"] or "").lower()
        if "public" in e:
            recos.append(
                "Il est recommandé de mobiliser au maximum les ressources disponibles dans l’établissement (clubs, dispositifs de remédiation, bibliothèque, associations) pour soutenir l’élève."
            )

    recos_uniques = list(dict.fromkeys(recos))
    return recos_uniques

# --------------------------------------------------------
# Modèle Pydantic dynamique (si tu veux l’utiliser plus tard)
# --------------------------------------------------------
def create_student_model(features: List[str]):
    """Crée dynamiquement un modèle Pydantic basé sur les features"""
    fields = {feat: (float, ...) for feat in features}
    return create_model("StudentData", **fields)


StudentData = None

# --------------------------------------------------------
# Endpoint de prédiction
# --------------------------------------------------------
@app.post("/predict")
async def predict(data: Dict[str, Any]):
    """
    Endpoint de prédiction binaire (0 = non admis, 1 = admis) + proba en % + recommandations.
    """
    try:
        if not hasattr(app.state, "model") or app.state.model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")

        df_input = pd.DataFrame([data])

        # Vérifier les colonnes manquantes
        missing_cols = [
            col for col in app.state.selected_features if col not in df_input.columns
        ]
        if missing_cols:
            logger.warning(f"Colonnes manquantes: {missing_cols}")
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes: {missing_cols}",
            )

        # Garder uniquement les features utiles
        df_input = df_input[app.state.selected_features]

        if df_input.isna().any().any():
            logger.warning("Données d'entrée contiennent des valeurs NaN")
            raise HTTPException(
                status_code=400,
                detail="Les données contiennent des valeurs manquantes",
            )

        # Prédiction brute
        prediction_array = app.state.model.predict(df_input)
        prediction = int(prediction_array[0])

        # Probabilités si dispo
        if hasattr(app.state.model, "predict_proba"):
            proba_array = app.state.model.predict_proba(df_input)
            # post-traitement binaire + recommandations
            row = df_input.iloc[0]
            resultat = proba_to_binary_output(proba_array, row)
        else:
            # Cas sans predict_proba : on fabrique une proba simple
            row = df_input.iloc[0]
            if prediction == 1:
                p_admis = 70.0
                p_non = 30.0
            else:
                p_admis = 30.0
                p_non = 70.0
            recos = build_recommendations(row, prediction)
            resultat = {
                "prediction": prediction,
                "prediction_label": "Admis" if prediction == 1 else "Non admis",
                "probabilities": {
                    "admis": p_admis,
                    "non_admis": p_non,
                },
                "confidence": p_admis if prediction == 1 else p_non,
                "recommendations": recos,
            }

        logger.info(
            f"Prédiction réussie: prediction={resultat['prediction']} label={resultat['prediction_label']}"
        )

        return {"success": True, **resultat}

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Erreur de validation des données: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur de format: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


# --------------------------------------------------------
# Gestion 404
# --------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint non trouvé", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
