from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# ---------------------------------------------------------
# Chemins de base du projet
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "pollution.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Détermination automatique des colonnes du dataset
# - Toutes les colonnes sauf la dernière = features
# - Dernière colonne = variable cible
# ---------------------------------------------------------
df_exemple = pd.read_csv(DATA_PATH)
toutes_les_colonnes = df_exemple.columns.tolist()

TARGET_COLUMN = toutes_les_colonnes[-1]
FEATURE_COLUMNS = toutes_les_colonnes[:-1]

print("=== Configuration du dataset ===")
print("Colonnes explicatives (FEATURE_COLUMNS) :", FEATURE_COLUMNS)
print("Colonne cible (TARGET_COLUMN)           :", TARGET_COLUMN)
print("=================================")

# Initialisation de l’app Flask
app = Flask(__name__)


# ---------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------
def charger_jeu_de_donnees():
    """
    Charge le fichier pollution.csv et sépare X (features) et y (cible).
    """
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def get_classifiers():
    """
    Retourne les classifieurs demandés dans l’énoncé.
    """
    return {
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "log_reg": LogisticRegression(max_iter=1000),
        "svm": SVC(probability=True),
        "naive_bayes": GaussianNB(),
    }


def entrainer_un_modele(nom_modele: str):
    """
    Entraîne un modèle donné, calcule les métriques,
    sauvegarde le modèle dans /models et retourne les résultats.
    """
    classifieurs = get_classifiers()
    if nom_modele not in classifieurs:
        raise ValueError(f"Modèle inconnu : {nom_modele}")

    # 1) Charger les données
    X, y = charger_jeu_de_donnees()

    # 2) Encoder la cible
    encodeur = LabelEncoder()
    y_enc = encodeur.fit_transform(y)

    # 3) Séparation entraînement / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )

    # 4) Pipeline = imputeur + normalisation + classifieur
    modele = classifieurs[nom_modele]

    pipeline = Pipeline(
        steps=[
            ("imputeur", SimpleImputer(strategy="median")),
            ("normalisation", StandardScaler()),
            ("classifieur", modele),
        ]
    )

    # 5) Apprentissage
    pipeline.fit(X_train, y_train)

    # 6) Prédiction sur l’ensemble de test
    y_pred = pipeline.predict(X_test)

    # 7) Métriques
    exactitude = float(accuracy_score(y_test, y_pred))
    precision_macro = float(
        precision_score(y_test, y_pred, average="macro", zero_division=0)
    )
    rappel_macro = float(
        recall_score(y_test, y_pred, average="macro", zero_division=0)
    )
    f1_macro = float(
        f1_score(y_test, y_pred, average="macro", zero_division=0)
    )
    matrice_conf = confusion_matrix(y_test, y_pred).tolist()

    indices_classes = np.unique(y_enc)
    noms_classes = encodeur.inverse_transform(indices_classes).tolist()

    # 8) Sauvegarde automatique du modèle dans /models
    objet_a_sauvegarder = {
        "pipeline": pipeline,
        "encoder": encodeur,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "classes": noms_classes,
    }
    chemin_modele = MODELS_DIR / f"{nom_modele}.pkl"
    joblib.dump(objet_a_sauvegarder, chemin_modele)

    # 9) Résultats renvoyés au frontend
    return {
        "model_name": nom_modele,
        "accuracy": exactitude,
        "precision": precision_macro,
        "recall": rappel_macro,
        "f1": f1_macro,
        "confusion_matrix": matrice_conf,
        "classes": noms_classes,
    }


# ---------------------------------------------------------
# Endpoints de l’API
# ---------------------------------------------------------
@app.route("/features", methods=["GET"])
def features():
    """
    Renvoie les features et la cible au frontend.
    """
    return jsonify({"features": FEATURE_COLUMNS, "target": TARGET_COLUMN})


@app.route("/train", methods=["POST"])
def train():
    """
    Entraîne un ou plusieurs modèles, sauvegarde chaque modèle dans /models
    et renvoie les métriques pour chaque modèle.
    """
    donnees = request.get_json(force=True) or {}
    tous_les_modeles = list(get_classifiers().keys())
    modeles_a_entrainer = donnees.get("models") or tous_les_modeles

    resultats = {}
    try:
        for nom in modeles_a_entrainer:
            metriques = entrainer_un_modele(nom)
            resultats[nom] = metriques
    except Exception as e:
        print("ERREUR PENDANT /train :", e)
        return jsonify({"error": str(e)}), 500

    return jsonify({"results": resultats})


@app.route("/models", methods=["GET"])
def models():
    """
    Renvoie la liste des modèles déjà sauvegardés dans /models.
    """
    fichiers = [f.name for f in MODELS_DIR.glob("*.pkl")]
    noms_sans_extension = [f.replace(".pkl", "") for f in fichiers]
    return jsonify({"models": noms_sans_extension})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Effectue une prédiction pour une ou plusieurs observations.
    """
    donnees = request.get_json(force=True) or {}

    nom_modele = donnees.get("model")
    X_input = donnees.get("data")

    if nom_modele is None:
        return jsonify({"error": "Champ 'model' manquant."}), 400
    if X_input is None:
        return jsonify({"error": "Champ 'data' manquant."}), 400

    chemin_modele = MODELS_DIR / f"{nom_modele}.pkl"
    if not chemin_modele.exists():
        return jsonify({"error": f"Modèle {nom_modele} introuvable."}), 404

    payload = joblib.load(chemin_modele)
    pipeline = payload["pipeline"]
    encodeur = payload["encoder"]
    features = payload["features"]

    # On accepte un dictionnaire (une observation) ou une liste de dictionnaires
    if isinstance(X_input, dict):
        df = pd.DataFrame([X_input])
    else:
        df = pd.DataFrame(X_input)

    df = df.reindex(columns=features)

    indices_predits = pipeline.predict(df)
    etiquettes = encodeur.inverse_transform(indices_predits).tolist()

    try:
        probabilites = pipeline.predict_proba(df).max(axis=1).tolist()
    except Exception:
        probabilites = None

    resultat = {"prediction": etiquettes}
    if probabilites is not None:
        resultat["proba"] = probabilites

    return jsonify(resultat)


# ---------------------------------------------------------
# Lancement du serveur Flask
# ---------------------------------------------------------
if __name__ == "__main__":
    # Le serveur écoute sur http://127.0.0.1:5000
    app.run(port=5000, debug=True)
