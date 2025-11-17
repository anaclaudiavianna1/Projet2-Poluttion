import streamlit as st
import pandas as pd
import requests
from pathlib import Path



# ---------------------------------------------------
# Configuration générale de la page Streamlit
# ---------------------------------------------------
st.set_page_config(page_title="Projet 2 – Qualité de l’air", layout="wide")

# URL de l’API Flask (backend)
BACKEND_URL = "http://127.0.0.1:5000"


# ---------------------------------------------------
# Fonctions utilitaires pour communiquer avec le backend
# ---------------------------------------------------
@st.cache_data
def get_features():
    """
    Récupère la liste des variables explicatives (features)
    et le nom de la variable cible auprès du backend.
    """
    r = requests.get(f"{BACKEND_URL}/features")
    r.raise_for_status()
    return r.json()


def list_models():
    """
    Récupère la liste des modèles sauvegardés auprès du backend.
    """
    r = requests.get(f"{BACKEND_URL}/models")
    r.raise_for_status()
    return r.json().get("models", [])


def train_selected(models):
    """
    Envoie au backend la liste des modèles à entraîner.
    Si le backend renvoie une erreur (code HTTP != 200),
    on affiche le message d’erreur au lieu de faire planter l’application.
    """
    payload = {"models": models}
    r = requests.post(f"{BACKEND_URL}/train", json=payload)

    # Gestion explicite des erreurs côté backend
    if r.status_code != 200:
        try:
            message = r.json().get("error", "Erreur inconnue côté backend.")
        except Exception:
            message = f"Code HTTP {r.status_code} sans message détaillé."
        st.error(f"Erreur pendant l'entraînement des modèles : {message}")
        return {}

    return r.json().get("results", {})


def predict_api(model_name, feat_dict):
    """
    Envoie une requête de prédiction au backend pour un modèle donné
    et un dictionnaire de caractéristiques (feat_dict).
    """
    payload = {"model": model_name, "data": feat_dict}
    r = requests.post(f"{BACKEND_URL}/predict", json=payload)

    if r.status_code != 200:
        try:
            message = r.json().get("error", "Erreur inconnue côté backend.")
        except Exception:
            message = f"Code HTTP {r.status_code} sans message détaillé."
        st.error(f"Erreur pendant la prédiction : {message}")
        return {}

    return r.json()


# ---------------------------------------------------
# Chargement des informations de features depuis le backend
# ---------------------------------------------------
features_info = get_features()
FEATURE_COLUMNS = features_info["features"]
TARGET_COLUMN = features_info["target"]


# ---------------------------------------------------
# Menu latéral (navigation entre les pages)
# ---------------------------------------------------
pages = ["Accueil", "Apprentissage & comparaison",
         "Prédiction", "Backend (API)"]
choice = st.sidebar.radio("Navigation", pages)


# ---------------------------------------------------
# 1) Page d’accueil
# ---------------------------------------------------
if choice == "Accueil":
    st.title("Tableau de bord – Projet 2 : Qualité de l’air")

    st.markdown("""
    Bienvenue dans l’application de **classification et prédiction de la qualité de l’air**.
    Les informations ci-dessous sont présentées sous forme de « cartes », comme des articles
    de nouvelles : un petit résumé visible, et les détails en cliquant.
    """)

    # ----------------- Carte 1 : Résumé du projet -----------------
    with st.container():
        st.subheader("📰 Résumé du projet")
        st.write(
            "Cette application permet d’analyser un jeu de données de pollution et "
            "de prédire la qualité de l’air à l’aide de plusieurs modèles "
            "d’apprentissage automatique."
        )

        with st.expander("Voir les objectifs détaillés du projet"):
            st.markdown("""
            **Objectifs détaillés :**

            - Charger le dataset de pollution (`pollution.csv`).
            - Entraîner plusieurs modèles de classification :
              **KNN, Decision Tree, Random Forest, Logistic Regression, SVM, Naïve Bayes**.
            - Comparer les performances :
              **accuracy, précision, rappel, F1-score, matrice de confusion**.
            - Sauvegarder les modèles entraînés au format `.pkl` dans le dossier `/models`.
            - Effectuer des prédictions en temps réel via l’interface Streamlit
              (formulaire ou fichier CSV).
            """)

    st.markdown("---")

    # ----------------- Carte 2 : Jeu de données -----------------
    with st.container():
        st.subheader("📊 Jeu de données – `pollution.csv`")
        st.write(
            "Le jeu de données contient des mesures de capteurs (polluants, météo, "
            "proximité de zones industrielles, densité de population, etc.) ainsi qu’un "
            "label de qualité de l’air."
        )

        with st.expander("Afficher / masquer un aperçu du dataset"):
            uploaded = st.file_uploader(
                "Charger un fichier CSV (optionnel – sinon `data/pollution.csv` sera utilisé)",
                type=["csv"],
                key="csv_accueil",
            )

            if uploaded is not None:
                df = pd.read_csv(uploaded)
                st.success("Fichier chargé depuis l’upload.")
                st.dataframe(df.head())
            else:
                data_path = Path(__file__).resolve().parents[1] / "data" / "pollution.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    st.info("Aucun fichier uploadé. Affichage de `data/pollution.csv` (5 premières lignes).")
                    st.dataframe(df.head())
                else:
                    st.warning("Aucun fichier `pollution.csv` trouvé dans le dossier `data`.")

    st.markdown("---")




# ---------------------------------------------------
# 2) Page « Apprentissage & comparaison »
# ---------------------------------------------------
if choice == "Apprentissage & comparaison":
    st.title("Apprentissage et comparaison des modèles")

    # Dictionnaire {nom interne : nom lisible}
    model_names = {
        "knn": "K-Nearest Neighbors",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "log_reg": "Logistic Regression",
        "svm": "SVM",
        "naive_bayes": "Naïve Bayes",
    }

    st.markdown("Sélectionnez les modèles à entraîner :")

    selected = st.multiselect(
        "Modèles",
        list(model_names.keys()),
        default=list(model_names.keys())
    )

    # On utilise l'état de session pour garder les résultats
    if "train_results" not in st.session_state:
        st.session_state["train_results"] = {}

    # --------- Bouton d'apprentissage ----------
    if st.button("Lancer l'apprentissage"):
        if not selected:
            st.error("Veuillez sélectionner au moins un modèle.")
        else:
            with st.spinner("Apprentissage en cours..."):
                results = train_selected(selected)

            if not results:
                st.error("Aucun résultat reçu du backend.")
            else:
                st.success("Apprentissage terminé !")
                # On sauvegarde les résultats dans la session
                st.session_state["train_results"] = results

    # --------- Affichage des résultats ----------
    results = st.session_state.get("train_results", {})

    if results:
        # Tableau des métriques
        df_metrics = pd.DataFrame.from_dict(results, orient="index")

        # On garde uniquement les scores principaux
        colonnes_scores = ["accuracy", "precision", "recall", "f1"]
        colonnes_scores = [c for c in colonnes_scores if c in df_metrics.columns]

        if colonnes_scores:
            df_metrics_display = df_metrics[colonnes_scores]

            st.subheader("Tableau comparatif des performances")
            st.dataframe(df_metrics_display.style.format("{:.3f}"))

            # Graphique des accuracies, si disponible
            if "accuracy" in df_metrics_display.columns:
                st.subheader("Graphique des accuracies")
                st.bar_chart(df_metrics_display["accuracy"])

        # Matrice de confusion
        st.subheader("Matrice de confusion")

        model_for_cm = st.selectbox(
            "Choisir un modèle pour afficher la matrice de confusion",
            list(results.keys())
        )

        cm = results[model_for_cm]["confusion_matrix"]
        classes = results[model_for_cm]["classes"]

        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        st.write("Lignes = valeurs réelles, colonnes = valeurs prédites")
        st.dataframe(cm_df)
    else:
        st.info("Aucun modèle n’a encore été entraîné. "
                "Choisissez des modèles et cliquez sur « Lancer l’apprentissage ».")    

# ---------------------------------------------------
# 3) Page « Prédiction »
# ---------------------------------------------------
if choice == "Prédiction":
    st.title("Prédiction en temps réel de la qualité de l’air")

    st.markdown("""
    Cette page permet de :
    - choisir un **modèle sauvegardé** (fichier .pkl dans le dossier `/backend/models`) ;
    - saisir manuellement de nouvelles mesures **ou** charger un fichier CSV ;
    - afficher la prédiction du niveau de qualité de l’air  
      (par exemple : **Bonne, Modérée, Mauvaise, Dangereuse**).
    """)

    # 1) Liste des modèles déjà entraînés
    models_files = list_models()

    if not models_files:
        st.warning("Aucun modèle sauvegardé. "
                   "Allez dans « Apprentissage & comparaison » pour entraîner au moins un modèle.")
    else:
        # Sélection d’un modèle sauvegardé
        selected_model = st.selectbox("Choisir un modèle entraîné", models_files)

        # Choix de la source des données : formulaire ou CSV
        mode = st.radio("Source des nouvelles mesures :", ["Saisie manuelle", "Fichier CSV"])

        # ---------- Mode : saisie manuelle ----------
        if mode == "Saisie manuelle":
            st.subheader("Formulaire – entrer manuellement les valeurs des capteurs")

            cols = st.columns(2)
            valeurs = {}

            for i, feat in enumerate(FEATURE_COLUMNS):
                with cols[i % 2]:
                    v = st.number_input(feat, value=0.0, format="%.4f")
                valeurs[feat] = v

            if st.button("Prédire à partir du formulaire"):
                with st.spinner("Prédiction en cours..."):
                    res = predict_api(selected_model, valeurs)

                if res and "prediction" in res:
                    label = res["prediction"][0]
                    st.success(f"Niveau de qualité de l’air prédit : **{label}**")
                    if "proba" in res:
                        st.write(f"Probabilité associée : {res['proba'][0]:.2f}")
                else:
                    st.error(res.get("error", "Erreur inconnue."))

        # ---------- Mode : fichier CSV ----------
        else:
            st.subheader("Charger un fichier CSV avec de nouvelles mesures")

            uploaded = st.file_uploader(
                "Le fichier doit contenir au moins les colonnes suivantes : "
                + ", ".join(FEATURE_COLUMNS),
                type=["csv"]
            )

            if uploaded is not None:
                df_new = pd.read_csv(uploaded)
                st.write("Aperçu des données :")
                st.dataframe(df_new.head())

                # Choix de la ligne pour la prédiction
                max_index = len(df_new) - 1
                index_ligne = st.number_input(
                    "Indice de la ligne à prédire (0 = première ligne)",
                    min_value=0,
                    max_value=max_index,
                    value=0,
                    step=1,
                )

                st.info("Pour respecter l’énoncé, on effectue une prédiction en temps réel "
                        "sur la ligne sélectionnée du fichier CSV.")

                if st.button("Prédire pour la ligne sélectionnée"):
                    ligne = df_new.iloc[index_ligne]
                    valeurs = ligne[FEATURE_COLUMNS].to_dict()

                    with st.spinner("Prédiction en cours..."):
                        res = predict_api(selected_model, valeurs)

                    if res and "prediction" in res:
                        label = res["prediction"][0]
                        st.success(f"Prédiction pour la ligne {index_ligne} : **{label}**")
                        if "proba" in res:
                            st.write(f"Probabilité associée : {res['proba'][0]:.2f}")
                    else:
                        st.error(res.get("error", "Erreur inconnue."))


# ---------------------------------------------------
# 4) Page « Backend (API) »
# ---------------------------------------------------
if choice == "Backend (API)":
    st.title("Backend – API Flask")

    st.markdown(f"""
    Cette page documente l’API REST du backend Flask.

    - URL de base : `{BACKEND_URL}`  

    **Endpoints principaux :**

    1. `POST /train` : entraîner un ou plusieurs modèles  
       Corps JSON :
       ```json
       {{"models": ["knn", "svm", "random_forest"]}}
       ```
       Si `"models"` est omis, tous les modèles disponibles sont entraînés.

    2. `POST /predict` : renvoyer la prédiction pour une entrée donnée  
       Corps JSON :
       ```json
       {{
         "model": "random_forest",
         "data": {{"PM2.5": 10.0, "PM10": 20.0, "...": 0.0}}
       }}
       ```

    3. `GET /models` : lister les modèles disponibles (fichiers `.pkl` dans `/backend/models`).
    """)

    st.subheader("Réponse de `/features` (variables explicatives et cible)")
    st.json(features_info)

    st.subheader("Modèles actuellement sauvegardés (`GET /models`)")
    try:
        st.json({"models": list_models()})
    except Exception as e:
        st.error(f"Impossible de contacter le backend : {e}")

    st.markdown("---")
    st.subheader("Tester rapidement l’API")

    col1, col2 = st.columns(2)

    # Test /train
    with col1:
        st.markdown("**Tester `/train` avec un seul modèle (par ex. `decision_tree`)**")
        modele_test = st.selectbox(
            "Modèle à entraîner pour le test",
            ["knn", "decision_tree", "random_forest", "log_reg", "svm", "naive_bayes"],
            key="backend_train_model",
        )
        if st.button("Lancer /train (test API)"):
            with st.spinner("Appel de l’API /train..."):
                res = train_selected([modele_test])
            if res:
                st.success("Entraînement API réussi.")
                st.write(res)

    # Test /predict
    with col2:
        st.markdown("**Tester `/predict` avec des valeurs fictives**")

        # Construire un dictionnaire de valeurs fictives (zéro)
        valeurs_fictives = {feat: 0.0 for feat in FEATURE_COLUMNS}