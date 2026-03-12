# Estimateur de prix immobilier

Estimation de prix à partir des données DVF (data.gouv.fr) — 1M+ transactions, France entière, 2020-2024.

**[Application en ligne](https://estimation-immobiliere-par-intelligence-artificielle.streamlit.app/)**



## Stack

- ML : scikit-learn (RandomForest, ExtraTrees, Ridge), pandas, numpy, joblib
- Frontend : Streamlit + CSS personnalisé
- Déploiement : Streamlit Cloud (continu via Git)

## Modélisation

**Ensemble RF (55%) + ExtraTrees (30%) + Ridge (15%)**, poids fixés par validation croisée. RF capte les non-linéarités sur données hétérogènes, ET réduit la variance par randomisation accrue, Ridge stabilise les prédictions aux extrêmes.

**Transformation logarithmique du prix cible** (`log1p` à l'entraînement, `expm1` à la prédiction) — les prix immobiliers ont une distribution très asymétrique avec quelques valeurs extrêmes légitimes ; le log normalise cette distribution et réduit l'impact des outliers sur l'erreur.

**RobustScaler** (médiane/IQR) plutôt que StandardScaler — même raison : les châteaux et biens atypiques ne doivent pas distordre la normalisation des features.

**Correction post-prédiction en deux étapes** plutôt qu'encodage géographique fin en feature (risque de surapprentissage) :
- Étape 1 (`modelisation.py`) : facteur correctif global calculé sur l'écart moyen train/prédiction
- Étape 2 (`app.py`) : blend ML + prix de référence par zone (~33 zones : arrondissements parisiens, départements IDF, grandes métropoles), avec intensité variable selon l'écart entre prédiction et marché

## Résultats

R² > 0.35 — MAE ~80K€ — < 200ms par estimation — ~33 zones de référence

## Structure

- `nettoyage_donnees.py` — ETL : 20M → 1M lignes filtrées
- `modelisation.py` — Entraînement, feature engineering, sauvegarde des artefacts
- `app.py` — Interface Streamlit
- `models/` — Artefacts ML (non versionnés)
- `data/` — Dataset DVF brut (non versionné)

## Lancement

```bash
pip install streamlit pandas scikit-learn numpy joblib

python nettoyage_donnees.py   # nécessite les fichiers DVF bruts dans data/
python modelisation.py
streamlit run app.py
```



**Auteur :** Christophe Bidouj
