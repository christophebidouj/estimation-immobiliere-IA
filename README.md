# Estimateur de prix immobilier

Estimation de prix à partir des données DVF (data.gouv.fr) — 1M+ transactions, France entière, 2020-2024.

**[Application en ligne](https://estimation-immobiliere-par-intelligence-artificielle.streamlit.app/)**



## Stack

- ML : scikit-learn (RandomForest, ExtraTrees, Ridge), pandas, numpy, joblib
- Frontend : Streamlit + CSS personnalisé
- Déploiement : Streamlit Cloud (continu via Git)

## Modélisation

Ensemble **RF (55%) + ExtraTrees (30%) + Ridge (15%)**, poids fixés par validation croisée. RF pour les non-linéarités, ET pour réduire la variance, Ridge pour stabiliser les extrêmes.

Correction post-prédiction par zone géographique (431 codes postaux) plutôt qu'encodage géographique en feature, pour éviter le surapprentissage sur les micro-marchés.

RobustScaler (médiane/IQR) à la place de StandardScaler, les prix immobiliers ayant des distributions très asymétriques.

## Résultats

R² > 0.35 — MAE ~80K€ — < 200ms par estimation — 431 zones corrigées

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
