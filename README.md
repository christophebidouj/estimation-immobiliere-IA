# 🏠 Estimateur Immobilier par Intelligence Artificielle

Système d'estimation de prix immobilier utilisant des modèles de machine learning et les données publiques des valeurs foncières françaises. Ce projet pédagogique démontre un pipeline complet de data science avec interface web moderne.

**Technologies :** Python, Scikit-learn, Streamlit, Ensemble Methods, Feature Engineering

---

## 🎯 Vue d'ensemble

Ce projet développe une solution d'estimation immobilière basée sur l'analyse de plus d'un million de transactions immobilières françaises (2020-2024). Il illustre une approche complète combinant preprocessing de données, modélisation par ensemble learning, et déploiement via interface web.

### Objectifs du projet :
- **Pipeline de données** : ETL pour traitement de datasets volumineux
- **Modélisation avancée** : Ensemble de modèles avec correction automatique des biais
- **Interface moderne** : Application web avec UX/UI soignée
- **Code professionnel** : Structure modulaire, gestion d'erreurs, documentation

## 🏗️ Architecture technique

### Stack technologique
```
Backend ML     : scikit-learn, pandas, numpy, joblib
Frontend       : Streamlit, CSS/HTML personnalisé
Data Pipeline  : ETL personnalisé, validation automatique
Déploiement    : Streamlit Cloud
```

### Composants principaux

#### 1. **Pipeline de données** (`nettoyage_donnees.py`)
- **ETL automatisé** avec gestion d'erreurs complète
- **Validation de données** : codes postaux, valeurs foncières, cohérence géographique
- **Optimisation mémoire** : traitement par chunks pour gros volumes
- **Audit complet** : statistiques de qualité à chaque étape

#### 2. **Moteur de ML** (`modelisation.py`)
- **Feature Engineering** : création de 15+ variables dérivées
- **Ensemble Learning** : Random Forest + Extra Trees + Ridge Regression
- **Preprocessing** : LabelEncoder, RobustScaler, gestion des outliers
- **Gestion des modèles** : sauvegarde, métadonnées de performance

#### 3. **Interface utilisateur** (`app.py`)
- **Design moderne** : système de design personnalisé, animations CSS
- **Validation temps réel** : contrôles de cohérence métier
- **UX intelligente** : recommandations contextuelles, retours visuels
- **Responsive design** : adaptation multi-device

---

## 🔬 Modélisation & Performance

### Approche algorithmique

**Ensemble Method** combinant :
- **Random Forest** (55%) : robustesse sur données hétérogènes
- **Extra Trees** (30%) : réduction de variance, diversité des prédictions  
- **Ridge Regression** (15%) : régularisation linéaire, stabilité

### Feature Engineering

```python
# Variables de base transformées
surface_par_piece, log_surface, surface_x_pieces

# Variables géographiques
departement → encodage avec gestion des nouveaux codes

# Variables temporelles
saison, recent, annee → encodage cyclique

# Segmentation métier
petit_logement, grand_logement, avec_terrain
```

### Métriques de performance
- **R² Score** : > 0.35 (objectif qualité)
- **Correction des biais** : facteur automatique selon écart marché
- **Robustesse** : validation croisée, gestion des outliers
- **Cohérence** : test sur données non vues, validation géographique

### Innovation : Correction Automatique v3
Algorithme développé pour ajuster les prédictions selon :
- **Prix de référence** par zone géographique (431 codes postaux)
- **Facteurs correctifs** adaptatifs selon type de bien
- **Seuils de sécurité** pour éviter les aberrations par région

---

## 📊 Données & Traitement

### Dataset
- **Source** : Données publiques DVF (data.gouv.fr)
- **Volume** : 1M+ transactions analysées
- **Période** : 2020-2024
- **Couverture** : France entière, 431 codes postaux

### Pipeline de traitement
```
Données brutes (20M+ lignes)
    ↓ Nettoyage automatisé
Données validées (5M+ lignes)  
    ↓ Feature engineering
Dataset ML (1M+ lignes)
    ↓ Échantillonnage stratifié
Training set (500K lignes)
```

### Qualité des données
- **Taux de données valides** : 85%+ après nettoyage
- **Couverture géographique** : Toutes régions françaises
- **Distribution temporelle** : Équilibrée sur 2020-2024

---

## 🚀 Interface & Expérience Utilisateur

### Fonctionnalités développées

**Estimation intelligente :**
- Validation temps réel des saisies utilisateur
- Calcul instantané avec retours progressifs
- Correction automatique selon contexte marché

**Analyses intégrées :**
- Métriques détaillées : fourchette, prix/m², benchmark régional
- Analyse géographique contextuelle
- Indicateurs de cohérence marché

**Interface moderne :**
- Design system avec thème sombre
- Animations CSS fluides
- Affichage progressif des informations

### Architecture frontend
```
Streamlit Core
    ↓ CSS personnalisé (animations, responsive)
Composants Interface
    ↓ Gestion d'état & validation
Logique Métier
    ↓ API Modèle ML
Prédictions Temps Réel
```

---

## 💼 Déploiement & Mise en ligne

### Infrastructure
- **Hébergement** : Streamlit Cloud
- **Déploiement** : Automatique via Git sur Streamlit Cloud
- **Architecture** : Chaque requête est indépendante
- **Monitoring** : Analytics Streamlit intégrées

### Gestion des modèles
- **Sauvegarde** : joblib avec compression
- **Métadonnées** : performances et paramètres inclus
- **Validation** : vérification de cohérence avec prix de référence
- **Évolution** : pipeline de mise à jour des modèles

---

## 📋 Structure du projet

```
estimateur-immobilier-ia/
├── 📁 data/                          # Datasets (non versionnés)
│   └── valeurs_foncieres_nettoye/    
├── 📁 models/                        # Artefacts ML
│   ├── modele_ia.pkl                 # Modèle ensemble
│   ├── encodeurs.pkl                 # Encodeurs de features  
│   ├── normaliseur.pkl               # Pipeline de normalisation
│   └── infos_modele.pkl              # Métadonnées du modèle
├── 🔧 nettoyage_donnees.py           # Pipeline ETL
├── 🤖 modelisation.py                # Pipeline d'entraînement ML  
├── 🎨 app.py                         # Application Streamlit
└── 📖 README.md                      # Documentation
```

---

## 🔧 Installation & Utilisation

### Prérequis
```bash
Python 3.11+
pip install streamlit pandas scikit-learn numpy joblib
```

### Pipeline complet
```bash
# 1. Nettoyage des données
python nettoyage_donnees.py

# 2. Entraînement du modèle
python modelisation.py

# 3. Lancement de l'application
streamlit run app.py
```

### API du modèle
```python
from modelisation import estimer_prix_immobilier

prix = estimer_prix_immobilier(
    surface=75, pieces=3, dept='75',
    terrain=0, type_local='Appartement', recent=1
)
# Retourne: estimation en euros
```

---

## 📈 Résultats & Performances

### Métriques techniques
- **Temps de réponse** : < 200ms pour estimation complète
- **Précision** : MAE ~80K€ sur biens standards
- **Robustesse** : gestion automatique des scénarios extrêmes
- **Disponibilité** : déployé en continu sur Streamlit Cloud

### Fonctionnalités métier
- **Estimation instantanée** avec niveau de confiance
- **Analyse comparative** du marché local
- **Interface intuitive** pour utilisateurs non-techniques
- **Transparence** des facteurs de calcul

### Cas d'usage
- **Pré-estimation** pour agences immobilières
- **Validation de cohérence** des prix de marché
- **Outil d'aide à la négociation** pour particuliers
- **Base de développement** pour applications PropTech

---

## 🎯 Perspectives d'amélioration

### Enrichissement des données
- **Données géographiques** : INSEE, proximité transports
- **Variables qualitatives** : état du bien, prestations, vue
- **Données temps réel** : tendances marché actuelles
- **Segmentation fine** : micro-marchés locaux

### Améliorations techniques
- **Modèles avancés** : gradient boosting, réseaux de neurones
- **API REST** : service web pour intégrations externes
- **Cache intelligent** : optimisation des prédictions fréquentes
- **Tests unitaires** : validation de la qualité du code

---

## 🏆 Compétences développées

### Data Science & ML
- **Feature Engineering** sur données immobilières complexes
- **Ensemble Methods** avec optimisation des poids
- **Correction de biais** et validation statistique
- **Déploiement de modèles** en environnement web

### Développement logiciel
- **Architecture modulaire** avec séparation des responsabilités
- **Qualité de code** : documentation, gestion d'erreurs, tests
- **Développement UI/UX** : interface moderne et responsive
- **Déploiement web** : versioning Git et mise en ligne automatique

### Expertise métier
- **Analyse immobilière** : compréhension des facteurs de prix
- **Validation de données** : cohérence métier et géographique
- **Expérience utilisateur** : workflow optimisé pour cas d'usage réels
- **Intelligence métier** : métriques et indicateurs pertinents

---

## 📞 À propos

**Auteur :** Christophe Bidouj  
**Objectif :** Démonstration de compétences en Data Science et développement d'applications ML

> Ce projet illustre une approche complète de développement de solution d'intelligence artificielle, de la conception à la mise en production, avec un focus sur la qualité du code et l'expérience utilisateur.

🔗 **Application en ligne :** [Tester l'estimateur](https://estimation-immobiliere-par-intelligence-artificielle.streamlit.app/)

---

*Projet pédagogique développé en 2024 - Démonstration de compétences en Data Science et ML Engineering*