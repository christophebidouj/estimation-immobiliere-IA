# 🏠 Estimation du prix des biens immobiliers (France entière)

Ce projet vise à prédire le prix de vente de biens immobiliers résidentiels à partir des données publiques des **valeurs foncières**.  
Il repose sur l'utilisation de modèles de machine learning, en particulier un **HistGradientBoostingRegressor**, pour fournir une estimation indicative du prix à partir de caractéristiques simples comme :

- la surface habitable,
- le nombre de pièces,
- la surface du terrain,
- la date de vente,
- et un code postal simplifié.

L'application finale est accessible via une interface interactive **Streamlit**.



## 🎯 Objectif

- Nettoyer, transformer et enrichir les données DVF brutes issues du site officiel [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/).
- Développer un **modèle d’estimation robuste**, capable de s’adapter à des biens partout en France.
- Créer une **application web intuitive**, testable par tous, pour simuler le prix d’un bien immobilier.
- Comprendre et expérimenter l’ensemble de la chaîne **data science / machine learning / déploiement**.



## 🧠 Pourquoi ce projet ?

Ce projet a été réalisé dans le cadre d’un **apprentissage personnel de la data science et des outils Python associés** (pandas, scikit-learn, Streamlit…). 

> 🧩 Il m’a permis d'explorer concrètement les problématiques de nettoyage de données massives, de modélisation prédictive, de gestion des limites de performances, et de création d'une interface interactive pour utilisateurs.

> 💬 Ce projet vise avant tout à illustrer une démarche complète : du nettoyage des données jusqu’au déploiement d’un modèle fonctionnel.



## ⚙️ Fonctionnement de l’application

L'application développée avec **Streamlit** permet à un utilisateur de renseigner les caractéristiques d’un bien et d’obtenir une estimation instantanée.

1. Le modèle (`model_hist_gradient_boosting.pkl`) est chargé au lancement.
2. Les valeurs du formulaire sont transformées pour correspondre au format du modèle d'entraînement.
3. L'estimation est calculée et présentée à l'utilisateur.
4. Un encart optionnel détaille les performances réelles du modèle utilisé (R², MAE, etc.).

⚠️ **Limites connues** :
- Estimation indicative, R² = **0.55**.
- Le modèle ne capte pas toutes les spécificités locales ou l’état du bien.
- L’interface fonctionne avec un **code postal simplifié** (3 premiers chiffres), pas au niveau adresse.



## 🗃️ Données

**Source officielle** : [https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/)

Données traitées (2020–2024) :
- `valeur_fonciere` (cible) – prix de vente
- `surface_reelle_bati`, `nombre_pieces_principales`, `surface_terrain`
- `date_mutation` → transformée en `année`, `mois`, `saison`
- `type_local` – Appartement ou Maison
- `code_postal_simplifie` – les 3 premiers chiffres uniquement



## 🔍 Traitement & Modélisation

### 📦 Nettoyage :
- Suppression des lignes incomplètes ou aberrantes (prix < 10 000 € ou > 1,5M €, surface < 10 m² ou > 500 m²…).
- Uniformisation des codes postaux valides (431 conservés sur ~20M de lignes).
- Échantillonnage à 5% pour accélérer l'entraînement (≈ 1,4M lignes).

### 🛠️ Modèle :
- **HistGradientBoostingRegressor**
- **Encodage des variables :** Target Encoding
- R² sur le jeu test : **0.55**
- MAE : **82 765 €**
- MSE : **15 008 322 552**

Le modèle est sauvegardé et utilisé dans l'application via `joblib`.



## 💡 Interface interactive

L'application permet de :
- Saisir les caractéristiques du bien immobilier
- Obtenir une estimation immédiate
- Consulter les performances du modèle
- Être averti du caractère indicatif de la prédiction

> L'application a été développée avec l’aide d’outils d’IA, à des fins pédagogiques et d’apprentissage technique.



## 🔗 Mise en ligne

L’application est disponible ici : (https://estimation-immobiliere-par-intelligence-artificielle.streamlit.app/)



## 🛠️ Environnement & outils

- **Langage** : Python 3.11
- **Data Science** : pandas, numpy, scikit-learn
- **Interface** : Streamlit
- **IDE** : Visual Studio Code



## 📁 Structure du projet

projet-prix-maisons/
├── data/ # Données brutes ou nettoyées (non trackées)
├── nettoyage_donnees_VF.py # Script de nettoyage
├── modelisation.py # Entraînement du modèle
├── app_streamlit.py # Interface web interactive
├── model_hist_gbr.pkl # Modèle sauvegardé
├── README.md # Ce fichier




## 🙋‍♂️ Auteur

Projet développé par **Christophe Bidouj**, dans le cadre d’un **apprentissage personnel de la data science appliquée à un cas concret**.  
Il s’agit d’un projet d’exploration pédagogique, visant à mettre en pratique des compétences en manipulation de données, modélisation, et création d'applications interactives.





## 🧾 Licence

Ce projet est librement réutilisable dans un cadre pédagogique ou personnel.
