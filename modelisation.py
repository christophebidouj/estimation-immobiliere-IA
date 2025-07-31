"""
=================================================================
MODÉLISATION IA IMMOBILIÈRE - VERSION PÉDAGOGIQUE POUR DÉBUTANTS
=================================================================

Ce fichier montre comment créer un modèle d'IA pour estimer les prix immobiliers.
Objectif pédagogique : comprendre les étapes de base du Machine Learning.

Étapes principales :
1. Charger les données
2. Nettoyer les données  
3. Créer des variables (features)
4. Entraîner le modèle IA
5. Évaluer les performances
6. Faire des prédictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configuration simple
TAILLE_ECHANTILLON = 500_000  # Nombre de lignes à utiliser
FICHIER_DONNEES = "data/valeurs_foncieres_nettoye/valeurs_foncieres_nettoye_2020_2024.csv"
DOSSIER_MODELES = "models"

def charger_donnees_immobilieres(fichier, taille_max):
    """
    ÉTAPE 1 : Charger les données immobilières
    
    Cette fonction lit un fichier CSV contenant des ventes immobilières
    et garde seulement les meilleures données pour l'IA.
    """
    print("📁 Chargement des données immobilières...")
    
    # Lire le fichier par petits morceaux (plus efficace)
    morceaux = []
    lignes_lues = 0
    
    for morceau in pd.read_csv(fichier, chunksize=100_000, low_memory=False):
        lignes_lues += len(morceau)
        morceaux.append(morceau)
        
        # Arrêter si on a assez de données
        if lignes_lues >= taille_max:
            break
    
    # Combiner tous les morceaux
    donnees = pd.concat(morceaux, ignore_index=True)
    print(f"✅ {len(donnees):,} lignes chargées")
    
    return donnees

def nettoyer_donnees(donnees):
    """
    ÉTAPE 2 : Nettoyer les données
    
    On garde seulement les ventes avec toutes les informations nécessaires
    et on supprime les valeurs aberrantes (trop grandes ou trop petites).
    """
    print("🧹 Nettoyage des données...")
    
    # Garde seulement les lignes avec les infos essentielles
    donnees_propres = donnees[
        donnees['Valeur fonciere'].notna() &           # Prix existe
        donnees['Surface reelle bati'].notna() &       # Surface existe  
        donnees['Nombre pieces principales'].notna()   # Nombre de pièces existe
    ]
    
    # Supprime les valeurs aberrantes (prix et surface impossibles)
    donnees_propres = donnees_propres[
        (donnees_propres['Valeur fonciere'] >= 12_000) &        # Prix mini 12k€
        (donnees_propres['Valeur fonciere'] <= 4_000_000) &     # Prix maxi 4M€
        (donnees_propres['Surface reelle bati'] >= 10) &        # Surface mini 10m²
        (donnees_propres['Surface reelle bati'] <= 800) &       # Surface maxi 800m²
        (donnees_propres['Nombre pieces principales'] >= 1) &   # Mini 1 pièce
        (donnees_propres['Nombre pieces principales'] <= 15)    # Maxi 15 pièces
    ]
    
    print(f"✅ {len(donnees_propres):,} lignes après nettoyage")
    return donnees_propres

def creer_variables_departement(donnees):
    """
    Créer la variable département à partir du code postal
    Exemple : 75001 → département 75 (Paris)
    """
    if 'Code postal corrigé' in donnees.columns:
        donnees['dept'] = donnees['Code postal corrigé'].astype(str).str[:2]
        
        # Garde seulement les départements avec assez de données
        comptes_dept = donnees['dept'].value_counts()
        dept_frequents = comptes_dept[comptes_dept >= 100].index.tolist()
        donnees['dept'] = donnees['dept'].apply(
            lambda x: x if x in dept_frequents else 'autres'
        )
    else:
        donnees['dept'] = 'autres'
    
    return donnees

def creer_variables_temporelles(donnees):
    """
    Créer des variables sur les dates de vente
    L'IA peut apprendre si certaines périodes ont des prix différents
    """
    if 'Date mutation' in donnees.columns:
        donnees['Date mutation'] = pd.to_datetime(donnees['Date mutation'], errors='coerce')
        donnees = donnees.dropna(subset=['Date mutation'])
        
        # Extraire l'année et le mois
        donnees['annee'] = donnees['Date mutation'].dt.year
        donnees['mois'] = donnees['Date mutation'].dt.month
        
        # Variable "récent" : 1 si vente après 2023, 0 sinon
        donnees['recent'] = (donnees['annee'] >= 2023).astype(int)
        
        # Créer les saisons
        saisons = {12: 'hiver', 1: 'hiver', 2: 'hiver', 
                  3: 'printemps', 4: 'printemps', 5: 'printemps',
                  6: 'ete', 7: 'ete', 8: 'ete',
                  9: 'automne', 10: 'automne', 11: 'automne'}
        donnees['saison'] = donnees['mois'].map(saisons)
    else:
        # Valeurs par défaut si pas de date
        donnees['recent'] = 0
        donnees['saison'] = 'ete'
        donnees['annee'] = 2022
    
    return donnees

def creer_variables_ia(donnees):
    """
    ÉTAPE 3 : Créer les variables pour l'IA
    
    L'IA a besoin de variables numériques pour apprendre.
    On transforme nos données brutes en variables utiles.
    """
    print("🔧 Création des variables pour l'IA...")
    
    # Variables de base (directement du dataset)
    variables = pd.DataFrame({
        'surface': donnees['Surface reelle bati'],
        'pieces': donnees['Nombre pieces principales'],
    })
    
    # Ajouter le terrain si disponible
    if 'Surface terrain' in donnees.columns:
        variables['terrain'] = donnees['Surface terrain'].fillna(0)
        variables['avec_terrain'] = (variables['terrain'] > 50).astype(int)
    else:
        variables['terrain'] = 0
        variables['avec_terrain'] = 0
    
    # Variables calculées (l'IA apprend mieux avec ces transformations)
    variables['surface_par_piece'] = variables['surface'] / np.maximum(variables['pieces'], 1)
    variables['log_surface'] = np.log1p(variables['surface'])  # log(surface+1)
    variables['surface_x_pieces'] = variables['surface'] * variables['pieces']
    
    # Variables de catégorie (petite/moyenne/grande surface)
    variables['petit_logement'] = (variables['surface'] < 40).astype(int)
    variables['grand_logement'] = (variables['surface'] > 120).astype(int)
    
    # Ajouter les variables créées précédemment
    for col in ['dept', 'recent', 'saison', 'annee']:
        if col in donnees.columns:
            variables[col] = donnees[col]
    
    if 'Type local' in donnees.columns:
        variables['type_local'] = donnees['Type local'].fillna('Appartement')
    else:
        variables['type_local'] = 'Appartement'
    
    print(f"✅ {len(variables.columns)} variables créées")
    return variables, donnees['Valeur fonciere']

def entrainer_modele_ia(X_train, y_train):
    """
    ÉTAPE 4 : Entraîner le modèle d'IA
    
    On utilise 3 algorithmes différents et on les combine :
    - Random Forest : très bon pour l'immobilier
    - Extra Trees : apporte de la diversité  
    - Ridge : évite le sur-apprentissage
    """
    print("🤖 Entraînement du modèle d'IA...")
    
    # Transformation logarithmique du prix (l'IA apprend mieux comme ça)
    y_log = np.log1p(y_train)
    
    # Algorithme 1 : Random Forest (forêt aléatoire)
    foret_aleatoire = RandomForestRegressor(
        n_estimators=100,        # Nombre d'arbres dans la forêt
        max_depth=20,           # Profondeur maximale des arbres
        min_samples_split=10,   # Minimum d'exemples pour diviser
        min_samples_leaf=5,     # Minimum d'exemples dans une feuille
        random_state=42,        # Pour reproduire les résultats
        n_jobs=-1              # Utilise tous les processeurs
    )
    
    # Algorithme 2 : Extra Trees (arbres extra-aléatoires)
    arbres_extra = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=18,
        min_samples_split=12,
        random_state=43,
        n_jobs=-1
    )
    
    # Algorithme 3 : Ridge (régression linéaire régularisée)
    regression_ridge = Ridge(alpha=75.0)
    
    # Combinaison des 3 algorithmes (ensemble)
    modele_combine = VotingRegressor([
        ('foret', foret_aleatoire),      # Poids 55%
        ('extra', arbres_extra),         # Poids 30% 
        ('ridge', regression_ridge)      # Poids 15%
    ], weights=[0.55, 0.3, 0.15])
    
    # Entraînement du modèle combiné
    modele_combine.fit(X_train, y_log)
    
    print("✅ Modèle d'IA entraîné")
    return modele_combine

def evaluer_modele(modele, X_train, X_test, y_train, y_test):
    """
    ÉTAPE 5 : Évaluer les performances du modèle
    
    On mesure si le modèle fait de bonnes prédictions avec des métriques :
    - R² : qualité globale (0 = mauvais, 1 = parfait)
    - MAE : erreur moyenne en euros
    """
    print("📊 Évaluation des performances...")
    
    # Prédictions sur les données d'entraînement et de test
    y_pred_log_train = modele.predict(X_train)
    y_pred_log_test = modele.predict(X_test)
    
    # Retransformation vers les prix réels
    y_pred_train = np.expm1(y_pred_log_train)
    y_pred_test = np.expm1(y_pred_log_test)
    
    # Correction du biais (le modèle sous-estime souvent)
    facteur_correction = np.mean(y_train) / np.mean(y_pred_train)
    facteur_correction = max(0.5, min(2.0, facteur_correction))  # Limite sécurité
    
    y_pred_train_corrige = y_pred_train * facteur_correction
    y_pred_test_corrige = y_pred_test * facteur_correction
    
    # Calcul des métriques
    r2_train = r2_score(y_train, y_pred_train_corrige)
    r2_test = r2_score(y_test, y_pred_test_corrige)
    mae_train = mean_absolute_error(y_train, y_pred_train_corrige)
    mae_test = mean_absolute_error(y_test, y_pred_test_corrige)
    
    # Mesure du sur-apprentissage
    suraprentissage = abs(r2_train - r2_test)
    
    # Affichage des résultats
    print(f"\n🏆 RÉSULTATS DU MODÈLE :")
    print(f"📈 R² (qualité) : {r2_test:.3f} (plus proche de 1 = mieux)")
    print(f"💰 Erreur moyenne : {mae_test:,.0f}€")
    print(f"🎯 Sur-apprentissage : {suraprentissage:.3f} (plus petit = mieux)")
    
    # Interprétation des résultats
    if r2_test > 0.35:
        print("✅ Très bon modèle !")
    elif r2_test > 0.25:
        print("✅ Bon modèle")
    else:
        print("⚠️ Modèle à améliorer")
    
    return {
        'r2_test': r2_test,
        'mae_test': mae_test,
        'facteur_correction': facteur_correction,
        'suraprentissage': suraprentissage
    }

def sauvegarder_modele(modele, encodeurs, normaliseur, colonnes, resultats):
    """
    Sauvegarder le modèle entraîné pour l'utiliser plus tard
    """
    print("💾 Sauvegarde du modèle...")
    
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    
    # Sauvegarde des composants
    joblib.dump(modele, f'{DOSSIER_MODELES}/modele_ia.pkl')
    joblib.dump(encodeurs, f'{DOSSIER_MODELES}/encodeurs.pkl')
    joblib.dump(normaliseur, f'{DOSSIER_MODELES}/normaliseur.pkl')
    joblib.dump(colonnes, f'{DOSSIER_MODELES}/colonnes.pkl')
    
    # Métadonnées
    infos_modele = {
        'performance': resultats,
        'description': 'Modèle IA simplifié pour débutants',
        'facteur_correction': resultats['facteur_correction']
    }
    joblib.dump(infos_modele, f'{DOSSIER_MODELES}/infos_modele.pkl')
    
    print("✅ Modèle sauvegardé avec succès !")

def estimer_prix_immobilier(surface, pieces, dept='75', terrain=0, type_local='Appartement', recent=1):
    """
    ÉTAPE 6 : Faire une prédiction avec le modèle entraîné
    
    Cette fonction utilise le modèle sauvegardé pour estimer le prix
    d'un bien immobilier selon ses caractéristiques.
    """
    try:
        # Chargement du modèle et de ses composants
        modele = joblib.load(f'{DOSSIER_MODELES}/modele_ia.pkl')
        encodeurs = joblib.load(f'{DOSSIER_MODELES}/encodeurs.pkl')
        normaliseur = joblib.load(f'{DOSSIER_MODELES}/normaliseur.pkl')
        colonnes = joblib.load(f'{DOSSIER_MODELES}/colonnes.pkl')
        infos = joblib.load(f'{DOSSIER_MODELES}/infos_modele.pkl')
        
        print(f"🏠 Estimation pour : {surface}m², {pieces} pièces, dept {dept}")
        
        # Création du vecteur de caractéristiques
        donnees_bien = {
            'surface': surface,
            'pieces': pieces,
            'terrain': terrain,
            'avec_terrain': 1 if terrain > 50 else 0,
            'surface_par_piece': surface / max(pieces, 1),
            'log_surface': np.log1p(surface),
            'surface_x_pieces': surface * pieces,
            'petit_logement': 1 if surface < 40 else 0,
            'grand_logement': 1 if surface > 120 else 0,
            'dept': dept,
            'recent': recent,
            'saison': 'ete',
            'annee': 2024,
            'type_local': type_local
        }
        
        # Conversion en DataFrame
        df_bien = pd.DataFrame([donnees_bien])
        
        # Encodage des variables catégorielles (texte → chiffres)
        for colonne in ['dept', 'type_local', 'saison']:
            if colonne in encodeurs:
                encodeur = encodeurs[colonne]
                valeur = df_bien[colonne].iloc[0]
                try:
                    if hasattr(encodeur, 'classes_') and valeur in encodeur.classes_:
                        df_bien[colonne] = encodeur.transform([valeur])[0]
                    else:
                        df_bien[colonne] = 0
                except:
                    df_bien[colonne] = 0
        
        # Ajout des colonnes manquantes
        for col in colonnes:
            if col not in df_bien.columns:
                df_bien[col] = 0
        
        # Réorganisation des colonnes dans le bon ordre
        df_final = df_bien[colonnes]
        
        # Normalisation des données
        df_normalise = normaliseur.transform(df_final)
        
        # Prédiction
        prediction_log = modele.predict(df_normalise)[0]
        prix_brut = np.expm1(prediction_log)
        
        # Application de la correction
        facteur_correction = infos['facteur_correction']
        prix_final = prix_brut * facteur_correction
        
        # Vérification de cohérence avec les prix du marché
        prix_final = verifier_coherence_prix(prix_final, surface, dept, type_local, terrain)
        
        print(f"💰 Prix estimé : {prix_final:,.0f}€")
        print(f"📐 Prix au m² : {prix_final/surface:,.0f}€/m²")
        
        return prix_final
        
    except FileNotFoundError:
        print("❌ Modèle non trouvé. Entraînez d'abord le modèle.")
        return None
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

def verifier_coherence_prix(prix, surface, dept, type_local, terrain):
    """
    Vérification de cohérence pour éviter les prix aberrants
    """
    # Prix de référence par m² selon la région
    prix_reference_m2 = {
        '75': 9200,   # Paris
        '92': 6500,   # Hauts-de-Seine
        '13': 3500,   # Marseille
        '69': 4200,   # Lyon
        '33': 3800,   # Bordeaux
    }
    
    prix_m2_ref = prix_reference_m2.get(dept, 3000)  # 3000€/m² par défaut
    prix_attendu = prix_m2_ref * surface
    
    # Ajustements selon le type
    if type_local == 'Maison':
        prix_attendu *= 1.05  # +5% pour les maisons
    if terrain > 200:
        prix_attendu *= 1.1   # +10% pour grand terrain
    
    # Si le prix est trop différent, on ajuste
    ratio = prix / prix_attendu
    if ratio > 2.0:  # Trop cher
        prix_corrige = prix * 0.4 + prix_attendu * 0.6
    elif ratio < 0.5:  # Trop bon marché
        prix_corrige = prix * 0.6 + prix_attendu * 0.4
    else:
        prix_corrige = prix
    
    return prix_corrige

def entrainer_modele_complet():
    """
    PIPELINE COMPLET : Toutes les étapes d'entraînement du modèle IA
    """
    print("🚀 ENTRAÎNEMENT COMPLET DU MODÈLE IA")
    print("=" * 50)
    
    try:
        # Étape 1 : Charger les données
        donnees_brutes = charger_donnees_immobilieres(FICHIER_DONNEES, TAILLE_ECHANTILLON)
        
        # Étape 2 : Nettoyer les données
        donnees_propres = nettoyer_donnees(donnees_brutes)
        donnees_propres = creer_variables_departement(donnees_propres)
        donnees_propres = creer_variables_temporelles(donnees_propres)
        
        # Étape 3 : Créer les variables pour l'IA
        X, y = creer_variables_ia(donnees_propres)
        
        print(f"📊 Dataset final : {len(X):,} lignes, {len(X.columns)} variables")
        print(f"💰 Prix : de {y.min():,.0f}€ à {y.max():,.0f}€")
        
        # Division train/test (80% entraînement, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Encodage des variables catégorielles
        X_train_encode = X_train.copy()
        X_test_encode = X_test.copy()
        encodeurs = {}
        
        colonnes_texte = X_train_encode.select_dtypes(include=['object']).columns
        for col in colonnes_texte:
            encodeur = LabelEncoder()
            X_train_encode[col] = encodeur.fit_transform(X_train_encode[col].astype(str))
            
            # Encoder le test en gérant les nouvelles valeurs
            valeurs_test = X_test_encode[col].astype(str)
            valeurs_encodees = []
            for val in valeurs_test:
                try:
                    valeurs_encodees.append(encodeur.transform([val])[0])
                except:
                    valeurs_encodees.append(0)  # Valeur par défaut
            X_test_encode[col] = valeurs_encodees
            encodeurs[col] = encodeur
        
        # Normalisation des données
        normaliseur = RobustScaler()
        X_train_normalise = normaliseur.fit_transform(X_train_encode)
        X_test_normalise = normaliseur.transform(X_test_encode)
        
        # Conversion en DataFrame
        X_train_final = pd.DataFrame(X_train_normalise, columns=X_train_encode.columns)
        X_test_final = pd.DataFrame(X_test_normalise, columns=X_train_encode.columns)
        
        # Étape 4 : Entraîner le modèle
        modele = entrainer_modele_ia(X_train_final, y_train)
        
        # Étape 5 : Évaluer les performances
        resultats = evaluer_modele(modele, X_train_final, X_test_final, y_train, y_test)
        
        # Étape 6 : Sauvegarder si satisfaisant
        if resultats['r2_test'] > 0.25:  # Seuil minimum
            sauvegarder_modele(modele, encodeurs, normaliseur, X_train_final.columns.tolist(), resultats)
            print("\n🎉 MODÈLE ENTRAÎNÉ AVEC SUCCÈS !")
            return True
        else:
            print("\n⚠️ Performances insuffisantes, modèle non sauvegardé")
            return False
            
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement : {e}")
        return False

def tester_modele():
    """
    Tester le modèle avec quelques exemples
    """
    print("\n🧪 TESTS DU MODÈLE")
    print("=" * 30)
    
    exemples = [
        {"desc": "Studio parisien", "params": {"surface": 25, "pieces": 1, "dept": "75"}},
        {"desc": "Appartement parisien", "params": {"surface": 60, "pieces": 3, "dept": "75"}},
        {"desc": "Maison banlieue", "params": {"surface": 120, "pieces": 5, "dept": "92", "terrain": 300, "type_local": "Maison"}},
        {"desc": "Appartement Lyon", "params": {"surface": 75, "pieces": 3, "dept": "69"}},
    ]
    
    for exemple in exemples:
        print(f"\n📍 {exemple['desc']} :")
        prix = estimer_prix_immobilier(**exemple['params'])
        if prix:
            surface = exemple['params']['surface']
            print(f"   📊 {prix/surface:,.0f}€/m²")

if __name__ == "__main__":
    print("🏠 MODÈLE IA IMMOBILIER - VERSION PÉDAGOGIQUE")
    print("=" * 55)
    print("📚 Ce code montre les bases du Machine Learning appliqué à l'immobilier")
    print()
    
    # Entraînement du modèle
    succes = entrainer_modele_complet()
    
    # Test si l'entraînement a réussi
    if succes:
        tester_modele()
    
    print("\n💡 Pour utiliser ce modèle dans Streamlit, lancez app.py")