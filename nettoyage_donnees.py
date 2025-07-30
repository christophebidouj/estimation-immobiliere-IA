"""
Script de nettoyage de données immobilières pour une application d'estimation de prix par IA
======================================================================================

Ce script transforme des données brutes de valeurs foncières en données propres
utilisables pour l'entraînement d'un modèle d'IA d'estimation de prix immobilier.

Objectifs principaux:
1. Charger et examiner les données brutes
2. Nettoyer et standardiser les codes postaux
3. Convertir les valeurs foncières au bon format numérique
4. Sauvegarder les données nettoyées

Auteur: [christophe bidouj]
"""

# =============================================================================
# IMPORTATION DES BIBLIOTHÈQUES
# =============================================================================

import pandas as pd  # Pour la manipulation des données tabulaires (DataFrames)
import numpy as np   # Pour les opérations mathématiques et la gestion des valeurs manquantes


# =============================================================================
# CONFIGURATION ET CONSTANTES
# =============================================================================

# Chemins des fichiers - centraliser ici facilite la maintenance
INPUT_FILE_PATH = "data/valeurs_foncieres_nettoye/ValeursFoncieres_2020_2024.csv"
OUTPUT_FILE_PATH = "data/valeurs_foncieres_nettoye/valeurs_foncieres_nettoye_2020_2024.csv"

# Paramètres de traitement
MAX_POSTAL_CODES_TO_DISPLAY = 50  # Nombre max de codes postaux à afficher dans l'audit
TOP_CODES_TO_DISPLAY = 10         # Nombre de codes postaux les plus fréquents à afficher


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def load_real_estate_data(file_path):
    """
    Charge le fichier CSV contenant les données de valeurs foncières.
    
    Args:
        file_path (str): Chemin vers le fichier CSV à charger
        
    Returns:
        pandas.DataFrame: DataFrame contenant les données chargées
        
    Note:
        low_memory=False évite les avertissements de type de données mixtes
    """
    print("🔄 Chargement des données...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"✅ Données chargées avec succès: {len(df):,} lignes")
        return df
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier {file_path} n'a pas été trouvé")
        raise
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        raise


def audit_postal_codes(df, column_name='Code postal'):
    """
    Effectue un audit des codes postaux pour comprendre leur état actuel.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        column_name (str): Nom de la colonne contenant les codes postaux
    """
    print(f"\n📊 AUDIT DES CODES POSTAUX - Colonne: '{column_name}'")
    print("=" * 60)
    
    # Statistiques générales
    total_values = len(df[column_name])
    non_null_values = df[column_name].notna().sum()
    unique_values = df[column_name].nunique()
    
    print(f"📈 Valeurs totales: {total_values:,}")
    print(f"📈 Valeurs non nulles: {non_null_values:,}")
    print(f"📈 Valeurs uniques: {unique_values:,}")
    
    # Exemples de codes postaux
    print(f"\n🔍 Exemples de codes postaux (max {MAX_POSTAL_CODES_TO_DISPLAY}):")
    unique_codes = df[column_name].dropna().unique()[:MAX_POSTAL_CODES_TO_DISPLAY]
    print(unique_codes)


def correct_postal_code(postal_code):
    """
    Corrige et standardise un code postal français.
    
    Un code postal français valide doit avoir exactement 5 chiffres.
    Cette fonction tente de corriger les codes postaux malformés.
    
    Args:
        postal_code: Code postal à corriger (peut être de n'importe quel type)
        
    Returns:
        str or np.nan: Code postal corrigé (5 chiffres) ou NaN si impossible à corriger
        
    Exemples:
        correct_postal_code("75001") -> "75001"
        correct_postal_code("7500") -> "75000" (complété avec des zéros)
        correct_postal_code("750") -> NaN (trop court)
        correct_postal_code("abc") -> NaN (pas de chiffres)
    """
    # Gérer les valeurs manquantes (NaN, None, etc.)
    if pd.isna(postal_code):
        return np.nan
    
    # Convertir en string et supprimer les espaces
    postal_code_str = str(postal_code).strip()
    
    # Extraire uniquement les chiffres
    digits_only = ''.join(filter(str.isdigit, postal_code_str))
    
    # Vérifier si on a au moins 4 chiffres (minimum pour être corrigible)
    if len(digits_only) >= 4:
        # Compléter avec des zéros à droite et tronquer à 5 chiffres
        corrected_code = digits_only.ljust(5, '0')[:5]
        return corrected_code
    
    # Si moins de 4 chiffres, impossible à corriger
    return np.nan


def clean_postal_codes(df, input_column='Code postal', output_column='Code postal corrigé'):
    """
    Nettoie tous les codes postaux d'un DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        input_column (str): Nom de la colonne source
        output_column (str): Nom de la colonne de destination
        
    Returns:
        pandas.DataFrame: DataFrame avec la colonne de codes postaux corrigés ajoutée
    """
    print(f"\n🧹 NETTOYAGE DES CODES POSTAUX")
    print("=" * 40)
    
    # Appliquer la correction à chaque code postal
    print("🔄 Application de la correction...")
    df[output_column] = df[input_column].apply(correct_postal_code)
    
    # Statistiques post-correction
    valid_codes = df[output_column].notna().sum()
    unique_codes = df[output_column].nunique()
    
    print(f"✅ Codes postaux corrigés valides: {valid_codes:,}")
    print(f"✅ Codes postaux uniques après correction: {unique_codes:,}")
    
    return df


def show_top_postal_codes(df, column_name='Code postal corrigé', top_n=TOP_CODES_TO_DISPLAY):
    """
    Affiche les codes postaux les plus fréquents.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        column_name (str): Nom de la colonne à analyser
        top_n (int): Nombre de codes à afficher
    """
    print(f"\n🏆 TOP {top_n} des codes postaux les plus fréquents:")
    print("-" * 50)
    top_codes = df[column_name].value_counts().head(top_n)
    for code, count in top_codes.items():
        print(f"📍 {code}: {count:,} occurrences")


def clean_property_values(df, column_name='Valeur fonciere'):
    """
    Nettoie et convertit les valeurs foncières au format numérique.
    
    En France, les nombres décimaux utilisent la virgule comme séparateur.
    Cette fonction convertit au format international (point) puis en numérique.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        column_name (str): Nom de la colonne contenant les valeurs foncières
        
    Returns:
        pandas.DataFrame: DataFrame avec les valeurs foncières nettoyées
    """
    print(f"\n💰 NETTOYAGE DES VALEURS FONCIÈRES")
    print("=" * 40)
    
    print("🔄 Remplacement des virgules par des points...")
    # Remplacer les virgules par des points (format français -> international)
    df[column_name] = df[column_name].str.replace(',', '.', regex=False)
    
    print("🔄 Conversion en format numérique...")
    # Convertir en numérique, les valeurs non convertibles deviennent NaN
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    # Statistiques post-conversion
    valid_values = df[column_name].notna().sum()
    print(f"✅ Valeurs foncières valides après conversion: {valid_values:,}")
    
    return df


def save_cleaned_data(df, output_path):
    """
    Sauvegarde le DataFrame nettoyé dans un fichier CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame à sauvegarder
        output_path (str): Chemin de destination du fichier
    """
    print(f"\n💾 SAUVEGARDE DES DONNÉES")
    print("=" * 30)
    
    try:
        df.to_csv(output_path, index=False)
        print(f"✅ Fichier sauvegardé avec succès: {output_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        raise


def display_final_summary(df, cleaned_column='Code postal corrigé'):
    """
    Affiche un résumé final du processus de nettoyage.
    
    Args:
        df (pandas.DataFrame): DataFrame final
        cleaned_column (str): Nom de la colonne nettoyée principale
    """
    print(f"\n📊 RÉSUMÉ FINAL")
    print("=" * 20)
    
    total_rows = len(df)
    valid_postal_codes = df[cleaned_column].notna().sum()
    
    print(f"📦 Nombre total de lignes: {total_rows:,}")
    print(f"📮 Codes postaux valides: {valid_postal_codes:,}")
    print(f"📈 Taux de codes postaux valides: {(valid_postal_codes/total_rows)*100:.2f}%")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """
    Fonction principale qui orchestre tout le processus de nettoyage.
    
    Cette approche modulaire facilite la maintenance et les tests.
    """
    print("🏠 NETTOYAGE DES DONNÉES IMMOBILIÈRES")
    print("=" * 50)
    
    try:
        # Étape 1: Chargement des données
        df = load_real_estate_data(INPUT_FILE_PATH)
        
        # Étape 2: Audit initial des codes postaux
        audit_postal_codes(df, 'Code postal')
        
        # Étape 3: Nettoyage des codes postaux
        df = clean_postal_codes(df)
        
        # Étape 4: Audit post-nettoyage
        audit_postal_codes(df, 'Code postal corrigé')
        show_top_postal_codes(df)
        
        # Étape 5: Nettoyage des valeurs foncières
        df = clean_property_values(df)
        
        # Étape 6: Sauvegarde
        save_cleaned_data(df, OUTPUT_FILE_PATH)
        
        # Étape 7: Résumé final
        display_final_summary(df)
        
        print("\n🎉 PROCESSUS TERMINÉ AVEC SUCCÈS!")
        
    except Exception as e:
        print(f"\n💥 ERREUR FATALE: {e}")
        print("Le processus a été interrompu.")
        raise


# =============================================================================
# POINT D'ENTRÉE DU SCRIPT
# =============================================================================

if __name__ == "__main__":
    """
    Point d'entrée du script.
    
    Cette structure permet d'importer ce fichier comme module sans exécuter main()
    automatiquement, ce qui est utile pour les tests et la réutilisabilité.
    """
    main()