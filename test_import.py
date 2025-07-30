"""
Test de diagnostic d'import
===========================
Fichier de test pour diagnostiquer les problèmes d'import
"""

import sys
import os

print("🔍 DIAGNOSTIC D'IMPORT")
print("=" * 50)

# 1. Vérification du dossier courant
print(f"📁 Dossier courant: {os.getcwd()}")

# 2. Vérification des fichiers
fichiers_requis = [
    'modelisation_optimal.py',
    'models/modele_production_final.pkl',
    'models/encoders_production_final.pkl',
    'models/scaler_production_final.pkl',
    'models/features_production_final.pkl',
    'models/metadata_production_final.pkl'
]

print("\n📋 Vérification des fichiers:")
for fichier in fichiers_requis:
    if os.path.exists(fichier):
        print(f"✅ {fichier}")
    else:
        print(f"❌ {fichier} - MANQUANT")

# 3. Test d'import
print("\n🧪 Test d'import:")
try:
    print("Tentative d'import du module...")
    import modelisation_optimal
    print("✅ Import du module réussi")
    
    print("Tentative d'import de la fonction...")
    from modelisation_optimal import predict_price_production
    print("✅ Import de la fonction réussi")
    
    print("Test de la fonction...")
    prix = predict_price_production(80, 3, '75', 0, 'Appartement', 0)
    if prix:
        print(f"✅ Fonction fonctionne - Prix test: {prix:,.0f}€")
    else:
        print("❌ Fonction retourne None")
        
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")

print("\n" + "=" * 50)
print("Lancez ce fichier avec: python test_import.py")