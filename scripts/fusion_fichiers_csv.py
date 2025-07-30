#fusion_fichiers_csv.py
import pandas as pd

# Liste des chemins vers les fichiers CSV déjà convertis
chemins_csv = [
    "data/valeurs_foncieres_nettoye/ValeursFoncieres-2020.csv",
    "data/valeurs_foncieres_nettoye/ValeursFoncieres-2021.csv",
    "data/valeurs_foncieres_nettoye/ValeursFoncieres-2022.csv",
    "data/valeurs_foncieres_nettoye/ValeursFoncieres-2023.csv",
    "data/valeurs_foncieres_nettoye/ValeursFoncieres-2024.csv"
]

# Fusion des fichiers CSV
df_liste = []
for chemin in chemins_csv:
    print(f"Chargement de {chemin}...")
    df = pd.read_csv(chemin, low_memory=False)
    df_liste.append(df)

df_complet = pd.concat(df_liste, ignore_index=True)

# Sauvegarde éventuelle du fichier fusionné (optionnelle)
df_complet.to_csv("data/valeurs_foncieres_nettoye/ValeursFoncieres_2020_2024.csv", index=False)

print("✔ Fusion terminée. Le fichier combiné contient", len(df_complet), "lignes.")
