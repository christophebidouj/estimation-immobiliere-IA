#txt_to_csv.py

import pandas as pd

# Liste des années à traiter
annees = [2020, 2021, 2022, 2023, 2024]

for annee in annees:
    fichier_txt = f"data/valeurs_foncieres_brutes/ValeursFoncieres-{annee}.txt"
    fichier_csv = f"data/valeurs_foncieres_nettoye/ValeursFoncieres-{annee}.csv"
    
    print(f"Lecture de {fichier_txt}...")
    df = pd.read_csv(fichier_txt, sep="|", encoding="utf-8", low_memory=False)

    print(f"→ Sauvegarde en {fichier_csv}")
    df.to_csv(fichier_csv, index=False)