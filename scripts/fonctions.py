#fonctions

def verification_concordance_donnees(model, input_data):
    """
    Vérifie la concordance entre les colonnes attendues par le modèle et celles fournies dans input_data.

    Args:
        model: un modèle sklearn avec l'attribut `feature_names_in_`
        input_data: un DataFrame pandas contenant les données à vérifier

    Affiche les colonnes manquantes ou supplémentaires.
    """

    # Vérification préalable
    if not hasattr(model, "feature_names_in_"):
        print("❌ Le modèle ne contient pas l'attribut 'feature_names_in_'.")
        return

    missing_cols = set(model.feature_names_in_) - set(input_data.columns)
    extra_cols = set(input_data.columns) - set(model.feature_names_in_)

    print("✔️ Colonnes attendues par le modèle :", len(model.feature_names_in_))
    print("✔️ Colonnes fournies :", len(input_data.columns))

    if missing_cols:
        print("❌ Colonnes manquantes dans input_data :", missing_cols)
    if extra_cols:
        print("⚠️ Colonnes supplémentaires dans input_data :", extra_cols)
    if not missing_cols and not extra_cols:
        print("✅ Les colonnes correspondent parfaitement au modèle.")
