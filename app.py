"""
===========================================================
INTERFACE STREAMLIT IMMOBILIÈRE - VERSION PÉDAGOGIQUE
===========================================================

Ce fichier créer une belle interface web avec Streamlit
pour utiliser notre modèle d'IA immobilière.

Streamlit transforme du code Python en application web !
"""

import streamlit as st
import numpy as np
import warnings

# Supprimer les warnings sklearn pour une interface plus propre
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Configuration de la page web
st.set_page_config(
    page_title="Estimateur Immobilier IA",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le bouton d'estimation et le bloc de résultat
st.markdown("""
<style>
    /* Style pour le bouton d'estimation */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Animation de pulsation pour attirer l'attention */
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
        50% { box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6); }
        100% { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
    }
    
    .stButton > button {
        animation: pulse 2s infinite !important;
    }
    
    /* Style pour le bloc de résultat - adaptatif selon le thème */
    .price-result-box {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white !important;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .price-value {
        color: white !important;
        margin: 0 !important;
        font-size: 2.8em !important;
        font-weight: bold !important;
    }
    
    .price-label {
        color: white !important;
        margin: 10px 0 !important;
        font-size: 1.2em !important;
        opacity: 0.9;
    }
    
    .price-description {
        color: white !important;
        margin: 0 !important;
        font-size: 0.95em !important;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Import de notre modèle IA
try:
    from modelisation import estimer_prix_immobilier
except ImportError:
    st.error("❌ Impossible d'importer le modèle. Vérifiez que 'modelisation.py' est présent.")
    st.stop()

def formater_affichage_ville(code_postal):
    """
    Fonction pour afficher joliment la localisation
    Exemple : 75001 → "Paris 1er"
    """
    if not code_postal or len(code_postal) != 5:
        return "Paris"
    
    if code_postal.startswith('75'):
        arrondissement = code_postal[2:4]
        if arrondissement and arrondissement != '00':
            numero = arrondissement.lstrip('0')
            if numero:
                return f"Paris {numero}ème"
        return "Paris"
    
    departement = code_postal[:2]
    return f"Département {departement}"

def calculer_prix_realiste(prix_ia, surface, code_postal, type_bien, terrain=0):
    """
    Fonction qui corrige les prix de l'IA pour qu'ils soient plus réalistes
    en les comparant aux prix moyens du marché français.
    """
    
    # Prix moyens réels par m² en France (données 2024)
    prix_moyen_m2 = {
        # Paris par arrondissement
        '75001': 12000, '75002': 10500, '75003': 9800, '75004': 11500,
        '75005': 10200, '75006': 12500, '75007': 12000, '75008': 13000,
        '75009': 9200, '75010': 8500, '75011': 8800, '75012': 8000,
        '75013': 7500, '75014': 8500, '75015': 8800, '75016': 11000,
        '75017': 9200, '75018': 7500, '75019': 7000, '75020': 8000,
        '75': 9200,  # Paris en général
        
        # Île-de-France
        '92': 6500, '93': 3800, '94': 4200, '77': 3000, 
        '78': 3800, '91': 3400, '95': 3200,
        
        # Grandes villes
        '06': 5200,  # Nice
        '13': 3500,  # Marseille
        '69': 4200,  # Lyon
        '33': 3800,  # Bordeaux
        '31': 3400,  # Toulouse
    }
    
    # Récupérer le prix de référence
    dept = code_postal[:2] if len(code_postal) > 2 else code_postal
    prix_ref_m2 = prix_moyen_m2.get(code_postal, prix_moyen_m2.get(dept, 2800))
    
    # Calculer le prix de référence pour ce bien
    prix_reference = prix_ref_m2 * surface
    
    # Ajustements selon le type de bien
    if type_bien == 'Maison' and surface > 0:
        # Les maisons sont généralement plus chères que les appartements
        facteur_maison = 1.08  # +8%
        
        # Ajustement selon la taille
        if surface > 150:
            facteur_maison *= 0.95  # Légère décote pour très grandes surfaces
        elif surface < 80:
            facteur_maison *= 1.05  # Prime pour petites maisons (plus rares)
        
        # Bonus pour le terrain
        if terrain > 200:
            facteur_maison *= 1.02  # +2% pour grand terrain
        
        prix_reference *= facteur_maison
    
    elif type_bien == 'Local':
        prix_reference *= 0.75  # -25% pour local commercial
    elif type_bien == 'Dépendance':
        prix_reference *= 0.6   # -40% pour dépendance
    
    print(f"🔍 Prix IA: {prix_ia:,.0f}€, Prix référence: {prix_reference:,.0f}€")
    
    # Logique de correction intelligente
    if prix_ia <= 0:
        return prix_reference
    
    ratio = prix_ia / prix_reference
    
    # Correction selon l'écart avec le marché
    if 0.7 <= ratio <= 1.3:
        # L'IA est cohérente, correction légère
        prix_final = prix_ia * 0.85 + prix_reference * 0.15
    elif 1.3 < ratio <= 1.8:
        # L'IA surestime modérément
        prix_final = prix_ia * 0.5 + prix_reference * 0.5
    elif ratio > 1.8:
        # L'IA surestime beaucoup
        prix_final = prix_ia * 0.3 + prix_reference * 0.7
    else:  # ratio < 0.7
        # L'IA sous-estime
        prix_final = prix_ia * 0.7 + prix_reference * 0.3
    
    # Sécurité finale : limites absolues
    prix_min = prix_ref_m2 * surface * 0.4
    prix_max = prix_ref_m2 * surface * 1.8
    
    if type_bien == 'Maison':
        prix_max *= 1.1  # Plus de tolérance pour les maisons
    
    prix_final = max(prix_min, min(prix_max, prix_final))
    
    print(f"💰 Prix final corrigé: {prix_final:,.0f}€")
    return prix_final

def valider_donnees_utilisateur(surface, pieces, code_postal, terrain, type_bien):
    """
    Vérifier que les données saisies par l'utilisateur sont cohérentes
    """
    erreurs = []
    
    # Vérification code postal
    if not code_postal or len(code_postal) != 5 or not code_postal.isdigit():
        erreurs.append("⚠️ Le code postal doit contenir exactement 5 chiffres")
    
    # Vérification surface
    if surface < 8:
        erreurs.append("⚠️ La surface semble trop petite (minimum 8 m²)")
    elif surface > 1000:
        erreurs.append("⚠️ La surface semble très importante (plus de 1000 m²)")
    
    # Vérification cohérence terrain/surface
    if terrain >= 0 and surface > terrain and type_bien != 'Appartement' and type_bien!= 'Local' :
        erreurs.append("⚠️ La surface habitable ne peut pas être plus grande que le terrain")
    
    # Vérifications spéciales pour les maisons
    if type_bien == 'Maison':
        if pieces < 3:
            erreurs.append("💡 Une maison a généralement au moins 3 pièces")
        if surface < 50:
            erreurs.append("💡 La surface semble petite pour une maison")
    
    return erreurs

# ==========================================
# INTERFACE UTILISATEUR STREAMLIT
# ==========================================

# Titre principal de l'application
st.title("🏡 Estimation Immobilière")
st.markdown("*Estimation de prix par Intelligence Artificielle (France)*")

# Barre latérale avec informations
with st.sidebar:
    st.markdown("### 📊 Modèle IA")
    st.info("✅ Modèle entraîné sur 1M+ transactions")
    st.info("🔧 Correction automatique v3")
    st.info("📍 Couverture nationale France")
    st.info("🛡️ Sécurités anti-aberrations")
    
    st.markdown("### 💡 Conseil")
    st.caption("Pour une estimation précise, renseignez tous les champs disponibles")

# Formulaire principal
st.markdown("### 📋 Caractéristiques de votre bien immobilier")

# Première ligne : infos de base
col1, col2 = st.columns(2)

with col1:
    surface = st.number_input(
        "**Surface habitable (m²)**",
        min_value=8.0, 
        max_value=1000.0, 
        value=75.0,
        step=1.0,
        help="Surface réelle habitable (hors garage, cave, grenier)"
    )
    
    pieces = st.selectbox(
        "**Nombre de pièces principales**",
        options=list(range(1, 16)),
        index=3,  # 4 pièces par défaut
        help="Chambres + salon + séjour (cuisine et salle de bain non comprises)"
    )

with col2:
    code_postal = st.text_input(
        "**Code postal**",
        value="75001",
        max_chars=5,
        help="Code postal français à 5 chiffres"
    )
    
    type_bien = st.selectbox(
        "**Type de bien**",
        options=['Appartement', 'Maison', 'Local', 'Dépendance'],
        index=1,  # Maison par défaut
        help="Type de local immobilier"
    )

# Deuxième ligne : détails complémentaires
col3, col4 = st.columns(2)

with col3:
    # Terrain seulement si pertinent
    if type_bien in ['Maison', 'Dépendance']:
        terrain = st.number_input(
            "**Surface terrain (m²)**",
            min_value=0.0,
            max_value=10000.0,
            value=300.0 if type_bien == 'Maison' else 0.0,
            step=10.0,
            help="Surface du terrain privé"
        )
    else:
        terrain = 0.0
        st.markdown("**Surface terrain**")
        st.caption("Non applicable pour ce type de bien")

with col4:
    annee = st.selectbox(
        "**Année de référence**",
        options=[2024, 2023, 2022, 2021, 2020],
        help="Année pour l'estimation des prix"
    )

# Séparateur visuel
st.markdown("---")

# Bouton d'estimation bien visible
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    bouton_estimer = st.button(
        "🔍 **ESTIMER LE PRIX**",
        type="primary",
        use_container_width=True,
        help="Lancer l'estimation par IA"
    )

# Traitement quand on clique sur le bouton
if bouton_estimer:
    # Validation des données
    erreurs = valider_donnees_utilisateur(surface, pieces, code_postal, terrain, type_bien)
    
    if erreurs:
        # Afficher les erreurs
        for erreur in erreurs:
            if erreur.startswith("💡"):
                st.info(erreur)
            else:
                st.error(erreur)
    else:
        # Lancer l'estimation
        with st.spinner("🧠 L'IA analyse votre bien immobilier..."):
            try:
                # Paramètres pour l'IA
                recent = 1 if annee >= 2023 else 0
                
                # Appel du modèle IA
                prix_ia = estimer_prix_immobilier(
                    surface=surface,
                    pieces=pieces,
                    dept=code_postal[:2],
                    terrain=terrain,
                    type_local=type_bien,
                    recent=recent
                )
                
                if prix_ia is None:
                    st.error("❌ Erreur lors du calcul. Vérifiez que le modèle est entraîné.")
                else:
                    # Correction pour un prix plus réaliste
                    prix_final = calculer_prix_realiste(prix_ia, surface, code_postal, type_bien, terrain)
                    
                    # Vérification de cohérence
                    prix_m2_final = prix_final / surface
                    if prix_m2_final > 12000 and code_postal.startswith('75'):
                        st.warning(f"⚠️ Prix élevé pour Paris ({prix_m2_final:,.0f}€/m²)")
                    elif prix_m2_final > 8000 and not code_postal.startswith('75'):
                        st.warning(f"⚠️ Prix très élevé pour cette zone ({prix_m2_final:,.0f}€/m²)")
                    
                    # ===== AFFICHAGE DU RÉSULTAT =====
                    st.markdown("### 💰 Estimation du prix immobilier")
                    
                    # Prix principal avec un design adaptatif
                    st.markdown(f"""
                    <div class="price-result-box">
                        <h1 class="price-value">{prix_final:,.0f} €</h1>
                        <p class="price-label">Prix estimé</p>
                        <p class="price-description">Estimation par IA avec correction de marché optimisée</p>
                    </div>
                    """.replace(",", " "), unsafe_allow_html=True)
                    
                    # Métriques détaillées
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        # Fourchette d'estimation avec format plus compact
                        marge = prix_final * 0.20
                        prix_min = prix_final - marge
                        prix_max = prix_final + marge
                        
                        # Format K€ pour gagner de la place
                        prix_min_k = prix_min / 1000
                        prix_max_k = prix_max / 1000
                        
                        st.metric(
                            "📊 Fourchette",
                            f"{prix_min_k:.0f}K - {prix_max_k:.0f}K €",
                            help="Intervalle de confiance ±20%"
                        )
                    
                    with col_m2:
                        # Prix au mètre carré avec indicateur de cohérence amélioré
                        prix_m2 = prix_final / surface
                        dept = code_postal[:2] if len(code_postal) >= 2 else '75'
                        
                        # Seuils adaptés par zone
                        if dept == '75':  # Paris
                            seuil_bas, seuil_haut = 7000, 13000
                            ref_moy = 9200
                        elif dept in ['92', '93', '94']:  # Petite couronne
                            seuil_bas, seuil_haut = 3500, 7000
                            ref_moy = 5000
                        elif dept in ['77', '78', '91', '95']:  # Grande couronne
                            seuil_bas, seuil_haut = 2500, 4500
                            ref_moy = 3200
                        else:  # Province
                            seuil_bas, seuil_haut = 2000, 5000
                            ref_moy = 3000
                        
                        if prix_m2 > seuil_haut:
                            delta = "⚠️ Très élevé"
                        elif prix_m2 > ref_moy * 1.2:
                            delta = "📈 Haut de gamme"
                        elif prix_m2 < seuil_bas:
                            delta = "📉 Très bas"
                        elif prix_m2 < ref_moy * 0.8:
                            delta = "💰 Bon prix"
                        else:
                            delta = "✅ Normal"
                            
                        st.metric(
                            "📐 Prix au m²",
                            f"{prix_m2:,.0f}  €/m²".replace(",", " "),
                            delta=delta,
                            help="Prix par mètre carré habitable avec indicateur de cohérence"
                        )
                    
                    with col_m3:
                        if terrain > 0:
                            # Calcul plus réaliste du prix terrain
                            part_terrain = min(0.3, terrain / (surface * 4))  # Max 30% de la valeur
                            valeur_terrain_estimee = prix_final * part_terrain
                            terrain_m2 = valeur_terrain_estimee / terrain if terrain > 0 else 0
                            
                            st.metric(
                                "🌿 Terrain au m²",
                                f"{terrain_m2:,.0f} €/m²".replace(",", " "),
                                help="Estimation de la valeur du terrain (20-30% du prix total)"
                            )
                        else:
                            # Ratio qualité/prix selon la zone
                            if dept == '75':
                                ratio_ref = prix_m2 / 9200  # Prix moyen Paris réel
                                if ratio_ref < 0.8:
                                    qualite = "🔥 Très bon prix"
                                elif ratio_ref < 1.2:
                                    qualite = "✅ Prix correct"
                                else:
                                    qualite = "💸 Prix élevé"
                            else:
                                qualite = f"📍 {type_bien}"
                            
                            st.metric(
                                "🎯 Évaluation",
                                delta,
                                help="Évaluation qualité/prix pour la zone"
                            )
                    
                    # ===== ANALYSE GÉOGRAPHIQUE =====
                    st.markdown("### 📍 Analyse du marché local")
                    
                    localisation = formater_affichage_ville(code_postal)
                    
                    # Messages contextuels selon la zone
                    if code_postal.startswith('75'):
                        st.info(f"🏙️ **{localisation}** - Marché parisien premium, prix moyen ~9 200€/m²")
                    elif dept == '92':
                        st.info("💼 **Hauts-de-Seine** - Marché haut de gamme, prix moyen ~6 500€/m²")
                    elif dept in ['93', '94']:
                        st.info("🚇 **Petite couronne parisienne** - Marché dynamique, prix moyen ~4 000€/m²")
                    elif dept in ['77', '78', '91', '95']:
                        st.info("🏘️ **Grande couronne parisienne** - Marché résidentiel, prix moyen ~3 200€/m²")
                    elif dept in ['06', '13', '69', '33']:
                        st.info("🌟 **Grande métropole française** - Marché urbain attractif")
                    else:
                        st.info("🌍 **Marché local** - Estimation adaptée aux spécificités régionales")
                    
                    # Conseils selon le type de bien
                    if type_bien == 'Maison' and terrain > 0:
                        if terrain > 500:
                            st.success(f"🌳 Maison avec grand terrain ({terrain}m²) - Atout majeur pour la valeur")
                        elif terrain > 200:
                            st.info(f"🏡 Maison avec terrain confortable ({terrain}m²) - Bon équilibre")
                        else:
                            st.info(f"🏠 Maison avec petit terrain ({terrain}m²) - Optimisation d'espace")
                    
                    # ===== RECOMMANDATIONS PERSONNALISÉES =====
                    with st.expander("💡 Recommandations pour votre bien"):
                        
                        st.markdown("**Facteurs qui peuvent influencer le prix réel :**")
                        
                        # Recommandations selon le type
                        if type_bien == 'Maison':
                            st.write("🏡 **Pour une maison :**")
                            st.write("• État de la toiture et de l'isolation")
                            st.write("• Système de chauffage et performance énergétique")
                            st.write("• Aménagement et exposition du jardin")
                            st.write("• Garage, parking ou dépendances")
                            
                            if terrain > 300:
                                st.write("• 🌟 Votre grand terrain est un atout majeur")
                            if pieces >= 5:
                                st.write("• 🏠 Maison familiale : recherchée sur le marché")
                        
                        else:  # Appartement
                            st.write("🏢 **Pour un appartement :**")
                            st.write("• Étage et présence d'ascenseur")
                            st.write("• Balcon, terrasse ou loggia")
                            st.write("• Vue et luminosité")
                            st.write("• Standing de l'immeuble")
                            st.write("• Cave, parking ou box")
                        
                        # Recommandations géographiques
                        if dept == '75':
                            st.write("📍 **Spécificités parisiennes :**")
                            st.write("• Proximité métro cruciale")
                            st.write("• Hauteur sous plafond")
                            st.write("• Charges de copropriété")
                        
                        st.markdown("💰 **Pour une vente optimale :**")                        
                        st.write("• Considérer des travaux de rafraîchissement")
                        st.write("• Mettre en valeur les points forts uniques")
                        st.write("• Consulter plusieurs agents immobiliers locaux")
                    
                    # ===== AVERTISSEMENTS PROFESSIONNELS =====
                    st.markdown("### ⚠️ Informations importantes")
                    
                    st.warning(f"""
                    **Cette estimation est fournie à titre indicatif** et ne constitue pas une expertise immobilière officielle
                    
                    **Spécificités de cette estimation :**
                    • Correction automatique v3 appliquée (facteur {prix_final/prix_ia:.2f}x)                  
                                        
                    **Facteurs non pris en compte par l'IA :**
                    • État général du bien et travaux nécessaires
                    • Vue, exposition et prestations spéciales
                    • Environnement immédiat et nuisances
                    • Marché local très récent 
                    
                    **Pour une transaction :** Consultez un professionnel de l'immobilier pour une expertise détaillée 
                    tenant compte de tous les facteurs spécifiques à votre bien
                    """)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du calcul: {str(e)}")

# ===== AIDE ET INFORMATIONS =====
st.markdown("---")

# Section conseils avancés
with st.expander("🎯 Conseils d'expert immobilier"):
    st.markdown("""
    **💡 Maximiser la valeur de votre bien :**
    
    **Avant estimation officielle :**
    • Rassemblez tous les diagnostics (DPE, amiante, plomb...)
    • Listez les travaux récents et améliorations
    • Identifiez les atouts uniques (vue, exposition, calme...)
    
    **Pour négocier :**
    • Cette estimation donne un ordre de grandeur réaliste
    • Ajustez selon l'état réel et les prestations
    • Considérez la dynamique du marché local actuel
    
    **Signal d'alarme :**
    • Si estimation très différente des attentes → vérifiez les paramètres
    • Prix/m² > 15 000 € hors Paris → probablement surestimé
    • Écart > 30 % avec autres estimations → expertise conseillée
    """)

# Information sur la technologie
with st.expander("🧑‍💻 À propos du projet et de ses technologies"):
    st.markdown("""
    **Projet pédagogique d'exploration de données**
    *développé par **Bidouj Christophe***

    **Exploration** de l'IA appliquée aux données immobilières françaises

    **Expérimentations :**
    • Expérimentation sur dataset massif (1M+ transactions DVF)
    • Exploration pratique du machine learning en contexte réel
    • Étude des biais et corrections sur données immobilières
    • Expérimentation sur données territoriales et financières

    **Méthodes testées :**
    • Analyse de patterns géographiques de prix
    • Algorithmes de régression sur données complexes
    • Corrections automatiques d'estimations

    ⚠️ **Cadre pédagogique** : Expérimentation sur cas réel
    """)