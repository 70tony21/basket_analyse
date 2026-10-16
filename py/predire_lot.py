"""
Lit un fichier CSV de matchs À VENIR et produit un fichier CSV unique
regroupant toutes les prédictions déjà développées :

  - Quart-temps par le modèle ML (ml_prediction.py)
  - Quart-temps par le modèle Poisson/Skellam (fonctions de ml_prediction.py)
  - Score total et probabilité de victoire (predire_total.py)
  - Écart / handicap suggéré, et probabilité de couverture d'une ligne
    donnée si elle est fournie (predire_total.py)

Ce script réutilise les modèles déjà entraînés (fichiers .joblib) — il
faut donc avoir lancé au préalable :
    python ml_prediction.py entrainer --historique ... --saison-actuelle ...
    python predire_total.py entrainer --historique ... --saison-actuelle ...

Fichier d'entrée attendu (--matchs), avec un en-tête, par exemple :
    equipe_domicile,equipe_exterieur,date,ligne,ligne_total,cote_victoire_domicile,cote_victoire_exterieur,cote_handicap_domicile,cote_handicap_exterieur,cote_over,cote_under
    Paris Basketball,AS Monaco,15/03/2026,-5.5,165.5,1.65,2.25,1.90,1.90,1.90,1.90
    Cholet,Nanterre,15/03/2026,,,,,,,,

Colonnes :
  - equipe_domicile, equipe_exterieur : obligatoires
  - date (JJ/MM/AAAA ou AAAA-MM-JJ) : optionnelle. Si absente, le
    lendemain du dernier match connu dans vos fichiers est utilisé.
  - ligne : optionnelle. Handicap à tester pour l'équipe à domicile
    (ex : -5.5). Laissez vide si vous voulez juste la ligne suggérée
    par le modèle, sans tester une cote précise.
  - ligne_total : optionnelle. Ligne de total (Over/Under) à tester
    (ex : 165.5).
  - cote_victoire_domicile / cote_victoire_exterieur : optionnelles.
    Cotes du bookmaker pour le match gagnant — si fournies, le script
    calcule l'EV et signale une "value" éventuelle.
  - cote_handicap_domicile / cote_handicap_exterieur : optionnelles,
    nécessitent que `ligne` soit renseignée pour être exploitées.
  - cote_over / cote_under : optionnelles, nécessitent que
    `ligne_total` soit renseignée pour être exploitées.
  - ligne_mi_temps : optionnelle. Handicap mi-temps à tester (ex : -3.5).
  - ligne_total_mi_temps : optionnelle. Ligne de total mi-temps (Over/Under).

Le CSV de sortie inclut, pour chaque match : les 8 quart-temps prédits
par le ML et par Poisson, la probabilité de victoire par quart-temps
(Poisson/Skellam), le score total et son intervalle de confiance,
l'écart/handicap suggéré, l'Over/Under, un niveau de CONFIANCE basé
sur le volume d'historique disponible pour les deux équipes, et — pour
chaque cote fournie — l'EV et un indicateur de VALUE BET.

Prérequis :
    pip install pandas numpy scikit-learn joblib scipy
    (nécessite aussi ml_prediction.py et predire_total.py dans le même
    dossier, dont ce script réutilise les fonctions)

Utilisation :
    python predire_lot.py \
        --matchs prochains_matchs.csv \
        --historique betclic_elite_2020-2025.csv \
        --saison-actuelle betclic_elite_2025-2026.csv \
        --sortie predictions_lot.csv

Attention performance : pour chaque match du fichier d'entrée, ce
script reconstruit l'historique complet des features deux fois (une
fois pour le pipeline quart-temps, une fois pour le pipeline total) —
c'est la même logique que ml_prediction.py::predire et
predire_total.py::predire, simplement répétée pour chaque match. Avec
beaucoup de matchs dans le fichier d'entrée et beaucoup d'historique,
ça peut prendre du temps (plusieurs dizaines de secondes par match).
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from ml_prediction import (
    CIBLES as CIBLES_QUARTS,
    COLONNES_QUARTS,
    charger_et_fusionner,
    colonnes_features as colonnes_features_qt,
    construire_features,
    evaluer_confiance,
    normaliser_nom_equipe,
    predire_quarts_poisson,
    suggerer_noms_proches,
)
from ml_prediction import FICHIER_MODELE_REGRESSION as FICHIER_MODELE_QT_REGRESSION
from ml_prediction import FICHIER_MODELE_CLASSIFICATION as FICHIER_MODELE_QT_CLASSIFICATION

from predire_total import (
    charger_et_fusionner_total,
    calculer_intervalles,
    colonnes_features as colonnes_features_total,
    construire_features_total,
)
from predire_total import FICHIER_MODELE_REGRESSION as FICHIER_MODELE_TOTAL_REGRESSION
from predire_total import FICHIER_MODELE_CLASSIFICATION as FICHIER_MODELE_TOTAL_CLASSIFICATION
from predire_total import FICHIER_MODELE_ECART, FICHIER_MODELE_POINTS_TOTAUX

from predire_mi_temps_dediee import ajouter_cibles_mi_temps
from predire_mi_temps_dediee import FICHIER_MODELE_REGRESSION as FICHIER_MODELE_MT_REGRESSION
from predire_mi_temps_dediee import FICHIER_MODELE_CLASSIFICATION as FICHIER_MODELE_MT_CLASSIFICATION
from predire_mi_temps_dediee import FICHIER_MODELE_ECART as FICHIER_MODELE_MT_ECART
from predire_mi_temps_dediee import FICHIER_MODELE_POINTS_TOTAUX as FICHIER_MODELE_MT_POINTS_TOTAUX

SEUIL_VALUE = 0.03  # écart minimal (3%) entre notre probabilité et celle du bookmaker pour signaler une "value"


def evaluer_value(proba_modele, cote_bookmaker, seuil=SEUIL_VALUE):
    """
    Compare la probabilité de notre modèle à la cote proposée par un
    bookmaker. EV (espérance de gain pour 1 unité misée) = proba * cote - 1.
    Une EV > 0 signifie que le modèle voit plus de valeur que le
    marché ; on ne signale "value" que si l'écart dépasse `seuil`,
    pour limiter les faux positifs dus au bruit du modèle.
    """
    if proba_modele is None or cote_bookmaker is None or cote_bookmaker <= 1:
        return None, False
    ev = proba_modele * cote_bookmaker - 1
    return round(ev, 3), ev > seuil


def charger_modeles():
    try:
        return {
            "qt_regression": joblib.load(FICHIER_MODELE_QT_REGRESSION),
            "qt_classification": joblib.load(FICHIER_MODELE_QT_CLASSIFICATION),
            "total_regression": joblib.load(FICHIER_MODELE_TOTAL_REGRESSION),
            "total_classification": joblib.load(FICHIER_MODELE_TOTAL_CLASSIFICATION),
            "total_ecart": joblib.load(FICHIER_MODELE_ECART),
            "total_points": joblib.load(FICHIER_MODELE_POINTS_TOTAUX),
            "mi_temps_regression": joblib.load(FICHIER_MODELE_MT_REGRESSION),
            "mi_temps_classification": joblib.load(FICHIER_MODELE_MT_CLASSIFICATION),
            "mi_temps_ecart": joblib.load(FICHIER_MODELE_MT_ECART),
            "mi_temps_points": joblib.load(FICHIER_MODELE_MT_POINTS_TOTAUX),
        }
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Modèle introuvable ({exc.filename}). Avez-vous bien lancé au préalable :\n"
            f"  python ml_prediction.py entrainer --historique ... --saison-actuelle ...\n"
            f"  python predire_total.py entrainer --historique ... --saison-actuelle ...\n"
            f"  python predire_mi_temps_dediee.py entrainer --historique ... --saison-actuelle ...\n"
        )


def construire_ligne_fictive_qt(df: pd.DataFrame, equipe_domicile: str, equipe_exterieur: str,
                                 date_fictive: pd.Timestamp, df_features: pd.DataFrame, cols_x):
    """Reconstruit la ligne de features (pipeline quart-temps) pour un match fictif donné."""
    ligne_fictive = pd.DataFrame([{
        "equipe_domicile": equipe_domicile,
        "equipe_exterieur": equipe_exterieur,
        "_date": date_fictive,
        **{c: 0 for c in COLONNES_QUARTS},
    }])
    df_avec_fictif = pd.concat([df, ligne_fictive], ignore_index=True)
    features_avec_fictif = construire_features(df_avec_fictif)
    ligne_a_predire = features_avec_fictif.iloc[[-1]][cols_x]

    manquantes = ligne_a_predire.columns[ligne_a_predire.isna().any()].tolist()
    if manquantes:
        medianes = df_features[cols_x].median(numeric_only=True)
        ligne_a_predire = ligne_a_predire.fillna(medianes)
    return ligne_a_predire, manquantes


def construire_ligne_fictive_total(df: pd.DataFrame, equipe_domicile: str, equipe_exterieur: str,
                                    date_fictive: pd.Timestamp, df_features: pd.DataFrame, cols_x):
    """Reconstruit la ligne de features (pipeline score total) pour un match fictif donné."""
    ligne_fictive = pd.DataFrame([{
        "equipe_domicile": equipe_domicile,
        "equipe_exterieur": equipe_exterieur,
        "_date": date_fictive,
        "score_domicile": 0, "score_exterieur": 0,
    }])
    df_avec_fictif = pd.concat([df, ligne_fictive], ignore_index=True)
    features_avec_fictif = construire_features_total(df_avec_fictif)
    ligne_a_predire = features_avec_fictif.iloc[[-1]][cols_x]

    manquantes = ligne_a_predire.columns[ligne_a_predire.isna().any()].tolist()
    if manquantes:
        medianes = df_features[cols_x].median(numeric_only=True)
        ligne_a_predire = ligne_a_predire.fillna(medianes)
    return ligne_a_predire, manquantes


def predire_un_match(df_qt, df_features_qt, cols_x_qt, df_total, df_features_total, cols_x_total,
                      modeles, equipe_domicile_saisi, equipe_exterieur_saisi, date_saisie,
                      ligne_saisie, ligne_total_saisie, ligne_mi_temps_saisie,
                      ligne_total_mi_temps_saisie, cotes, format_date, bornes):
    resultat = {
        "equipe_domicile": equipe_domicile_saisi,
        "equipe_exterieur": equipe_exterieur_saisi,
    }

    equipe_domicile = normaliser_nom_equipe(equipe_domicile_saisi)
    equipe_exterieur = normaliser_nom_equipe(equipe_exterieur_saisi)

    noms_connus_qt = set(df_features_qt["equipe_domicile"]) | set(df_features_qt["equipe_exterieur"])
    for nom_saisi, nom_norm in [(equipe_domicile_saisi, equipe_domicile),
                                 (equipe_exterieur_saisi, equipe_exterieur)]:
        if nom_norm not in noms_connus_qt:
            suggestions = suggerer_noms_proches(nom_norm, list(noms_connus_qt))
            resultat["erreur"] = (
                f"Équipe introuvable : '{nom_saisi}'"
                + (f" (suggestions : {', '.join(suggestions)})" if suggestions else "")
            )
            return resultat

    if date_saisie and str(date_saisie).strip() and str(date_saisie).lower() != "nan":
        # format flexible : accepte aussi bien "25/09/2026" que
        # "2026-09-25" sans planter sur un format inattendu
        date_fictive = pd.to_datetime(date_saisie, dayfirst=True)
    else:
        date_fictive = df_qt["_date"].max() + pd.Timedelta(days=1)
    resultat["date"] = date_fictive.strftime("%d/%m/%Y")

    # --- Confiance de la prédiction (volume d'historique disponible) ---
    confiance = evaluer_confiance(df_total, equipe_domicile, equipe_exterieur, date_fictive)
    resultat["confiance"] = confiance["niveau_confiance"]
    resultat["nb_matchs_domicile"] = confiance["nb_matchs_domicile"]
    resultat["nb_matchs_exterieur"] = confiance["nb_matchs_exterieur"]
    resultat["nb_confrontations"] = confiance["nb_confrontations"]

    # ------------------------------------------------------------
    # Pipeline quart-temps : ML + Poisson (+ vainqueur par quart-temps)
    # ------------------------------------------------------------
    ligne_qt, manquantes_qt = construire_ligne_fictive_qt(
        df_qt, equipe_domicile, equipe_exterieur, date_fictive, df_features_qt, cols_x_qt
    )
    if manquantes_qt:
        resultat["avertissement_qt"] = f"valeurs comblées par la médiane : {', '.join(manquantes_qt)}"

    total_dom_ml, total_ext_ml = 0.0, 0.0
    for cible in CIBLES_QUARTS:
        pred = modeles["qt_regression"]["modeles"][cible].predict(ligne_qt)[0]
        resultat[f"ml_{cible}"] = round(float(pred), 1)
        if "domicile" in cible:
            total_dom_ml += pred
        else:
            total_ext_ml += pred
    resultat["ml_total_domicile"] = round(total_dom_ml, 1)
    resultat["ml_total_exterieur"] = round(total_ext_ml, 1)
    proba_victoire_ml = float(modeles["qt_classification"]["modele"].predict_proba(ligne_qt)[0][1])
    resultat["ml_proba_victoire_domicile"] = round(proba_victoire_ml, 3)

    # --- Mi-temps (modèle dédié, réutilise directement ligne_qt : mêmes
    # features que le pipeline quart-temps — voir predire_mi_temps_dediee.py) ---
    pred_mt_dom = float(modeles["mi_temps_regression"]["modeles"]["score_mi_temps_domicile"].predict(ligne_qt)[0])
    pred_mt_ext = float(modeles["mi_temps_regression"]["modeles"]["score_mi_temps_exterieur"].predict(ligne_qt)[0])
    proba_mener_mt = float(modeles["mi_temps_classification"]["modele"].predict_proba(ligne_qt)[0][1])
    marge_mt = float(modeles["mi_temps_ecart"]["modele"].predict(ligne_qt)[0])
    ecart_type_mt = modeles["mi_temps_ecart"]["ecart_type_residus"]
    points_totaux_mt = float(modeles["mi_temps_points"]["modele"].predict(ligne_qt)[0])

    resultat["mi_temps_score_domicile"] = round(pred_mt_dom, 1)
    resultat["mi_temps_score_exterieur"] = round(pred_mt_ext, 1)
    resultat["mi_temps_proba_mener_domicile"] = round(proba_mener_mt, 3)
    resultat["mi_temps_marge_predite_domicile"] = round(marge_mt, 1)
    resultat["mi_temps_ligne_suggeree_domicile"] = round(-marge_mt * 2) / 2
    resultat["mi_temps_points_totaux_predits"] = round(points_totaux_mt, 1)

    if (ligne_mi_temps_saisie is not None and str(ligne_mi_temps_saisie).strip()
            and str(ligne_mi_temps_saisie).lower() != "nan"):
        ligne_mt_val = float(ligne_mi_temps_saisie)
        proba_couverture_mt = float(norm.sf(-ligne_mt_val, loc=marge_mt, scale=ecart_type_mt))
        resultat["mi_temps_ligne_testee"] = ligne_mt_val
        resultat["mi_temps_proba_couverture_domicile"] = round(proba_couverture_mt, 3)
        resultat["mi_temps_proba_couverture_exterieur"] = round(1 - proba_couverture_mt, 3)

    if (ligne_total_mi_temps_saisie is not None and str(ligne_total_mi_temps_saisie).strip()
            and str(ligne_total_mi_temps_saisie).lower() != "nan"):
        ligne_totale_mt_val = float(ligne_total_mi_temps_saisie)
        proba_over_mt = float(norm.sf(ligne_totale_mt_val, loc=points_totaux_mt, scale=modeles["mi_temps_points"]["ecart_type_residus"]))
        resultat["mi_temps_ligne_total_testee"] = ligne_totale_mt_val
        resultat["mi_temps_proba_over"] = round(proba_over_mt, 3)
        resultat["mi_temps_proba_under"] = round(1 - proba_over_mt, 3)

    pred_poisson = predire_quarts_poisson(df_qt, equipe_domicile, equipe_exterieur)
    if pred_poisson is None:
        resultat["avertissement_poisson"] = "historique insuffisant pour au moins une équipe"
    else:
        total_dom_poisson, total_ext_poisson = 0.0, 0.0
        for cible in CIBLES_QUARTS:
            resultat[f"poisson_{cible}"] = round(float(pred_poisson[cible]), 1)
            if "domicile" in cible:
                total_dom_poisson += pred_poisson[cible]
            else:
                total_ext_poisson += pred_poisson[cible]
        resultat["poisson_total_domicile"] = round(total_dom_poisson, 1)
        resultat["poisson_total_exterieur"] = round(total_ext_poisson, 1)
        resultat["poisson_proba_victoire_domicile"] = round(
            float(pred_poisson["proba_victoire_domicile"]), 3
        )
        # vainqueur par quart-temps (Poisson/Skellam appliqué à chaque
        # quart-temps séparément — voir ml_prediction.py)
        for i in range(1, 5):
            resultat[f"proba_victoire_q{i}_domicile"] = round(
                float(pred_poisson[f"proba_victoire_q{i}_domicile"]), 3
            )

    # ------------------------------------------------------------
    # Pipeline score total dédié + handicap + Over/Under
    # ------------------------------------------------------------
    ligne_totale, manquantes_totale = construire_ligne_fictive_total(
        df_total, equipe_domicile, equipe_exterieur, date_fictive, df_features_total, cols_x_total
    )
    if manquantes_totale:
        resultat["avertissement_total"] = f"valeurs comblées par la médiane : {', '.join(manquantes_totale)}"

    pred_dom_total = modeles["total_regression"]["modeles"]["score_domicile"].predict(ligne_totale)[0]
    pred_ext_total = modeles["total_regression"]["modeles"]["score_exterieur"].predict(ligne_totale)[0]
    resultat["total_score_domicile"] = round(float(pred_dom_total), 1)
    resultat["total_score_exterieur"] = round(float(pred_ext_total), 1)
    proba_victoire_total = float(modeles["total_classification"]["modele"].predict_proba(ligne_totale)[0][1])
    resultat["total_proba_victoire_domicile"] = round(proba_victoire_total, 3)

    marge_predite = float(modeles["total_ecart"]["modele"].predict(ligne_totale)[0])
    ecart_type_residus = modeles["total_ecart"]["ecart_type_residus"]
    ligne_suggeree = round(-marge_predite * 2) / 2
    resultat["marge_predite_domicile"] = round(marge_predite, 1)
    resultat["ligne_suggeree_domicile"] = ligne_suggeree

    proba_couverture_dom = None
    if ligne_saisie is not None and str(ligne_saisie).strip() and str(ligne_saisie).lower() != "nan":
        ligne_val = float(ligne_saisie)
        proba_couverture_dom = float(norm.sf(-ligne_val, loc=marge_predite, scale=ecart_type_residus))
        resultat["ligne_testee"] = ligne_val
        resultat["proba_couverture_domicile"] = round(proba_couverture_dom, 3)
        resultat["proba_couverture_exterieur"] = round(1 - proba_couverture_dom, 3)

    moyenne_points_totaux = float(modeles["total_points"]["modele"].predict(ligne_totale)[0])
    ecart_type_points_totaux = modeles["total_points"]["ecart_type_residus"]
    resultat["points_totaux_predits"] = round(moyenne_points_totaux, 1)

    proba_over = None
    if ligne_total_saisie is not None and str(ligne_total_saisie).strip() and str(ligne_total_saisie).lower() != "nan":
        ligne_totale_val = float(ligne_total_saisie)
        proba_over = float(norm.sf(ligne_totale_val, loc=moyenne_points_totaux, scale=ecart_type_points_totaux))
        resultat["ligne_total_testee"] = ligne_totale_val
        resultat["proba_over"] = round(proba_over, 3)
        resultat["proba_under"] = round(1 - proba_over, 3)

    # --- Intervalle de score total par tranches (grille façon paris) ---
    intervalles = calculer_intervalles(moyenne_points_totaux, ecart_type_points_totaux, bornes)
    for i, intervalle in enumerate(intervalles, start=1):
        resultat[f"tranche_{i}_nom"] = intervalle["tranche"]
        resultat[f"tranche_{i}_proba"] = intervalle["probabilite"]
        resultat[f"tranche_{i}_cote"] = intervalle["cote_juste"]

    # ------------------------------------------------------------
    # Détecteur de value bet (uniquement pour les marchés où une cote
    # bookmaker a été fournie dans le fichier d'entrée)
    # ------------------------------------------------------------
    proba_victoire_moyenne = np.mean([
        p for p in [proba_victoire_ml, resultat.get("poisson_proba_victoire_domicile"), proba_victoire_total]
        if p is not None
    ])

    if cotes.get("cote_victoire_domicile") is not None:
        ev, value = evaluer_value(proba_victoire_moyenne, cotes["cote_victoire_domicile"])
        resultat["ev_victoire_domicile"] = ev
        resultat["value_victoire_domicile"] = value
    if cotes.get("cote_victoire_exterieur") is not None:
        ev, value = evaluer_value(1 - proba_victoire_moyenne, cotes["cote_victoire_exterieur"])
        resultat["ev_victoire_exterieur"] = ev
        resultat["value_victoire_exterieur"] = value

    if proba_couverture_dom is not None:
        if cotes.get("cote_handicap_domicile") is not None:
            ev, value = evaluer_value(proba_couverture_dom, cotes["cote_handicap_domicile"])
            resultat["ev_handicap_domicile"] = ev
            resultat["value_handicap_domicile"] = value
        if cotes.get("cote_handicap_exterieur") is not None:
            ev, value = evaluer_value(1 - proba_couverture_dom, cotes["cote_handicap_exterieur"])
            resultat["ev_handicap_exterieur"] = ev
            resultat["value_handicap_exterieur"] = value

    if proba_over is not None:
        if cotes.get("cote_over") is not None:
            ev, value = evaluer_value(proba_over, cotes["cote_over"])
            resultat["ev_over"] = ev
            resultat["value_over"] = value
        if cotes.get("cote_under") is not None:
            ev, value = evaluer_value(1 - proba_over, cotes["cote_under"])
            resultat["ev_under"] = ev
            resultat["value_under"] = value

    return resultat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchs", required=True,
                         help="CSV des matchs à venir (colonnes : equipe_domicile, "
                              "equipe_exterieur, date [optionnel], ligne [optionnel]).")
    parser.add_argument("--historique", required=True)
    parser.add_argument("--saison-actuelle", required=True)
    parser.add_argument("--colonne-date", default="date")
    parser.add_argument("--format-date", default="%d/%m/%Y")
    parser.add_argument("--sortie", default="../prediction/predictions_lot.csv")
    parser.add_argument("--bornes", default="100,110,120,130,140,150,160,170,180",
                         help="Bornes séparant les tranches de score total (grille façon paris), "
                              "séparées par des virgules. Voir predire_total.py.")
    args = parser.parse_args()

    print("Chargement des modèles déjà entraînés...")
    modeles = charger_modeles()

    print("Chargement et construction des features (pipeline quart-temps)...")
    df_qt = charger_et_fusionner(args.historique, args.saison_actuelle,
                                  args.colonne_date, args.format_date)
    df_features_qt = construire_features(df_qt)
    cols_x_qt = colonnes_features_qt(df_features_qt)

    print("Chargement et construction des features (pipeline score total)...")
    df_total = charger_et_fusionner_total(args.historique, args.saison_actuelle,
                                           args.colonne_date, args.format_date)
    df_features_total = construire_features_total(df_total)
    cols_x_total = colonnes_features_total(df_features_total)

    matchs_a_predire = pd.read_csv(args.matchs, dtype=str)
    print(f"\n{len(matchs_a_predire)} match(s) à prédire.\n")

    bornes = [int(b) for b in args.bornes.split(",")]

    colonnes_cotes = [
        "cote_victoire_domicile", "cote_victoire_exterieur",
        "cote_handicap_domicile", "cote_handicap_exterieur",
        "cote_over", "cote_under",
    ]

    resultats = []
    for i, ligne in matchs_a_predire.iterrows():
        print(f"[{i + 1}/{len(matchs_a_predire)}] {ligne['equipe_domicile']} vs "
              f"{ligne['equipe_exterieur']}...")

        cotes = {}
        for c in colonnes_cotes:
            valeur = ligne.get(c)
            if valeur is not None and str(valeur).strip() and str(valeur).lower() != "nan":
                cotes[c] = float(valeur)

        resultat = predire_un_match(
            df_qt, df_features_qt, cols_x_qt,
            df_total, df_features_total, cols_x_total,
            modeles,
            ligne["equipe_domicile"], ligne["equipe_exterieur"],
            ligne.get("date"), ligne.get("ligne"), ligne.get("ligne_total"),
            ligne.get("ligne_mi_temps"), ligne.get("ligne_total_mi_temps"),
            cotes,
            args.format_date,
            bornes,
        )
        if "erreur" in resultat:
            print(f"   /!\\ {resultat['erreur']}")
        resultats.append(resultat)

    pd.DataFrame(resultats).to_csv(args.sortie, index=False, encoding="mac_roman")
    print(f"\nTerminé. Prédictions écrites dans {args.sortie}.")


if __name__ == "__main__":
    main()
