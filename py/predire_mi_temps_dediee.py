"""
Prédit le score de la MI-TEMPS (Q1+Q2 cumulés), avant le match — à ne
pas confondre avec predire_mi_temps.py, qui suppose la mi-temps déjà
JOUÉE (score réel connu) pour prédire le vainqueur final.

Comme pour le score total (predire_total.py), on utilise des modèles
DÉDIÉS entraînés directement sur la cible "score à la mi-temps",
plutôt que de sommer les prédictions Q1 et Q2 individuelles — plus
précis pour cette cible précise, et ça débloque des marchés réels
(total mi-temps, handicap mi-temps, vainqueur 1ère période).

Contrairement à predire_total.py, ce script réutilise directement le
pipeline quart-temps de ml_prediction.py (mêmes features, mêmes
fichiers d'entrée) puisque prédire un score de mi-temps nécessite de
connaître les quarts-temps réels pour construire les cibles
d'entraînement — donc seuls les matchs avec détail quart-temps connu
sont exploitables ici (comme pour ml_prediction.py, contrairement à
predire_total.py qui accepte aussi les matchs sans détail).

Prérequis :
    pip install pandas numpy scikit-learn joblib scipy
    (nécessite aussi ml_prediction.py dans le même dossier)

Utilisation :
    python predire_mi_temps_dediee.py entrainer \
        --historique betclic_elite_2020-2025.csv \
        --saison-actuelle betclic_elite_2025-2026.csv

    python predire_mi_temps_dediee.py predire "Paris Basketball" "AS Monaco" \
        --historique betclic_elite_2020-2025.csv \
        --saison-actuelle betclic_elite_2025-2026.csv \
        --ligne -3.5 --ligne-total 85.5
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from ml_prediction import (
    charger_et_fusionner,
    colonnes_features as colonnes_features_qt,
    construire_features,
    evaluer_confiance,
    normaliser_nom_equipe,
    suggerer_noms_proches,
)

CIBLES = ["score_mi_temps_domicile", "score_mi_temps_exterieur"]

FICHIER_MODELE_REGRESSION = "../model_data/modele_mi_temps_score.joblib"
FICHIER_MODELE_CLASSIFICATION = "../model_data/modele_mi_temps_vainqueur.joblib"
FICHIER_MODELE_ECART = "../model_data/modele_mi_temps_ecart.joblib"
FICHIER_MODELE_POINTS_TOTAUX = "../model_data/modele_mi_temps_points_totaux.joblib"
FICHIER_RESULTATS_TEST = "../model_test/resultats_test_mi_temps.csv"
FICHIER_PREDICTIONS = "../model_data/predictions_mi_temps_a_venir.csv"


def ajouter_cibles_mi_temps(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les cibles de mi-temps à partir des colonnes q1/q2 déjà
    présentes dans df_features (les cibles habituelles du pipeline
    quart-temps de ml_prediction.py).
    """
    df = df_features.copy()
    df["score_mi_temps_domicile"] = df["q1_domicile"] + df["q2_domicile"]
    df["score_mi_temps_exterieur"] = df["q1_exterieur"] + df["q2_exterieur"]
    df["ecart_mi_temps"] = df["score_mi_temps_domicile"] - df["score_mi_temps_exterieur"]
    df["points_totaux_mi_temps"] = df["score_mi_temps_domicile"] + df["score_mi_temps_exterieur"]
    df["victoire_domicile_mi_temps"] = (df["ecart_mi_temps"] > 0).astype(int)
    return df


def charger_et_preparer(args):
    df_brut = charger_et_fusionner(args.historique, args.saison_actuelle,
                                    args.colonne_date, args.format_date)
    df_features = construire_features(df_brut)
    cols_x = colonnes_features_qt(df_features)  # calculé AVANT d'ajouter les cibles mi-temps
    df_features = ajouter_cibles_mi_temps(df_features)
    return df_brut, df_features, cols_x


# ---------------------------------------------------------------------
# Entraînement + évaluation
# ---------------------------------------------------------------------

def entrainer(args):
    _, df_features, cols_x = charger_et_preparer(args)

    df_utile = df_features.dropna(subset=cols_x).reset_index(drop=True)
    print(f"{len(df_utile)} matchs utilisables (avec détail quart-temps connu).")

    n_test = int(len(df_utile) * args.proportion_test)
    train = df_utile.iloc[:-n_test]
    test = df_utile.iloc[-n_test:]
    print(f"Entraînement sur {len(train)} matchs, évaluation sur {len(test)} "
          f"matchs les plus récents.\n")

    X_train, X_test = train[cols_x], test[cols_x]

    print("Régression (score à la mi-temps) :")
    modeles_regression = {}
    predictions_test = {}
    for cible in CIBLES:
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(X_train, train[cible])
        pred = m.predict(X_test)
        mae = mean_absolute_error(test[cible], pred)
        print(f"  {cible:28s} MAE = {mae:.2f} points")
        modeles_regression[cible] = m
        predictions_test[cible] = pred

    print("\nClassification (qui mène à la mi-temps) :")
    modele_clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    modele_clf.fit(X_train, train["victoire_domicile_mi_temps"])
    proba_test = modele_clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    acc = accuracy_score(test["victoire_domicile_mi_temps"], pred_test)
    ll = log_loss(test["victoire_domicile_mi_temps"], proba_test)
    print(f"  Précision = {acc:.1%}  |  Log-loss = {ll:.3f}")

    print("\nModèle dédié à l'écart mi-temps :")
    modele_ecart = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    modele_ecart.fit(X_train, train["ecart_mi_temps"])
    pred_ecart_test = modele_ecart.predict(X_test)
    mae_ecart = mean_absolute_error(test["ecart_mi_temps"], pred_ecart_test)
    ecart_type_ecart = float(np.std(test["ecart_mi_temps"].values - pred_ecart_test))
    print(f"  MAE = {mae_ecart:.2f}  |  écart-type des erreurs = {ecart_type_ecart:.2f} "
          f"(utilisé pour les probabilités de couverture de ligne)")

    print("\nModèle dédié aux points totaux mi-temps :")
    modele_points = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    modele_points.fit(X_train, train["points_totaux_mi_temps"])
    pred_points_test = modele_points.predict(X_test)
    mae_points = mean_absolute_error(test["points_totaux_mi_temps"], pred_points_test)
    ecart_type_points = float(np.std(test["points_totaux_mi_temps"].values - pred_points_test))
    print(f"  MAE = {mae_points:.2f}  |  écart-type des erreurs = {ecart_type_points:.2f} "
          f"(utilisé pour les probabilités Over/Under mi-temps)")

    resultats_test = pd.DataFrame({
        "date": test["date"].values,
        "equipe_domicile": test["equipe_domicile"].values,
        "equipe_exterieur": test["equipe_exterieur"].values,
    })
    for cible in CIBLES:
        resultats_test[f"{cible}_reel"] = test[cible].values
        resultats_test[f"{cible}_predit"] = np.round(predictions_test[cible], 1)
        resultats_test[f"{cible}_erreur"] = np.round(np.abs(test[cible].values - predictions_test[cible]), 1)
    resultats_test["victoire_mi_temps_reel"] = test["victoire_domicile_mi_temps"].values
    resultats_test["proba_victoire_mi_temps_predite"] = np.round(proba_test, 3)
    resultats_test["victoire_mi_temps_predite"] = pred_test
    resultats_test["prediction_correcte"] = (
        resultats_test["victoire_mi_temps_reel"] == resultats_test["victoire_mi_temps_predite"]
    )
    resultats_test["ecart_mi_temps_reel"] = test["ecart_mi_temps"].values
    resultats_test["ecart_mi_temps_predit"] = np.round(pred_ecart_test, 1)
    resultats_test["points_totaux_mi_temps_reel"] = test["points_totaux_mi_temps"].values
    resultats_test["points_totaux_mi_temps_predit"] = np.round(pred_points_test, 1)
    resultats_test.to_csv(FICHIER_RESULTATS_TEST, index=False)
    print(f"\nDétail des prédictions de test : {FICHIER_RESULTATS_TEST}")

    joblib.dump({"modeles": modeles_regression, "colonnes": cols_x}, FICHIER_MODELE_REGRESSION)
    joblib.dump({"modele": modele_clf, "colonnes": cols_x}, FICHIER_MODELE_CLASSIFICATION)
    joblib.dump({"modele": modele_ecart, "colonnes": cols_x, "ecart_type_residus": ecart_type_ecart},
                FICHIER_MODELE_ECART)
    joblib.dump({"modele": modele_points, "colonnes": cols_x, "ecart_type_residus": ecart_type_points},
                FICHIER_MODELE_POINTS_TOTAUX)
    print(f"Modèles sauvegardés : {FICHIER_MODELE_REGRESSION}, {FICHIER_MODELE_CLASSIFICATION}, "
          f"{FICHIER_MODELE_ECART}, {FICHIER_MODELE_POINTS_TOTAUX}")


# ---------------------------------------------------------------------
# Prédiction d'un match à venir
# ---------------------------------------------------------------------

def predire(args):
    df_brut, df_features, cols_x = charger_et_preparer(args)

    equipe_domicile = normaliser_nom_equipe(args.equipe_domicile)
    equipe_exterieur = normaliser_nom_equipe(args.equipe_exterieur)

    noms_connus = pd.concat([df_features["equipe_domicile"], df_features["equipe_exterieur"]]).unique().tolist()
    for nom_saisi, nom_norm in [(args.equipe_domicile, equipe_domicile), (args.equipe_exterieur, equipe_exterieur)]:
        if nom_norm not in noms_connus:
            suggestions = suggerer_noms_proches(nom_norm, noms_connus)
            message = f"Équipe introuvable : '{nom_saisi}'"
            message += f". Vouliez-vous dire : {', '.join(suggestions)} ?" if suggestions else ""
            raise SystemExit(message)

    if args.date:
        date_fictive = pd.to_datetime(args.date, dayfirst=True)
    else:
        date_fictive = df_brut["_date"].max() + pd.Timedelta(days=1)

    ligne_fictive = pd.DataFrame([{
        "equipe_domicile": equipe_domicile, "equipe_exterieur": equipe_exterieur,
        "_date": date_fictive,
        "q1_domicile": 0, "q1_exterieur": 0, "q2_domicile": 0, "q2_exterieur": 0,
        "q3_domicile": 0, "q3_exterieur": 0, "q4_domicile": 0, "q4_exterieur": 0,
    }])
    df_avec_fictif = pd.concat([df_brut, ligne_fictive], ignore_index=True)
    features_avec_fictif = construire_features(df_avec_fictif)
    ligne_a_predire = features_avec_fictif.iloc[[-1]][cols_x]

    colonnes_manquantes = ligne_a_predire.columns[ligne_a_predire.isna().any()].tolist()
    if colonnes_manquantes:
        medianes = df_features[cols_x].median(numeric_only=True)
        ligne_a_predire = ligne_a_predire.fillna(medianes)
        print(f"Attention : valeur(s) manquante(s) comblée(s) par la médiane pour : "
              f"{', '.join(colonnes_manquantes)}.\n")

    modele_reg = joblib.load(FICHIER_MODELE_REGRESSION)
    modele_clf = joblib.load(FICHIER_MODELE_CLASSIFICATION)
    modele_ecart = joblib.load(FICHIER_MODELE_ECART)
    modele_points = joblib.load(FICHIER_MODELE_POINTS_TOTAUX)

    pred_dom = float(modele_reg["modeles"]["score_mi_temps_domicile"].predict(ligne_a_predire)[0])
    pred_ext = float(modele_reg["modeles"]["score_mi_temps_exterieur"].predict(ligne_a_predire)[0])
    proba_dom = float(modele_clf["modele"].predict_proba(ligne_a_predire)[0][1])
    marge_predite = float(modele_ecart["modele"].predict(ligne_a_predire)[0])
    ecart_type_ecart = modele_ecart["ecart_type_residus"]
    points_totaux_predits = float(modele_points["modele"].predict(ligne_a_predire)[0])
    ecart_type_points = modele_points["ecart_type_residus"]

    print(f"\nPrédiction MI-TEMPS : {equipe_domicile} vs {equipe_exterieur}\n")
    print(f"  Score mi-temps estimé : {equipe_domicile} {pred_dom:.1f} - {pred_ext:.1f} {equipe_exterieur}")
    print(f"  Probabilité de mener à la mi-temps : {equipe_domicile} {proba_dom:.1%} | "
          f"{equipe_exterieur} {1 - proba_dom:.1%}")

    ligne_suggeree = round(-marge_predite * 2) / 2
    print(f"\n  Écart mi-temps prédit : {equipe_domicile} "
          f"{'+' if marge_predite >= 0 else ''}{marge_predite:.1f} {equipe_exterieur}")
    print(f"  Ligne de handicap mi-temps suggérée : {equipe_domicile} "
          f"{'+' if ligne_suggeree > 0 else ''}{ligne_suggeree:g} / "
          f"{equipe_exterieur} {'+' if -ligne_suggeree > 0 else ''}{-ligne_suggeree:g}")

    proba_couverture = None
    if args.ligne is not None:
        proba_couverture = float(norm.sf(-args.ligne, loc=marge_predite, scale=ecart_type_ecart))
        print(f"\n  Ligne testée : {equipe_domicile} {'+' if args.ligne > 0 else ''}{args.ligne:g}")
        print(f"  Probabilité de couverture {equipe_domicile} : {proba_couverture:.1%}")
        print(f"  Probabilité de couverture {equipe_exterieur} : {1 - proba_couverture:.1%}")

    print(f"\n  Points totaux mi-temps prédits : {points_totaux_predits:.1f} "
          f"(écart-type : {ecart_type_points:.1f})")
    proba_over = None
    if args.ligne_total is not None:
        proba_over = float(norm.sf(args.ligne_total, loc=points_totaux_predits, scale=ecart_type_points))
        print(f"  Ligne totale mi-temps testée : {args.ligne_total:g}")
        print(f"  Probabilité Over : {proba_over:.1%}  |  Probabilité Under : {1 - proba_over:.1%}")

    confiance = evaluer_confiance(df_brut, equipe_domicile, equipe_exterieur, date_fictive)
    print(f"\n  Confiance : {confiance['niveau_confiance']} "
          f"({confiance['nb_matchs_domicile']} matchs connus pour {equipe_domicile}, "
          f"{confiance['nb_matchs_exterieur']} pour {equipe_exterieur}, "
          f"{confiance['nb_confrontations']} confrontation(s) directe(s))")

    nouvelle_ligne = {
        "date_prediction": pd.Timestamp.today().strftime("%d/%m/%Y %H:%M"),
        "equipe_domicile": equipe_domicile, "equipe_exterieur": equipe_exterieur,
        "score_mi_temps_domicile_predit": round(pred_dom, 1),
        "score_mi_temps_exterieur_predit": round(pred_ext, 1),
        "proba_mener_domicile": round(proba_dom, 3),
        "marge_predite_domicile": round(marge_predite, 1),
        "ligne_suggeree_domicile": ligne_suggeree,
        "ligne_testee": args.ligne if args.ligne is not None else "",
        "proba_couverture_domicile": round(proba_couverture, 3) if proba_couverture is not None else "",
        "points_totaux_mi_temps_predits": round(points_totaux_predits, 1),
        "ligne_total_testee": args.ligne_total if args.ligne_total is not None else "",
        "proba_over": round(proba_over, 3) if proba_over is not None else "",
        "confiance": confiance["niveau_confiance"],
    }
    df_nouvelle_ligne = pd.DataFrame([nouvelle_ligne])
    try:
        df_predictions = pd.concat([pd.read_csv(FICHIER_PREDICTIONS), df_nouvelle_ligne], ignore_index=True)
    except FileNotFoundError:
        df_predictions = df_nouvelle_ligne
    df_predictions.to_csv(FICHIER_PREDICTIONS, index=False)
    print(f"\nPrédiction ajoutée à {FICHIER_PREDICTIONS}.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sous_commandes = parser.add_subparsers(dest="commande", required=True)

    p_entrainer = sous_commandes.add_parser("entrainer")
    p_entrainer.add_argument("--historique", required=True)
    p_entrainer.add_argument("--saison-actuelle", required=True)
    p_entrainer.add_argument("--colonne-date", default="date")
    p_entrainer.add_argument("--format-date", default="%d/%m/%Y")
    p_entrainer.add_argument("--proportion-test", type=float, default=0.15)
    p_entrainer.set_defaults(func=entrainer)

    p_predire = sous_commandes.add_parser("predire")
    p_predire.add_argument("equipe_domicile")
    p_predire.add_argument("equipe_exterieur")
    p_predire.add_argument("--historique", required=True)
    p_predire.add_argument("--saison-actuelle", required=True)
    p_predire.add_argument("--colonne-date", default="date")
    p_predire.add_argument("--format-date", default="%d/%m/%Y")
    p_predire.add_argument("--date", default=None)
    p_predire.add_argument("--ligne", type=float, default=None,
                            help="Handicap mi-temps à tester pour l'équipe à domicile.")
    p_predire.add_argument("--ligne-total", type=float, default=None,
                            help="Ligne de total (Over/Under) mi-temps à tester.")
    p_predire.set_defaults(func=predire)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
