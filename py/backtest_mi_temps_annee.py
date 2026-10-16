"""
Backtest du modèle mi-temps DÉDIÉ (predire_mi_temps_dediee.py) sur une
année civile donnée (2026 par défaut), avec RÉENTRAÎNEMENT PÉRIODIQUE
(mensuel par défaut) — même principe que backtest_lot_annee.py :
à chaque nouveau mois de l'année testée, tout ce qui est réellement
connu jusque-là (historique + mois déjà backtestés de la même année)
sert à réentraîner les modèles avant de prédire le mois suivant.

En prime, ce script entraîne aussi deux régressions Q1/Q2 INDÉPENDANTES
(comme le fait ml_prediction.py pour tous les quarts-temps) et compare
leur somme au modèle dédié — pour vérifier, comme on l'a fait pour le
score total, si prédire la mi-temps directement bat le fait de sommer
deux prédictions de quart-temps distinctes.

Prérequis :
    pip install pandas numpy scikit-learn scipy
    (nécessite aussi ml_prediction.py et predire_mi_temps_dediee.py
    dans le même dossier)

Utilisation :
    python backtest_mi_temps_annee.py \
        --historique betclic_elite_2020-2025.csv \
        --saison-actuelle betclic_elite_2025-2026.csv \
        --annee 2026 \
        --sortie backtest_mi_temps_2026.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from ml_prediction import charger_et_fusionner, colonnes_features as colonnes_features_qt, construire_features, evaluer_confiance
from predire_mi_temps_dediee import ajouter_cibles_mi_temps

PARAMS_MODELE = dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)


def entrainer_modeles_mi_temps(train, cols_x):
    """Modèles DÉDIÉS : score mi-temps, vainqueur, écart, points totaux."""
    m_dom = GradientBoostingRegressor(**PARAMS_MODELE)
    m_dom.fit(train[cols_x], train["score_mi_temps_domicile"])
    m_ext = GradientBoostingRegressor(**PARAMS_MODELE)
    m_ext.fit(train[cols_x], train["score_mi_temps_exterieur"])
    clf = GradientBoostingClassifier(**PARAMS_MODELE)
    clf.fit(train[cols_x], train["victoire_domicile_mi_temps"])
    m_ecart = GradientBoostingRegressor(**PARAMS_MODELE)
    m_ecart.fit(train[cols_x], train["ecart_mi_temps"])
    m_points = GradientBoostingRegressor(**PARAMS_MODELE)
    m_points.fit(train[cols_x], train["points_totaux_mi_temps"])
    return m_dom, m_ext, clf, m_ecart, m_points


def entrainer_modeles_q1_q2(train, cols_x):
    """Modèles INDÉPENDANTS par quart-temps (Q1, Q2 séparément), pour comparaison."""
    modeles = {}
    for cible in ["q1_domicile", "q1_exterieur", "q2_domicile", "q2_exterieur"]:
        m = GradientBoostingRegressor(**PARAMS_MODELE)
        m.fit(train[cols_x], train[cible])
        modeles[cible] = m
    return modeles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historique", required=True)
    parser.add_argument("--saison-actuelle", required=True)
    parser.add_argument("--colonne-date", default="date")
    parser.add_argument("--format-date", default="%d/%m/%Y")
    parser.add_argument("--annee", type=int, default=2026)
    parser.add_argument("--sortie", default="../model_test/backtest_mi_temps_annee.csv")
    parser.add_argument("--frequence", choices=["mensuel", "trimestriel", "annuel"], default="mensuel")
    args = parser.parse_args()

    date_debut = pd.Timestamp(f"{args.annee}-01-01")
    date_fin = pd.Timestamp(f"{args.annee}-12-31 23:59:59")

    print("Chargement et construction des features (pipeline quart-temps)...")
    df_brut = charger_et_fusionner(args.historique, args.saison_actuelle,
                                    args.colonne_date, args.format_date)
    df_features = construire_features(df_brut)
    cols_x = colonnes_features_qt(df_features)
    df_features = ajouter_cibles_mi_temps(df_features)
    df_utile = df_features.dropna(subset=cols_x).reset_index(drop=True)

    if len(df_utile[df_utile["date"] < date_debut]) == 0:
        raise SystemExit(f"Aucune donnée d'entraînement avant {args.annee} — vérifiez vos fichiers.")

    test_annee = df_utile[(df_utile["date"] >= date_debut) & (df_utile["date"] <= date_fin)]
    if len(test_annee) == 0:
        raise SystemExit(f"Aucun match trouvé en {args.annee} dans vos fichiers.")

    if args.frequence == "annuel":
        checkpoints = [date_debut]
    else:
        pas = {"mensuel": "MS", "trimestriel": "3MS"}[args.frequence]
        checkpoints = list(pd.date_range(date_debut, date_fin, freq=pas))
        if not checkpoints or checkpoints[0] != date_debut:
            checkpoints = [date_debut] + checkpoints
    bornes_periodes = checkpoints + [date_fin + pd.Timedelta(seconds=1)]

    print(f"\n{len(checkpoints)} réentraînement(s) prévu(s) ({args.frequence}), "
          f"aux dates : {', '.join(c.strftime('%d/%m/%Y') for c in checkpoints)}")
    print(f"\nBacktest de {len(test_annee)} match(s) sur {args.annee}...\n")

    resultats = []
    compteur_global = 0

    for p, debut_periode in enumerate(checkpoints, start=1):
        fin_periode = bornes_periodes[p]
        train_p = df_utile[df_utile["date"] < debut_periode]
        test_p = df_utile[(df_utile["date"] >= debut_periode) & (df_utile["date"] < fin_periode)]
        if len(test_p) == 0:
            continue

        print(f"--- Période {p}/{len(checkpoints)} : entraînement sur {len(train_p)} "
              f"matchs connus avant le {debut_periode.strftime('%d/%m/%Y')}, "
              f"{len(test_p)} match(s) à prédire ---")

        m_dom, m_ext, clf, m_ecart, m_points = entrainer_modeles_mi_temps(train_p, cols_x)
        modeles_q1_q2 = entrainer_modeles_q1_q2(train_p, cols_x)

        for _, row in test_p.sort_values("date").iterrows():
            compteur_global += 1
            dom, ext, date_match = row["equipe_domicile"], row["equipe_exterieur"], row["date"]
            if compteur_global % 25 == 0 or compteur_global == len(test_annee):
                print(f"  [{compteur_global}/{len(test_annee)}] {date_match.date()} {dom} - {ext}")

            x = row[cols_x].to_frame().T

            pred_dom = float(m_dom.predict(x)[0])
            pred_ext = float(m_ext.predict(x)[0])
            proba_dom = float(clf.predict_proba(x)[0][1])
            pred_ecart = float(m_ecart.predict(x)[0])
            pred_points = float(m_points.predict(x)[0])

            pred_q1_dom = float(modeles_q1_q2["q1_domicile"].predict(x)[0])
            pred_q1_ext = float(modeles_q1_q2["q1_exterieur"].predict(x)[0])
            pred_q2_dom = float(modeles_q1_q2["q2_domicile"].predict(x)[0])
            pred_q2_ext = float(modeles_q1_q2["q2_exterieur"].predict(x)[0])
            somme_dom = pred_q1_dom + pred_q2_dom
            somme_ext = pred_q1_ext + pred_q2_ext

            confiance = evaluer_confiance(df_brut, dom, ext, date_match)

            resultats.append({
                "date": date_match.strftime("%d/%m/%Y"),
                "equipe_domicile": dom, "equipe_exterieur": ext,
                "periode_entrainement": debut_periode.strftime("%d/%m/%Y"),
                "score_mi_temps_domicile_reel": row["score_mi_temps_domicile"],
                "score_mi_temps_exterieur_reel": row["score_mi_temps_exterieur"],
                "score_mi_temps_domicile_dedie": round(pred_dom, 1),
                "score_mi_temps_exterieur_dedie": round(pred_ext, 1),
                "score_mi_temps_domicile_somme_q1q2": round(somme_dom, 1),
                "score_mi_temps_exterieur_somme_q1q2": round(somme_ext, 1),
                "victoire_mi_temps_reelle": row["victoire_domicile_mi_temps"],
                "proba_victoire_mi_temps_dedie": round(proba_dom, 3),
                "victoire_mi_temps_predite": int(proba_dom >= 0.5),
                "ecart_mi_temps_reel": row["ecart_mi_temps"],
                "ecart_mi_temps_predit": round(pred_ecart, 1),
                "points_totaux_mi_temps_reel": row["points_totaux_mi_temps"],
                "points_totaux_mi_temps_predit": round(pred_points, 1),
                "confiance": confiance["niveau_confiance"],
            })

    df_resultats = pd.DataFrame(resultats)
    df_resultats.to_csv(args.sortie, index=False)
    print(f"\nDétail match par match écrit dans {args.sortie}.")

    print(f"\n{'=' * 60}")
    print(f"SYNTHÈSE DU BACKTEST MI-TEMPS — {args.annee}")
    print(f"{'=' * 60}\n")

    mae_dom_dedie = mean_absolute_error(df_resultats["score_mi_temps_domicile_reel"],
                                         df_resultats["score_mi_temps_domicile_dedie"])
    mae_dom_somme = mean_absolute_error(df_resultats["score_mi_temps_domicile_reel"],
                                         df_resultats["score_mi_temps_domicile_somme_q1q2"])
    mae_ext_dedie = mean_absolute_error(df_resultats["score_mi_temps_exterieur_reel"],
                                         df_resultats["score_mi_temps_exterieur_dedie"])
    mae_ext_somme = mean_absolute_error(df_resultats["score_mi_temps_exterieur_reel"],
                                         df_resultats["score_mi_temps_exterieur_somme_q1q2"])

    print(f"Score mi-temps ({len(df_resultats)} matchs) :")
    print(f"  {'':20s} {'Modèle dédié':>14s} {'Somme Q1+Q2':>14s}")
    print(f"  {'domicile (MAE)':20s} {mae_dom_dedie:14.2f} {mae_dom_somme:14.2f}")
    print(f"  {'extérieur (MAE)':20s} {mae_ext_dedie:14.2f} {mae_ext_somme:14.2f}")

    mae_ecart = mean_absolute_error(df_resultats["ecart_mi_temps_reel"], df_resultats["ecart_mi_temps_predit"])
    mae_points = mean_absolute_error(df_resultats["points_totaux_mi_temps_reel"],
                                      df_resultats["points_totaux_mi_temps_predit"])
    print(f"\n  MAE écart mi-temps (dédié) : {mae_ecart:.2f}")
    print(f"  MAE points totaux mi-temps (dédié) : {mae_points:.2f}")

    acc = accuracy_score(df_resultats["victoire_mi_temps_reelle"], df_resultats["victoire_mi_temps_predite"])
    ll = log_loss(df_resultats["victoire_mi_temps_reelle"], df_resultats["proba_victoire_mi_temps_dedie"],
                  labels=[0, 1])
    print(f"\n  Précision (qui mène à la mi-temps) : {acc:.1%}  |  Log-loss : {ll:.3f}")

    taux_domicile = df_resultats["victoire_mi_temps_reelle"].mean()
    print(f"  (référence naïve « domicile mène toujours » : {taux_domicile:.1%} des matchs)")

    print(f"\nRépartition de la confiance :")
    print(df_resultats["confiance"].value_counts().to_string())


if __name__ == "__main__":
    main()
