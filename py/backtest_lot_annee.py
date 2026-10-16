"""
Backtest du pipeline complet (QT ML, QT Poisson, score total, écart)
sur une année civile donnée (2026 par défaut). Les modèles sont
RÉENTRAÎNÉS PÉRIODIQUEMENT (mensuel par défaut) : à chaque nouveau
mois de l'année testée, tout ce qui est réellement connu jusque-là
(l'historique complet + les mois déjà backtestés de la même année)
sert à réentraîner les modèles avant de prédire le mois suivant.
C'est plus réaliste qu'un entraînement figé une fois pour toute
l'année : ça correspond à un usage où vous relanceriez régulièrement
"entrainer" au fil de la saison. Utilisez --frequence annuel pour
revenir à un entraînement unique (comportement plus simple, mais
moins réaliste).

Le modèle Poisson, lui, n'est jamais "entraîné" à proprement parler :
il est recalculé en glissant pour chaque match, avec seulement
l'historique réellement antérieur à ce match (comme dans
backtest_predictions.py et comparer_modeles.py).

Prérequis :
    pip install pandas numpy scikit-learn scipy
    (nécessite aussi ml_prediction.py et predire_total.py dans le même
    dossier, dont ce script réutilise les fonctions)

Utilisation :
    python backtest_lot_annee.py \
        --historique betclic_elite_2020-2025.csv \
        --saison-actuelle betclic_elite_2025-2026.csv \
        --annee 2026 \
        --sortie backtest_2026.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from ml_prediction import (
    CIBLES as CIBLES_QUARTS,
    charger_et_fusionner,
    colonnes_features as colonnes_features_qt,
    construire_features,
    evaluer_confiance,
    predire_quarts_poisson,
)
from predire_total import (
    calculer_intervalles,
    charger_et_fusionner_total,
    colonnes_features as colonnes_features_total,
    construire_features_total,
)
from predire_mi_temps_dediee import ajouter_cibles_mi_temps

PARAMS_MODELE = dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)


def entrainer_modeles_qt(train, cols_x):
    modeles = {}
    for cible in CIBLES_QUARTS:
        m = GradientBoostingRegressor(**PARAMS_MODELE)
        m.fit(train[cols_x], train[cible])
        modeles[cible] = m
    clf = GradientBoostingClassifier(**PARAMS_MODELE)
    clf.fit(train[cols_x], train["victoire_domicile"])
    return modeles, clf


def entrainer_modeles_total(train, cols_x):
    m_dom = GradientBoostingRegressor(**PARAMS_MODELE)
    m_dom.fit(train[cols_x], train["score_domicile"])
    m_ext = GradientBoostingRegressor(**PARAMS_MODELE)
    m_ext.fit(train[cols_x], train["score_exterieur"])
    clf = GradientBoostingClassifier(**PARAMS_MODELE)
    clf.fit(train[cols_x], train["victoire_domicile"])
    m_ecart = GradientBoostingRegressor(**PARAMS_MODELE)
    m_ecart.fit(train[cols_x], train["ecart"])
    m_points = GradientBoostingRegressor(**PARAMS_MODELE)
    m_points.fit(train[cols_x], train["points_totaux"])
    # écart-type des résidus, mesuré EN INTERNE sur le train (faute d'un
    # jeu de validation dédié à ce stade du backtest) : approximation
    # raisonnable pour alimenter la grille de tranches (calculer_intervalles)
    ecart_type_points = float(np.std(train["points_totaux"].values - m_points.predict(train[cols_x])))
    return m_dom, m_ext, clf, m_ecart, m_points, ecart_type_points


def entrainer_modeles_mi_temps(train, cols_x):
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


def tranche_de_valeur(valeur, bornes):
    """Nom de la tranche (même convention que calculer_intervalles) contenant `valeur`."""
    bornes = sorted(bornes)
    if valeur <= bornes[0]:
        return f"0-{bornes[0]}"
    for i in range(len(bornes) - 1):
        if bornes[i] < valeur <= bornes[i + 1]:
            return f"{bornes[i] + 1}-{bornes[i + 1]}"
    return f"{bornes[-1] + 1} et plus"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historique", required=True)
    parser.add_argument("--saison-actuelle", required=True)
    parser.add_argument("--colonne-date", default="date")
    parser.add_argument("--format-date", default="%d/%m/%Y")
    parser.add_argument("--annee", type=int, default=2026,
                         help="Année civile à backtester (les modèles sont entraînés "
                              "uniquement sur les données antérieures au 1er janvier).")
    parser.add_argument("--sortie", default="backtest_lot_annee.csv")
    parser.add_argument("--frequence", choices=["mensuel", "trimestriel", "annuel"], default="mensuel",
                         help="Fréquence de réentraînement pendant l'année testée. 'annuel' revient "
                              "au comportement d'origine (un seul entraînement, sur les données "
                              "antérieures au 1er janvier).")
    parser.add_argument("--bornes", default="100,110,120,130,140,150,160,170,180",
                         help="Bornes de la grille de score total (voir predire_total.py).")
    args = parser.parse_args()

    bornes = [int(b) for b in args.bornes.split(",")]

    date_debut = pd.Timestamp(f"{args.annee}-01-01")
    date_fin = pd.Timestamp(f"{args.annee}-12-31 23:59:59")

    # ------------------------------------------------------------
    # Chargement + features (les deux pipelines, comme predire_lot.py)
    # ------------------------------------------------------------
    print("Chargement et construction des features (pipeline quart-temps)...")
    df_qt = charger_et_fusionner(args.historique, args.saison_actuelle,
                                  args.colonne_date, args.format_date)
    df_features_qt = construire_features(df_qt)
    cols_x_qt = colonnes_features_qt(df_features_qt)
    df_features_qt = ajouter_cibles_mi_temps(df_features_qt)
    df_utile_qt = df_features_qt.dropna(subset=cols_x_qt).reset_index(drop=True)

    print("Chargement et construction des features (pipeline score total)...")
    df_total = charger_et_fusionner_total(args.historique, args.saison_actuelle,
                                           args.colonne_date, args.format_date)
    df_features_total = construire_features_total(df_total)
    cols_x_total = colonnes_features_total(df_features_total)
    df_utile_total = df_features_total.dropna(subset=cols_x_total).reset_index(drop=True)

    # ------------------------------------------------------------
    # Découpage train (avant l'année testée) / test (l'année testée)
    # ------------------------------------------------------------
    train_qt = df_utile_qt[df_utile_qt["date"] < date_debut]
    test_qt = df_utile_qt[(df_utile_qt["date"] >= date_debut) & (df_utile_qt["date"] <= date_fin)]

    if len(df_utile_qt[df_utile_qt["date"] < date_debut]) == 0 or len(df_utile_total[df_utile_total["date"] < date_debut]) == 0:
        raise SystemExit(f"Aucune donnée d'entraînement avant {args.annee} — vérifiez vos fichiers.")

    test_total_annee = df_utile_total[(df_utile_total["date"] >= date_debut) & (df_utile_total["date"] <= date_fin)]
    if len(test_total_annee) == 0:
        raise SystemExit(f"Aucun match trouvé en {args.annee} dans vos fichiers.")

    # ------------------------------------------------------------
    # Périodes de réentraînement : par défaut, un réentraînement au
    # début de chaque mois de l'année testée, avec TOUTES les données
    # connues jusque-là (y compris les mois déjà backtestés de la même
    # année) — c'est ce qui rend le backtest réaliste : le modèle
    # apprend au fil du temps, comme vous le feriez en le relançant
    # régulièrement.
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

    # index rapide des lignes QT par (equipe_domicile, equipe_exterieur, date)
    # pour retrouver, pour chaque match du pipeline total, ses éventuelles
    # features/valeurs réelles quart-temps
    test_qt_index = {
        (r["equipe_domicile"], r["equipe_exterieur"], r["date"]): r
        for _, r in df_utile_qt[
            (df_utile_qt["date"] >= date_debut) & (df_utile_qt["date"] <= date_fin)
        ].iterrows()
    }

    # ------------------------------------------------------------
    # Boucle par période : réentraînement puis prédiction des matchs
    # de cette période uniquement
    # ------------------------------------------------------------
    print(f"\nBacktest de {len(test_total_annee)} match(s) sur {args.annee}...\n")
    resultats = []
    compteur_global = 0

    for p, debut_periode in enumerate(checkpoints, start=1):
        fin_periode = bornes_periodes[p]

        train_qt_p = df_utile_qt[df_utile_qt["date"] < debut_periode]
        train_total_p = df_utile_total[df_utile_total["date"] < debut_periode]
        test_total_p = df_utile_total[
            (df_utile_total["date"] >= debut_periode) & (df_utile_total["date"] < fin_periode)
        ]
        if len(test_total_p) == 0:
            continue

        print(f"--- Période {p}/{len(checkpoints)} : entraînement sur {len(train_total_p)} "
              f"matchs connus avant le {debut_periode.strftime('%d/%m/%Y')}, "
              f"{len(test_total_p)} match(s) à prédire ---")

        modeles_qt_reg, modele_qt_clf = entrainer_modeles_qt(train_qt_p, cols_x_qt)
        modele_total_dom, modele_total_ext, modele_total_clf, modele_ecart, modele_points, ecart_type_points = (
            entrainer_modeles_total(train_total_p, cols_x_total)
        )
        modele_mt_dom, modele_mt_ext, modele_mt_clf, modele_mt_ecart, modele_mt_points = (
            entrainer_modeles_mi_temps(train_qt_p, cols_x_qt)
        )

        for _, row in test_total_p.sort_values("date").iterrows():
            compteur_global += 1
            dom, ext, date_match = row["equipe_domicile"], row["equipe_exterieur"], row["date"]
            if compteur_global % 25 == 0 or compteur_global == len(test_total_annee):
                print(f"  [{compteur_global}/{len(test_total_annee)}] {date_match.date()} {dom} - {ext}")

            resultat = {
                "date": date_match.strftime("%d/%m/%Y"),
                "equipe_domicile": dom,
                "equipe_exterieur": ext,
                "periode_entrainement": debut_periode.strftime("%d/%m/%Y"),
                "score_domicile_reel": row["score_domicile"],
                "score_exterieur_reel": row["score_exterieur"],
            }

            # --- Total : score, vainqueur, écart, points totaux ---
            x_total = row[cols_x_total].to_frame().T
            pred_dom = float(modele_total_dom.predict(x_total)[0])
            pred_ext = float(modele_total_ext.predict(x_total)[0])
            proba_dom_total = float(modele_total_clf.predict_proba(x_total)[0][1])
            pred_ecart = float(modele_ecart.predict(x_total)[0])
            pred_points = float(modele_points.predict(x_total)[0])

            resultat.update({
                "total_score_domicile_predit": round(pred_dom, 1),
                "total_score_exterieur_predit": round(pred_ext, 1),
                "total_proba_victoire_domicile": round(proba_dom_total, 3),
                "total_victoire_domicile_predite": int(proba_dom_total >= 0.5),
                "victoire_domicile_reelle": int(row["score_domicile"] > row["score_exterieur"]),
                "ecart_reel": row["ecart"],
                "ecart_predit": round(pred_ecart, 1),
                "points_totaux_reel": row["points_totaux"],
                "points_totaux_predit": round(pred_points, 1),
            })

            # --- Grille de score total par tranche ---
            intervalles = calculer_intervalles(pred_points, ecart_type_points, bornes)
            proba_par_tranche = {iv["tranche"]: iv["probabilite"] for iv in intervalles}
            tranche_reelle = tranche_de_valeur(row["points_totaux"], bornes)
            tranche_predite_top = max(proba_par_tranche, key=proba_par_tranche.get)
            proba_tranche_reelle = max(proba_par_tranche.get(tranche_reelle, 1e-6), 1e-6)
            resultat["tranche_reelle"] = tranche_reelle
            resultat["tranche_predite_top"] = tranche_predite_top
            resultat["tranche_top_correcte"] = int(tranche_reelle == tranche_predite_top)
            resultat["proba_tranche_reelle"] = round(proba_tranche_reelle, 4)
            resultat["log_score_tranche"] = round(-np.log(proba_tranche_reelle), 3)

            # --- Confiance ---
            confiance = evaluer_confiance(df_total, dom, ext, date_match)
            resultat["confiance"] = confiance["niveau_confiance"]

            # --- Quart-temps ML (si ce match a le détail QT) ---
            ligne_qt = test_qt_index.get((dom, ext, date_match))
            if ligne_qt is not None:
                x_qt = ligne_qt[cols_x_qt].to_frame().T
                total_dom_ml, total_ext_ml = 0.0, 0.0
                for cible in CIBLES_QUARTS:
                    pred = float(modeles_qt_reg[cible].predict(x_qt)[0])
                    reel = ligne_qt[cible]
                    resultat[f"ml_{cible}_reel"] = reel
                    resultat[f"ml_{cible}_predit"] = round(pred, 1)
                    resultat[f"ml_{cible}_erreur"] = round(abs(reel - pred), 1)
                    if "domicile" in cible:
                        total_dom_ml += pred
                    else:
                        total_ext_ml += pred
                resultat["ml_proba_victoire_domicile"] = round(
                    float(modele_qt_clf.predict_proba(x_qt)[0][1]), 3
                )

                # --- Mi-temps (modèle dédié, mêmes features que le QT) ---
                reel_mt_dom = ligne_qt["q1_domicile"] + ligne_qt["q2_domicile"]
                reel_mt_ext = ligne_qt["q1_exterieur"] + ligne_qt["q2_exterieur"]
                pred_mt_dom = float(modele_mt_dom.predict(x_qt)[0])
                pred_mt_ext = float(modele_mt_ext.predict(x_qt)[0])
                proba_mt_dom = float(modele_mt_clf.predict_proba(x_qt)[0][1])
                pred_mt_ecart = float(modele_mt_ecart.predict(x_qt)[0])
                pred_mt_points = float(modele_mt_points.predict(x_qt)[0])

                resultat.update({
                    "mi_temps_score_domicile_reel": reel_mt_dom,
                    "mi_temps_score_exterieur_reel": reel_mt_ext,
                    "mi_temps_score_domicile_predit": round(pred_mt_dom, 1),
                    "mi_temps_score_exterieur_predit": round(pred_mt_ext, 1),
                    "mi_temps_victoire_reelle": int(reel_mt_dom > reel_mt_ext),
                    "mi_temps_proba_mener_domicile": round(proba_mt_dom, 3),
                    "mi_temps_ecart_reel": reel_mt_dom - reel_mt_ext,
                    "mi_temps_ecart_predit": round(pred_mt_ecart, 1),
                    "mi_temps_points_totaux_reel": reel_mt_dom + reel_mt_ext,
                    "mi_temps_points_totaux_predit": round(pred_mt_points, 1),
                })

                # --- Poisson, recalculé en glissant avec l'historique
                # réellement antérieur à CE match précis ---
                historique_avant = df_qt[df_qt["_date"] < date_match]
                pred_poisson = predire_quarts_poisson(historique_avant, dom, ext)
                if pred_poisson is not None:
                    for cible in CIBLES_QUARTS:
                        resultat[f"poisson_{cible}_predit"] = round(float(pred_poisson[cible]), 1)
                        resultat[f"poisson_{cible}_erreur"] = round(
                            abs(ligne_qt[cible] - pred_poisson[cible]), 1
                        )
                    resultat["poisson_proba_victoire_domicile"] = round(
                        float(pred_poisson["proba_victoire_domicile"]), 3
                    )
                    # vainqueur par quart-temps (Poisson/Skellam par quart-temps)
                    for q_i in range(1, 5):
                        resultat[f"poisson_proba_victoire_q{q_i}_domicile"] = round(
                            float(pred_poisson[f"proba_victoire_q{q_i}_domicile"]), 3
                        )
                        resultat[f"victoire_q{q_i}_domicile_reelle"] = int(
                            ligne_qt[f"q{q_i}_domicile"] > ligne_qt[f"q{q_i}_exterieur"]
                        )

            resultats.append(resultat)

    df_resultats = pd.DataFrame(resultats)
    df_resultats.to_csv(args.sortie, index=False)
    print(f"\nDétail match par match écrit dans {args.sortie}.")

    # ------------------------------------------------------------
    # Synthèse
    # ------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"SYNTHÈSE DU BACKTEST — {args.annee}")
    print(f"{'=' * 60}\n")

    print(f"Score total (modèle dédié, {len(df_resultats)} matchs) :")
    print(f"  MAE score domicile : "
          f"{mean_absolute_error(df_resultats['score_domicile_reel'], df_resultats['total_score_domicile_predit']):.2f}")
    print(f"  MAE score extérieur : "
          f"{mean_absolute_error(df_resultats['score_exterieur_reel'], df_resultats['total_score_exterieur_predit']):.2f}")
    print(f"  MAE écart (marge de victoire) : "
          f"{mean_absolute_error(df_resultats['ecart_reel'], df_resultats['ecart_predit']):.2f}")
    print(f"  MAE points totaux : "
          f"{mean_absolute_error(df_resultats['points_totaux_reel'], df_resultats['points_totaux_predit']):.2f}")

    acc_tranche = df_resultats["tranche_top_correcte"].mean()
    log_score_moyen = df_resultats["log_score_tranche"].mean()
    print(f"\nGrille de score total par tranche :")
    print(f"  Précision (tranche la plus probable = tranche réelle) : {acc_tranche:.1%}")
    print(f"  Log-score moyen (plus bas = mieux) : {log_score_moyen:.3f}")

    acc_total = accuracy_score(df_resultats["victoire_domicile_reelle"], df_resultats["total_victoire_domicile_predite"])
    ll_total = log_loss(df_resultats["victoire_domicile_reelle"], df_resultats["total_proba_victoire_domicile"],
                         labels=[0, 1])
    print(f"  Précision vainqueur : {acc_total:.1%}  |  Log-loss : {ll_total:.3f}")

    sous_ens_qt = df_resultats.dropna(subset=["ml_proba_victoire_domicile"])
    if len(sous_ens_qt) > 0:
        print(f"\nQuart-temps ML ({len(sous_ens_qt)} matchs avec détail QT connu) :")
        for cible in CIBLES_QUARTS:
            col = f"ml_{cible}_erreur"
            if col in sous_ens_qt.columns:
                print(f"  MAE {cible:16s} : {sous_ens_qt[col].mean():.2f}")
        pred_bin_qt = (sous_ens_qt["ml_proba_victoire_domicile"] >= 0.5).astype(int)
        acc_qt = accuracy_score(sous_ens_qt["victoire_domicile_reelle"], pred_bin_qt)
        ll_qt = log_loss(sous_ens_qt["victoire_domicile_reelle"], sous_ens_qt["ml_proba_victoire_domicile"],
                          labels=[0, 1])
        print(f"  Précision vainqueur (ML, via quart-temps) : {acc_qt:.1%}  |  Log-loss : {ll_qt:.3f}")

    sous_ens_poisson = df_resultats.dropna(subset=["poisson_proba_victoire_domicile"]) \
        if "poisson_proba_victoire_domicile" in df_resultats.columns else pd.DataFrame()
    if len(sous_ens_poisson) > 0:
        print(f"\nQuart-temps Poisson ({len(sous_ens_poisson)} matchs) :")
        for cible in CIBLES_QUARTS:
            col = f"poisson_{cible}_erreur"
            if col in sous_ens_poisson.columns:
                print(f"  MAE {cible:16s} : {sous_ens_poisson[col].mean():.2f}")
        pred_bin_poisson = (sous_ens_poisson["poisson_proba_victoire_domicile"] >= 0.5).astype(int)
        acc_poisson = accuracy_score(sous_ens_poisson["victoire_domicile_reelle"], pred_bin_poisson)
        ll_poisson = log_loss(sous_ens_poisson["victoire_domicile_reelle"],
                               sous_ens_poisson["poisson_proba_victoire_domicile"], labels=[0, 1])
        print(f"  Précision vainqueur (Poisson) : {acc_poisson:.1%}  |  Log-loss : {ll_poisson:.3f}")

        print(f"\nVainqueur par quart-temps (Poisson) :")
        for q_i in range(1, 5):
            col_proba = f"poisson_proba_victoire_q{q_i}_domicile"
            col_reel = f"victoire_q{q_i}_domicile_reelle"
            if col_proba in sous_ens_poisson.columns:
                pred_bin_q = (sous_ens_poisson[col_proba] >= 0.5).astype(int)
                acc_q = accuracy_score(sous_ens_poisson[col_reel], pred_bin_q)
                ll_q = log_loss(sous_ens_poisson[col_reel], sous_ens_poisson[col_proba], labels=[0, 1])
                print(f"  Q{q_i} : précision {acc_q:.1%}  |  log-loss {ll_q:.3f}")

    print(f"\nRépartition de la confiance :")
    print(df_resultats["confiance"].value_counts().to_string())

    sous_ens_mt = df_resultats.dropna(subset=["mi_temps_proba_mener_domicile"]) \
        if "mi_temps_proba_mener_domicile" in df_resultats.columns else pd.DataFrame()
    if len(sous_ens_mt) > 0:
        print(f"\nMi-temps (modèle dédié, {len(sous_ens_mt)} matchs) :")
        mae_mt_dom = mean_absolute_error(sous_ens_mt["mi_temps_score_domicile_reel"],
                                          sous_ens_mt["mi_temps_score_domicile_predit"])
        mae_mt_ext = mean_absolute_error(sous_ens_mt["mi_temps_score_exterieur_reel"],
                                          sous_ens_mt["mi_temps_score_exterieur_predit"])
        mae_mt_ecart = mean_absolute_error(sous_ens_mt["mi_temps_ecart_reel"], sous_ens_mt["mi_temps_ecart_predit"])
        mae_mt_points = mean_absolute_error(sous_ens_mt["mi_temps_points_totaux_reel"],
                                             sous_ens_mt["mi_temps_points_totaux_predit"])
        print(f"  MAE score domicile : {mae_mt_dom:.2f}  |  MAE score extérieur : {mae_mt_ext:.2f}")
        print(f"  MAE écart : {mae_mt_ecart:.2f}  |  MAE points totaux : {mae_mt_points:.2f}")
        pred_bin_mt = (sous_ens_mt["mi_temps_proba_mener_domicile"] >= 0.5).astype(int)
        acc_mt = accuracy_score(sous_ens_mt["mi_temps_victoire_reelle"], pred_bin_mt)
        ll_mt = log_loss(sous_ens_mt["mi_temps_victoire_reelle"], sous_ens_mt["mi_temps_proba_mener_domicile"],
                          labels=[0, 1])
        print(f"  Précision (qui mène à la mi-temps) : {acc_mt:.1%}  |  Log-loss : {ll_mt:.3f}")


if __name__ == "__main__":
    main()
