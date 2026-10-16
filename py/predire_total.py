"""
Prédit le SCORE TOTAL du match (score_domicile, score_exterieur) et le
vainqueur, avec un modèle entraîné DIRECTEMENT sur le score final —
indépendant des modèles par quart-temps de ml_prediction.py.

Pourquoi un pipeline séparé plutôt que de sommer les 4 quarts-temps
prédits (comme le fait ml_prediction.py) :

  1. Statistiquement, sommer 4 prédictions indépendantes n'est pas la
     façon la plus précise de prédire un total — un modèle qui vise
     directement le score final peut être meilleur sur cette cible
     précise (voir l'échange précédent sur MAE quart par quart vs MAE
     totale).

  2. Surtout : ce pipeline n'exige PAS le détail par quart-temps pour
     s'entraîner — seulement score_domicile/score_exterieur. Si vous
     ajoutez des saisons anciennes dont vous n'avez que le score final
     (pas les quarts-temps), ml_prediction.py les ignorerait
     totalement (dropna sur les colonnes de quart-temps), alors que ce
     script peut s'en servir pleinement.

Les features (forme récente, Elo, classement, repos, confrontations
directes) sont donc recalculées ici sur la base des POINTS TOTAUX
marqués/encaissés, pas quart-temps par quart-temps — de sorte que
n'importe quel match avec juste un score final utilisable peut nourrir
l'historique de chaque équipe.

Prérequis :
    pip install pandas numpy scikit-learn joblib
    (nécessite aussi ml_prediction.py dans le même dossier, dont ce
    script réutilise la normalisation des noms d'équipe)

Utilisation :
    # Entraîner et évaluer
    python predire_total.py entrainer \
        --historique betclic_elite_2020-2025.csv \
        --saison-actuelle betclic_elite_2025-2026.csv

    # Prédire un match à venir
    python predire_total.py predire "Paris Basketball" "AS Monaco" \
        --historique betclic_elite_2020-2025.csv \
        --saison-actuelle betclic_elite_2025-2026.csv
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from ml_prediction import normaliser_nom_equipe, parser_dates_flexible, suggerer_noms_proches, evaluer_confiance

FENETRE_FORME = 10
ELO_INITIAL = 1500
ELO_K = 20

CIBLES = ["score_domicile", "score_exterieur"]

FICHIER_MODELE_REGRESSION = "../model_data/modele_total_score.joblib"
FICHIER_MODELE_CLASSIFICATION = "../model_data/modele_total_vainqueur.joblib"
FICHIER_MODELE_ECART = "../model_data/modele_total_ecart.joblib"
FICHIER_MODELE_POINTS_TOTAUX = "../model_data/modele_points_totaux.joblib"
FICHIER_FEATURES = "../model_data/features_total_construites.csv"
FICHIER_RESULTATS_TEST = "../model_test/resultats_test_total.csv"
FICHIER_PREDICTIONS = "../prediction/predictions_total_a_venir.csv"


# ---------------------------------------------------------------------
# Chargement — n'exige QUE le score final, pas le détail quart-temps
# ---------------------------------------------------------------------

def charger_et_fusionner_total(chemin_historique: str, chemin_saison_actuelle: str,
                                colonne_date: str, format_date: str) -> pd.DataFrame:
    hist = pd.read_csv(chemin_historique, dtype=str)
    actu = pd.read_csv(chemin_saison_actuelle, dtype=str)

    colonnes_communes = [c for c in actu.columns if c in hist.columns]
    df = pd.concat([hist[colonnes_communes], actu[colonnes_communes]], ignore_index=True)

    for c in ("equipe_domicile", "equipe_exterieur"):
        if c in df.columns:
            df[c] = df[c].apply(normaliser_nom_equipe)

    for c in CIBLES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nb_avant = len(df)
    df = df.dropna(subset=CIBLES).copy()
    if len(df) < nb_avant:
        print(f"{nb_avant - len(df)} ligne(s) écartée(s) : score final manquant.")

    df["_date"] = parser_dates_flexible(df[colonne_date], format_date)
    nb_avant = len(df)
    df = df.dropna(subset=["_date"])
    if len(df) < nb_avant:
        print(f"{nb_avant - len(df)} match(s) écarté(s) : date illisible.")

    # Dédoublonnage (voir ml_prediction.py::charger_et_fusionner pour
    # l'explication complète) : un même match présent à la fois dans
    # l'historique et la saison en cours ne doit être compté qu'une
    # fois. keep="last" garde la version issue de la saison en cours.
    nb_avant = len(df)
    df = df.drop_duplicates(subset=["equipe_domicile", "equipe_exterieur", "_date"], keep="last")
    if len(df) < nb_avant:
        print(f"{nb_avant - len(df)} match(s) en double écarté(s) (présents à la fois "
              f"dans l'historique et la saison en cours).")

    df = df.sort_values("_date").reset_index(drop=True)
    print(f"{len(df)} matchs exploitables au total (avec ou sans détail quart-temps).")
    return df


# ---------------------------------------------------------------------
# Construction des features, basées sur le score TOTAL uniquement
# ---------------------------------------------------------------------

def calculer_saison(date_match) -> str:
    annee, mois = date_match.year, date_match.month
    return f"{annee}-{annee + 1}" if mois >= 8 else f"{annee - 1}-{annee}"


def construire_features_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Équivalent de construire_features (ml_prediction.py) mais entièrement
    basé sur les points TOTAUX marqués/encaissés — aucune dépendance au
    détail par quart-temps, pour pouvoir exploiter des matchs dont on
    n'a que le score final.
    """
    historique_par_equipe = {}
    elo = {}
    derniere_date = {}
    stats_saison = {}

    lignes = []

    for _, m in df.iterrows():
        dom, ext = m["equipe_domicile"], m["equipe_exterieur"]
        date_match = m["_date"]
        saison_match = calculer_saison(date_match)

        for equipe in (dom, ext):
            historique_par_equipe.setdefault(equipe, [])
            elo.setdefault(equipe, ELO_INITIAL)
            stats_saison.setdefault((saison_match, equipe), {
                "victoires": 0, "matchs": 0, "diff_points": 0,
            })

        def classement_saison(equipe: str):
            s = stats_saison[(saison_match, equipe)]
            if s["matchs"] == 0:
                return {"classement": np.nan, "taux_victoire": np.nan, "diff_points_moyen": np.nan}
            classement = sorted(
                [
                    (nom, val["victoires"], val["diff_points"])
                    for (sais, nom), val in stats_saison.items()
                    if sais == saison_match and val["matchs"] > 0
                ],
                key=lambda t: (-t[1], -t[2]),
            )
            rang = next((i + 1 for i, (nom, _, _) in enumerate(classement) if nom == equipe), np.nan)
            return {
                "classement": rang,
                "taux_victoire": s["victoires"] / s["matchs"],
                "diff_points_moyen": s["diff_points"] / s["matchs"],
            }

        def forme_recente(equipe: str, cote: str):
            """Moyenne des FENETRE_FORME derniers matchs joués côté `cote`, en points TOTAUX."""
            matchs_passes = [
                h for h in historique_par_equipe[equipe][-FENETRE_FORME:] if h["position"] == cote
            ]
            if not matchs_passes:
                return {"marque": np.nan, "encaisse": np.nan}
            return {
                "marque": np.mean([h["marque"] for h in matchs_passes]),
                "encaisse": np.mean([h["encaisse"] for h in matchs_passes]),
            }

        def repos(equipe: str):
            if equipe not in derniere_date:
                return np.nan
            return (date_match - derniere_date[equipe]).days

        def stats_h2h(equipe_a: str, equipe_b: str):
            confrontations = [h for h in historique_par_equipe[equipe_a] if h["adversaire"] == equipe_b]
            if not confrontations:
                return {"marque": np.nan, "encaisse": np.nan, "nb": 0}
            return {
                "marque": np.mean([h["marque"] for h in confrontations]),
                "encaisse": np.mean([h["encaisse"] for h in confrontations]),
                "nb": len(confrontations),
            }

        forme_dom = forme_recente(dom, "domicile")
        forme_ext = forme_recente(ext, "exterieur")
        h2h_dom = stats_h2h(dom, ext)
        classement_dom = classement_saison(dom)
        classement_ext = classement_saison(ext)
        repos_dom = repos(dom)
        repos_ext = repos(ext)

        ligne = {
            "date": date_match,
            "equipe_domicile": dom,
            "equipe_exterieur": ext,
            "elo_domicile": elo[dom],
            "elo_exterieur": elo[ext],
            "diff_elo": elo[dom] - elo[ext],
            "repos_domicile": repos_dom,
            "repos_exterieur": repos_ext,
            "back_to_back_domicile": int(repos_dom <= 1) if not np.isnan(repos_dom) else np.nan,
            "back_to_back_exterieur": int(repos_ext <= 1) if not np.isnan(repos_ext) else np.nan,
            "nb_confrontations": h2h_dom["nb"],
            "forme_dom_marque": forme_dom["marque"],
            "forme_dom_encaisse": forme_dom["encaisse"],
            "forme_ext_marque": forme_ext["marque"],
            "forme_ext_encaisse": forme_ext["encaisse"],
            "h2h_marque_dom": h2h_dom["marque"],
            "h2h_encaisse_dom": h2h_dom["encaisse"],
            "classement_domicile": classement_dom["classement"],
            "classement_exterieur": classement_ext["classement"],
            "diff_classement": (
                classement_ext["classement"] - classement_dom["classement"]
                if not (np.isnan(classement_dom["classement"]) or np.isnan(classement_ext["classement"]))
                else np.nan
            ),
            "taux_victoire_saison_domicile": classement_dom["taux_victoire"],
            "taux_victoire_saison_exterieur": classement_ext["taux_victoire"],
            "diff_points_moyen_saison_domicile": classement_dom["diff_points_moyen"],
            "diff_points_moyen_saison_exterieur": classement_ext["diff_points_moyen"],
            "score_domicile": m["score_domicile"],
            "score_exterieur": m["score_exterieur"],
            "ecart": m["score_domicile"] - m["score_exterieur"],
            "points_totaux": m["score_domicile"] + m["score_exterieur"],
            "victoire_domicile": int(m["score_domicile"] > m["score_exterieur"]),
        }
        lignes.append(ligne)

        historique_par_equipe[dom].append({
            "position": "domicile", "adversaire": ext,
            "marque": m["score_domicile"], "encaisse": m["score_exterieur"],
        })
        historique_par_equipe[ext].append({
            "position": "exterieur", "adversaire": dom,
            "marque": m["score_exterieur"], "encaisse": m["score_domicile"],
        })

        s_dom, s_ext = stats_saison[(saison_match, dom)], stats_saison[(saison_match, ext)]
        s_dom["matchs"] += 1
        s_ext["matchs"] += 1
        diff = m["score_domicile"] - m["score_exterieur"]
        s_dom["diff_points"] += diff
        s_ext["diff_points"] -= diff
        if diff > 0:
            s_dom["victoires"] += 1
        else:
            s_ext["victoires"] += 1

        attendu_dom = 1 / (1 + 10 ** ((elo[ext] - elo[dom]) / 400))
        resultat_dom = 1.0 if diff > 0 else 0.0
        elo[dom] += ELO_K * (resultat_dom - attendu_dom)
        elo[ext] += ELO_K * ((1 - resultat_dom) - (1 - attendu_dom))

        derniere_date[dom] = date_match
        derniere_date[ext] = date_match

    return pd.DataFrame(lignes)


def calculer_intervalles(moyenne: float, ecart_type: float, bornes):
    """
    Découpe une loi normale N(moyenne, ecart_type) en tranches selon
    les `bornes` fournies (ex : [100, 110, 120, ...]), et renvoie pour
    chaque tranche sa probabilité et la cote décimale "juste"
    équivalente (1 / probabilité) — comme sur les grilles de paris par
    tranche de points.

    La première tranche est "0 à bornes[0]", la dernière est
    "bornes[-1]+1 et plus", les tranches intermédiaires sont
    "bornes[i]+1 à bornes[i+1]".
    """
    bornes = sorted(bornes)
    intervalles = []

    proba = float(norm.cdf(bornes[0], loc=moyenne, scale=ecart_type))
    intervalles.append({
        "tranche": f"0-{bornes[0]}", "borne_basse": None, "borne_haute": bornes[0],
        "probabilite": proba,
    })

    for i in range(len(bornes) - 1):
        b_min, b_max = bornes[i], bornes[i + 1]
        proba = float(
            norm.cdf(b_max, loc=moyenne, scale=ecart_type)
            - norm.cdf(b_min, loc=moyenne, scale=ecart_type)
        )
        intervalles.append({
            "tranche": f"{b_min + 1}-{b_max}", "borne_basse": b_min, "borne_haute": b_max,
            "probabilite": proba,
        })

    proba = float(norm.sf(bornes[-1], loc=moyenne, scale=ecart_type))
    intervalles.append({
        "tranche": f"{bornes[-1] + 1} et plus", "borne_basse": bornes[-1], "borne_haute": None,
        "probabilite": proba,
    })

    for intervalle in intervalles:
        p = intervalle["probabilite"]
        intervalle["cote_juste"] = round(1 / p, 2) if p > 0.0001 else 999.0
        intervalle["probabilite"] = round(p, 4)

    return intervalles


def colonnes_features(df_features: pd.DataFrame):
    exclure = {"date", "equipe_domicile", "equipe_exterieur", "score_domicile",
               "score_exterieur", "ecart", "points_totaux", "victoire_domicile"}
    return [c for c in df_features.columns if c not in exclure]


# ---------------------------------------------------------------------
# Entraînement + évaluation
# ---------------------------------------------------------------------

def entrainer(args):
    df = charger_et_fusionner_total(args.historique, args.saison_actuelle,
                                     args.colonne_date, args.format_date)
    df_features = construire_features_total(df)
    df_features.to_csv(FICHIER_FEATURES, index=False)

    cols_x = colonnes_features(df_features)
    df_utile = df_features.dropna(subset=cols_x).reset_index(drop=True)
    print(f"{len(df_utile)} matchs utilisables après exclusion des tout premiers "
          f"matchs de chaque équipe (historique insuffisant).")

    n_test = int(len(df_utile) * args.proportion_test)
    train = df_utile.iloc[:-n_test]
    test = df_utile.iloc[-n_test:]
    print(f"Entraînement sur {len(train)} matchs, évaluation sur {len(test)} "
          f"matchs les plus récents.\n")

    X_train, X_test = train[cols_x], test[cols_x]

    modeles_regression = {}
    predictions_test = {}
    print("Régression (score total) :")
    for cible in CIBLES:
        modele = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        modele.fit(X_train, train[cible])
        pred = modele.predict(X_test)
        mae = mean_absolute_error(test[cible], pred)
        print(f"  {cible:16s} MAE = {mae:.2f} points")
        modeles_regression[cible] = modele
        predictions_test[cible] = pred

    mae_total_ecart = mean_absolute_error(
        test["score_domicile"] - test["score_exterieur"],
        predictions_test["score_domicile"] - predictions_test["score_exterieur"],
    )
    print(f"  {'écart (dom-ext)':16s} MAE = {mae_total_ecart:.2f} points  "
          f"(déduit des 2 régressions ci-dessus)")

    # --- modèle DÉDIÉ à l'écart, entraîné directement sur cette cible
    # plutôt que déduit de deux régressions séparées ---
    modele_ecart = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    modele_ecart.fit(X_train, train["ecart"])
    pred_ecart_test = modele_ecart.predict(X_test)
    mae_ecart_dedie = mean_absolute_error(test["ecart"], pred_ecart_test)
    print(f"  {'écart (dédié)':16s} MAE = {mae_ecart_dedie:.2f} points  "
          f"(modèle entraîné directement sur l'écart)")

    # écart-type des résidus du modèle dédié : nécessaire pour estimer
    # la probabilité qu'une équipe couvre une ligne de handicap donnée
    # (on modélise l'écart réel comme une gaussienne centrée sur la
    # prédiction, de cet écart-type)
    residus_ecart = test["ecart"].values - pred_ecart_test
    ecart_type_residus = float(np.std(residus_ecart))
    print(f"  Écart-type des erreurs du modèle dédié : {ecart_type_residus:.2f} points "
          f"(utilisé pour les probabilités de couverture de ligne)")

    # --- modèle dédié aux POINTS TOTAUX (domicile + extérieur),
    # nécessaire pour prédire une distribution par tranche de score
    # total (ex : "151 à 160 points") ---
    modele_points_totaux = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    modele_points_totaux.fit(X_train, train["points_totaux"])
    pred_points_totaux_test = modele_points_totaux.predict(X_test)
    mae_points_totaux = mean_absolute_error(test["points_totaux"], pred_points_totaux_test)
    ecart_type_points_totaux = float(np.std(test["points_totaux"].values - pred_points_totaux_test))
    print(f"\nRégression (points totaux du match, domicile + extérieur) :")
    print(f"  MAE = {mae_points_totaux:.2f} points")
    print(f"  Écart-type des erreurs : {ecart_type_points_totaux:.2f} points "
          f"(utilisé pour les probabilités par tranche)")

    print("\nClassification (vainqueur) :")
    modele_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    modele_clf.fit(X_train, train["victoire_domicile"])
    proba_test = modele_clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(test["victoire_domicile"], pred_test)
    ll = log_loss(test["victoire_domicile"], proba_test)
    print(f"  Précision = {acc:.1%}")
    print(f"  Log-loss  = {ll:.3f}")

    resultats_test = pd.DataFrame({
        "date": test["date"].values,
        "equipe_domicile": test["equipe_domicile"].values,
        "equipe_exterieur": test["equipe_exterieur"].values,
    })
    for cible in CIBLES:
        resultats_test[f"{cible}_reel"] = test[cible].values
        resultats_test[f"{cible}_predit"] = np.round(predictions_test[cible], 1)
        resultats_test[f"{cible}_erreur"] = np.round(
            np.abs(test[cible].values - predictions_test[cible]), 1
        )
    resultats_test["victoire_domicile_reel"] = test["victoire_domicile"].values
    resultats_test["proba_victoire_domicile_predite"] = np.round(proba_test, 3)
    resultats_test["victoire_domicile_predite"] = pred_test
    resultats_test["prediction_correcte"] = (
        resultats_test["victoire_domicile_reel"] == resultats_test["victoire_domicile_predite"]
    )
    resultats_test["ecart_reel"] = test["ecart"].values
    resultats_test["ecart_predit"] = np.round(pred_ecart_test, 1)
    resultats_test.to_csv(FICHIER_RESULTATS_TEST, index=False)
    print(f"\nDétail des prédictions de test : {FICHIER_RESULTATS_TEST}")

    joblib.dump({"modeles": modeles_regression, "colonnes": cols_x}, FICHIER_MODELE_REGRESSION)
    joblib.dump({"modele": modele_clf, "colonnes": cols_x}, FICHIER_MODELE_CLASSIFICATION)
    joblib.dump({"modele": modele_ecart, "colonnes": cols_x, "ecart_type_residus": ecart_type_residus},
                FICHIER_MODELE_ECART)
    joblib.dump(
        {"modele": modele_points_totaux, "colonnes": cols_x, "ecart_type_residus": ecart_type_points_totaux},
        FICHIER_MODELE_POINTS_TOTAUX,
    )
    print(f"Modèles sauvegardés : {FICHIER_MODELE_REGRESSION}, {FICHIER_MODELE_CLASSIFICATION}, "
          f"{FICHIER_MODELE_ECART}, {FICHIER_MODELE_POINTS_TOTAUX}")


# ---------------------------------------------------------------------
# Prédiction d'un match à venir
# ---------------------------------------------------------------------

def predire(args):
    df = charger_et_fusionner_total(args.historique, args.saison_actuelle,
                                     args.colonne_date, args.format_date)
    df_features = construire_features_total(df)
    cols_x = colonnes_features(df_features)

    equipe_domicile = normaliser_nom_equipe(args.equipe_domicile)
    equipe_exterieur = normaliser_nom_equipe(args.equipe_exterieur)

    noms_connus = pd.concat([
        df_features["equipe_domicile"], df_features["equipe_exterieur"]
    ]).unique().tolist()

    for nom_saisi, nom_normalise in [(args.equipe_domicile, equipe_domicile),
                                      (args.equipe_exterieur, equipe_exterieur)]:
        if nom_normalise not in noms_connus:
            suggestions = suggerer_noms_proches(nom_normalise, noms_connus)
            message = f"Équipe introuvable : '{nom_saisi}'"
            if suggestions:
                message += f". Vouliez-vous dire : {', '.join(suggestions)} ?"
            else:
                message += ". Noms disponibles : " + ", ".join(sorted(noms_connus))
            raise SystemExit(message)

    if args.date:
        # format flexible : accepte aussi bien "25/09/2026" que
        # "2026-09-25" sans planter sur un format inattendu
        date_fictive = pd.to_datetime(args.date, dayfirst=True)
    else:
        date_fictive = df["_date"].max() + pd.Timedelta(days=1)

    ligne_fictive = pd.DataFrame([{
        "equipe_domicile": equipe_domicile,
        "equipe_exterieur": equipe_exterieur,
        "_date": date_fictive,
        "score_domicile": 0, "score_exterieur": 0,
    }])
    df_avec_fictif = pd.concat([df, ligne_fictive], ignore_index=True)
    features_avec_fictif = construire_features_total(df_avec_fictif)
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

    pred_dom = modele_reg["modeles"]["score_domicile"].predict(ligne_a_predire)[0]
    pred_ext = modele_reg["modeles"]["score_exterieur"].predict(ligne_a_predire)[0]
    proba_dom = modele_clf["modele"].predict_proba(ligne_a_predire)[0][1]
    marge_predite = modele_ecart["modele"].predict(ligne_a_predire)[0]
    ecart_type_residus = modele_ecart["ecart_type_residus"]

    print(f"\nPrédiction (score total, indépendante des quart-temps) : "
          f"{equipe_domicile} vs {equipe_exterieur}\n")
    print(f"  Score estimé : {equipe_domicile} {pred_dom:.1f} - {pred_ext:.1f} {equipe_exterieur}")
    print(f"  Probabilité de victoire {equipe_domicile} (domicile) : {proba_dom:.1%}")
    print(f"  Probabilité de victoire {equipe_exterieur} (extérieur) : {1 - proba_dom:.1%}")

    # --- Handicap ---
    # marge_predite > 0 -> domicile favori. La ligne de handicap
    # "naturelle" pour le favori est l'opposé de la marge prédite,
    # arrondie au demi-point (convention habituelle en paris sportifs,
    # pour éviter les push/égalités exactes).
    ligne_suggeree_dom = round(-marge_predite * 2) / 2
    print(f"\n  Écart prédit (modèle dédié) : {equipe_domicile} "
          f"{'+' if marge_predite >= 0 else ''}{marge_predite:.1f} {equipe_exterieur}")
    print(f"  Ligne de handicap suggérée : {equipe_domicile} "
          f"{'+' if ligne_suggeree_dom > 0 else ''}{ligne_suggeree_dom:g} / "
          f"{equipe_exterieur} {'+' if -ligne_suggeree_dom > 0 else ''}{-ligne_suggeree_dom:g}")

    proba_couverture_ligne = None
    if args.ligne is not None:
        # convention : --ligne est le handicap DE L'ÉQUIPE À DOMICILE,
        # tel qu'affiché chez un bookmaker (négatif si favori). Le
        # domicile "couvre" si (marge réelle + ligne) > 0.
        proba_couverture_ligne = float(
            norm.sf(-args.ligne, loc=marge_predite, scale=ecart_type_residus)
        )
        signe = "+" if args.ligne > 0 else ""
        print(f"\n  Ligne testée : {equipe_domicile} {signe}{args.ligne:g}")
        print(f"  Probabilité que {equipe_domicile} couvre cette ligne : "
              f"{proba_couverture_ligne:.1%}")
        print(f"  Probabilité que {equipe_exterieur} couvre cette ligne "
              f"({-args.ligne:+g}) : {1 - proba_couverture_ligne:.1%}")
        print(f"  (hypothèse : l'écart réel suit une loi normale centrée sur la "
              f"prédiction, d'écart-type {ecart_type_residus:.1f} points, estimé "
              f"sur l'échantillon de test lors de l'entraînement)")

    # --- Intervalle de score total (par tranches, comme une grille de paris) ---
    modele_points_totaux = joblib.load(FICHIER_MODELE_POINTS_TOTAUX)
    moyenne_points_totaux = float(modele_points_totaux["modele"].predict(ligne_a_predire)[0])
    ecart_type_points_totaux = modele_points_totaux["ecart_type_residus"]

    bornes = [int(b) for b in args.bornes.split(",")]
    intervalles = calculer_intervalles(moyenne_points_totaux, ecart_type_points_totaux, bornes)

    print(f"\nPoints totaux du match — moyenne prédite : {moyenne_points_totaux:.1f} "
          f"(écart-type : {ecart_type_points_totaux:.1f})")
    print(f"  {'Tranche':>14s}   {'Probabilité':>11s}   {'Cote juste':>10s}")
    for intervalle in intervalles:
        print(f"  {intervalle['tranche']:>14s}   {intervalle['probabilite']:>10.1%}   "
              f"{intervalle['cote_juste']:>10.2f}")

    proba_over_ligne_totale = None
    if args.ligne_total is not None:
        # Over/Under direct sur une ligne précise (ex : 165.5), sans
        # avoir à lire toute la grille de tranches — pratique si vous
        # avez juste une cote de bookmaker à comparer.
        proba_over_ligne_totale = float(
            norm.sf(args.ligne_total, loc=moyenne_points_totaux, scale=ecart_type_points_totaux)
        )
        print(f"\n  Ligne totale testée : {args.ligne_total:g} points")
        print(f"  Probabilité Over {args.ligne_total:g} : {proba_over_ligne_totale:.1%}")
        print(f"  Probabilité Under {args.ligne_total:g} : {1 - proba_over_ligne_totale:.1%}")

    # --- Confiance de la prédiction ---
    confiance = evaluer_confiance(df, equipe_domicile, equipe_exterieur, date_fictive)
    print(f"\n  Confiance : {confiance['niveau_confiance']} "
          f"({confiance['nb_matchs_domicile']} matchs connus pour {equipe_domicile}, "
          f"{confiance['nb_matchs_exterieur']} pour {equipe_exterieur}, "
          f"{confiance['nb_confrontations']} confrontation(s) directe(s))")

    nom_fichier_intervalles = (
        f"../prediction/intervalles/intervalles_total_{equipe_domicile}_vs_{equipe_exterieur}.csv"
    ).replace(" ", "_")
    pd.DataFrame(intervalles).to_csv(nom_fichier_intervalles, index=False)
    print(f"  Détail par tranche écrit dans {nom_fichier_intervalles}")

    nouvelle_ligne = {
        "date_prediction": pd.Timestamp.today().strftime("%d/%m/%Y %H:%M"),
        "equipe_domicile": equipe_domicile,
        "equipe_exterieur": equipe_exterieur,
        "score_domicile_predit": round(float(pred_dom), 1),
        "score_exterieur_predit": round(float(pred_ext), 1),
        "proba_victoire_domicile": round(float(proba_dom), 3),
        "proba_victoire_exterieur": round(float(1 - proba_dom), 3),
        "marge_predite_domicile": round(float(marge_predite), 1),
        "ligne_suggeree_domicile": ligne_suggeree_dom,
        "ligne_testee": args.ligne if args.ligne is not None else "",
        "proba_couverture_domicile": (
            round(proba_couverture_ligne, 3) if proba_couverture_ligne is not None else ""
        ),
        "points_totaux_predits": round(moyenne_points_totaux, 1),
        "ligne_total_testee": args.ligne_total if args.ligne_total is not None else "",
        "proba_over": round(proba_over_ligne_totale, 3) if proba_over_ligne_totale is not None else "",
        "confiance": confiance["niveau_confiance"],
        "nb_matchs_domicile": confiance["nb_matchs_domicile"],
        "nb_matchs_exterieur": confiance["nb_matchs_exterieur"],
        "nb_confrontations": confiance["nb_confrontations"],
    }
    df_nouvelle_ligne = pd.DataFrame([nouvelle_ligne])
    try:
        df_predictions = pd.concat(
            [pd.read_csv(FICHIER_PREDICTIONS), df_nouvelle_ligne], ignore_index=True
        )
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
                            help="Ligne de handicap à tester, exprimée pour l'équipe à "
                                 "domicile comme chez un bookmaker (ex : -5.5 si domicile "
                                 "favori de 5.5 points, +3 si domicile outsider de 3 points).")
    p_predire.add_argument("--bornes", default="100,110,120,130,140,150,160,170,180",
                            help="Bornes séparant les tranches de score total, séparées par "
                                 "des virgules (ex : '100,110,120,...,180'). Crée "
                                 "automatiquement une tranche 'sous la première borne' et "
                                 "une tranche 'au-dessus de la dernière'.")
    p_predire.add_argument("--ligne-total", type=float, default=None,
                            help="Ligne de total (Over/Under) à tester directement, "
                                 "ex : 165.5.")
    p_predire.set_defaults(func=predire)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
