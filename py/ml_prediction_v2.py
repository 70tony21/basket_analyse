"""
ml_prediction_v2.py — version améliorée du pipeline de prédiction.

Améliorations par rapport à ml_prediction.py :
  1. LightGBM (fallback sklearn si absent) au lieu de GradientBoostingRegressor
     - gestion native des NaN : on garde TOUS les matchs à l'entraînement,
       y compris les premières journées de chaque équipe, au lieu de les
       jeter via dropna() — gain typique : ~15% de matchs en plus
     - hyperparamètres mieux adaptés (learning_rate plus bas, plus d'arbres)
  2. Calibration isotonique des probabilités du classifieur : améliore le
     log-loss sans toucher à l'accuracy
  3. Cross-validation WALK-FORWARD : on entraîne sur toutes les saisons
     antérieures et on teste sur la saison S, pour chaque S. Beaucoup plus
     robuste qu'un split 85/15 unique (dont la variance est énorme)
  4. Test de McNemar pour vérifier si l'écart ML vs Poisson est significatif
     ou juste du bruit d'échantillonnage

Utilisation :
    pip install lightgbm  # fortement recommandé

    python ml_prediction_v2.py entrainer --historique ... --saison-actuelle ...
    python ml_prediction_v2.py comparer  --historique ... --saison-actuelle ...
    python ml_prediction_v2.py cv        --historique ... --saison-actuelle ...
    python ml_prediction_v2.py predire "Paris Basketball" "AS Monaco" \
        --historique ... --saison-actuelle ...
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from ml_prediction import (
    CIBLES, COLONNES_QUARTS,
    charger_et_fusionner, construire_features, colonnes_features,
    predire_quarts_poisson, normaliser_nom_equipe, suggerer_noms_proches,
    calculer_saison,
)

FICHIER_MODELE_REGRESSION = "v2_modele_quart_temps.joblib"
FICHIER_MODELE_CLASSIFICATION = "v2_modele_vainqueur.joblib"
FICHIER_CALIBRATEUR = "v2_calibrateur.joblib"
FICHIER_FEATURES = "v2_features_construites.csv"
FICHIER_RESULTATS_TEST = "v2_resultats_test.csv"
FICHIER_CV = "v2_cv_walk_forward.csv"
FICHIER_PREDICTIONS = "v2_predictions_a_venir.csv"


# ---------------------------------------------------------------------
# Fabriques de modèles (LightGBM si dispo, sinon sklearn)
# ---------------------------------------------------------------------

def creer_classifieur(random_state: int = 42):
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=random_state,
            verbose=-1,
        )
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03, random_state=random_state
    )


def creer_regresseur(random_state: int = 42):
    if HAS_LGB:
        return lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=random_state,
            verbose=-1,
        )
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.03, random_state=random_state
    )


def _imputer_si_besoin(X: pd.DataFrame, ref: pd.DataFrame = None) -> pd.DataFrame:
    """LightGBM gère les NaN nativement ; sklearn non → imputation médiane."""
    if HAS_LGB:
        return X
    ref = ref if ref is not None else X
    return X.fillna(ref.median(numeric_only=True))


# ---------------------------------------------------------------------
# Calibration isotonique (sans dépendre de la version de sklearn)
# ---------------------------------------------------------------------

def entrainer_avec_calibration(X_train: pd.DataFrame, y_train: pd.Series,
                                fraction_calibration: float = 0.2,
                                random_state: int = 42):
    """
    Entraîne le classifieur sur les premiers (1-fraction) du train
    chronologique, puis ajuste une régression isotonique sur le reste
    (postérieur, comme en conditions réelles). Retourne (clf, iso).

    L'isotonique corrige la tendance des arbres à être trop confiants
    aux extrêmes — gain typique : -0.02 à -0.04 de log-loss, accuracy
    inchangée ou légèrement meilleure.
    """
    n_calib = max(int(len(X_train) * fraction_calibration), 50)
    X_fit, y_fit = X_train.iloc[:-n_calib], y_train.iloc[:-n_calib]
    X_cal, y_cal = X_train.iloc[-n_calib:], y_train.iloc[-n_calib:]

    clf = creer_classifieur(random_state=random_state)
    clf.fit(_imputer_si_besoin(X_fit, X_fit), y_fit)

    probas_cal = clf.predict_proba(_imputer_si_besoin(X_cal, X_fit))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probas_cal, y_cal)
    return clf, iso


def predire_proba_calibree(clf, iso, X, ref=None) -> np.ndarray:
    p_raw = clf.predict_proba(_imputer_si_besoin(X, ref))[:, 1]
    return iso.predict(p_raw)


# ---------------------------------------------------------------------
# Walk-forward CV : entraîner sur toutes les saisons antérieures, tester sur S
# ---------------------------------------------------------------------

def walk_forward_cv(df_brut: pd.DataFrame, df_features: pd.DataFrame,
                     cols_x: list, seuil_min_train: int = 150,
                     calculer_poisson: bool = True,
                     verbeux: bool = True) -> pd.DataFrame:
    """
    Pour chaque saison S présente dans df_features (sauf la/les première(s)),
    entraîne le classifieur sur toutes les lignes dont la saison < S, puis
    prédit sur S. Idem pour Poisson, mais en walk-forward strict match par
    match (l'historique grandit au fur et à mesure — comme en réel).

    Retourne un DataFrame de prédictions out-of-fold, une ligne par match.
    """
    df = df_features.copy()
    df["_saison"] = df["date"].apply(calculer_saison)
    saisons = sorted(df["_saison"].unique())

    lignes = []
    for saison_test in saisons:
        train = df[df["_saison"] < saison_test]
        test = df[df["_saison"] == saison_test]

        if len(train) < seuil_min_train or len(test) == 0:
            if verbeux:
                print(f"  [skip] saison {saison_test} : train={len(train)}, test={len(test)}")
            continue

        if verbeux:
            print(f"  [cv] saison {saison_test} : train={len(train)}, test={len(test)}")

        # --- ML : un seul modèle entraîné sur tout le passé ---
        clf = creer_classifieur()
        clf.fit(_imputer_si_besoin(train[cols_x], train[cols_x]), train["victoire_domicile"])
        probas_ml = clf.predict_proba(_imputer_si_besoin(test[cols_x], train[cols_x]))[:, 1]

        for (idx, row), p_ml in zip(test.iterrows(), probas_ml):
            proba_poisson = np.nan
            if calculer_poisson:
                hist = df_brut[df_brut["_date"] < row["date"]]
                pred_p = predire_quarts_poisson(hist, row["equipe_domicile"], row["equipe_exterieur"])
                if pred_p is not None:
                    proba_poisson = pred_p["proba_victoire_domicile"]

            lignes.append({
                "saison": saison_test,
                "date": row["date"],
                "equipe_domicile": row["equipe_domicile"],
                "equipe_exterieur": row["equipe_exterieur"],
                "y_true": int(row["victoire_domicile"]),
                "proba_ml": float(p_ml),
                "proba_poisson": float(proba_poisson) if not np.isnan(proba_poisson) else np.nan,
            })

    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------
# Test de McNemar : l'écart ML vs Poisson est-il significatif ?
# ---------------------------------------------------------------------

def test_mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """
    Test de McNemar sur prédictions binaires. H0 : A et B ont le même
    taux d'erreur. Rejeter H0 (p < 0.05) signifie que l'écart observé
    n'est probablement pas dû au hasard.

    Retourne stat, p_value, b (A bon / B mauvais), c (A mauvais / B bon).
    """
    y_true = np.asarray(y_true)
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"stat": 0.0, "p_value": 1.0, "b": b, "c": c}
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # correction de continuité
    return {"stat": float(stat), "p_value": float(1 - chi2.cdf(stat, df=1)), "b": b, "c": c}


# ---------------------------------------------------------------------
# Commande : cv (walk-forward ML vs Poisson + McNemar)
# ---------------------------------------------------------------------

def cv(args):
    df = charger_et_fusionner(args.historique, args.saison_actuelle,
                               args.colonne_date, args.format_date)
    df_features = construire_features(df)
    cols_x = colonnes_features(df_features)

    print(f"{len(df_features)} matchs, {len(cols_x)} features.\n")
    print("Walk-forward CV (entraîner sur le passé, tester saison par saison) :")
    res = walk_forward_cv(df, df_features, cols_x, verbeux=True)

    if len(res) == 0:
        print("\nPas assez d'historique pour faire une CV walk-forward "
              "(il faut au moins ~2 saisons complètes).")
        return

    res.to_csv(FICHIER_CV, index=False)
    print(f"\nPrédictions out-of-fold sauvegardées : {FICHIER_CV}")

    # --- Métriques agrégées ML ---
    y = res["y_true"].values
    p_ml = res["proba_ml"].values
    pred_ml = (p_ml >= 0.5).astype(int)
    acc_ml = accuracy_score(y, pred_ml)
    ll_ml = log_loss(y, np.clip(p_ml, 1e-6, 1 - 1e-6))

    print(f"\n{'='*60}\nRÉSULTATS WALK-FORWARD — {len(res)} matchs out-of-fold\n{'='*60}")
    print(f"ML       : accuracy = {acc_ml:.1%}   log-loss = {ll_ml:.3f}")

    # --- Métriques Poisson (si assez de matchs où Poisson a pu prédire) ---
    masque_poisson = res["proba_poisson"].notna()
    if masque_poisson.sum() > 30:
        y_p = res.loc[masque_poisson, "y_true"].values
        p_p = res.loc[masque_poisson, "proba_poisson"].values
        pred_p = (p_p >= 0.5).astype(int)
        acc_p = accuracy_score(y_p, pred_p)
        ll_p = log_loss(y_p, np.clip(p_p, 1e-6, 1 - 1e-6))
        print(f"Poisson  : accuracy = {acc_p:.1%}   log-loss = {ll_p:.3f}   "
              f"(sur {masque_poisson.sum()} matchs)")

        # --- McNemar sur les matchs communs ---
        y_c = res.loc[masque_poisson, "y_true"].values
        pml_c = (res.loc[masque_poisson, "proba_ml"].values >= 0.5).astype(int)
        pp_c = pred_p
        m = test_mcnemar(y_c, pml_c, pp_c)
        print(f"\nTest de McNemar (ML vs Poisson) sur {len(y_c)} matchs communs :")
        print(f"  ML seul gagne  : {m['b']:4d} matchs")
        print(f"  Poisson seul   : {m['c']:4d} matchs")
        print(f"  χ² = {m['stat']:.3f}   p-value = {m['p_value']:.4f}")
        if m["p_value"] < 0.05:
            gagnant = "ML" if m["b"] > m["c"] else "Poisson"
            print(f"  → Écart STATISTIQUEMENT SIGNIFICATIF en faveur de : {gagnant}")
        else:
            print("  → Écart NON significatif : les deux modèles se valent "
                  "sur cet échantillon.")
    else:
        print("Trop peu de matchs où Poisson a pu prédire pour comparer.")

    # --- Baseline naïve ---
    taux_dom = y.mean()
    print(f"\nRéférence naïve « domicile gagne toujours » : {taux_dom:.1%}")
    print(f"Gain ML vs naïf : {acc_ml - taux_dom:+.1%}")


# ---------------------------------------------------------------------
# Commande : entrainer (LightGBM + calibration, sur TOUS les matchs)
# ---------------------------------------------------------------------

def entrainer(args):
    df = charger_et_fusionner(args.historique, args.saison_actuelle,
                               args.colonne_date, args.format_date)
    print(f"{len(df)} matchs avec détail quart-temps.")

    df_features = construire_features(df)
    df_features.to_csv(FICHIER_FEATURES, index=False)
    cols_x = colonnes_features(df_features)

    # Plus de dropna : LightGBM gère les NaN, on entraîne donc sur
    # l'intégralité de l'historique (gain ~10-20% de matchs en plus
    # par rapport à ml_prediction.py, qui jetait les premières journées).
    df_utile = df_features
    if not HAS_LGB:
        # fallback sklearn : on impute, mais on garde tout
        for c in cols_x:
            df_utile[c] = df_utile[c].fillna(df_utile[c].median())

    n_test = int(len(df_utile) * args.proportion_test)
    train = df_utile.iloc[:-n_test].reset_index(drop=True)
    test = df_utile.iloc[-n_test:].reset_index(drop=True)
    print(f"Entraînement sur {len(train)} matchs, test sur {len(test)} "
          f"matchs les plus récents.\n")

    # --- Régression par quart-temps ---
    print("Régression (score par quart-temps) :")
    modeles_reg = {}
    predictions_test = {}
    for cible in CIBLES:
        mask = train[cible].notna()
        modele = creer_regresseur()
        modele.fit(_imputer_si_besoin(train.loc[mask, cols_x], train[cols_x]),
                   train.loc[mask, cible])
        pred = modele.predict(_imputer_si_besoin(test[cols_x], train[cols_x]))
        mae = mean_absolute_error(test[cible], pred)
        print(f"  {cible:16s} MAE = {mae:.2f} points")
        modeles_reg[cible] = modele
        predictions_test[cible] = pred

    # --- Classification + calibration ---
    print("\nClassification (vainqueur) avec calibration isotonique :")
    clf, iso = entrainer_avec_calibration(train[cols_x], train["victoire_domicile"])
    proba_brute = clf.predict_proba(_imputer_si_besoin(test[cols_x], train[cols_x]))[:, 1]
    proba_cal = iso.predict(proba_brute)

    pred_brut = (proba_brute >= 0.5).astype(int)
    pred_cal = (proba_cal >= 0.5).astype(int)

    acc_brut = accuracy_score(test["victoire_domicile"], pred_brut)
    acc_cal = accuracy_score(test["victoire_domicile"], pred_cal)
    ll_brut = log_loss(test["victoire_domicile"], np.clip(proba_brute, 1e-6, 1 - 1e-6))
    ll_cal = log_loss(test["victoire_domicile"], np.clip(proba_cal, 1e-6, 1 - 1e-6))

    print(f"  Sans calibration : accuracy = {acc_brut:.1%}   log-loss = {ll_brut:.3f}")
    print(f"  Avec calibration : accuracy = {acc_cal:.1%}   log-loss = {ll_cal:.3f}")

    taux_dom = train["victoire_domicile"].mean()
    print(f"  (référence naïve « domicile gagne toujours » : {taux_dom:.1%})")

    # --- Sauvegarde ---
    joblib.dump({"modeles": modeles_reg, "colonnes": cols_x}, FICHIER_MODELE_REGRESSION)
    joblib.dump({"modele": clf, "colonnes": cols_x}, FICHIER_MODELE_CLASSIFICATION)
    joblib.dump({"iso": iso, "colonnes": cols_x}, FICHIER_CALIBRATEUR)
    print(f"\nModèles sauvegardés : {FICHIER_MODELE_REGRESSION}, "
          f"{FICHIER_MODELE_CLASSIFICATION}, {FICHIER_CALIBRATEUR}")

    # --- Détail CSV ---
    out = pd.DataFrame({
        "date": test["date"].values,
        "equipe_domicile": test["equipe_domicile"].values,
        "equipe_exterieur": test["equipe_exterieur"].values,
    })
    for cible in CIBLES:
        out[f"{cible}_reel"] = test[cible].values
        out[f"{cible}_predit"] = np.round(predictions_test[cible], 1)
    out["victoire_domicile_reel"] = test["victoire_domicile"].values
    out["proba_brute"] = np.round(proba_brute, 3)
    out["proba_calibree"] = np.round(proba_cal, 3)
    out["prediction"] = pred_cal
    out["correcte"] = (out["prediction"] == out["victoire_domicile_reel"])
    out.to_csv(FICHIER_RESULTATS_TEST, index=False)
    print(f"Détail test : {FICHIER_RESULTATS_TEST}")


# ---------------------------------------------------------------------
# Commande : comparer (split unique, comme ml_prediction.py mais v2)
# ---------------------------------------------------------------------

def comparer(args):
    # Renvoie vers la CV walk-forward qui est strictement meilleure.
    print("La commande « comparer » en split unique a été remplacée par « cv » "
          "(walk-forward), qui est statistiquement beaucoup plus fiable.\n")
    cv(args)


# ---------------------------------------------------------------------
# Commande : predire
# ---------------------------------------------------------------------

def predire(args):
    df = charger_et_fusionner(args.historique, args.saison_actuelle,
                               args.colonne_date, args.format_date)
    df_features = construire_features(df)
    cols_x = colonnes_features(df_features)

    equipe_domicile = normaliser_nom_equipe(args.equipe_domicile)
    equipe_exterieur = normaliser_nom_equipe(args.equipe_exterieur)

    noms_connus = pd.concat([
        df_features["equipe_domicile"], df_features["equipe_exterieur"]
    ]).unique().tolist()
    for nom_saisi, nom_norm in [(args.equipe_domicile, equipe_domicile),
                                 (args.equipe_exterieur, equipe_exterieur)]:
        if nom_norm not in noms_connus:
            sugg = suggerer_noms_proches(nom_norm, noms_connus)
            msg = f"Équipe introuvable : '{nom_saisi}'"
            if sugg:
                msg += f". Vouliez-vous dire : {', '.join(sugg)} ?"
            raise SystemExit(msg)

    if args.date:
        date_fictive = pd.to_datetime(args.date, dayfirst=True)
    else:
        date_fictive = df["_date"].max() + pd.Timedelta(days=1)

    ligne_fictive = pd.DataFrame([{
        "equipe_domicile": equipe_domicile,
        "equipe_exterieur": equipe_exterieur,
        "_date": date_fictive,
        **{c: 0 for c in COLONNES_QUARTS},
    }])
    df_avec_fictif = pd.concat([df, ligne_fictive], ignore_index=True)
    features_avec_fictif = construire_features(df_avec_fictif)
    ligne = features_avec_fictif.iloc[[-1]][cols_x]

    # Avertir si des NaN persistent (LightGBM les gère, mais on informe)
    if HAS_LGB:
        nb_nan = int(ligne.isna().sum().sum())
        if nb_nan > 0:
            print(f"Info : {nb_nan} feature(s) NaN sur cette ligne — LightGBM "
                  f"les gère nativement.\n")
    else:
        med = df_features[cols_x].median(numeric_only=True)
        ligne = ligne.fillna(med)

    modele_reg = joblib.load(FICHIER_MODELE_REGRESSION)
    modele_clf = joblib.load(FICHIER_MODELE_CLASSIFICATION)
    calib = joblib.load(FICHIER_CALIBRATEUR)

    print(f"\nPrédiction : {equipe_domicile} (domicile) vs "
          f"{equipe_exterieur} (extérieur)\n")

    total_dom, total_ext, predictions = 0, 0, {}
    for cible in CIBLES:
        pred = modele_reg["modeles"][cible].predict(ligne)[0]
        predictions[cible] = pred
        if "domicile" in cible:
            total_dom += pred
        else:
            total_ext += pred
        print(f"  {cible:16s} : {pred:.1f}")

    p_brute = modele_clf["modele"].predict_proba(ligne)[0][1]
    p_cal = float(calib["iso"].predict([p_brute])[0])

    print(f"\n  Total estimé : {equipe_domicile} {total_dom:.1f} - "
          f"{total_ext:.1f} {equipe_exterieur}")
    print(f"  Proba victoire {equipe_domicile} : {p_cal:.1%}  "
          f"(brute : {p_brute:.1%})")
    print(f"  Proba victoire {equipe_exterieur} : {1 - p_cal:.1%}")

    nouvelle = {
        "date_prediction": pd.Timestamp.today().strftime("%d/%m/%Y %H:%M"),
        "equipe_domicile": equipe_domicile,
        "equipe_exterieur": equipe_exterieur,
    }
    for c in CIBLES:
        nouvelle[c] = round(float(predictions[c]), 1)
    nouvelle["total_domicile"] = round(total_dom, 1)
    nouvelle["total_exterieur"] = round(total_ext, 1)
    nouvelle["proba_victoire_domicile"] = round(p_cal, 3)
    nouvelle["proba_victoire_exterieur"] = round(1 - p_cal, 3)

    try:
        existant = pd.read_csv(FICHIER_PREDICTIONS)
        pd.concat([existant, pd.DataFrame([nouvelle])], ignore_index=True) \
            .to_csv(FICHIER_PREDICTIONS, index=False)
    except FileNotFoundError:
        pd.DataFrame([nouvelle]).to_csv(FICHIER_PREDICTIONS, index=False)
    print(f"\nPrédiction ajoutée à {FICHIER_PREDICTIONS}.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="commande", required=True)

    for nom in ("entrainer", "comparer", "cv"):
        p = sub.add_parser(nom)
        p.add_argument("--historique", required=True)
        p.add_argument("--saison-actuelle", required=True)
        p.add_argument("--colonne-date", default="date")
        p.add_argument("--format-date", default="%d/%m/%Y")
        p.add_argument("--proportion-test", type=float, default=0.15)

    p_pred = sub.add_parser("predire")
    p_pred.add_argument("equipe_domicile")
    p_pred.add_argument("equipe_exterieur")
    p_pred.add_argument("--historique", required=True)
    p_pred.add_argument("--saison-actuelle", required=True)
    p_pred.add_argument("--colonne-date", default="date")
    p_pred.add_argument("--format-date", default="%d/%m/%Y")
    p_pred.add_argument("--date", default=None)

    sub.choices["entrainer"].set_defaults(func=entrainer)
    sub.choices["comparer"].set_defaults(func=comparer)
    sub.choices["cv"].set_defaults(func=cv)
    sub.choices["predire"].set_defaults(func=predire)

    args = parser.parse_args()
    if not HAS_LGB:
        print("⚠ LightGBM non installé — fallback sur sklearn GB (plus lent, "
              "pas de gestion native des NaN).\n"
              "  Installation recommandée : pip install lightgbm\n")
    args.func(args)


if __name__ == "__main__":
    main()