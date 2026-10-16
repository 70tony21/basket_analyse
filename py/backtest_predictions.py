"""
Parcourt la saison en cours match par match (dans l'ordre chronologique)
et, pour chacun :

  1. Prédit le score par quart-temps en se basant UNIQUEMENT sur
     l'historique connu jusque-là (fichier 1 : saisons passées + les
     matchs de la saison en cours déjà traités).
  2. Enregistre cette prédiction dans le fichier de la saison en cours
     (fichier 2), dans des colonnes pred_q1_domicile, pred_q1_exterieur,
     etc.
  3. Si le match a déjà été joué (score réel disponible dans le
     fichier 2), ajoute son résultat réel au fichier historique
     (fichier 1), sauvegarde ce fichier, puis le RECHARGE avant de
     passer au match suivant — donc le match qu'on vient de jouer
     participe aux prédictions des matchs suivants.

C'est une simulation "en aveugle, au fil de la saison" : pour chaque
match, le modèle ne voit que ce qui s'est passé AVANT ce match.
Cela permet de comparer prédiction vs réalité match par match.

Le script est relançable : les matchs déjà prédits (colonne
pred_q1_domicile déjà remplie dans le fichier 2) sont sautés lors
d'une nouvelle exécution, donc vous pouvez le relancer au fur et à
mesure que le scraper de la saison en cours ajoute de nouveaux
matchs.

Le modèle de prédiction repose sur une loi de POISSON : le score de
chaque équipe à chaque quart-temps est modélisé comme une variable de
Poisson dont le paramètre lambda est la moyenne pondérée habituelle
(fraîcheur des matchs + confrontations directes). Cela permet de
produire, en plus de la moyenne :
  - le score le plus probable (mode de la loi, colonnes proba_qi_*)
  - un intervalle de confiance à 90 % (colonnes qi_domicile_min/max)
  - la probabilité de victoire de chaque équipe sur le match entier,
    calculée sans simulation grâce à la loi de SKELLAM (la différence
    de deux lois de Poisson indépendantes suit une loi de Skellam).

Prérequis :
    pip install pandas numpy scipy

Utilisation :
    python backtest_predictions.py \
        --historique betclic_elite_thesportsdb_2022_2025.csv \
        --saison-actuelle betclic_elite_2025_2026.csv \
        --demi-vie 10 --poids-h2h 0.35
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

COLONNES_QUARTS_REQUISES = [
    "q1_domicile", "q1_exterieur", "q2_domicile", "q2_exterieur",
    "q3_domicile", "q3_exterieur", "q4_domicile", "q4_exterieur",
]

# Correspondance de noms de colonnes entre les deux fichiers, si jamais
# ils ne portent pas exactement le même nom (ex: "journee" côté saison
# actuelle vs "round" côté historique). Complétez si besoin.
CORRESPONDANCE_COLONNES = {
    "journee": "round",
}


# ---------------------------------------------------------------------
# Chargement / sauvegarde
# ---------------------------------------------------------------------

def charger_csv(chemin: str) -> pd.DataFrame:
    return pd.read_csv(chemin, dtype=str)


def sauvegarder_csv(df: pd.DataFrame, chemin: str):
    df.to_csv(chemin, index=False)


def matchs_avec_quarts_connus(df: pd.DataFrame) -> pd.DataFrame:
    """Sous-ensemble des matchs dont le détail par quart-temps est renseigné."""
    if df.empty:
        return df
    presentes = [c for c in COLONNES_QUARTS_REQUISES if c in df.columns]
    if len(presentes) < len(COLONNES_QUARTS_REQUISES):
        return df.iloc[0:0]
    return df.dropna(subset=COLONNES_QUARTS_REQUISES)


# ---------------------------------------------------------------------
# Prédiction (moyennes pondérées par fraîcheur + confrontations directes)
# ---------------------------------------------------------------------

def poids_fraicheur(n: int, demi_vie: int) -> np.ndarray:
    if n == 0:
        return np.array([])
    rangs = np.arange(n - 1, -1, -1)  # 0 = plus récent
    return 0.5 ** (rangs / demi_vie)


def moyenne_ponderee(valeurs, poids) -> float:
    valeurs = pd.to_numeric(pd.Series(valeurs), errors="coerce")
    poids = np.asarray(poids, dtype=float)
    masque = ~valeurs.isna()
    if masque.sum() == 0 or poids[masque].sum() == 0:
        return np.nan
    return np.average(valeurs[masque], weights=poids[masque])


def moyennes_equipe(historique: pd.DataFrame, equipe: str, demi_vie: int):
    dom = historique[historique["equipe_domicile"] == equipe]
    ext = historique[historique["equipe_exterieur"] == equipe]

    poids_dom = poids_fraicheur(len(dom), demi_vie)
    poids_ext = poids_fraicheur(len(ext), demi_vie)

    marques_dom = {f"q{i}": moyenne_ponderee(dom[f"q{i}_domicile"], poids_dom) for i in range(1, 5)}
    encaisses_dom = {f"q{i}": moyenne_ponderee(dom[f"q{i}_exterieur"], poids_dom) for i in range(1, 5)}
    marques_ext = {f"q{i}": moyenne_ponderee(ext[f"q{i}_exterieur"], poids_ext) for i in range(1, 5)}
    encaisses_ext = {f"q{i}": moyenne_ponderee(ext[f"q{i}_domicile"], poids_ext) for i in range(1, 5)}

    return {
        "marques_domicile": marques_dom, "encaisses_domicile": encaisses_dom,
        "marques_exterieur": marques_ext, "encaisses_exterieur": encaisses_ext,
    }


def confrontations_directes(historique: pd.DataFrame, equipe_a: str, equipe_b: str, demi_vie: int):
    mask = (
        ((historique["equipe_domicile"] == equipe_a) & (historique["equipe_exterieur"] == equipe_b))
        | ((historique["equipe_domicile"] == equipe_b) & (historique["equipe_exterieur"] == equipe_a))
    )
    h2h = historique[mask].reset_index(drop=True)
    nb = len(h2h)
    if nb == 0:
        return None, None, 0

    poids = poids_fraicheur(nb, demi_vie)

    def stats_pour(equipe):
        marques, encaisses = {}, {}
        for i in range(1, 5):
            q = f"q{i}"
            vm, ve = [], []
            for idx, row in h2h.iterrows():
                if row["equipe_domicile"] == equipe:
                    vm.append(row[f"{q}_domicile"])
                    ve.append(row[f"{q}_exterieur"])
                else:
                    vm.append(row[f"{q}_exterieur"])
                    ve.append(row[f"{q}_domicile"])
            marques[q] = moyenne_ponderee(vm, poids)
            encaisses[q] = moyenne_ponderee(ve, poids)
        return {"marques": marques, "encaisses": encaisses}

    return stats_pour(equipe_a), stats_pour(equipe_b), nb


def predire_quarts(historique: pd.DataFrame, equipe_dom: str, equipe_ext: str,
                    demi_vie: int, poids_h2h: float):
    """
    Modélise le score de chaque équipe, à chaque quart-temps, comme une
    variable aléatoire de POISSON de paramètre lambda (le nombre moyen
    de points attendus, calculé comme avant : moyenne pondérée par la
    fraîcheur des matchs + confrontations directes).

    La moyenne pondérée n'est donc plus la prédiction finale : c'est le
    paramètre lambda de la loi de Poisson. Cette loi nous permet de
    déduire, en plus de la moyenne :
      - le score le PLUS PROBABLE (le mode de la distribution, qui
        n'est pas forcément égal à la moyenne arrondie) ;
      - un intervalle de confiance à 90 % (du 5e au 95e percentile).

    Pour le résultat du MATCH complet, la somme de 4 variables de
    Poisson indépendantes est elle-même une loi de Poisson (lambda
    total = somme des 4 lambdas). La différence entre deux lois de
    Poisson indépendantes suit une loi de SKELLAM, ce qui permet de
    calculer directement, sans simulation, la probabilité de victoire
    de chaque équipe sur l'ensemble du match.

    Retourne un dict avec, pour chaque quart-temps :
        pred_qi_domicile / pred_qi_exterieur   -> lambda (points attendus)
        proba_qi_domicile / proba_qi_exterieur -> score le plus probable
        qi_domicile_min / qi_domicile_max      -> intervalle 90 %
        qi_exterieur_min / qi_exterieur_max    -> intervalle 90 %
    et pour le match entier :
        total_domicile_attendu / total_exterieur_attendu
        proba_victoire_domicile / proba_victoire_exterieur / proba_egalite
    """
    stats_dom = moyennes_equipe(historique, equipe_dom, demi_vie)
    stats_ext = moyennes_equipe(historique, equipe_ext, demi_vie)

    h2h_dom, h2h_ext, nb_h2h = confrontations_directes(historique, equipe_dom, equipe_ext, demi_vie)
    poids_h2h_eff = poids_h2h if nb_h2h > 0 else 0.0

    resultat = {}
    lambda_dom_total, lambda_ext_total = 0.0, 0.0

    for i in range(1, 5):
        q = f"q{i}"
        general_dom = np.nanmean([stats_dom["marques_domicile"][q], stats_ext["encaisses_exterieur"][q]])
        general_ext = np.nanmean([stats_ext["marques_exterieur"][q], stats_dom["encaisses_domicile"][q]])

        if nb_h2h > 0:
            h2h_dom_val = np.nanmean([h2h_dom["marques"][q], h2h_ext["encaisses"][q]])
            h2h_ext_val = np.nanmean([h2h_ext["marques"][q], h2h_dom["encaisses"][q]])
            lambda_dom = (1 - poids_h2h_eff) * general_dom + poids_h2h_eff * h2h_dom_val
            lambda_ext = (1 - poids_h2h_eff) * general_ext + poids_h2h_eff * h2h_ext_val
        else:
            lambda_dom, lambda_ext = general_dom, general_ext

        if np.isnan(lambda_dom) or np.isnan(lambda_ext):
            # pas assez d'historique : on ne peut pas construire de loi
            # de Poisson exploitable pour ce quart-temps
            resultat[f"pred_{q}_domicile"] = ""
            resultat[f"pred_{q}_exterieur"] = ""
            resultat[f"proba_{q}_domicile"] = ""
            resultat[f"proba_{q}_exterieur"] = ""
            resultat[f"{q}_domicile_min"] = ""
            resultat[f"{q}_domicile_max"] = ""
            resultat[f"{q}_exterieur_min"] = ""
            resultat[f"{q}_exterieur_max"] = ""
            continue

        lambda_dom_total += lambda_dom
        lambda_ext_total += lambda_ext

        # mode de la loi de Poisson = score le plus probable
        mode_dom = int(poisson.ppf(0.5, lambda_dom))
        mode_ext = int(poisson.ppf(0.5, lambda_ext))

        # intervalle de confiance à 90 %
        min_dom, max_dom = poisson.ppf([0.05, 0.95], lambda_dom)
        min_ext, max_ext = poisson.ppf([0.05, 0.95], lambda_ext)

        resultat[f"pred_{q}_domicile"] = round(lambda_dom, 1)
        resultat[f"pred_{q}_exterieur"] = round(lambda_ext, 1)
        resultat[f"proba_{q}_domicile"] = mode_dom
        resultat[f"proba_{q}_exterieur"] = mode_ext
        resultat[f"{q}_domicile_min"] = int(min_dom)
        resultat[f"{q}_domicile_max"] = int(max_dom)
        resultat[f"{q}_exterieur_min"] = int(min_ext)
        resultat[f"{q}_exterieur_max"] = int(max_ext)

    if lambda_dom_total > 0 and lambda_ext_total > 0:
        # Skellam : distribution de (score_domicile - score_exterieur)
        # quand les deux scores suivent des lois de Poisson indépendantes
        # de paramètres lambda_dom_total et lambda_ext_total.
        proba_victoire_ext = skellam.cdf(-1, lambda_dom_total, lambda_ext_total)
        proba_egalite = skellam.pmf(0, lambda_dom_total, lambda_ext_total)
        proba_victoire_dom = 1 - proba_victoire_ext - proba_egalite

        resultat["total_domicile_attendu"] = round(lambda_dom_total, 1)
        resultat["total_exterieur_attendu"] = round(lambda_ext_total, 1)
        resultat["proba_victoire_domicile"] = round(float(proba_victoire_dom), 3)
        resultat["proba_victoire_exterieur"] = round(float(proba_victoire_ext), 3)
        resultat["proba_egalite"] = round(float(proba_egalite), 3)
    else:
        resultat["total_domicile_attendu"] = ""
        resultat["total_exterieur_attendu"] = ""
        resultat["proba_victoire_domicile"] = ""
        resultat["proba_victoire_exterieur"] = ""
        resultat["proba_egalite"] = ""

    return resultat


# ---------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------

def construire_ligne_pour_historique(colonnes_historique, ligne_saison: pd.Series) -> dict:
    """
    Construit une ligne à ajouter au fichier historique à partir d'une
    ligne du fichier de la saison en cours, en alignant les noms de
    colonnes (avec CORRESPONDANCE_COLONNES pour les cas particuliers).
    """
    nouvelle_ligne = {}
    for col in colonnes_historique:
        if col in ligne_saison.index:
            nouvelle_ligne[col] = ligne_saison[col]
            continue
        # cherche une colonne source correspondante via le mapping
        source = next((k for k, v in CORRESPONDANCE_COLONNES.items() if v == col), None)
        if source and source in ligne_saison.index:
            nouvelle_ligne[col] = ligne_saison[source]
        else:
            nouvelle_ligne[col] = ""
    return nouvelle_ligne


def main():
    parser = argparse.ArgumentParser(
        description="Prédiction match par match avec mise à jour incrémentale de l'historique."
    )
    parser.add_argument("--historique", required=True,
                         help="CSV des saisons passées (sera complété au fil de l'exécution).")
    parser.add_argument("--saison-actuelle", required=True,
                         help="CSV de la saison en cours (les prédictions y seront ajoutées).")
    parser.add_argument("--demi-vie", type=int, default=10)
    parser.add_argument("--poids-h2h", type=float, default=0.35)
    parser.add_argument("--colonne-date", default="date",
                         help="Nom de la colonne date à utiliser pour trier "
                              "les matchs de la saison en cours (défaut: 'date').")
    parser.add_argument("--format-date", default="%d/%m/%Y",
                         help="Format de la colonne date (défaut: JJ/MM/AAAA).")
    args = parser.parse_args()

    saison = charger_csv(args.saison_actuelle)

    # colonnes de prédiction : on les crée si elles n'existent pas encore
    colonnes_pred = []
    for i in range(1, 5):
        q = f"q{i}"
        colonnes_pred += [
            f"pred_{q}_domicile", f"pred_{q}_exterieur",
            f"proba_{q}_domicile", f"proba_{q}_exterieur",
            f"{q}_domicile_min", f"{q}_domicile_max",
            f"{q}_exterieur_min", f"{q}_exterieur_max",
        ]
    colonnes_pred += [
        "total_domicile_attendu", "total_exterieur_attendu",
        "proba_victoire_domicile", "proba_victoire_exterieur", "proba_egalite",
    ]
    for col in colonnes_pred:
        if col not in saison.columns:
            saison[col] = ""

    # tri chronologique des matchs de la saison en cours
    saison["_date_triee"] = pd.to_datetime(
        saison[args.colonne_date], format=args.format_date, errors="coerce"
    )
    saison = saison.sort_values("_date_triee", na_position="last").reset_index(drop=True)

    nb_traites = 0
    nb_predits_seulement = 0

    for idx in saison.index:
        ligne = saison.loc[idx]

        # déjà traité lors d'une exécution précédente -> on saute
        if str(ligne.get("pred_q1_domicile", "")).strip() not in ("", "nan"):
            continue

        equipe_dom = ligne["equipe_domicile"]
        equipe_ext = ligne["equipe_exterieur"]

        # on recharge l'historique à CHAQUE itération, comme demandé,
        # pour repartir d'un état frais garanti à jour sur disque
        historique = charger_csv(args.historique)
        historique_valide = matchs_avec_quarts_connus(historique)

        print(f"Match : {equipe_dom} - {equipe_ext} "
              f"({ligne.get(args.colonne_date, '')}) "
              f"[historique : {len(historique_valide)} matchs connus]")

        prediction = predire_quarts(
            historique_valide, equipe_dom, equipe_ext, args.demi_vie, args.poids_h2h
        )

        for col, valeur in prediction.items():
            saison.at[idx, col] = str(valeur)

        print(f"   Prédiction (λ Poisson) : "
              f"Q1 {prediction['pred_q1_domicile']}-{prediction['pred_q1_exterieur']} | "
              f"Q2 {prediction['pred_q2_domicile']}-{prediction['pred_q2_exterieur']} | "
              f"Q3 {prediction['pred_q3_domicile']}-{prediction['pred_q3_exterieur']} | "
              f"Q4 {prediction['pred_q4_domicile']}-{prediction['pred_q4_exterieur']}")
        print(f"   Score le plus probable par quart-temps : "
              f"Q1 {prediction['proba_q1_domicile']}-{prediction['proba_q1_exterieur']} | "
              f"Q2 {prediction['proba_q2_domicile']}-{prediction['proba_q2_exterieur']} | "
              f"Q3 {prediction['proba_q3_domicile']}-{prediction['proba_q3_exterieur']} | "
              f"Q4 {prediction['proba_q4_domicile']}-{prediction['proba_q4_exterieur']}")
        if prediction["proba_victoire_domicile"] != "":
            print(f"   Total attendu : {equipe_dom} {prediction['total_domicile_attendu']} - "
                  f"{prediction['total_exterieur_attendu']} {equipe_ext} | "
                  f"P(victoire {equipe_dom}) = {prediction['proba_victoire_domicile']:.1%} | "
                  f"P(victoire {equipe_ext}) = {prediction['proba_victoire_exterieur']:.1%}")

        # sauvegarde immédiate de la prédiction dans le fichier saison
        sauvegarder_csv(saison.drop(columns=["_date_triee"]), args.saison_actuelle)

        # le match a-t-il déjà été joué (score réel + quarts connus) ?
        a_un_resultat_reel = all(
            str(ligne.get(c, "")).strip() not in ("", "nan") for c in COLONNES_QUARTS_REQUISES
        )

        if not a_un_resultat_reel:
            print("   Match pas encore joué : prédiction enregistrée, "
                  "rien à ajouter à l'historique pour l'instant.\n")
            nb_predits_seulement += 1
            continue

        # ajout du résultat réel à l'historique
        nouvelle_ligne = construire_ligne_pour_historique(historique.columns, ligne)
        historique = pd.concat([historique, pd.DataFrame([nouvelle_ligne])], ignore_index=True)
        sauvegarder_csv(historique, args.historique)

        print(f"   Résultat réel ajouté à l'historique "
              f"({equipe_dom} {ligne['score_domicile']} - "
              f"{ligne['score_exterieur']} {equipe_ext}).\n")

        nb_traites += 1

    print(f"\nTerminé. {nb_traites} match(s) joué(s) traité(s) et ajouté(s) à l'historique, "
          f"{nb_predits_seulement} match(s) à venir seulement prédit(s).")


if __name__ == "__main__":
    main()
