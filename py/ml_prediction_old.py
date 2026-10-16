"""
Modèle de machine learning pour prédire :
  - le score de chaque équipe à chaque quart-temps (régression)
  - le vainqueur du match (classification, probabilité de victoire
    de l'équipe à domicile)

------------------------------------------------------------------
ÉTAPE 1 : construction des features (variables explicatives)
------------------------------------------------------------------
Pour chaque match, on calcule des statistiques basées UNIQUEMENT sur
les matchs précédant ce match (jamais sur le match lui-même ni sur
des matchs postérieurs) :

  - forme récente (moyenne des N derniers matchs) de points marqués
    et encaissés, par quart-temps, à domicile et à l'extérieur
  - un rating de type Elo, mis à jour après chaque match
  - le nombre de jours de repos depuis le match précédent
  - les statistiques de confrontation directe entre les deux équipes

C'est le point le plus important du script : toute variable qui
utiliserait, même indirectement, une information du match à prédire
ou d'un match futur fausserait complètement l'évaluation (le modèle
semblerait excellent en test mais serait inutilisable en réel).

------------------------------------------------------------------
ÉTAPE 1bis : stacking — Poisson comme feature du ML
------------------------------------------------------------------
En plus des features ci-dessus, chaque match reçoit aussi la
prédiction du modèle Poisson (mêmes formules que backtest_predictions.py),
recalculée en glissant sur l'historique connu à cet instant. Le modèle
ML dispose ainsi, en plus de ses propres variables, du résumé que
donnerait un statisticien avec une calculatrice — libre à lui de s'en
servir tel quel, de le corriger, ou de l'ignorer si d'autres variables
sont plus informatives.

------------------------------------------------------------------
ÉTAPE 2 : entraînement
------------------------------------------------------------------
  - Régression (GradientBoostingRegressor, un par cible) pour
    q1_domicile, q1_exterieur, ..., q4_exterieur
  - Classification (GradientBoostingClassifier) pour la victoire à
    domicile (0/1), qui donne directement une probabilité

Le découpage entraînement/test se fait de façon CHRONOLOGIQUE (les
X % de matchs les plus anciens servent à l'entraînement, les plus
récents à l'évaluation) — jamais un découpage aléatoire, qui
mélangerait passé et futur.

------------------------------------------------------------------
ÉTAPE 3 : prédiction d'un match à venir
------------------------------------------------------------------
Utilise les dernières statistiques connues des deux équipes pour
produire une prédiction.

------------------------------------------------------------------
ÉTAPE 4 : comparaison avec le modèle Poisson
------------------------------------------------------------------
La sous-commande "comparer" évalue le modèle ML ET le modèle Poisson
(moyenne pondérée par fraîcheur + confrontations directes, cf.
backtest_predictions.py) sur EXACTEMENT le même échantillon de test
chronologique, avec les mêmes métriques (MAE par quart-temps,
accuracy et log-loss pour le vainqueur). Le modèle Poisson est
recalculé en walk-forward (son "historique" grandit match après
match au fil du test, comme il le ferait en conditions réelles).

Prérequis :
    pip install pandas numpy scikit-learn joblib scipy

Utilisation :
    # Entraîner et évaluer le modèle
    python ml_prediction.py entrainer \
        --historique betclic_elite_thesportsdb_2022_2025.csv \
        --saison-actuelle betclic_elite_2025_2026.csv

    # Prédire un match à venir avec le modèle déjà entraîné
    python ml_prediction.py predire "Paris Basketball" "AS Monaco" \
        --historique betclic_elite_thesportsdb_2022_2025.csv \
        --saison-actuelle betclic_elite_2025_2026.csv

    # Comparer ML vs Poisson sur le même échantillon de test
    python ml_prediction.py comparer \
        --historique betclic_elite_thesportsdb_2022_2025.csv \
        --saison-actuelle betclic_elite_2025_2026.csv
"""

import argparse
import difflib
import unicodedata

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

COLONNES_QUARTS = [
    "q1_domicile", "q1_exterieur", "q2_domicile", "q2_exterieur",
    "q3_domicile", "q3_exterieur", "q4_domicile", "q4_exterieur",
]
CIBLES = COLONNES_QUARTS  # ce que le modèle de régression doit prédire

# ---------------------------------------------------------------------
# Normalisation des noms d'équipe
# ---------------------------------------------------------------------
# TheSportsDB et coteur.com ne nomment pas les équipes de la même façon
# (ex : "AS Monaco Basket" vs "AS Monaco"). Sans cette table, le même
# club apparaît comme deux entités totalement différentes selon la
# source du match, ce qui fragmente silencieusement tout l'historique
# (Elo, forme, confrontations directes, classement...) de ce club.
#
# Clé   : variante du nom, en minuscules et sans accents
# Valeur : nom canonique utilisé partout dans ce pipeline
NOMS_CANONIQUES = {
    "as monaco basket": "AS Monaco",
    "as monaco": "AS Monaco",
    "sluc nancy basket": "Nancy",
    "sluc nancy basket association": "Nancy",
    "nancy": "Nancy",
    "essm le portel": "Le Portel",
    "le portel": "Le Portel",
    "saint-quentin basket-ball": "Saint-Quentin",
    "saint-quentin": "Saint-Quentin",
    "le mans sarthe basket": "Le Mans",
    "le mans": "Le Mans",
    "lyon-villeurbanne": "ASVEL",
    "asvel": "ASVEL",
    "elan chalon": "Chalon-sur-Saone",
    "chalon-sur-saone": "Chalon-sur-Saone",
    "jda dijon bourgogne": "Dijon",
    "dijon": "Dijon",
    "cholet basket": "Cholet",
    "cholet": "Cholet",
    "jeunesse laique bourg basket": "Bourg-en-Bresse",
    "bourg-en-bresse": "Bourg-en-Bresse",
    "gravelines-dunkerque": "Gravelines-Dunkerque",
    "limoges csp": "Limoges",
    "limoges": "Limoges",
    "strasbourg ig": "Strasbourg",
    "strasbourg": "Strasbourg",
    "nanterre": "Nanterre",
    "paris basketball": "Paris Basketball",
    "boulazac": "Boulazac",
    # clubs qui ont quitté l'élite mais restent dans l'historique passé
    # (pas de fusion nécessaire, mais on les normalise quand même pour
    # la cohérence de casse/accents)
    "chorale roanne basket": "Roanne",
    "ada blois": "Blois",
    "metropolitans 92": "Metropolitans 92",
    "fos provence basket": "Fos Provence",
}


def normaliser_nom_equipe(nom: str) -> str:
    """
    Convertit un nom d'équipe (quelle que soit sa source) vers son nom
    canonique, via NOMS_CANONIQUES. Insensible à la casse et aux
    accents. Si le nom n'est pas répertorié, il est renvoyé tel quel
    (au lieu de planter), pour ne pas bloquer sur un club qu'on
    n'aurait pas encore recensé — mais dans ce cas, pensez à l'ajouter
    à la table ci-dessus.
    """
    if not isinstance(nom, str):
        return nom
    cle = unicodedata.normalize("NFKD", nom.strip().lower()).encode("ascii", "ignore").decode()
    return NOMS_CANONIQUES.get(cle, nom.strip())


def suggerer_noms_proches(nom_recherche: str, noms_disponibles, n=5):
    """Utilisé pour donner un message d'erreur utile quand une équipe est introuvable."""
    return difflib.get_close_matches(nom_recherche, noms_disponibles, n=n, cutoff=0.4)

FENETRE_FORME = 10          # nombre de matchs récents pour la "forme"
ELO_INITIAL = 1500
ELO_K = 20                  # vitesse d'ajustement du rating Elo

FICHIER_MODELE_REGRESSION = "../model_data/modele_quart_temps.joblib"
FICHIER_MODELE_CLASSIFICATION = "../model_data/modele_vainqueur.joblib"
FICHIER_FEATURES = "../model_data/features_construites.csv"
FICHIER_RESULTATS_TEST = "../model_test/resultats_test_ml.csv"
FICHIER_PREDICTIONS = "../prediction/predictions_a_venir.csv"


# ---------------------------------------------------------------------
# Chargement des données brutes
# ---------------------------------------------------------------------

def parser_dates_flexible(serie: pd.Series, format_prioritaire: str) -> pd.Series:
    """
    Convertit une colonne de dates qui peut mélanger plusieurs formats
    (typiquement : "26/09/2025" côté scraper coteur.com pour la saison
    en cours, et "2023-09-16 14:00:00" côté TheSportsDB pour les
    saisons passées). On tente d'abord le format fourni, puis on
    retente les valeurs non converties avec un format ISO, puis avec
    l'inférence automatique de pandas en dernier recours.
    """
    dates = pd.to_datetime(serie, format=format_prioritaire, errors="coerce")

    a_retenter = dates.isna() & serie.notna()
    if a_retenter.any():
        dates.loc[a_retenter] = pd.to_datetime(
            serie[a_retenter], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    a_retenter = dates.isna() & serie.notna()
    if a_retenter.any():
        dates.loc[a_retenter] = pd.to_datetime(
            serie[a_retenter], dayfirst=True, errors="coerce"
        )

    return dates


def charger_et_fusionner(chemin_historique: str, chemin_saison_actuelle: str,
                          colonne_date: str, format_date: str) -> pd.DataFrame:
    """
    Charge les deux fichiers, ne garde que les matchs dont le détail
    quart-temps est connu, et les fusionne en un seul DataFrame trié
    chronologiquement.

    Les deux fichiers peuvent avoir des formats de date différents
    (ex : DD/MM/AAAA pour la saison en cours, ISO pour l'historique
    TheSportsDB) : voir parser_dates_flexible.
    """
    hist = pd.read_csv(chemin_historique, dtype=str)
    actu = pd.read_csv(chemin_saison_actuelle, dtype=str)

    colonnes_communes = [c for c in actu.columns if c in hist.columns]
    df = pd.concat([hist[colonnes_communes], actu[colonnes_communes]], ignore_index=True)

    # normalisation des noms d'équipe AVANT toute agrégation, pour ne
    # pas fragmenter l'historique d'un même club entre ses différentes
    # variantes de nom selon la source
    for c in ("equipe_domicile", "equipe_exterieur"):
        if c in df.columns:
            df[c] = df[c].apply(normaliser_nom_equipe)

    for c in COLONNES_QUARTS + ["score_domicile", "score_exterieur"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=COLONNES_QUARTS).copy()

    df["_date"] = parser_dates_flexible(df[colonne_date], format_date)
    nb_avant = len(df)
    df = df.dropna(subset=["_date"])
    if len(df) < nb_avant:
        print(f"Attention : {nb_avant - len(df)} match(s) écarté(s) car leur date "
              f"n'a pas pu être interprétée (colonne '{colonne_date}').")
    df = df.sort_values("_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Construction des features, match par match, dans l'ordre chronologique
# ---------------------------------------------------------------------

def calculer_saison(date_match) -> str:
    """
    Détermine la saison (ex: "2024-2025") à partir d'une date, en
    supposant qu'une saison de basket va d'août à juin. Nécessaire car
    les fichiers fusionnés n'ont pas forcément tous une colonne
    "saison" exploitable (elle est perdue si absente de l'un des deux
    fichiers lors de la fusion).
    """
    annee, mois = date_match.year, date_match.month
    if mois >= 8:
        return f"{annee}-{annee + 1}"
    return f"{annee - 1}-{annee}"


def construire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parcourt les matchs dans l'ordre chronologique et construit, pour
    CHAQUE match, un jeu de variables explicatives basées seulement
    sur ce qui est connu avant ce match. Retourne un DataFrame de
    features (une ligne par match) avec les cibles jointes.
    """
    historique_par_equipe = {}   # equipe -> liste de matchs joués (dicts)
    elo = {}                     # equipe -> rating Elo courant
    derniere_date = {}           # equipe -> date du dernier match joué
    stats_saison = {}            # (saison, equipe) -> {"victoires","matchs","diff_points"}
    lignes_brutes_historique = []  # lignes brutes des matchs déjà traités (pour le modèle Poisson)

    lignes_features = []

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

        def classement_et_forme_saison(equipe: str):
            """
            Classement (rang, 1 = meilleure équipe), taux de victoire
            et différentiel de points moyen de l'équipe DANS LA SAISON
            EN COURS, calculés uniquement sur les matchs déjà joués
            cette saison avant le match actuel.
            """
            s = stats_saison[(saison_match, equipe)]
            if s["matchs"] == 0:
                return {"classement": np.nan, "taux_victoire": np.nan, "diff_points_moyen": np.nan}

            # classement parmi toutes les équipes ayant déjà joué cette
            # saison, trié par (victoires desc, diff_points desc)
            classement_toutes_equipes = sorted(
                [
                    (cle_equipe, val["victoires"], val["diff_points"])
                    for (cle_saison, cle_equipe), val in stats_saison.items()
                    if cle_saison == saison_match and val["matchs"] > 0
                ],
                key=lambda t: (-t[1], -t[2]),
            )
            rang = next(
                (i + 1 for i, (nom, _, _) in enumerate(classement_toutes_equipes) if nom == equipe),
                np.nan,
            )
            return {
                "classement": rang,
                "taux_victoire": s["victoires"] / s["matchs"],
                "diff_points_moyen": s["diff_points"] / s["matchs"],
            }

        def forme_recente(equipe: str, cote: str):
            """
            Moyenne, sur les FENETRE_FORME derniers matchs de l'équipe
            (tous confondus, domicile + extérieur), des points
            marqués/encaissés par quart-temps du côté `cote`
            ("domicile" si l'équipe jouait à domicile lors de ce
            match passé, "exterieur" sinon) — on reconstruit donc la
            forme "à domicile" et "à l'extérieur" séparément.
            """
            matchs_passes = [
                h for h in historique_par_equipe[equipe][-FENETRE_FORME:]
                if h["position"] == cote
            ]
            if not matchs_passes:
                return {f"marque_{q}": np.nan for q in range(1, 5)} | \
                       {f"encaisse_{q}": np.nan for q in range(1, 5)}
            resultat = {}
            for q in range(1, 5):
                resultat[f"marque_{q}"] = np.mean([h[f"marque_q{q}"] for h in matchs_passes])
                resultat[f"encaisse_{q}"] = np.mean([h[f"encaisse_q{q}"] for h in matchs_passes])
            return resultat

        def repos(equipe: str):
            if equipe not in derniere_date:
                return np.nan
            return (date_match - derniere_date[equipe]).days

        def stats_h2h(equipe_a: str, equipe_b: str):
            """Moyenne des confrontations directes passées (tous les matchs déjà en historique)."""
            confrontations = [
                h for h in historique_par_equipe[equipe_a]
                if h["adversaire"] == equipe_b
            ]
            if not confrontations:
                return {f"h2h_marque_{q}": np.nan for q in range(1, 5)} | {"h2h_nb": 0}
            resultat = {"h2h_nb": len(confrontations)}
            for q in range(1, 5):
                resultat[f"h2h_marque_{q}"] = np.mean([h[f"marque_q{q}"] for h in confrontations])
            return resultat

        forme_dom = forme_recente(dom, "domicile")
        forme_ext = forme_recente(ext, "exterieur")
        h2h_dom = stats_h2h(dom, ext)
        classement_dom = classement_et_forme_saison(dom)
        classement_ext = classement_et_forme_saison(ext)
        repos_dom = repos(dom)
        repos_ext = repos(ext)

        # --- Stacking : le modèle Poisson (moyennes pondérées +
        # confrontations directes, cf. backtest_predictions.py) est
        # recalculé en glissant sur l'historique déjà connu, et sa
        # prédiction est fournie au modèle ML comme une feature parmi
        # d'autres, plutôt que comme un concurrent séparé. Le ML peut
        # ainsi apprendre à corriger les biais systématiques de Poisson
        # plutôt que de repartir de zéro.
        if lignes_brutes_historique:
            historique_df_poisson = pd.DataFrame(lignes_brutes_historique)
            pred_poisson = predire_quarts_poisson(historique_df_poisson, dom, ext)
        else:
            pred_poisson = None

        if pred_poisson is None:
            features_poisson = {f"poisson_{c}": np.nan for c in COLONNES_QUARTS}
            features_poisson["poisson_proba_victoire_domicile"] = np.nan
        else:
            features_poisson = {f"poisson_{c}": pred_poisson[c] for c in COLONNES_QUARTS}
            features_poisson["poisson_proba_victoire_domicile"] = pred_poisson["proba_victoire_domicile"]

        ligne = {
            "date": date_match,
            "equipe_domicile": dom,
            "equipe_exterieur": ext,
            "elo_domicile": elo[dom],
            "elo_exterieur": elo[ext],
            "diff_elo": elo[dom] - elo[ext],
            "repos_domicile": repos_dom,
            "repos_exterieur": repos_ext,
            # back-to-back : match précédent la veille ou l'avant-veille
            "back_to_back_domicile": int(repos_dom <= 1) if not np.isnan(repos_dom) else np.nan,
            "back_to_back_exterieur": int(repos_ext <= 1) if not np.isnan(repos_ext) else np.nan,
            "nb_confrontations": h2h_dom["h2h_nb"],
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
            **features_poisson,
        }
        for q in range(1, 5):
            ligne[f"forme_dom_marque_q{q}"] = forme_dom[f"marque_{q}"]
            ligne[f"forme_dom_encaisse_q{q}"] = forme_dom[f"encaisse_{q}"]
            ligne[f"forme_ext_marque_q{q}"] = forme_ext[f"marque_{q}"]
            ligne[f"forme_ext_encaisse_q{q}"] = forme_ext[f"encaisse_{q}"]
            ligne[f"h2h_marque_dom_q{q}"] = h2h_dom.get(f"h2h_marque_{q}", np.nan)

        # cibles (ce qu'on veut prédire)
        for c in COLONNES_QUARTS:
            ligne[c] = m[c]
        ligne["victoire_domicile"] = int(m["q1_domicile"] + m["q2_domicile"]
                                          + m["q3_domicile"] + m["q4_domicile"]
                                          > m["q1_exterieur"] + m["q2_exterieur"]
                                          + m["q3_exterieur"] + m["q4_exterieur"])

        lignes_features.append(ligne)

        # -- mise à jour de l'état APRÈS avoir construit les features
        #    (pour que le match courant ne s'auto-alimente pas)
        total_dom = m["q1_domicile"] + m["q2_domicile"] + m["q3_domicile"] + m["q4_domicile"]
        total_ext = m["q1_exterieur"] + m["q2_exterieur"] + m["q3_exterieur"] + m["q4_exterieur"]

        s_dom = stats_saison[(saison_match, dom)]
        s_ext = stats_saison[(saison_match, ext)]
        s_dom["matchs"] += 1
        s_ext["matchs"] += 1
        s_dom["diff_points"] += total_dom - total_ext
        s_ext["diff_points"] += total_ext - total_dom
        if total_dom > total_ext:
            s_dom["victoires"] += 1
        else:
            s_ext["victoires"] += 1

        lignes_brutes_historique.append({
            "equipe_domicile": dom, "equipe_exterieur": ext,
            **{c: m[c] for c in COLONNES_QUARTS},
        })

        historique_par_equipe[dom].append({
            "position": "domicile", "adversaire": ext,
            **{f"marque_q{q}": m[f"q{q}_domicile"] for q in range(1, 5)},
            **{f"encaisse_q{q}": m[f"q{q}_exterieur"] for q in range(1, 5)},
        })
        historique_par_equipe[ext].append({
            "position": "exterieur", "adversaire": dom,
            **{f"marque_q{q}": m[f"q{q}_exterieur"] for q in range(1, 5)},
            **{f"encaisse_q{q}": m[f"q{q}_domicile"] for q in range(1, 5)},
        })

        # mise à jour Elo (formule standard, avec avantage du terrain
        # implicite car on ne l'ajoute pas explicitement ici)
        attendu_dom = 1 / (1 + 10 ** ((elo[ext] - elo[dom]) / 400))
        resultat_reel_dom = 1.0 if total_dom > total_ext else 0.0
        elo[dom] = elo[dom] + ELO_K * (resultat_reel_dom - attendu_dom)
        elo[ext] = elo[ext] + ELO_K * ((1 - resultat_reel_dom) - (1 - attendu_dom))

        derniere_date[dom] = date_match
        derniere_date[ext] = date_match

    return pd.DataFrame(lignes_features)


def colonnes_features(df_features: pd.DataFrame):
    exclure = {"date", "equipe_domicile", "equipe_exterieur", *CIBLES, "victoire_domicile"}
    return [c for c in df_features.columns if c not in exclure]


# ---------------------------------------------------------------------
# Entraînement + évaluation
# ---------------------------------------------------------------------

def entrainer_evaluer_ml(train: pd.DataFrame, test: pd.DataFrame, cols_x: list, verbeux: bool = True):
    """
    Entraîne les modèles ML (régression par quart-temps + classification
    du vainqueur) sur `train`, puis les évalue sur `test`.

    Retourne (modeles_regression, modele_clf, metriques) où metriques
    est un dict avec les MAE par cible, l'accuracy et le log-loss.
    """
    X_train, X_test = train[cols_x], test[cols_x]

    modeles_regression = {}
    mae_par_cible = {}
    for cible in CIBLES:
        modele = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        modele.fit(X_train, train[cible])
        pred = modele.predict(X_test)
        mae_par_cible[cible] = mean_absolute_error(test[cible], pred)
        modeles_regression[cible] = modele
        if verbeux:
            print(f"  {cible:16s} MAE = {mae_par_cible[cible]:.2f} points")

    modele_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    modele_clf.fit(X_train, train["victoire_domicile"])
    proba_test = modele_clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(test["victoire_domicile"], pred_test)
    ll = log_loss(test["victoire_domicile"], proba_test)

    metriques = {"mae": mae_par_cible, "accuracy": acc, "log_loss": ll}
    return modeles_regression, modele_clf, metriques


# ---------------------------------------------------------------------
# Modèle de référence : Poisson + confrontations directes
# (mêmes formules que backtest_predictions.py, reprises ici pour que
# la comparaison tourne sur exactement le même jeu de test)
# ---------------------------------------------------------------------

DEMI_VIE_POISSON = 10
POIDS_H2H_POISSON = 0.35


def poids_fraicheur(n: int, demi_vie: int) -> np.ndarray:
    if n == 0:
        return np.array([])
    rangs = np.arange(n - 1, -1, -1)
    return 0.5 ** (rangs / demi_vie)


def moyenne_ponderee_poisson(valeurs, poids) -> float:
    valeurs = pd.to_numeric(pd.Series(valeurs), errors="coerce")
    poids = np.asarray(poids, dtype=float)
    masque = ~valeurs.isna()
    if masque.sum() == 0 or poids[masque].sum() == 0:
        return np.nan
    return np.average(valeurs[masque], weights=poids[masque])


def moyennes_equipe_poisson(historique: pd.DataFrame, equipe: str, demi_vie: int):
    dom = historique[historique["equipe_domicile"] == equipe]
    ext = historique[historique["equipe_exterieur"] == equipe]
    poids_dom = poids_fraicheur(len(dom), demi_vie)
    poids_ext = poids_fraicheur(len(ext), demi_vie)

    marques_dom = {f"q{i}": moyenne_ponderee_poisson(dom[f"q{i}_domicile"], poids_dom) for i in range(1, 5)}
    encaisses_dom = {f"q{i}": moyenne_ponderee_poisson(dom[f"q{i}_exterieur"], poids_dom) for i in range(1, 5)}
    marques_ext = {f"q{i}": moyenne_ponderee_poisson(ext[f"q{i}_exterieur"], poids_ext) for i in range(1, 5)}
    encaisses_ext = {f"q{i}": moyenne_ponderee_poisson(ext[f"q{i}_domicile"], poids_ext) for i in range(1, 5)}
    return {
        "marques_domicile": marques_dom, "encaisses_domicile": encaisses_dom,
        "marques_exterieur": marques_ext, "encaisses_exterieur": encaisses_ext,
    }


def confrontations_directes_poisson(historique: pd.DataFrame, equipe_a: str, equipe_b: str, demi_vie: int):
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
            for _, row in h2h.iterrows():
                if row["equipe_domicile"] == equipe:
                    vm.append(row[f"{q}_domicile"]); ve.append(row[f"{q}_exterieur"])
                else:
                    vm.append(row[f"{q}_exterieur"]); ve.append(row[f"{q}_domicile"])
            marques[q] = moyenne_ponderee_poisson(vm, poids)
            encaisses[q] = moyenne_ponderee_poisson(ve, poids)
        return {"marques": marques, "encaisses": encaisses}

    return stats_pour(equipe_a), stats_pour(equipe_b), nb


def moyenne_sans_avertissement(valeurs) -> float:
    """
    Équivalent de np.nanmean, mais sans le RuntimeWarning "Mean of
    empty slice" quand toutes les valeurs sont NaN (cas normal en
    tout début d'historique reconstitué, quand une équipe n'a pas
    encore de matchs enregistrés dans une configuration donnée).
    """
    valeurs = np.asarray(valeurs, dtype=float)
    if np.all(np.isnan(valeurs)):
        return np.nan
    return float(np.nanmean(valeurs))


def predire_quarts_poisson(historique: pd.DataFrame, equipe_dom: str, equipe_ext: str,
                            demi_vie: int = DEMI_VIE_POISSON, poids_h2h: float = POIDS_H2H_POISSON):
    """
    Reproduction du modèle Poisson de backtest_predictions.py : renvoie
    les lambda par quart-temps + la probabilité de victoire à domicile
    (via la loi de Skellam), ou None si l'historique est insuffisant.
    """
    stats_dom = moyennes_equipe_poisson(historique, equipe_dom, demi_vie)
    stats_ext = moyennes_equipe_poisson(historique, equipe_ext, demi_vie)
    h2h_dom, h2h_ext, nb_h2h = confrontations_directes_poisson(historique, equipe_dom, equipe_ext, demi_vie)
    poids_h2h_eff = poids_h2h if nb_h2h > 0 else 0.0

    lambdas = {}
    lambda_dom_total, lambda_ext_total = 0.0, 0.0

    for i in range(1, 5):
        q = f"q{i}"
        general_dom = moyenne_sans_avertissement([stats_dom["marques_domicile"][q], stats_ext["encaisses_exterieur"][q]])
        general_ext = moyenne_sans_avertissement([stats_ext["marques_exterieur"][q], stats_dom["encaisses_domicile"][q]])

        if nb_h2h > 0:
            h2h_dom_val = moyenne_sans_avertissement([h2h_dom["marques"][q], h2h_ext["encaisses"][q]])
            h2h_ext_val = moyenne_sans_avertissement([h2h_ext["marques"][q], h2h_dom["encaisses"][q]])
            lambda_dom = (1 - poids_h2h_eff) * general_dom + poids_h2h_eff * h2h_dom_val
            lambda_ext = (1 - poids_h2h_eff) * general_ext + poids_h2h_eff * h2h_ext_val
        else:
            lambda_dom, lambda_ext = general_dom, general_ext

        if np.isnan(lambda_dom) or np.isnan(lambda_ext):
            return None

        lambdas[f"q{i}_domicile"] = lambda_dom
        lambdas[f"q{i}_exterieur"] = lambda_ext
        lambda_dom_total += lambda_dom
        lambda_ext_total += lambda_ext

        # probabilité de victoire SUR CE QUART-TEMPS précis (Skellam
        # appliqué aux lambdas du quart, pas au total du match) —
        # renormalisée hors égalité, comme pour le match complet
        proba_egalite_q = skellam.pmf(0, lambda_dom, lambda_ext)
        proba_ext_q = skellam.cdf(-1, lambda_dom, lambda_ext)
        proba_dom_q = 1 - proba_ext_q - proba_egalite_q
        denominateur = 1 - float(proba_egalite_q)
        lambdas[f"proba_victoire_{q}_domicile"] = float(proba_dom_q) / denominateur if denominateur > 0 else np.nan
        lambdas[f"proba_victoire_{q}_exterieur"] = float(proba_ext_q) / denominateur if denominateur > 0 else np.nan

    proba_victoire_ext = skellam.cdf(-1, lambda_dom_total, lambda_ext_total)
    proba_egalite = skellam.pmf(0, lambda_dom_total, lambda_ext_total)
    proba_victoire_dom = 1 - proba_victoire_ext - proba_egalite

    lambdas["proba_victoire_domicile"] = float(proba_victoire_dom) / (1 - float(proba_egalite))
    return lambdas


def evaluer_confiance(df: pd.DataFrame, equipe_domicile: str, equipe_exterieur: str,
                       date_ref: pd.Timestamp = None):
    """
    Indicateur de confiance simple et transparent, basé sur le VOLUME
    D'HISTORIQUE réellement disponible pour ce match précis — pas sur
    une estimation de variance du modèle. L'idée : une prédiction sur
    un match entre deux équipes qui ont chacune 40+ matchs connus et
    se sont déjà affrontées plusieurs fois est plus fiable qu'une
    prédiction avec 8 matchs connus et 0 confrontation directe.

    Retourne un dict avec le nombre de matchs connus de chaque équipe,
    le nombre de confrontations directes, et un niveau qualitatif
    ("Faible", "Moyenne", "Élevée").
    """
    df_avant = df[df["_date"] < date_ref] if date_ref is not None else df

    nb_dom = len(df_avant[
        (df_avant["equipe_domicile"] == equipe_domicile) | (df_avant["equipe_exterieur"] == equipe_domicile)
    ])
    nb_ext = len(df_avant[
        (df_avant["equipe_domicile"] == equipe_exterieur) | (df_avant["equipe_exterieur"] == equipe_exterieur)
    ])
    nb_h2h = len(df_avant[
        ((df_avant["equipe_domicile"] == equipe_domicile) & (df_avant["equipe_exterieur"] == equipe_exterieur))
        | ((df_avant["equipe_domicile"] == equipe_exterieur) & (df_avant["equipe_exterieur"] == equipe_domicile))
    ])

    profondeur = min(nb_dom, nb_ext)
    if profondeur < 10:
        niveau = "Faible"
    elif profondeur < 30:
        niveau = "Moyenne"
    else:
        niveau = "Élevée"
    if nb_h2h == 0 and niveau == "Élevée":
        niveau = "Moyenne"  # aucune confrontation directe : on dégrade d'un cran

    return {
        "nb_matchs_domicile": nb_dom,
        "nb_matchs_exterieur": nb_ext,
        "nb_confrontations": nb_h2h,
        "niveau_confiance": niveau,
    }


def evaluer_poisson(df_brut: pd.DataFrame, test: pd.DataFrame):
    """
    Évalue le modèle Poisson sur les mêmes matchs que `test`, en ne
    donnant à chaque prédiction que les matchs de df_brut strictement
    antérieurs à la date du match évalué (même logique "aucune fuite
    du futur" que pour le ML).
    """
    erreurs_par_cible = {c: [] for c in CIBLES}
    vrais_labels, probas_predites = [], []

    for _, row in test.iterrows():
        date_match = row["date"]
        dom, ext = row["equipe_domicile"], row["equipe_exterieur"]
        historique = df_brut[df_brut["_date"] < date_match]

        pred = predire_quarts_poisson(historique, dom, ext)
        if pred is None:
            continue

        for c in CIBLES:
            erreurs_par_cible[c].append(abs(pred[c] - row[c]))

        vrais_labels.append(row["victoire_domicile"])
        probas_predites.append(pred["proba_victoire_domicile"])

    mae_par_cible = {c: float(np.mean(v)) if v else np.nan for c, v in erreurs_par_cible.items()}
    probas_predites = np.clip(probas_predites, 1e-6, 1 - 1e-6)  # évite log(0)
    acc = accuracy_score(vrais_labels, [1 if p >= 0.5 else 0 for p in probas_predites])
    ll = log_loss(vrais_labels, probas_predites)

    return {"mae": mae_par_cible, "accuracy": acc, "log_loss": ll, "nb_matchs": len(vrais_labels)}


# ---------------------------------------------------------------------
# Comparaison ML vs Poisson (commande "comparer")
# ---------------------------------------------------------------------

def comparer(args):
    df = charger_et_fusionner(args.historique, args.saison_actuelle,
                               args.colonne_date, args.format_date)
    df_features = construire_features(df)
    cols_x = colonnes_features(df_features)

    df_utile = df_features.dropna(subset=cols_x).reset_index(drop=True)

    n_test = int(len(df_utile) * args.proportion_test)
    train = df_utile.iloc[:-n_test]
    test = df_utile.iloc[-n_test:].reset_index(drop=True)

    print(f"{len(train)} matchs d'entraînement, {len(test)} matchs de test "
          f"(les plus récents).\n")

    print("Entraînement et évaluation du modèle de machine learning...")
    _, _, metriques_ml = entrainer_evaluer_ml(train, test, cols_x, verbeux=False)

    print("Évaluation du modèle Poisson (walk-forward)...")
    metriques_poisson = evaluer_poisson(df, test)

    print(f"\n{'Cible':16s} {'MAE Poisson':>14s} {'MAE ML':>10s} {'Meilleur':>10s}")
    print("-" * 54)
    for c in CIBLES:
        mae_p = metriques_poisson["mae"][c]
        mae_m = metriques_ml["mae"][c]
        meilleur = "ML" if mae_m < mae_p else "Poisson"
        print(f"{c:16s} {mae_p:14.2f} {mae_m:10.2f} {meilleur:>10s}")

    print("\nPrédiction du vainqueur :")
    print(f"{'':16s} {'Poisson':>14s} {'ML':>10s}")
    print(f"{'Précision':16s} {metriques_poisson['accuracy']:14.1%} {metriques_ml['accuracy']:10.1%}")
    print(f"{'Log-loss':16s} {metriques_poisson['log_loss']:14.3f} {metriques_ml['log_loss']:10.3f}")

    taux_domicile = train["victoire_domicile"].mean()
    print(f"\n(Référence naïve « domicile gagne toujours » sur le train : "
          f"{taux_domicile:.1%})")
    if metriques_poisson["nb_matchs"] < len(test):
        ignores = len(test) - metriques_poisson["nb_matchs"]
        print(f"({ignores} match(s) exclu(s) de l'évaluation Poisson : "
              f"historique insuffisant à ce stade de la saison)")


# ---------------------------------------------------------------------
# Entraînement (commande "entrainer")
# ---------------------------------------------------------------------

def entrainer(args):
    df = charger_et_fusionner(args.historique, args.saison_actuelle,
                               args.colonne_date, args.format_date)
    print(f"{len(df)} matchs avec détail quart-temps disponibles.")

    df_features = construire_features(df)
    df_features.to_csv(FICHIER_FEATURES, index=False)
    print(f"Features construites et sauvegardées dans {FICHIER_FEATURES}.")

    cols_x = colonnes_features(df_features)

    # on ne peut entraîner que sur les lignes où les features sont
    # complètes (les tout premiers matchs de chaque équipe n'ont pas
    # encore assez d'historique -> on les exclut)
    df_utile = df_features.dropna(subset=cols_x).reset_index(drop=True)
    print(f"{len(df_utile)} matchs utilisables après exclusion des tout "
          f"premiers matchs de chaque équipe (historique insuffisant).")

    # découpage CHRONOLOGIQUE (jamais aléatoire)
    n_test = int(len(df_utile) * args.proportion_test)
    train = df_utile.iloc[:-n_test]
    test = df_utile.iloc[-n_test:]
    print(f"Entraînement sur {len(train)} matchs, évaluation sur {len(test)} "
          f"matchs les plus récents.\n")

    X_train, X_test = train[cols_x], test[cols_x]

    # --- régression : un modèle par quart-temps/équipe ---
    modeles_regression = {}
    predictions_test = {}
    print("Régression (score par quart-temps) :")
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

    # --- classification : vainqueur ---
    print("\nClassification (vainqueur) :")
    modele_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    modele_clf.fit(X_train, train["victoire_domicile"])
    proba_test = modele_clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(test["victoire_domicile"], pred_test)
    ll = log_loss(test["victoire_domicile"], proba_test)
    print(f"  Précision (accuracy) = {acc:.1%}")
    print(f"  Log-loss             = {ll:.3f}")

    # référence naïve : toujours prédire l'équipe à domicile gagnante
    taux_domicile = train["victoire_domicile"].mean()
    print(f"  (référence naïve « domicile gagne toujours » sur le train : "
          f"{taux_domicile:.1%} des matchs)")

    # --- CSV : prédictions vs réalité sur l'échantillon de test ---
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
    resultats_test.to_csv(FICHIER_RESULTATS_TEST, index=False)
    print(f"\nDétail des prédictions sur l'échantillon de test : {FICHIER_RESULTATS_TEST}")

    joblib.dump({"modeles": modeles_regression, "colonnes": cols_x}, FICHIER_MODELE_REGRESSION)
    joblib.dump({"modele": modele_clf, "colonnes": cols_x}, FICHIER_MODELE_CLASSIFICATION)
    print(f"Modèles sauvegardés : {FICHIER_MODELE_REGRESSION}, {FICHIER_MODELE_CLASSIFICATION}")


# ---------------------------------------------------------------------
# Prédiction d'un match à venir
# ---------------------------------------------------------------------

def predire(args):
    df = charger_et_fusionner(args.historique, args.saison_actuelle,
                               args.colonne_date, args.format_date)
    df_features = construire_features(df)  # état le plus à jour possible

    cols_x = colonnes_features(df_features)

    # on normalise les noms saisis en argument, exactement comme les
    # noms des deux fichiers sources (voir NOMS_CANONIQUES) — sinon
    # "AS Monaco Basket" tapé à la main ne correspondrait à rien si
    # l'historique ne connaît que "AS Monaco", et inversement.
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
            if nom_normalise != nom_saisi:
                message += f" (normalisée en '{nom_normalise}')"
            if suggestions:
                message += f". Vouliez-vous dire : {', '.join(suggestions)} ?"
            else:
                message += ". Noms disponibles : " + ", ".join(sorted(noms_connus))
            raise SystemExit(message)

    # reconstruction d'une ligne de features pour CE match précis en
    # rejouant construire_features sur l'historique + un match fictif,
    # ce qui garantit une cohérence totale avec l'entraînement (même
    # logique, mêmes colonnes).
    #
    # Date utilisée : le lendemain du dernier match connu dans vos
    # fichiers, PAS la date réelle du jour (pd.Timestamp.today()).
    # Utiliser la date réelle est risqué : si vous lancez ce script
    # après la fin de la saison couverte par vos fichiers (ou pendant
    # l'intersaison), la date réelle tombe dans une saison qui n'a
    # encore AUCUN match enregistré -> classement, taux de victoire et
    # différentiel de points valent NaN pour toutes les équipes, ce
    # qui fait planter le modèle. Utilisez --date pour forcer une date
    # précise si besoin (ex : simuler un match à une date passée).
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
        **{c: 0 for c in COLONNES_QUARTS},  # valeurs bidons, non utilisées pour prédire
    }])
    df_avec_fictif = pd.concat([df, ligne_fictive], ignore_index=True)
    features_avec_fictif = construire_features(df_avec_fictif)
    ligne_a_predire = features_avec_fictif.iloc[[-1]][cols_x]

    # filet de sécurité : si une variable manque quand même (ex : une
    # des deux équipes n'a encore jamais joué dans la saison de
    # date_fictive), on comble avec la médiane connue plutôt que de
    # planter — mais on prévient clairement l'utilisateur, car la
    # prédiction est alors moins fiable sur ce point précis.
    colonnes_manquantes = ligne_a_predire.columns[ligne_a_predire.isna().any()].tolist()
    if colonnes_manquantes:
        medianes = df_features[cols_x].median(numeric_only=True)
        ligne_a_predire = ligne_a_predire.fillna(medianes)
        print(f"Attention : valeur(s) manquante(s) comblée(s) par la médiane pour : "
              f"{', '.join(colonnes_manquantes)} (historique insuffisant pour ce point précis).\n")

    modele_reg = joblib.load(FICHIER_MODELE_REGRESSION)
    modele_clf = joblib.load(FICHIER_MODELE_CLASSIFICATION)

    print(f"\nPrédiction : {equipe_domicile} (domicile) vs "
          f"{equipe_exterieur} (extérieur)\n")

    total_dom, total_ext = 0, 0
    predictions_quarts = {}
    for cible in CIBLES:
        pred = modele_reg["modeles"][cible].predict(ligne_a_predire)[0]
        predictions_quarts[cible] = pred
        if "domicile" in cible:
            total_dom += pred
        else:
            total_ext += pred
        print(f"  {cible:16s} : {pred:.1f}")

    proba_dom = modele_clf["modele"].predict_proba(ligne_a_predire)[0][1]
    print(f"\n  Total estimé : {equipe_domicile} {total_dom:.1f} - "
          f"{total_ext:.1f} {equipe_exterieur}")
    print(f"  Probabilité de victoire {equipe_domicile} (domicile) : {proba_dom:.1%}")
    print(f"  Probabilité de victoire {equipe_exterieur} (extérieur) : {1 - proba_dom:.1%}")

    # --- CSV : on ajoute cette prédiction à l'historique des
    # prédictions déjà faites (le fichier s'enrichit au fil du temps,
    # au lieu d'être écrasé à chaque exécution) ---
    nouvelle_ligne = {
        "date_prediction": pd.Timestamp.today().strftime("%d/%m/%Y %H:%M"),
        "equipe_domicile": equipe_domicile,
        "equipe_exterieur": equipe_exterieur,
    }
    for cible in CIBLES:
        nouvelle_ligne[cible] = round(predictions_quarts[cible], 1)
    nouvelle_ligne["total_domicile"] = round(total_dom, 1)
    nouvelle_ligne["total_exterieur"] = round(total_ext, 1)
    nouvelle_ligne["proba_victoire_domicile"] = round(float(proba_dom), 3)
    nouvelle_ligne["proba_victoire_exterieur"] = round(float(1 - proba_dom), 3)

    df_nouvelle_ligne = pd.DataFrame([nouvelle_ligne])
    try:
        predictions_existantes = pd.read_csv(FICHIER_PREDICTIONS)
        df_predictions = pd.concat([predictions_existantes, df_nouvelle_ligne], ignore_index=True)
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
    p_entrainer.add_argument("--proportion-test", type=float, default=0.15,
                              help="Proportion des matchs les plus récents réservée au test.")
    p_entrainer.set_defaults(func=entrainer)

    p_predire = sous_commandes.add_parser("predire")
    p_predire.add_argument("equipe_domicile")
    p_predire.add_argument("equipe_exterieur")
    p_predire.add_argument("--historique", required=True)
    p_predire.add_argument("--saison-actuelle", required=True)
    p_predire.add_argument("--colonne-date", default="date")
    p_predire.add_argument("--format-date", default="%d/%m/%Y")
    p_predire.add_argument("--date", default=None,
                            help="Date du match à prédire (JJ/MM/AAAA). Par défaut : "
                                 "le lendemain du dernier match connu dans vos fichiers "
                                 "(PAS la date réelle du jour).")
    p_predire.set_defaults(func=predire)

    p_comparer = sous_commandes.add_parser("comparer")
    p_comparer.add_argument("--historique", required=True)
    p_comparer.add_argument("--saison-actuelle", required=True)
    p_comparer.add_argument("--colonne-date", default="date")
    p_comparer.add_argument("--format-date", default="%d/%m/%Y")
    p_comparer.add_argument("--proportion-test", type=float, default=0.15)
    p_comparer.set_defaults(func=comparer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
