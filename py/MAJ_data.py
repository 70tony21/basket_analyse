"""
maj_data.py — Met à jour le fichier lnb_2026-2027.csv avec les résultats
des matchs terminés, et génère prochains_match.csv pour les jours à venir.

Utilisation :
    python maj_data.py                          # 2 jours par défaut
    python maj_data.py --jours 3                # 3 jours à venir
    python maj_data.py --csv chemin/lnb.csv     # chemin custom
    python maj_data.py --no-scrape              # ne scrape pas, juste le prochain CSV

Prérequis :
    pip install requests beautifulsoup4
"""

import argparse
import csv
import io
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ============================================================
# CONFIG
# ============================================================
ID_LIGUE = "4423"  # French LNB sur TheSportsDB
BASE_SEASON_URL = f"https://www.thesportsdb.com/season/{ID_LIGUE}-french-lnb/{{saison}}?all=1&csv=1"
BASE_EVENT_URL = "https://www.thesportsdb.com/event/{id_event}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Chemins par défaut
CSV_SAISON_DEFAUT = Path("../raw_data/lnb_2026-2027.csv")
CSV_PROCHAINS_DEFAUT = Path("../raw_data/prochains_match.csv")

# Colonnes du CSV saison (identiques à ton fichier actuel)
CHAMPS_SAISON = [
    "saison", "round", "date", "equipe_domicile", "equipe_exterieur",
    "score_domicile", "score_exterieur",
    "q1_domicile", "q1_exterieur", "q2_domicile", "q2_exterieur",
    "q3_domicile", "q3_exterieur", "q4_domicile", "q4_exterieur",
    "prolongation_domicile", "prolongation_exterieur",
]

# Colonnes du CSV prochains matchs (avec les lignes de paris + cotes)
CHAMPS_PROCHAINS = [
    "saison", "round", "date", "equipe_domicile", "equipe_exterieur",
    "ligne", "ligne_total",
    "cote_victoire_domicile", "cote_victoire_exterieur",
    "cote_handicap_domicile", "cote_handicap_exterieur",
    "cote_over", "cote_under",
    "ligne_mi_temps", "ligne_total_mi_temps",
]

# Motif quart-temps (repris du scraper d'origine)
MOTIF_QUARTS = re.compile(
    r"([\w\séÉèàâîôûç\-'\.]+?)\s+Quarters:\s*\n?\s*([\d\s]+?)\n"
    r".*?"
    r"([\w\séÉèàâîôûç\-'\.]+?)\s+Quarters:\s*\n?\s*([\d\s]+?)\n",
    re.IGNORECASE | re.DOTALL,
)


# ============================================================
# LECTURE / ÉCRITURE CSV
# ============================================================
def lire_csv(chemin: Path):
    """Lit un CSV et renvoie une liste de dicts."""
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    with chemin.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ecrire_csv(chemin: Path, champs: list, lignes: list):
    """Écrit un CSV avec un ordre de colonnes fixe."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=champs)
        writer.writeheader()
        for ligne in lignes:
            # Ne garder que les champs attendus (au cas où il y aurait du rab)
            writer.writerow({k: ligne.get(k, "") for k in champs})


# ============================================================
# SCRAPING THESPORTSDB
# ============================================================
def recuperer_calendrier_saison(saison: str):
    """
    Télécharge le CSV récapitulatif d'une saison sur TheSportsDB.
    Retourne une liste de dicts (idEvent, Home Team, Away Team, etc.)
    """
    url = BASE_SEASON_URL.format(saison=saison)
    print(f"  → Téléchargement du calendrier {saison} : {url}")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    zone_texte = soup.find("textarea", id="myInput")
    if zone_texte is None:
        raise RuntimeError(
            "Impossible de trouver <textarea id='myInput'> sur la page saison."
        )

    contenu_csv = zone_texte.get_text()
    lecteur = csv.DictReader(io.StringIO(contenu_csv))
    matchs = list(lecteur)
    print(f"  → {len(matchs)} matchs dans le calendrier officiel.")
    return matchs


def extraire_quarts(id_event: str):
    """
    Récupère les scores par quart-temps sur la page publique de l'événement.
    Retourne un dict {"dom": [q1..q4], "ext": [q1..q4],
                      "prolongation_dom": int, "prolongation_ext": int}
    ou None si non trouvé.
    """
    url = BASE_EVENT_URL.format(id_event=id_event)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    texte = soup.get_text("\n", strip=True)

    m = MOTIF_QUARTS.search(texte)
    if not m:
        return None

    _, quarts_1, _, quarts_2 = m.groups()
    periodes_dom = [int(x) for x in quarts_1.split()]
    periodes_ext = [int(x) for x in quarts_2.split()]

    if len(periodes_dom) < 4 or len(periodes_ext) < 4:
        return None

    return {
        "dom": periodes_dom[:4],
        "ext": periodes_ext[:4],
        "prolongation_dom": sum(periodes_dom[4:]),
        "prolongation_ext": sum(periodes_ext[4:]),
    }


# ============================================================
# MATCHING LOCAL ↔ THESPORTSDB
# ============================================================
def normaliser_nom(nom: str) -> str:
    """
    Normalise un nom d'équipe pour comparer localement (accents, casse,
    suffixes communs type 'Basket', 'Basket-Ball'...).
    """
    n = nom.lower().strip()
    # Retirer accents simples
    remplacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a", "ä": "a",
        "î": "i", "ï": "i",
        "ô": "o", "ö": "o",
        "û": "u", "ü": "u", "ù": "u",
        "ç": "c",
    }
    for k, v in remplacements.items():
        n = n.replace(k, v)
    # Enlever suffixes courants
    for suffixe in [" basket-ball", " basket", " bball", "-dunkerque"]:
        n = n.replace(suffixe, "")
    # Enlever espaces/ponctuation multiples
    n = re.sub(r"[\s\-'\.]+", " ", n).strip()
    return n


def trouver_match_thesportsdb(match_local: dict, matchs_tsdb: list):
    """
    Retrouve la ligne TheSportsDB correspondant à un match local,
    en comparant les noms d'équipes normalisés.
    """
    dom_local = normaliser_nom(match_local["equipe_domicile"])
    ext_local = normaliser_nom(match_local["equipe_exterieur"])

    for m in matchs_tsdb:
        dom_tsdb = normaliser_nom(m.get("Home Team", ""))
        ext_tsdb = normaliser_nom(m.get("Away Team", ""))
        if dom_local == dom_tsdb and ext_local == ext_tsdb:
            return m
    return None


# ============================================================
# LOGIQUE PRINCIPALE
# ============================================================
def parser_date(date_str: str):
    """Parse la date au format 'YYYY-MM-DD HH:MM:SS'. Retourne None si invalide."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def match_est_termine(match: dict, maintenant: datetime) -> bool:
    """
    Un match est considéré terminé si :
      - ses scores sont déjà renseignés, OU
      - sa date est passée de plus de 3h (durée max d'un match + marge)
    """
    if match.get("score_domicile") and match.get("score_exterieur"):
        return True
    d = parser_date(match.get("date", ""))
    if d and d < maintenant - timedelta(hours=3):
        return True
    return False


def maj_saison(csv_saison: Path, scrape: bool = True):
    """
    Met à jour le CSV saison en scrapant les matchs terminés manquants.
    Retourne la liste des lignes mises à jour.
    """
    print(f"\n=== Mise à jour de {csv_saison} ===")
    lignes = lire_csv(csv_saison)
    maintenant = datetime.now()

    # Trouver les matchs terminés mais sans scores
    a_completer = []
    for i, m in enumerate(lignes):
        d = parser_date(m.get("date", ""))
        scores_vides = not (m.get("score_domicile") and m.get("score_exterieur"))
        if d and d < maintenant - timedelta(hours=3) and scores_vides:
            a_completer.append((i, m))

    print(f"  → {len(a_completer)} match(s) terminé(s) à compléter.")

    if not a_completer or not scrape:
        if not scrape:
            print("  → Scraping désactivé (--no-scrape).")
        return lignes

    # Récupérer le calendrier TheSportsDB de la saison
    saison = lignes[0]["saison"] if lignes else None
    if not saison:
        print("  ⚠ Aucune saison détectée dans le CSV.")
        return lignes

    try:
        matchs_tsdb = recuperer_calendrier_saison(saison)
    except Exception as exc:
        print(f"  ⚠ Impossible de récupérer le calendrier TheSportsDB : {exc}")
        return lignes

    # Compléter chaque match
    nb_ok = 0
    nb_echec = 0
    for idx, match_local in a_completer:
        match_tsdb = trouver_match_thesportsdb(match_local, matchs_tsdb)
        if not match_tsdb:
            print(f"  ⚠ Non trouvé sur TheSportsDB : "
                  f"{match_local['equipe_domicile']} vs {match_local['equipe_exterieur']}")
            nb_echec += 1
            continue

        id_event = match_tsdb.get("idEvent")
        if not id_event:
            nb_echec += 1
            continue

        # Score final
        ligne = lignes[idx]
        ligne["score_domicile"] = match_tsdb.get("Home Score", "")
        ligne["score_exterieur"] = match_tsdb.get("Away Score", "")

        # Quart-temps
        try:
            quarts = extraire_quarts(id_event)
        except Exception as exc:
            print(f"  ⚠ Erreur scraping {id_event} : {exc}")
            quarts = None

        if quarts:
            ligne.update({
                "q1_domicile": quarts["dom"][0], "q1_exterieur": quarts["ext"][0],
                "q2_domicile": quarts["dom"][1], "q2_exterieur": quarts["ext"][1],
                "q3_domicile": quarts["dom"][2], "q3_exterieur": quarts["ext"][2],
                "q4_domicile": quarts["dom"][3], "q4_exterieur": quarts["ext"][3],
                "prolongation_domicile": quarts["prolongation_dom"],
                "prolongation_exterieur": quarts["prolongation_ext"],
            })
        else:
            # Score final OK mais quart-temps non dispo
            ligne["prolongation_domicile"] = 0
            ligne["prolongation_exterieur"] = 0

        nb_ok += 1
        print(f"  ✓ {match_local['equipe_domicile']} vs "
              f"{match_local['equipe_exterieur']} : "
              f"{ligne['score_domicile']}-{ligne['score_exterieur']}")
        time.sleep(1)  # pause respectueuse

    print(f"  → {nb_ok} match(s) mis à jour, {nb_echec} échec(s).")

    # Réécriture du CSV
    ecrire_csv(csv_saison, CHAMPS_SAISON, lignes)
    print(f"  → Fichier réécrit : {csv_saison}")

    return lignes


def generer_prochains_matchs(lignes_saison: list, csv_sortie: Path, nb_jours: int = 2):
    """
    Génère prochains_match.csv avec les matchs à venir dans les N prochains jours.
    Les colonnes de paris / cotes sont laissées vides (à remplir à la main).
    """
    print(f"\n=== Génération de {csv_sortie} ({nb_jours} jour(s)) ===")
    maintenant = datetime.now()
    limite = maintenant + timedelta(days=nb_jours)

    a_venir = []
    for m in lignes_saison:
        d = parser_date(m.get("date", ""))
        if not d:
            continue
        scores_vides = not (m.get("score_domicile") and m.get("score_exterieur"))
        if maintenant <= d <= limite and scores_vides:
            a_venir.append(m)

    print(f"  → {len(a_venir)} match(s) dans les {nb_jours} prochain(s) jour(s).")

    # Construire les lignes de sortie
    lignes_sortie = []
    for m in a_venir:
        lignes_sortie.append({
            "saison": m.get("saison", ""),
            "round": m.get("round", ""),
            "date": m.get("date", ""),
            "equipe_domicile": m.get("equipe_domicile", ""),
            "equipe_exterieur": m.get("equipe_exterieur", ""),
            "ligne": "",
            "ligne_total": "",
            "cote_victoire_domicile": "",
            "cote_victoire_exterieur": "",
            "cote_handicap_domicile": "",
            "cote_handicap_exterieur": "",
            "cote_over": "",
            "cote_under": "",
            "ligne_mi_temps": "",
            "ligne_total_mi_temps": "",
        })

    ecrire_csv(csv_sortie, CHAMPS_PROCHAINS, lignes_sortie)
    print(f"  → Fichier écrit : {csv_sortie}")

    # Petit aperçu
    if lignes_sortie:
        print("\n  Aperçu des prochains matchs :")
        for m in lignes_sortie[:10]:
            print(f"    {m['date']} — {m['equipe_domicile']} vs {m['equipe_exterieur']}")
        if len(lignes_sortie) > 10:
            print(f"    ... et {len(lignes_sortie) - 10} autre(s).")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MAJ données LNB + prochains matchs")
    parser.add_argument("--csv", type=Path, default=CSV_SAISON_DEFAUT,
                        help=f"CSV de la saison (défaut : {CSV_SAISON_DEFAUT})")
    parser.add_argument("--sortie", type=Path, default=CSV_PROCHAINS_DEFAUT,
                        help=f"CSV des prochains matchs (défaut : {CSV_PROCHAINS_DEFAUT})")
    parser.add_argument("--jours", type=int, default=2,
                        help="Nombre de jours à venir (défaut : 2)")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Ne pas scraper TheSportsDB, juste générer prochains_matchs")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"❌ Fichier introuvable : {args.csv}")
        sys.exit(1)

    # 1) Mise à jour du CSV saison
    if args.no_scrape:
        lignes = lire_csv(args.csv)
    else:
        try:
            lignes = maj_saison(args.csv, scrape=True)
        except Exception as exc:
            print(f"❌ Erreur lors de la MAJ saison : {exc}")
            print("   → On continue avec le CSV tel quel.")
            lignes = lire_csv(args.csv)

    # 2) Génération du CSV prochains matchs
    generer_prochains_matchs(lignes, args.sortie, nb_jours=args.jours)

    print("\n✅ Terminé.")


if __name__ == "__main__":
    main()