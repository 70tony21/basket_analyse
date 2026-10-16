"""https://aka.ms/vscode-workspace-trust
Récupère, pour une ou plusieurs saisons passées de Betclic ELITE (LNB),
les résultats avec le détail par quart-temps, en scrapant les pages
publiques de TheSportsDB.com (pas l'API gratuite, qui est verrouillée
sur un exemple de démonstration).

Fonctionnement :
  1. Télécharge le CSV récapitulatif de la saison, qui liste tous les
     matchs (idEvent, date, équipes, score final) en une requête :
     https://www.thesportsdb.com/season/4423-french-lnb/{saison}?all=1&csv=1
  2. Pour chaque idEvent, visite la page publique
     https://www.thesportsdb.com/event/{idEvent}
     et en extrait le bloc "Home Team Quarters:" / "Away Team Quarters:"
     de la section "Result Description".
  3. Écrit un CSV par saison.

Prérequis (à installer une seule fois) :
    pip install requests beautifulsoup4

Utilisation :
    python thesportsdb_scraper.py 2023-2024 2022-2023

Sortie :
    betclic_elite_thesportsdb_2023-2024.csv
    betclic_elite_thesportsdb_2022-2023.csv

Note : l'ID de ligue 4423 correspond à "French LNB" sur TheSportsDB.
Le nom du championnat a changé plusieurs fois dans le temps (Pro A,
Jeep Élite, Betclic Élite) mais c'est la même ligue/le même ID côté
TheSportsDB, seule la saison (format AAAA-AAAA) change dans l'URL.
"""

import csv
import io
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

ID_LIGUE = "4423"  # French LNB sur TheSportsDB
BASE_SEASON_URL = f"https://www.thesportsdb.com/season/{ID_LIGUE}-french-lnb/{{saison}}?all=1&csv=1"
BASE_EVENT_URL = "https://www.thesportsdb.com/event/{id_event}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Repère le bloc quart-temps affiché sur la page publique de l'événement,
# de la forme :
#   Paris Basketball Quarters:
#   15 26 25 24
#   AS Monaco Basket Quarters:
#   28 25 24 25
MOTIF_QUARTS = re.compile(
    r"([\w\séÉèàâîôûç\-'\.]+?)\s+Quarters:\s*\n?\s*([\d\s]+?)\n"
    r".*?"
    r"([\w\séÉèàâîôûç\-'\.]+?)\s+Quarters:\s*\n?\s*([\d\s]+?)\n",
    re.IGNORECASE | re.DOTALL,
)


def recuperer_calendrier_saison(saison: str):
    """
    Télécharge et parse le CSV récapitulatif d'une saison.
    Le CSV est fourni par TheSportsDB à l'intérieur d'un
    <textarea id="myInput"> sur la page (pas mélangé au reste du HTML),
    donc on cible directement cet élément plutôt que de scanner le
    texte de toute la page.

    Retourne une liste de dicts avec au moins idEvent, strTimestamp,
    Round, Home Team, Home Score, Away Team, Away Score.
    """
    url = BASE_SEASON_URL.format(saison=saison)
    print(f"Récupération du calendrier {saison} : {url}")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    zone_texte = soup.find("textarea", id="myInput")
    if zone_texte is None:
        raise RuntimeError(
            "Impossible de trouver le bloc CSV (<textarea id='myInput'>) "
            "sur la page — la structure du site a peut-être changé."
        )

    # .get_text() décode automatiquement les entités HTML (ex: &eacute;)
    contenu_csv = zone_texte.get_text()

    lecteur = csv.DictReader(io.StringIO(contenu_csv))
    matchs = list(lecteur)
    print(f"  {len(matchs)} matchs trouvés dans le calendrier.")
    return matchs


def extraire_quarts(id_event: str):
    """
    Va chercher la page publique de l'événement et tente d'en extraire
    les scores par quart-temps de chaque équipe, ainsi que la
    prolongation éventuelle.

    Sur TheSportsDB, la ligne "Quarters:" contient 4 nombres (Q1 à Q4)
    ou plus s'il y a eu une ou plusieurs prolongations, ex :
        Strasbourg Quarters:
        23 22 29 29 11
    (les 4 premiers nombres sont les quart-temps, le(s) suivant(s) sont
    le score de chaque prolongation — PAS un score cumulé). S'il y a
    plusieurs prolongations, leurs scores sont additionnés en une
    seule valeur de prolongation.

    Retourne un dict :
        {"dom": [q1,q2,q3,q4], "ext": [q1,q2,q3,q4],
         "prolongation_dom": int, "prolongation_ext": int}
    ou None si le détail n'a pas été trouvé (ancien match sans détail,
    ou mise en page différente).
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

    prolongation_dom = sum(periodes_dom[4:])  # 0 si pas de prolongation
    prolongation_ext = sum(periodes_ext[4:])

    return {
        "dom": periodes_dom[:4], "ext": periodes_ext[:4],
        "prolongation_dom": prolongation_dom,
        "prolongation_ext": prolongation_ext,
    }


def traiter_saison(saison: str):
    matchs = recuperer_calendrier_saison(saison)
    fichier_sortie = f"../raw_data/LNB/betclic_elite_thesportsdb_{saison}.csv"

    champs = [
        "saison", "round", "date", "equipe_domicile", "equipe_exterieur",
        "score_domicile", "score_exterieur",
        "q1_domicile", "q1_exterieur", "q2_domicile", "q2_exterieur",
        "q3_domicile", "q3_exterieur", "q4_domicile", "q4_exterieur",
        "prolongation_domicile", "prolongation_exterieur",
    ]

    echecs = []

    with open(fichier_sortie, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=champs)
        writer.writeheader()

        for i, match in enumerate(matchs, start=1):
            id_event = match["idEvent"]
            print(f"[{i}/{len(matchs)}] {match['Home Team']} vs "
                  f"{match['Away Team']} (id {id_event})")

            ligne = {
                "saison": saison,
                "round": match.get("Round", ""),
                "date": match.get("strTimestamp", ""),
                "equipe_domicile": match["Home Team"],
                "equipe_exterieur": match["Away Team"],
                "score_domicile": match["Home Score"],
                "score_exterieur": match["Away Score"],
                "q1_domicile": "", "q1_exterieur": "",
                "q2_domicile": "", "q2_exterieur": "",
                "q3_domicile": "", "q3_exterieur": "",
                "q4_domicile": "", "q4_exterieur": "",
                "prolongation_domicile": 0, "prolongation_exterieur": 0,
            }

            try:
                quarts = extraire_quarts(id_event)
            except Exception as exc:  # noqa: BLE001
                print(f"   Erreur : {exc}")
                quarts = None

            if quarts is None:
                echecs.append(id_event)
            else:
                ligne.update({
                    "q1_domicile": quarts["dom"][0], "q1_exterieur": quarts["ext"][0],
                    "q2_domicile": quarts["dom"][1], "q2_exterieur": quarts["ext"][1],
                    "q3_domicile": quarts["dom"][2], "q3_exterieur": quarts["ext"][2],
                    "q4_domicile": quarts["dom"][3], "q4_exterieur": quarts["ext"][3],
                    "prolongation_domicile": quarts["prolongation_dom"],
                    "prolongation_exterieur": quarts["prolongation_ext"],
                })

            writer.writerow(ligne)
            time.sleep(1)  # pause pour rester correct envers le site

    print(f"\nSaison {saison} terminée. Fichier écrit : {fichier_sortie}")
    if echecs:
        print(f"  {len(echecs)} match(s) sans détail quart-temps trouvé "
              f"(ids: {', '.join(echecs[:20])}{'...' if len(echecs) > 20 else ''})")


def main():
    if len(sys.argv) < 2:
        print("Utilisation : python thesportsdb_scraper.py 2023-2024 2022-2023 ...")
        sys.exit(1)

    for saison in sys.argv[1:]:
        traiter_saison(saison)


if __name__ == "__main__":
    main()
