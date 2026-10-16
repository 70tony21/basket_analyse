import argparse
from pathlib import Path

import pandas as pd

"""
"Fusionne tous les fichiers CSV d'un dossier et les trie par date."
"""
def main():
    # Arguments du terminal
    parser = argparse.ArgumentParser(
        description="Fusionne tous les fichiers CSV d'un dossier et les trie par date."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Nom du fichier CSV de sortie."
    )

    args = parser.parse_args()

    # Dossier contenant les CSV
    dossier = Path(".")

    # Fichier de sortie
    fichier_output = Path(args.output)

    # Rechercher tous les CSV du dossier
    fichiers_csv = list(dossier.glob("*.csv"))

    # Exclure le fichier de sortie
    fichiers_csv = [
        fichier
        for fichier in fichiers_csv
        if fichier.resolve() != fichier_output.resolve()
    ]

    if not fichiers_csv:
        raise FileNotFoundError(
            "Aucun fichier CSV trouvé dans le dossier."
        )

    # Lire les fichiers
    dfs = []

    for fichier in sorted(fichiers_csv):
        print(f"Lecture : {fichier.name}")
        df = pd.read_csv(fichier)
        dfs.append(df)

    # Fusionner
    fusion = pd.concat(dfs, ignore_index=True)

    # Convertir la colonne date
    fusion["date"] = pd.to_datetime(
        fusion["date"],
        errors="coerce"
    )

    # Trier par date croissante
    fusion = fusion.sort_values(
        by="date",
        ascending=True
    )

    # Sauvegarder
    fusion.to_csv(fichier_output, index=False)

    print()
    print(f"{len(fichiers_csv)} fichiers fusionnés.")
    print(f"{len(fusion)} lignes au total.")
    print(f"Fichier créé : {fichier_output}")


if __name__ == "__main__":
    main()
