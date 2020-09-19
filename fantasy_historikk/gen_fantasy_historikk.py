# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:20:09 2019

@author: Eirik
"""

import re
import pandas as pd
import numpy as np

pattern = re.compile(
    r"(\d{1,2})\s+(.*)\s+(Win|Loss)\s+(\d{1,3}\.\d{2}) - (\d{1,3}\.\d{2})"
)

weekly_results = []

path = "C:/Users/Eirik/Documents/Python_Scripts/Fantasy/fantasy_historikk"

for year in range(2008, 2019):

    with open(f"{path}/fantasy_{year}.txt") as f:
        lines = f.readlines()

    lines = [pattern.findall(line)[0] for line in lines]

    cols = ["Week", "Opponent", "Result", "Pts For", "Pts Against"]

    df = pd.DataFrame(lines, columns=cols,)

    df["Week"] = df["Week"].astype(int)
    df["Pts For"] = df["Pts For"].astype(float)
    df["Pts Against"] = df["Pts Against"].astype(float)
    df["Year"] = year
    df["Opponent"] = df["Opponent"].str.strip()
    df["Result"] = df["Result"] == "Win"
    df["Result"] = df["Result"].astype(int)
    df = df.rename(columns={"Result": "Win"})
    weeks = df["Week"].max()

    teams = df["Opponent"].unique()
    n_teams = len(teams)

    df["Team"] = np.NaN
    for i in range(n_teams):
        team = set(teams) - set(
            df.loc[i * weeks : (i + 1) * weeks - 1, "Opponent"].values
        )
        if len(team) is not 1:
            raise ValueError("A unique value for team was not found")
        else:
            df.loc[i * weeks : (i + 1) * weeks - 1, "Team"] = team

    if (df.groupby("Opponent").size() != weeks).any():
        raise RuntimeError(
            "Schedule is not consistent. "
            + "Not all teams play the same number of games."
        )

    weekly_results.append(df)

weekly_results = pd.concat(weekly_results, axis=0, sort=False, ignore_index=True)
weekly_results = weekly_results[
    ["Year", "Team", "Week", "Opponent", "Win", "Pts For", "Pts Against"]
]

# Verify that 'weekly_results' is correct by computing the standings from it
# and comparing with the ones from the Yahoo website.
for year in range(2008, 2019):

    standings_from_results = weekly_results.loc[
        weekly_results["Year"] == year, :
    ].copy()
    standings_from_results["Loss"] = 1 - standings_from_results["Win"]
    standings_from_results = (
        standings_from_results.groupby("Team")["Win", "Loss", "Pts For", "Pts Against"]
        .sum()
        .sort_index()
    )

    standings_from_results[["Pts For", "Pts Against"]] = np.round(
        standings_from_results[["Pts For", "Pts Against"]], decimals=2
    )

    standings = pd.read_csv(f"{path}/fantasy_{year}_standings.txt", sep="\t")
    cols = standings.columns.values
    stripped_cols = [col.strip() for col in cols]
    standings = standings.rename(columns=dict(zip(cols, stripped_cols)))
    standings[["Win", "Loss"]] = (
        standings["W-L-T"].str.extract(r"(\d{1,2})-(\d{1,2})-(?:\d{1,2})").astype(int)
    )
    standings[["Pts For", "Pts Against"]] = np.round(
        standings[["Pts For", "Pts Against"]].astype(float), decimals=2
    )
    standings = standings[["Team", "Win", "Loss", "Pts For", "Pts Against"]]
    standings["Team"] = standings["Team"].str.strip()
    standings = standings.set_index(keys="Team", verify_integrity=True).sort_index()

    print(f"For the year {year}.")
    print("Final standings from Yahoo data:")
    print(standings)
    print("")
    print("Final standings from Yahoo results data:")
    print(standings_from_results)
    print("")

    if standings.equals(other=standings_from_results):
        print("The two standings match.\n")
    else:
        raise RuntimeError(f"Mismatch between computed and Yahoo standings for {year}.")
    print("*" * 50)


weekly_results.to_pickle("./weekly_results.pkl")
