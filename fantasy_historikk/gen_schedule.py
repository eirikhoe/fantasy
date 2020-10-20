import numpy as np
import pandas as pd
import re
from pathlib import Path


def secondsToStr(time):
    hours, time = divmod(time, 3600)
    minutes, time = divmod(time, 60)
    seconds, milliseconds = divmod(time, 1)
    milliseconds = milliseconds * 1000
    return "{h:d}:{m:02d}:{s:02d}.{ms:03d}".format(
        h=int(hours), m=int(minutes), s=int(seconds), ms=int(milliseconds)
    )


year = 2020

path = Path(__file__).parent
pattern = re.compile(r"^(\d{1,2})\s+((?:\S+ )+)")

with open(path / f"fantasy_{year}.txt") as f:
    lines = f.readlines()

lines = [pattern.findall(line)[0] for line in lines]

cols = ["Week", "Opponent"]

schedule = pd.DataFrame(lines, columns=cols,)

schedule["Week"] = schedule["Week"].astype(int)
schedule["Opponent"] = schedule["Opponent"].str.strip()
weeks_season = schedule["Week"].max()

teams = schedule["Opponent"].unique()
n_teams = len(teams)

schedule["Team"] = np.NaN
for i in range(n_teams):
    team = set(teams) - set(
        schedule.loc[i * weeks_season : (i + 1) * weeks_season - 1, "Opponent"].values
    )
    if len(team) is not 1:
        raise ValueError("A unique value for team was not found")
    else:
        schedule.loc[i * weeks_season : (i + 1) * weeks_season - 1, "Team"] = team.pop()

if (schedule.groupby("Opponent").size() != weeks_season).any():
    raise RuntimeError(
        "Schedule is not consistent. " + "Not all teams play the same number of games."
    )


schedule.to_csv(f"schedule_{year}.csv", index=False)

