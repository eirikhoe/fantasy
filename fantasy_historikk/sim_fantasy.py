import numpy as np
import pandas as pd
import re

from scipy.stats import gamma
import matplotlib.pyplot as plt
import time
import pickle


def seconds_to_str(time):
    hours, time = divmod(time, 3600)
    minutes, time = divmod(time, 60)
    seconds, milliseconds = divmod(time, 1)
    milliseconds = milliseconds * 1000
    return "{h:d}:{m:02d}:{s:02d}.{ms:03d}".format(
        h=int(hours), m=int(minutes), s=int(seconds), ms=int(milliseconds)
    )


path = "C:/Users/Eirik/Documents/Python_Scripts/Fantasy/fantasy_historikk"

schedule = pd.read_csv(f"{path}/schedule_2019.csv")

weeks_season = schedule["Week"].max()

teams = schedule["Opponent"].unique()
n_teams = len(teams)

schedule = schedule.pivot(index="Team", columns="Week", values="Opponent").sort_index()

standings = pd.read_csv(f"{path}/fantasy_curr_standings.txt", sep="\t")
cols = standings.columns.values
stripped_cols = [col.strip() for col in cols]
standings = standings.rename(columns=dict(zip(cols, stripped_cols)))
standings[["Win", "Loss"]] = (
    standings["W-L-T"].str.extract(r"(\d{1,2})-(\d{1,2})-(?:\d{1,2})").astype(int)
)
standings[["Pts For", "Pts Agnst"]] = np.round(
    standings[["Pts For", "Pts Agnst"]].astype(float), decimals=2
)
standings = standings[["Team", "Win", "Loss", "Pts For", "Pts Agnst"]]
standings["Team"] = standings["Team"].str.strip()
weeks_played = standings.loc[0, "Win"] + standings.loc[0, "Loss"]
standings = standings.sort_values(by="Team").reset_index(drop=True)

# Map schedule to numbers
team_map = dict(zip(standings["Team"].values, standings.index.values))
for col in schedule.columns.values:
    schedule[col] = schedule[col].map(team_map)
schedule = schedule.values

print(standings)

points = standings["Pts For"].values
wins = standings["Win"].values

gamma_shape, loc, gamma_scale_prior = (28.546715229170655, 0, 4.918714525643943)
rate_hyp_init_shape = 250
rate_hyp_init_rate = rate_hyp_init_shape * gamma_scale_prior

rate_hyp_shape = rate_hyp_init_shape + weeks_played * gamma_shape
rate_hyp_rate = rate_hyp_init_rate + points

make_playoffs = np.zeros(n_teams, dtype=np.int64)
n_sim = int(5e6)

start_time = time.perf_counter()

pts_week = np.zeros(n_teams)
dist = np.zeros((n_teams, n_sim * (weeks_season - weeks_played)))
n_dist = 0
leverage_num = np.zeros((n_teams,2))
leverage_den = np.zeros((n_teams,2))
for n in np.arange(n_sim):
    sim_pts = points.copy()
    sim_wins = wins.copy()
    for week in np.arange(weeks_played, weeks_season):
        for team in range(n_teams):
            pts_week[team] = gamma.rvs(
                gamma_shape,
                scale=gamma.rvs(rate_hyp_shape, scale=rate_hyp_rate[team] ** (-1))
                ** (-1),
            )
        dist[:, n_dist] = pts_week
        n_dist += 1
        win_week = (pts_week > pts_week[schedule[:, week]]).astype(int)
        sim_wins += win_week
        sim_pts += pts_week
        if week == weeks_played:
            win_initial_week = win_week
    placement = np.lexsort((sim_pts, sim_wins))[::-1]
    make_playoffs[placement[:4]] += 1
    leverage_den[np.arange(n_teams),win_initial_week] +=1
    leverage_num[placement[:4],win_initial_week[placement[:4]]] +=1

    if (not (n % 5e4)) and (n > 0):
        runtime = seconds_to_str(time.perf_counter() - start_time)
        print()
        print(f"{int(n/1e3)}k rows processed. Elapsed time: {runtime}")
        eta = seconds_to_str((n_sim - n) * (time.perf_counter() - start_time) / n)
        print(f"ETA: {eta}")
        standings["Make playoffs"] = 100 * (make_playoffs / (n+1))
        print(f"Current probabilites:")
        print(standings.set_index("Team")["Make playoffs"])

standings["Make playoffs"] = 100 * (make_playoffs / n_sim)

x = np.linspace(np.min(dist[0, :]), np.max(dist[0, :]), 100)
pdf_fitted = gamma.pdf(x, gamma_shape, loc, gamma_scale_prior)
leverage = 100*leverage_num/leverage_den

n_rows = 3
fig, ax = plt.subplots(n_rows, int(np.ceil(n_teams / n_rows)), sharex=True, sharey=True)
for i in range(n_teams):
    ax[i % n_rows, i // n_rows].hist(
        dist[i, :], bins=100, density=True, histtype="step"
    )
    ax[i % n_rows, i // n_rows].plot(x, pdf_fitted, color="r")
    ax[i % n_rows, i // n_rows].set_title(standings.loc[i, "Team"])
    ax[i % n_rows, i // n_rows].set_xlim(70, 250)
fig.show()

playoff_probs = standings.set_index("Team")["Make playoffs"]
last_week = pd.read_pickle("./last_week_prob.pkl")
change = playoff_probs - last_week

print()
print(f"Playoff odds after week {weeks_played}:")
for i,team in enumerate(playoff_probs.index.values):
    print(f"{team}: {playoff_probs[team]:.1f}% ({change[team]:+.1f} pp.)"
    + f" [{leverage[i,0]:.1f}%, {leverage[i,1]:.1f}%]")

playoff_probs.to_pickle("./last_week_prob.pkl")

