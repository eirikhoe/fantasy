import numpy as np
import pandas as pd
import re

from scipy.stats import gamma
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent

gamma_shape, loc, gamma_scale_prior = (28.911131362547057, 0, 4.8412050199817385)

rate_hyp_init_shape = 260

rate_hyp_init_rate = rate_hyp_init_shape * gamma_scale_prior

hyp_param = (rate_hyp_init_shape, 0, rate_hyp_init_rate ** (-1))

n = int(1e5)
p = np.zeros(n)
for i in range(n):
    p[i] = gamma.rvs(
                gamma_shape,
                scale=gamma.rvs(rate_hyp_init_shape, scale=rate_hyp_init_rate ** (-1))
                ** (-1),
            )
print(p[p>150.28].shape[0]/p.shape[0])

pct = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
n_vals = len(pct)
n_rows = 2
fig, ax = plt.subplots(n_rows, int(np.ceil(n_vals / n_rows)))
x = np.linspace(50, 250, 100)
for i in range(n_vals):
    rate = gamma.ppf(pct[i], *hyp_param)
    point_pdf = gamma.pdf(x, gamma_shape, 0, rate ** (-1))
    ax[i % n_rows, i // n_rows].plot(x, point_pdf, color="r")
    ax[i % n_rows, i // n_rows].set_title(f"{pct[i]}%")
    ax[i % n_rows, i // n_rows].set_xlim(50, 250)
    print(f"{pct[i]:.3f}%: {gamma.mean(gamma_shape,0,rate**(-1))}")
fig.show()

schedules = pd.read_pickle(f"{path}/weekly_results.pkl")

data = schedules.loc[
    (schedules["Year"] >= 2015) & (schedules["Team"] != "Average Joes"), :
]

yearly_pt_average = (
    data.groupby(by=["Year", "Team"])["Pts For"].mean().sort_values().reset_index()
)

yearly_pt_prediction = yearly_pt_average["Pts For"].values.copy()
n = len(yearly_pt_prediction)
for i in np.arange(n):
    rate = gamma.ppf(1 - ((i + 1) / (n + 1)), *hyp_param)
    yearly_pt_prediction[i] = gamma.mean(gamma_shape, 0, rate ** (-1))
yearly_pt_average["Prediction"] = yearly_pt_prediction
yearly_pt_average["Error"] = np.abs(yearly_pt_average["Pts For"] - yearly_pt_average["Prediction"])

print(yearly_pt_average)
print(f'Mean error = {yearly_pt_average["Error"].mean()}')