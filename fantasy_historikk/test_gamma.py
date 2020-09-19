import numpy as np
import pandas as pd
import re

from scipy.stats import gamma
import matplotlib.pyplot as plt

path  = 'C:/Users/Eirik/Documents/Python_Scripts/Fantasy/fantasy_historikk'

gamma_shape,loc,gamma_scale_prior = (28.546715229170655, 0, 4.918714525643943) 

rate_hyp_init_shape = 250

rate_hyp_init_rate = rate_hyp_init_shape*gamma_scale_prior

hyp_param = (rate_hyp_init_shape,0,rate_hyp_init_rate**(-1))

pct = [1/35,0.1,0.25,0.5,0.75,0.9,1-(1/35)]
n_vals = len(pct)
n_rows = 2
fig,ax = plt.subplots(n_rows,int(np.ceil(n_vals/n_rows)))
x = np.linspace(50, 250, 100)
for i in range(n_vals):
    rate = gamma.ppf(pct[i], *hyp_param)
    point_pdf = gamma.pdf(x, gamma_shape,0,rate**(-1))
    ax[i % n_rows,i//n_rows].plot(x, point_pdf, color='r')
    ax[i % n_rows,i//n_rows].set_title(f'{pct[i]}%')
    ax[i % n_rows,i//n_rows].set_xlim(50,250)
    print(f'{pct[i]:.3f}%: {gamma.mean(gamma_shape,0,rate**(-1))}')
fig.show()

schedules = pd.read_pickle(f"{path}/schedule.pkl")

data =  schedules.loc[(schedules['Year'] >= 2015) & (schedules['Team'] != 'Average Joes'),:]

yearly_pt_average = data.groupby(by=['Year','Team'])['Pts For'].mean().sort_values().reset_index()

print(yearly_pt_average)
print(f'Mean points = {yearly_pt_average["Pts For"].mean()}')
