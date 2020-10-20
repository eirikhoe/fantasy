import numpy as np
import pandas as pd

from scipy.stats import gamma
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent


schedules = pd.read_pickle(path / "weekly_results.pkl")

# Current scoring setting started in 2015
data =  schedules.loc[(schedules['Year'] >= 2015) & (schedules['Team'] != 'Average Joes'),'Pts For']
param = gamma.fit(data,floc=0)

x = np.linspace(data.min(), data.max(), 100)
pdf_fitted = gamma.pdf(x, *param)
plt.plot(x, pdf_fitted, color='r')
plt.hist(data, density=True, bins=25)
plt.show()

print(param)