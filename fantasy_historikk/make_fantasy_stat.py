import pandas as pd 
import numpy as np

path  = 'C:/Users/Eirik/Documents/Python_Scripts/Fantasy/fantasy_historikk'
results = pd.read_pickle(f'{path}/weekly_results.pkl')

results['Exp Win'] = (results.groupby(by=['Year','Week'])['Pts For'].rank(ascending=True)-1)/(results.groupby(by=['Year','Week'])['Pts For'].transform(np.size)-1)
luck = results.groupby(by=['Year','Team'])[['Win','Exp Win']].sum()
luck['Luck'] = luck['Win']-luck['Exp Win'] 