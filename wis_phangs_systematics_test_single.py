import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wis = pd.read_csv('/Users/administrator/Astro/LLAMA/ALMA/comp_samples/wis_new.csv')
phangs = pd.read_csv('/Users/administrator/Astro/LLAMA/ALMA/comp_samples/phangs_new.csv')
wis_metrics = pd.read_csv('/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0_metrics_wis.csv')
phangs_metrics = pd.read_csv('/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0_metrics_phangs.csv')

wis = pd.merge(wis_metrics, wis, left_on='name', right_on='Name',how='left')
phangs = pd.merge(phangs_metrics, phangs, left_on='name', right_on='Name',how='left')



for column, df in zip(['Gini','Asymmetry','Smoothness_davis_y'],[wis,phangs]):
    x_col = column+'_x'
    y_col = column+'_y'
    x = df[x_col]
    y = df[y_col]
    plt.scatter(x,y)
    plt.axis([0,1,0,1])
    plt.plot([0, 1], [0, 1], "k--", label="1:1")
    plt.show()








