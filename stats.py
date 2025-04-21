import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
def get_spearman(df, x, y):
    temp = df[[x, y]].dropna()
    # Compute Spearman correlation
    print (f"{y} vs {x}")
    corr, p_value = spearmanr(temp[x], temp[y])
    print(f"Spearman Correlation: {corr:.3f}, p-value: {p_value:.3f}")
    return corr, p_value

def get_granger_causality(df, x, y, maxlag = 7, verbose = False):
# Perform Granger causality test (using up to 3 lags)
    temp = df[[x, y]].dropna()
    results= grangercausalitytests(temp, maxlag=maxlag, verbose = verbose)
    sig_results = {}
    for lag, stats in results.items():
        for test, vals in stats[0].items():
            if vals[1] < 0.05:
                try:
                    sig_results[lag][test] = vals
                except KeyError:
                    sig_results[lag] = {}
                    sig_results[lag][test] = vals
    return sig_results