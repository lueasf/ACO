import numpy as np
import yfinance as yf
from scipy.optimize import minimize
 
symbols = ['AAPL','MSFT','GOOGL','AMZN']
data = yf.download(symbols, '2022-01-01','2025-01-01')['Close']
rets = data.pct_change().dropna()
cov = rets.cov().values * 252
n = len(symbols)
 
def risk_contributions(w, cov):
    vol = np.sqrt(w @ cov @ w)
    return w * (cov @ w) / vol
 
def rp_obj(w, cov):
    rc = risk_contributions(w, cov)
    return np.sum((rc - rc.mean())**2)
 
cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1},)
bounds = [(0,1)]*n
# Solve
w0 = np.ones(n)/n
res = minimize(rp_obj, w0, args=(cov,), method='SLSQP',
               bounds=bounds, constraints=cons,
               options={'ftol':1e-12,'disp':True})
w_rp = res.x

import matplotlib.pyplot as plt
plt.bar(symbols, w_rp); plt.title("RP weights"); plt.savefig("rp_weights.png")