import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
 
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(symbols, start='2020-01-01', end='2025-01-01')['Close']
returns = data.pct_change().dropna()
 
cov = returns.cov().values * 252 # annualisation
n = len(symbols)
 
def portfolio_volatility(w, cov):
    return np.sqrt(w.dot(cov).dot(w))
 
def risk_contributions(w, cov):
    port_vol = portfolio_volatility(w, cov)
    # RC_i = w_i * (Σ w)_i / port_vol
    marginal = cov.dot(w)
    return w * marginal / port_vol

# Solveur risk parity par méthode log-barrière
def solve_risk_parity_barrier(cov, x0, k=10.0, beta=100.0, tol=1e-6, max_iter=50):
    w = x0.copy()
    for it in range(max_iter): 
        port_vol = portfolio_volatility(w, cov)
        rc = risk_contributions(w, cov)
        target = port_vol / n
        grad_f = 2.0 * ((rc - target) * (cov.dot(w)) / port_vol - (rc - target) * w.dot(cov).dot(rc) / (port_vol**3))
        hess_f = np.diag(2.0 * ((cov.dot(w)/port_vol)**2))
        grad_bar = -k * (1.0 / w)
        hess_bar = np.diag(k * (1.0 / w**2))
        s = np.sum(w) - 1.0
        grad_pen = 2 * beta * s * np.ones(n)
        hess_pen = 2 * beta * np.ones((n, n))
        grad_L = grad_f + grad_bar + grad_pen
        hess_L = hess_f + hess_bar + hess_pen
        dw = -np.linalg.solve(hess_L, grad_L)
        alpha = 1.0
        neg_idx = dw < 0
        if np.any(neg_idx):
            alpha = min(alpha, np.min(-w[neg_idx] / dw[neg_idx]) * 0.99)
        w += alpha * dw
        if np.linalg.norm(dw) < tol:
            break
    return w

# Initialisation
w0 = np.ones(n) / n
w_rp = solve_risk_parity_barrier(cov, w0)
w_rp /= np.sum(w_rp)

rc = risk_contributions(w_rp, cov)
plt.figure(figsize=(8,5))
plt.bar(symbols, w_rp, label='Poids')
plt.ylabel('Poids')
plt.title('Portefeuille Risk Parity via Log-Barrier')
plt.grid(axis='y')
plt.savefig('risk_parity_weights.png')

plt.figure(figsize=(8,5))
plt.bar(symbols, rc, color='orange', label='Risk Contribution')
plt.ylabel('Contribution au risque')
plt.title('Contributions au Risque')
plt.grid(axis='y')
plt.savefig('risk_parity_contributions.png')