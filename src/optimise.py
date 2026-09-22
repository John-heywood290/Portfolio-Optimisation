from unittest import result

import numpy as np
from scipy.optimize import minimize
from .metrics import portfolio_return, portfolio_vol, sharpe_ratio

def min_variance(cov):
    """Find the long-only portfolio weights with the lowest volatility."""
    n = len(cov)                     

    result = minimize(
        fun=lambda w: portfolio_vol(w, cov),                 
        x0=np.ones(n) / n,                      
        method="SLSQP",
        bounds=[(0, 1) for _ in range(n)],                   
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1}
    )
    if not result.success:
       raise ValueError(result.message)

    return result.x

def max_sharpe(mu, cov):
    """Find the long-only portfolio weights with the highest Sharpe ratio."""
    n = len(cov)                     

    result = minimize(
        fun=lambda w: -sharpe_ratio(w, mu, cov),                 
        x0=np.ones(n) / n,                      
        method="SLSQP",
        bounds=[(0, 1) for _ in range(n)],                   
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1}
    )
    if not result.success:
       raise ValueError(result.message)

    return result.x

def efficient_portfolio(mu, cov, target_return):
    """Find the long-only portfolio weights for a given target return."""
    n = len(cov)

    result = minimize(
        fun=lambda w: portfolio_vol(w, cov),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1) for _ in range(n)],
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: portfolio_return(w, mu) - target_return}
        ]
    )
    if not result.success:
        raise ValueError(result.message)

    return result.x