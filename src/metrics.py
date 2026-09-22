import numpy as np
def portfolio_return(weights, mu):
    """
    Calculate the expected return of a portfolio.

    Parameters:
    weights (numpy.ndarray): An array of weights for each asset in the portfolio.
    mu (numpy.ndarray): An array of expected returns for each asset.

    Returns:
    float: The expected return of the portfolio.
    """
    return np.dot(weights, mu)

def portfolio_vol(weights,cov):
    """
    Calculate the volatility (standard deviation) of a portfolio.

    Parameters:
    weights (numpy.ndarray): An array of weights for each asset in the portfolio.
    cov (numpy.ndarray): The covariance matrix of asset returns.

    Returns:
    float: The volatility of the portfolio.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

def sharpe_ratio(weights, mu, cov, risk_free_rate=0):
    """
    Calculate the Sharpe ratio of a portfolio.

    Parameters:
    weights (numpy.ndarray): An array of weights for each asset in the portfolio.
    mu (numpy.ndarray): An array of expected returns for each asset.
    cov (numpy.ndarray): The covariance matrix of asset returns.
    risk_free_rate (float): The risk-free rate of return. Default is 0.

    Returns:
    float: The Sharpe ratio of the portfolio.
    """
    port_return = portfolio_return(weights, mu)
    port_vol = portfolio_vol(weights, cov)
    return (port_return - risk_free_rate) / port_vol

def max_drawdown(daily_returns):
    """Largest peak-to-trough fall of a daily return series."""
    wealth = (1 + daily_returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return drawdown.min()