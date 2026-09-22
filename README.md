# Portfolio Optimisation

Mean-variance portfolio optimisation on 10 US-listed ETFs, testing whether
optimised portfolios actually outperform a simple equal-weight allocation.

## Data

- Daily adjusted close prices from Yahoo Finance (via `yfinance`)
- Period: 1 Sept 2016 – 31 Aug 2026 (10 years, 2,511 daily returns)
- No missing values, so no forward-filling was required
- The window includes two major stress events: the 2020 Covid crash and the
  2022 rate shock

| Ticker | Asset class | Annual return | Annual volatility |
|---|---|---|---|
| SPY | US large-cap equities | 15.9% | 18.0% |
| QQQ | US tech equities | 21.5% | 22.6% |
| EFA | Developed ex-US equities | 10.5% | 17.0% |
| EEM | Emerging markets equities | 10.4% | 20.8% |
| TLT | Long-term US Treasuries | −1.4% | 14.8% |
| IEF | 7–10yr US Treasuries | 0.7% | 6.6% |
| LQD | Investment-grade corporate bonds | 2.4% | 8.7% |
| GLD | Gold | 13.2% | 16.3% |
| VNQ | US real estate (REITs) | 7.0% | 20.8% |
| DBC | Broad commodities | 11.2% | 17.9% |

Returns are annualised as mean daily return × 252; volatility as daily
standard deviation × √252.

## Correlation analysis

![Correlation heatmap](data/correlation_heatmap.png)

- The equity ETFs are highly correlated with each other (e.g. SPY–QQQ at
  [0.93]), so holding several offers little diversification.
- SPY–TLT correlation of −0.14: long-term Treasuries still diversified US
  equities over the period, though the relationship was weak. That's likely
  due in part to 2022, when stocks and bonds fell together as rates rose.
- Gold shows low correlation with equities (e.g. [0.14] with SPY), consistent
  with its role as a diversifying asset.

![Average correlation](data/avg_correlation.png)

- **Prediction for optimisation:** TLT (0.15) and DBC (0.15) have the lowest
  average correlation with the other ETFs, followed by IEF (0.18) and GLD
  (0.22), while equities cluster at 0.39–0.43. I expected the optimiser to
  lean on bonds, commodities and gold for diversification, and to favour one
  equity ETF over holding several.

## Diversification benefit

An equal-weight portfolio (10% in each ETF) is used as the benchmark.

| Portfolio | Return | Volatility | Sharpe |
|---|---|---|---|
| Equal weight | 9.1% | 10.6% | 0.86 |
| SPY only | 15.9% | 18.0% | 0.88 |

- The equal-weight return is the simple average of the ten ETFs, but its
  volatility (10.6%) is 35% below their average volatility (16.4%). That
  reduction comes purely from imperfect correlations between assets.
- Over this decade, simply holding SPY was slightly better risk-adjusted than
  naive diversification, reflecting the exceptional run in US equities.

## Random portfolios (Monte Carlo)

![Monte Carlo](data/monte_carlo.png)

5,000 random long-only portfolios, with weights drawn from a Dirichlet
distribution so that all weight combinations are equally likely. The
equal-weight portfolio (red) sits inside the cloud rather than on its upper
edge, showing that better risk/return combinations exist.

## Optimised portfolios

Optimised with `scipy.optimize.minimize` (SLSQP), long-only (weights between
0 and 1) and fully invested (weights sum to 1).

### Minimum variance

| Return | Volatility | Sharpe |
|---|---|---|
| 3.3% | 5.6% | 0.59 |

Weights: IEF 79.3%, DBC 11.1%, SPY 9.6%, all others 0%.

- Volatility of 5.6% is below even the least volatile single ETF (IEF, 6.6%):
  combining low-correlation assets reduces risk below any one of them alone.
- LQD and TLT receive nothing despite being bonds. Both are more volatile
  than IEF and highly correlated with it, so the optimiser picks the best
  asset from a correlated group and ignores the rest.
- **Prediction partly confirmed:** bonds and commodities were used, but only
  one bond fund, and gold received no weight.
- The result is concentrated (7 of 10 assets at 0%) and has a lower Sharpe
  ratio than equal weight. Minimising risk pushed the portfolio into bonds,
  which had a poor decade.

## Assumptions

- Sharpe ratios assume a risk-free rate of 0. Using actual T-bill rates would
  lower all Sharpe ratios.
- All statistics so far are **in-sample**: estimated and evaluated on the same
  data. An out-of-sample test follows.

## Optimisation vs Random Portfolio Findings## 

- The best of 5,000 random portfolios reached a Sharpe ratio of 1.15, against the optimiser's 1.21. The optimal portfolios are concentrated in a few assets, which random sampling rarely produces, so even a large random search falls short. This is why optimisation is used rather than random search