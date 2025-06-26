import numpy as np
from typing import Union, List, Dict


def check_sharpe_ratio(
        returns: Union[List[float], np.ndarray],
        risk_free_rate: Union[float, np.ndarray] = 0.0,
        threshold: float = 1.0
) -> Dict[str, object]:
    """
    Calculates the Sharpe Ratio of a strategy and evaluates it against a threshold.

    Args:
        returns: The returns of the strategy (list or np.ndarray).
        risk_free_rate: The risk-free rate. Defaults to 0.0.
        threshold: Minimum acceptable Sharpe ratio. Defaults to 1.0.

    Returns:
        dict: A dictionary containing the Sharpe Ratio and a pass/fail flag.
    """
    returns = np.atleast_1d(returns).astype(float)
    risk_free_rate = np.atleast_1d(risk_free_rate).astype(float)
    excess_returns = returns - risk_free_rate
    std = np.std(excess_returns, ddof=1)
    sharpe_ratio = np.mean(excess_returns) / std if std > 0 else np.nan

    return {
        "metric": "Sharpe Ratio",
        "value": round(sharpe_ratio, 3),
        "pass": sharpe_ratio >= threshold
    }

def max_drawdown(equity_curve: Union[List[float], np.ndarray]) -> float:
    """
    Calculates the maximum drawdown of an equity curve.

    Args:
        equity_curve: The equity curve of the strategy (list or np.ndarray).

    Returns:
        float: The maximum drawdown as a percentage.
    """
    equity_curve = np.atleast_1d(equity_curve).astype(float)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown_value = np.min(drawdown)
    return round(max_drawdown_value * 100, 2)




