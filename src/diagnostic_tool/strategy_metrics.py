from sys import pycache_prefix

import numpy as np
from typing import Union, List, Dict
import warnings

import pathlib
import shutil

def check_sharpe_ratio(
        returns: Union[List[float], np.ndarray],
        risk_free_rate: Union[float, np.ndarray] = 0.0,
        threshold: float = 1.0,
        verbose: bool = False
) -> Dict[str, object]:
    """
    Calculates the Sharpe Ratio of a strategy and evaluates it against a threshold (return per unit of volatility).

    Args:
        returns: The returns of the strategy (list or np.ndarray).
        risk_free_rate: The risk-free rate. Defaults to 0.0.
        threshold: Minimum acceptable Sharpe ratio. Defaults to 1.0.

    Returns:
        dict: A dictionary containing the Sharpe Ratio and a pass/fail flag.
    """
    returns = np.atleast_1d(returns).astype(float)
    # Handle scalar risk-free rate
    if np.isscalar(risk_free_rate):
        excess_returns = returns - risk_free_rate
    else:
        risk_free_rate = np.atleast_1d(risk_free_rate).astype(float)
        excess_returns = returns - risk_free_rate
    std = np.std(excess_returns, ddof=1)
    sharpe_ratio = np.mean(excess_returns) / std if std > 0 else np.nan

    sharpe_results= {
        "metric": "Sharpe Ratio",
        "value": round(sharpe_ratio, 3),
        "pass": sharpe_ratio >= threshold
    }
    if verbose:
        print(f"metric: {sharpe_results['metric']}\n value: {sharpe_results['value']}\n pass: {sharpe_results['pass']}")
    return sharpe_results
def check_max_drawdown(equity_curve: Union[List[float], np.ndarray]) -> float:
    """
    Calculates the maximum drawdown of an equity curve.

    Args:
        equity_curve: The equity curve of the strategy (list or np.ndarray).

    Returns:
        float: The maximum drawdown as a percentage.
    """
    equity_curve = np.atleast_1d(equity_curve).astype(float)
    peak = np.maximum.accumulate(equity_curve)
    print(f"Equity Curve: {equity_curve}")
    print(f"Peak: {peak}")
    print(f"Any zeros in peak: {np.any(peak == 0)}")
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'): # A context manager to temporarily control how NumPy handles floating-point errors
        drawdown = np.where(peak > 0, (equity_curve - peak) / peak, 0)
    max_drawdown_value = np.min(drawdown)
    return round(max_drawdown_value * 100, 2)


def check_calmar_ratio(
        returns: Union[List[float], np.ndarray],
        risk_free_rate: Union[float, np.ndarray] = 0.0,
        threshold: float = 1.0
) -> Dict[str, object]:
    """
    Calculates the Calmar Ratio of a strategy and evaluates it against a threshold (return per unit of maximum drawdown).

    Args:
        returns: The returns of the strategy (list or np.ndarray).
        risk_free_rate: The risk-free rate. Defaults to 0.0.
        threshold: Minimum acceptable Calmar ratio. Defaults to 1.0.

    Returns:
        dict: A dictionary containing the Calmar Ratio and a pass/fail flag.
    """
    annual_factor = 252  # Assuming daily returns
    returns = np.atleast_1d(returns).astype(float)
    risk_free_rate = np.atleast_1d(risk_free_rate).astype(float)
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    annualised_return = mean_return * annual_factor
    max_dd = check_max_drawdown(excess_returns)
    calmar_ratio = annualised_return / max_dd if max_dd != 0 else np.nan

    return {
        "metric": "Calmar Ratio",
        "value": round(calmar_ratio, 3),
        "pass": calmar_ratio >= threshold
    }

def check_sortino_ratio(
        returns: Union[List[float],np.ndarray],
        risk_free_rate: Union[float, np.array] = 0.0,
        threshold: float = 1.0,
) -> Dict[str, object]:
    """
    Calculates the Sortino Ratio of a strategy and evaluates it against a threshold (return per unit of downside risk).

    Args:
        returns: The returns of the strategy (list or np.ndarray).
        risk_free_rate: The risk-free rate. Defaults to 0.0.
        threshold: Minimum acceptable Sortino ratio. Defaults to 1.0.
    Returns:
        dict: A dictionary containing the Sortino Ratio and a pass/fail flag.
    """
    returns = np.atleast_1d(returns).astype(float)
    risk_free_rate = np.atleast_1d(risk_free_rate).astype(float)
    excess_returns = returns - risk_free_rate
    downside_returns = np.minimum(excess_returns,0)
    downside_std = np.sqrt(np.mean(downside_returns**2))
    sortino_ratio = np.mean(excess_returns)/ downside_std if downside_std > 0 else np.nan
    return {
        "metric": "Sortino Ratio",
        "value": round(sortino_ratio, 3),
        "pass": sortino_ratio >= threshold
    }

def check_omega_ratio(
        returns: Union[List[float],np.ndarray],
        risk_free_rate: Union[float, np.ndarray] = 0.0,
        threshold: float = 1.0,

) -> Dict[str, object]:
    '''
    Calculates the Omega Ratio of a strategy and evaluates it against a threshold (probability-weighted return per unit of loss). It considers all
    moments - mean, variance, skewness and kurtosis - of the return distribution.
    Args:
        returns: The returns of the strategy (list or np.ndarray).
        risk_free_rate: The risk-free rate. Defaults to 0.0.
        threshold: Minimum acceptable Omega ratio. Defaults to 1.0.
    Returns:
        dict: A dictionary containing the Omega Ratio and a pass/fail flag.
    '''
    returns = np.atleast_1d(returns).astype(float)
    risk_free_rate = float(np.mean(np.atleast_1d(risk_free_rate)))
    excess_returns = returns - risk_free_rate
    gains = excess_returns[excess_returns >= 0]
    losses = excess_returns[excess_returns < 0]
    omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else np.nan
    return {
        "metric": "Omega Ratio",
        "value": round(omega_ratio, 3),
        "pass": omega_ratio >= threshold
    }

def check_cvar(
        returns: Union[List[float],np.ndarray],
        risk_free_rate: Union[float, np.ndarray] = 0.0,
        threshold: float = 0.0
) -> Dict[str, object]:
    '''
    Calculates the Conditional Value at Risk (CVaR) of a strategy and evaluates it against a threshold (expected loss in the worst-case scenario).
    Args:
        returns: The returns of the strategy (list or np.ndarray).
        risk_free_rate: The risk-free rate. Defaults to 0.0.
        threshold: Minimum acceptable CVaR. Defaults to 0.0.
    Returns:
        dict: A dictionary containing the CVaR and a pass/fail flag.
    '''
    returns = np.atleast_1d(returns).astype(float)
    risk_free_rate = float(np.mean(np.atleast_1d(risk_free_rate)))
    excess_returns = returns - risk_free_rate
    sorted_returns = np.sort(excess_returns)
    n = len(sorted_returns)
    if n == 0:
        warnings.warn("No returns data provided for CVaR calculation.")
        return {"metric": "CVaR", "value": np.nan, "pass": False}
    threshold_index = int(0.05 * n)  # 5% worst-case scenario
    if threshold_index >= n:
        warnings.warn("Not enough data to calculate CVaR at the 5% level.")
        return {"metric": "CVaR", "value": np.nan, "pass": False}
    cvar_value = np.mean(sorted_returns[:threshold_index])
    return {
        "metric": "CVaR",
        "value": round(cvar_value, 3),
        "pass": cvar_value >= threshold
    }





