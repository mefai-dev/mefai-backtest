"""Performance statistics and report formatting.

Provides functions for computing risk adjusted metrics and
generating human readable performance summaries.
"""

from __future__ import annotations

import numpy as np
import structlog

from .types import BacktestResult

logger = structlog.get_logger(__name__)


def annualized_sharpe(
    returns: np.ndarray,
    timeframe_minutes: int = 60,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio scaled by timeframe.

    The annualization factor is derived from the number of bars per year.
    A year is assumed to be 365.25 days of continuous trading (crypto markets).

    Args:
        returns: Array of per bar fractional returns.
        timeframe_minutes: Duration of each bar in minutes.
        risk_free_rate: Annualized risk free rate (default 0).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if standard deviation is zero.
    """
    if len(returns) < 2:
        return 0.0

    bars_per_year = (365.25 * 24 * 60) / timeframe_minutes
    excess = returns - (risk_free_rate / bars_per_year)
    std = np.std(excess, ddof=1)

    if std == 0.0:
        return 0.0

    return float(np.mean(excess) / std * np.sqrt(bars_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown as a fraction of peak equity.

    Args:
        equity_curve: Array of equity values over time.

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.25 means 25% drawdown).
    """
    if len(equity_curve) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / np.where(
        running_max > 0, running_max, 1.0
    )
    return float(np.max(drawdowns))


def calmar_ratio(total_return: float, max_dd: float) -> float:
    """Compute Calmar ratio (annualized return over max drawdown).

    Args:
        total_return: Total fractional return.
        max_dd: Maximum drawdown as a fraction.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    if max_dd == 0.0:
        return 0.0
    return total_return / max_dd


def profit_factor(trades: list) -> float:
    """Compute profit factor (gross profit divided by gross loss).

    Args:
        trades: List of TradeRecord objects.

    Returns:
        Profit factor. Returns 0.0 if there are no losing trades.
    """
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def format_report(result: BacktestResult) -> str:
    """Generate a human readable performance report.

    Args:
        result: A completed BacktestResult object.

    Returns:
        Multiline string with formatted performance metrics.
    """
    cfg = result.config
    initial = cfg.initial_capital if cfg else 10_000.0
    total_return = result.total_pnl / initial if initial > 0 else 0.0
    pf = profit_factor(result.trades)

    lines = [
        "=" * 60,
        "  BACKTEST PERFORMANCE REPORT",
        "=" * 60,
        "",
        f"  Initial Capital:      {initial:>14.2f}",
        f"  Final Equity:         {initial + result.total_pnl:>14.2f}",
        f"  Total PnL:            {result.total_pnl:>14.2f}",
        f"  Total Return:         {total_return:>13.2%}",
        f"  Total Fees Paid:      {result.total_fees:>14.2f}",
        "",
        f"  Sharpe Ratio:         {result.sharpe_ratio:>14.4f}",
        f"  Max Drawdown:         {result.max_drawdown:>13.2%}",
        f"  Profit Factor:        {pf:>14.4f}",
        "",
        f"  Number of Trades:     {result.num_trades:>14d}",
        f"  Win Rate:             {result.win_rate:>13.2%}",
        f"  Avg Trade Duration:   {result.avg_trade_duration:>14.1f} bars",
        f"  Liquidations:         {result.liquidations:>14d}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)
