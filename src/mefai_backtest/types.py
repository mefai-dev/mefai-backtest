"""Type definitions for the backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run.

    All monetary values are in quote currency (e.g. USDT).
    Fees are expressed as fractions (0.0004 means 4 bps).
    Leverage applies to notional position sizing.
    Funding rate is per interval and only charged while a position is open.
    Liquidation threshold is the fraction of equity loss that triggers forced close.

    Attributes:
        initial_capital: Starting equity in quote currency.
        fee_rate: Trading fee as a fraction of notional value.
        slippage_base_bps: Base slippage in basis points before volatility scaling.
        leverage: Position leverage multiplier.
        funding_rate: Per interval funding rate charged on open positions.
        liquidation_threshold: Equity drawdown fraction that triggers liquidation.
        timeframe_minutes: Bar duration in minutes (used for annualized Sharpe).
        volatility_lookback: Number of bars for rolling volatility estimate.
    """

    initial_capital: float = 10_000.0
    fee_rate: float = 0.0004
    slippage_base_bps: float = 1.0
    leverage: float = 1.0
    funding_rate: float = 0.0001
    liquidation_threshold: float = 0.90
    timeframe_minutes: int = 60
    volatility_lookback: int = 20


@dataclass
class TradeRecord:
    """A single completed trade.

    Attributes:
        entry_bar: Bar index where the trade was opened.
        exit_bar: Bar index where the trade was closed.
        direction: 1 for long and -1 for short.
        entry_price: Price at entry after slippage.
        exit_price: Price at exit after slippage.
        pnl: Realized profit or loss in quote currency.
        fees_paid: Total fees paid for this round trip.
        duration_bars: Number of bars the trade was held.
        exit_reason: Why the trade was closed (signal or liquidation).
    """

    entry_bar: int = 0
    exit_bar: int = 0
    direction: int = 1
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    fees_paid: float = 0.0
    duration_bars: int = 0
    exit_reason: str = "signal"


@dataclass
class BacktestResult:
    """Complete results from a backtest run.

    Contains equity curve and trade log and summary statistics.
    All arrays are numpy ndarrays aligned to the input price series.

    Attributes:
        equity_curve: Equity value at each bar.
        returns: Per bar fractional returns.
        trades: List of completed trade records.
        total_pnl: Net profit or loss.
        total_fees: Sum of all fees paid.
        sharpe_ratio: Annualized Sharpe ratio adjusted for timeframe.
        max_drawdown: Maximum peak to trough drawdown as a fraction.
        win_rate: Fraction of trades that were profitable.
        num_trades: Total number of completed trades.
        liquidations: Number of times liquidation was triggered.
        avg_trade_duration: Mean trade duration in bars.
        config: The configuration used for this run.
        extra: Arbitrary metadata dict.
    """

    equity_curve: Any = None
    returns: Any = None
    trades: list[TradeRecord] = field(default_factory=list)
    total_pnl: float = 0.0
    total_fees: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    liquidations: int = 0
    avg_trade_duration: float = 0.0
    config: BacktestConfig | None = None
    extra: dict[str, Any] = field(default_factory=dict)
