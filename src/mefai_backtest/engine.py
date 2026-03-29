"""Vectorized backtesting engine with realistic market simulation.

This module implements a production grade backtesting engine designed
for crypto perpetual futures. It includes ten key improvements over
naive backtesting approaches:

1. Notional fee model: fees scale with position size not just capital
2. Compounding equity: position sizes grow and shrink with equity
3. Liquidation detection: forced close when equity drops below threshold
4. Timeframe aware Sharpe: annualization factor adapts to bar duration
5. Funding only when position open: no phantom funding on flat periods
6. Volatility scaled slippage: slippage increases with recent volatility
7. Trade duration tracking: records how long each trade was held
8. Full docstrings: every method is documented
9. Structured logging: all events logged via structlog
10. Walk forward integration: results are compatible with WalkForwardOptimizer
"""

from __future__ import annotations

import numpy as np
import structlog

from .stats import annualized_sharpe, max_drawdown
from .types import BacktestConfig, BacktestResult, TradeRecord

logger = structlog.get_logger(__name__)


class BacktestEngine:
    """Vectorized backtesting engine for crypto perpetual futures.

    Processes a price series and a signal series bar by bar with
    realistic fee and slippage and funding and liquidation modeling.

    The signal array should contain:
        1  = long
       -1  = short
        0  = flat (close any open position)

    Usage::

        engine = BacktestEngine(config)
        result = engine.run(prices, signals)

    Args:
        config: BacktestConfig with simulation parameters.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Initialize the engine with a configuration.

        Args:
            config: Simulation parameters. Uses defaults if None.
        """
        self.config = config or BacktestConfig()
        logger.info(
            "engine.init",
            capital=self.config.initial_capital,
            fee_rate=self.config.fee_rate,
            leverage=self.config.leverage,
        )

    def _compute_volatility(self, prices: np.ndarray) -> np.ndarray:
        """Compute rolling volatility for slippage scaling.

        Uses log returns with a lookback window defined in config.
        The first few bars use expanding window until enough data exists.

        Args:
            prices: Array of close prices.

        Returns:
            Array of rolling volatility values (same length as prices).
        """
        lookback = self.config.volatility_lookback
        log_returns = np.zeros(len(prices))
        log_returns[1:] = np.log(prices[1:] / prices[:-1])

        vol = np.zeros(len(prices))
        for i in range(1, len(prices)):
            start = max(0, i - lookback)
            window = log_returns[start : i + 1]
            if len(window) > 1:
                vol[i] = np.std(window, ddof=1)

        # Replace zeros with a small default to avoid zero slippage everywhere
        median_vol = np.median(vol[vol > 0]) if np.any(vol > 0) else 0.001
        vol[vol == 0] = median_vol

        return vol

    def _apply_slippage(
        self, price: float, direction: int, volatility: float
    ) -> float:
        """Apply volatility scaled slippage to a fill price.

        Slippage is proportional to base_bps scaled by the ratio of
        current volatility to a baseline of 1%.

        Args:
            price: Raw price before slippage.
            direction: 1 for buy and -1 for sell.
            volatility: Current rolling volatility.

        Returns:
            Adjusted fill price after slippage.
        """
        baseline_vol = 0.01
        vol_ratio = volatility / baseline_vol if baseline_vol > 0 else 1.0
        slippage_bps = self.config.slippage_base_bps * vol_ratio
        slippage_frac = slippage_bps / 10_000.0

        # Buys get worse (higher) price and sells get worse (lower) price
        return price * (1.0 + direction * slippage_frac)

    def _compute_fee(self, notional_value: float) -> float:
        """Compute trading fee based on notional position value.

        This is improvement #1: fees scale with the actual notional
        value of the trade rather than just the capital deployed.

        Args:
            notional_value: Absolute notional value of the trade.

        Returns:
            Fee amount in quote currency.
        """
        return abs(notional_value) * self.config.fee_rate

    def run(self, prices: np.ndarray, signals: np.ndarray) -> BacktestResult:
        """Execute a full backtest over the given price and signal series.

        Iterates bar by bar tracking equity and positions and fees.
        Position sizes compound with equity (improvement #2).
        Liquidation is checked each bar (improvement #3).
        Funding is only charged when a position is open (improvement #5).

        Args:
            prices: 1D numpy array of close prices.
            signals: 1D numpy array of position signals (1/0/-1).

        Returns:
            BacktestResult with equity curve and trades and statistics.

        Raises:
            ValueError: If prices and signals have different lengths.
        """
        prices = np.asarray(prices, dtype=np.float64)
        signals = np.asarray(signals, dtype=np.float64)

        if len(prices) != len(signals):
            raise ValueError(
                f"prices length {len(prices)} != signals length {len(signals)}"
            )

        n = len(prices)
        if n == 0:
            return BacktestResult(
                equity_curve=np.array([]),
                returns=np.array([]),
                config=self.config,
            )

        volatility = self._compute_volatility(prices)

        equity = np.zeros(n)
        equity[0] = self.config.initial_capital
        returns_arr = np.zeros(n)

        current_equity = self.config.initial_capital
        position_dir = 0  # 1 long / -1 short / 0 flat
        position_size = 0.0  # number of units held
        entry_price = 0.0
        entry_bar = 0
        total_fees = 0.0
        trades: list[TradeRecord] = []
        liquidations = 0
        trade_fees_accumulated = 0.0

        logger.info("engine.run.start", bars=n)

        for i in range(n):
            prev_equity = current_equity
            sig = int(signals[i])

            # --- Check liquidation (improvement #3) ---
            if position_dir != 0 and i > 0:
                price_change = (prices[i] - prices[i - 1]) / prices[i - 1]
                position_pnl = (
                    position_dir
                    * position_size
                    * prices[i - 1]
                    * price_change
                    * self.config.leverage
                )
                current_equity += position_pnl

                # Funding only when position is open (improvement #5)
                funding_cost = (
                    abs(position_size)
                    * prices[i]
                    * self.config.funding_rate
                    * self.config.leverage
                )
                current_equity -= funding_cost
                total_fees += funding_cost
                trade_fees_accumulated += funding_cost

                # Check if liquidated
                loss_frac = (
                    1.0 - current_equity / self.config.initial_capital
                    if self.config.initial_capital > 0
                    else 0.0
                )
                equity_from_peak = 1.0 - (
                    current_equity / max(np.max(equity[:i]), self.config.initial_capital)
                )
                if current_equity <= 0 or equity_from_peak >= self.config.liquidation_threshold:
                    logger.warning(
                        "engine.liquidation",
                        bar=i,
                        equity=current_equity,
                    )
                    exit_price = self._apply_slippage(
                        prices[i], -position_dir, volatility[i]
                    )
                    trades.append(
                        TradeRecord(
                            entry_bar=entry_bar,
                            exit_bar=i,
                            direction=position_dir,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl=current_equity - prev_equity,
                            fees_paid=trade_fees_accumulated,
                            duration_bars=i - entry_bar,
                            exit_reason="liquidation",
                        )
                    )
                    liquidations += 1
                    position_dir = 0
                    position_size = 0.0
                    trade_fees_accumulated = 0.0

                    if current_equity <= 0:
                        current_equity = 0.0
                        equity[i:] = 0.0
                        returns_arr[i] = -1.0
                        break

            # --- Process signal changes ---
            if sig != position_dir:
                # Close existing position if any
                if position_dir != 0:
                    exit_price = self._apply_slippage(
                        prices[i], -position_dir, volatility[i]
                    )
                    notional = position_size * exit_price * self.config.leverage
                    close_fee = self._compute_fee(notional)
                    current_equity -= close_fee
                    total_fees += close_fee
                    trade_fees_accumulated += close_fee

                    trade_pnl = current_equity - (
                        equity[entry_bar] if entry_bar < len(equity) else self.config.initial_capital
                    )
                    trades.append(
                        TradeRecord(
                            entry_bar=entry_bar,
                            exit_bar=i,
                            direction=position_dir,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl=current_equity - equity[entry_bar],
                            fees_paid=trade_fees_accumulated,
                            duration_bars=i - entry_bar,
                            exit_reason="signal",
                        )
                    )
                    trade_fees_accumulated = 0.0
                    position_dir = 0
                    position_size = 0.0

                # Open new position if signal is not flat
                if sig != 0 and current_equity > 0:
                    entry_price = self._apply_slippage(
                        prices[i], sig, volatility[i]
                    )
                    # Compounding: size based on current equity (improvement #2)
                    position_size = current_equity / entry_price
                    notional = position_size * entry_price * self.config.leverage
                    open_fee = self._compute_fee(notional)
                    current_equity -= open_fee
                    total_fees += open_fee
                    trade_fees_accumulated = open_fee
                    position_dir = sig
                    entry_bar = i

            equity[i] = current_equity
            if i > 0 and equity[i - 1] > 0:
                returns_arr[i] = (equity[i] - equity[i - 1]) / equity[i - 1]

        # Close any remaining position at end
        if position_dir != 0:
            exit_price = self._apply_slippage(
                prices[-1], -position_dir, volatility[-1]
            )
            notional = position_size * exit_price * self.config.leverage
            close_fee = self._compute_fee(notional)
            current_equity -= close_fee
            total_fees += close_fee
            trade_fees_accumulated += close_fee
            equity[-1] = current_equity

            trades.append(
                TradeRecord(
                    entry_bar=entry_bar,
                    exit_bar=n - 1,
                    direction=position_dir,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=current_equity - equity[entry_bar],
                    fees_paid=trade_fees_accumulated,
                    duration_bars=(n - 1) - entry_bar,
                    exit_reason="signal",
                )
            )

        # Compute statistics
        sharpe = annualized_sharpe(
            returns_arr, self.config.timeframe_minutes
        )
        mdd = max_drawdown(equity)
        winning = [t for t in trades if t.pnl > 0]
        win_rate = len(winning) / len(trades) if trades else 0.0
        avg_dur = (
            np.mean([t.duration_bars for t in trades]) if trades else 0.0
        )

        result = BacktestResult(
            equity_curve=equity,
            returns=returns_arr,
            trades=trades,
            total_pnl=current_equity - self.config.initial_capital,
            total_fees=total_fees,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            win_rate=win_rate,
            num_trades=len(trades),
            liquidations=liquidations,
            avg_trade_duration=float(avg_dur),
            config=self.config,
        )

        logger.info(
            "engine.run.complete",
            trades=result.num_trades,
            pnl=result.total_pnl,
            sharpe=result.sharpe_ratio,
        )

        return result
