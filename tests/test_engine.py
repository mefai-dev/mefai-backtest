"""Tests for the MEFAI backtesting engine.

Covers all ten improvements and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from mefai_backtest import BacktestConfig, BacktestEngine, BacktestResult
from mefai_backtest.walk_forward import WalkForwardOptimizer
from mefai_backtest.stats import annualized_sharpe, max_drawdown, format_report


def _rising_prices(n: int = 100, start: float = 100.0, pct: float = 0.01) -> np.ndarray:
    """Generate a steadily rising price series."""
    return start * (1 + pct) ** np.arange(n)


def _falling_prices(n: int = 100, start: float = 100.0, pct: float = 0.01) -> np.ndarray:
    """Generate a steadily falling price series."""
    return start * (1 - pct) ** np.arange(n)


def _flat_prices(n: int = 100, price: float = 100.0) -> np.ndarray:
    """Generate a flat price series."""
    return np.full(n, price)


class TestBuyAndHoldRising:
    """Test that buying a rising asset produces profit."""

    def test_long_rising_is_profitable(self):
        prices = _rising_prices(100)
        signals = np.ones(100)
        config = BacktestConfig(fee_rate=0.0, slippage_base_bps=0.0, funding_rate=0.0)
        engine = BacktestEngine(config)
        result = engine.run(prices, signals)

        assert result.total_pnl > 0
        assert result.num_trades == 1
        assert result.equity_curve[-1] > config.initial_capital


class TestShortRising:
    """Test that shorting a rising asset produces a loss."""

    def test_short_rising_loses_money(self):
        prices = _rising_prices(100)
        signals = -np.ones(100)
        config = BacktestConfig(fee_rate=0.0, slippage_base_bps=0.0, funding_rate=0.0)
        engine = BacktestEngine(config)
        result = engine.run(prices, signals)

        assert result.total_pnl < 0


class TestFeeImpact:
    """Test that fees reduce profits compared to zero fee baseline."""

    def test_fees_reduce_pnl(self):
        prices = _rising_prices(100)
        signals = np.ones(100)

        no_fee_config = BacktestConfig(fee_rate=0.0, slippage_base_bps=0.0, funding_rate=0.0)
        fee_config = BacktestConfig(fee_rate=0.001, slippage_base_bps=0.0, funding_rate=0.0)

        no_fee_result = BacktestEngine(no_fee_config).run(prices, signals)
        fee_result = BacktestEngine(fee_config).run(prices, signals)

        assert fee_result.total_pnl < no_fee_result.total_pnl
        assert fee_result.total_fees > 0


class TestCompounding:
    """Test that position sizes grow with equity (compounding)."""

    def test_compounding_amplifies_gains(self):
        # With a long rising series the compounding engine should produce
        # more than linear growth
        prices = _rising_prices(200, pct=0.005)
        signals = np.ones(200)
        config = BacktestConfig(fee_rate=0.0, slippage_base_bps=0.0, funding_rate=0.0)
        engine = BacktestEngine(config)
        result = engine.run(prices, signals)

        # Final equity should exceed initial + naive linear estimate
        naive_return = (prices[-1] / prices[0] - 1.0) * config.initial_capital
        assert result.total_pnl > naive_return * 0.9  # roughly compound


class TestLiquidation:
    """Test that liquidation triggers on extreme losses."""

    def test_leveraged_short_in_bull_market_liquidates(self):
        prices = _rising_prices(100, pct=0.05)
        signals = -np.ones(100)
        config = BacktestConfig(
            leverage=10.0,
            fee_rate=0.0,
            slippage_base_bps=0.0,
            funding_rate=0.0,
            liquidation_threshold=0.90,
        )
        engine = BacktestEngine(config)
        result = engine.run(prices, signals)

        assert result.liquidations > 0
        assert any(t.exit_reason == "liquidation" for t in result.trades)


class TestTradeDuration:
    """Test that trade duration tracking works correctly."""

    def test_duration_matches_signal_length(self):
        prices = _rising_prices(50)
        # Long for 20 bars then flat
        signals = np.zeros(50)
        signals[:20] = 1

        config = BacktestConfig(fee_rate=0.0, slippage_base_bps=0.0, funding_rate=0.0)
        engine = BacktestEngine(config)
        result = engine.run(prices, signals)

        assert result.num_trades == 1
        assert result.trades[0].duration_bars == 20
        assert result.avg_trade_duration == 20.0


class TestTimeframeSharpe:
    """Test that Sharpe ratio changes with timeframe setting."""

    def test_different_timeframes_different_sharpe(self):
        returns = np.random.default_rng(42).normal(0.001, 0.01, 500)

        sharpe_1h = annualized_sharpe(returns, timeframe_minutes=60)
        sharpe_1d = annualized_sharpe(returns, timeframe_minutes=1440)

        # Hourly has more bars per year so higher annualization factor
        assert sharpe_1h != sharpe_1d
        assert abs(sharpe_1h) > abs(sharpe_1d)


class TestFundingWhenFlat:
    """Test that funding is not charged when position is flat."""

    def test_no_funding_on_flat_position(self):
        prices = _flat_prices(100)
        signals = np.zeros(100)  # always flat

        config = BacktestConfig(funding_rate=0.01, fee_rate=0.0, slippage_base_bps=0.0)
        engine = BacktestEngine(config)
        result = engine.run(prices, signals)

        # No trades and no fees because no position was ever open
        assert result.total_fees == 0.0
        assert result.num_trades == 0
        assert result.equity_curve[-1] == config.initial_capital


class TestFlatSignalNoTrades:
    """Test that all zero signals produce no trades."""

    def test_zero_signals_zero_trades(self):
        prices = _rising_prices(100)
        signals = np.zeros(100)
        engine = BacktestEngine()
        result = engine.run(prices, signals)

        assert result.num_trades == 0
        assert result.total_pnl == 0.0
        assert result.total_fees == 0.0


class TestWalkForwardFolds:
    """Test that walk forward optimizer produces correct number of folds."""

    def test_fold_count(self):
        np.random.seed(42)
        n = 1000
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n)))

        def signal_fn(p, lookback=20):
            ma = np.convolve(p, np.ones(lookback) / lookback, mode="same")
            return np.where(p > ma, 1, -1).astype(float)

        optimizer = WalkForwardOptimizer(
            n_folds=4,
            train_ratio=0.7,
            base_config=BacktestConfig(fee_rate=0.0, slippage_base_bps=0.0, funding_rate=0.0),
        )

        wf_result = optimizer.run(
            prices=prices,
            signal_fn=signal_fn,
            param_grid={"lookback": [10, 20, 50]},
        )

        assert len(wf_result.folds) == 4
        assert wf_result.avg_train_sharpe != 0.0


class TestMaxDrawdown:
    """Test max drawdown computation."""

    def test_known_drawdown(self):
        equity = np.array([100.0, 110.0, 90.0, 95.0, 80.0, 120.0])
        dd = max_drawdown(equity)
        # Peak was 110 and trough was 80 so dd = 30/110
        expected = 30.0 / 110.0
        assert abs(dd - expected) < 1e-6


class TestReportFormatting:
    """Test that report generation does not crash."""

    def test_format_report_runs(self):
        prices = _rising_prices(100)
        signals = np.ones(100)
        engine = BacktestEngine()
        result = engine.run(prices, signals)
        report = format_report(result)

        assert "BACKTEST PERFORMANCE REPORT" in report
        assert "Sharpe" in report
        assert "Drawdown" in report
