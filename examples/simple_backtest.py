"""Simple backtest example using MEFAI Backtest engine.

Generates a synthetic price series and runs a moving average
crossover strategy through the backtester.
"""

import numpy as np

from mefai_backtest import BacktestConfig, BacktestEngine
from mefai_backtest.stats import format_report


def moving_average_signals(
    prices: np.ndarray, fast: int = 10, slow: int = 30
) -> np.ndarray:
    """Generate signals from a dual moving average crossover.

    Returns 1 when fast MA is above slow MA and -1 otherwise.
    """
    fast_ma = np.convolve(prices, np.ones(fast) / fast, mode="same")
    slow_ma = np.convolve(prices, np.ones(slow) / slow, mode="same")
    signals = np.where(fast_ma > slow_ma, 1.0, -1.0)
    # Flatten early bars where MAs are unreliable
    signals[: slow + 5] = 0.0
    return signals


def main() -> None:
    """Run a simple backtest on synthetic data."""
    # Generate synthetic trending price data
    rng = np.random.default_rng(42)
    n = 2000
    log_returns = rng.normal(0.0002, 0.015, n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))

    # Configure the engine
    config = BacktestConfig(
        initial_capital=10_000.0,
        fee_rate=0.0004,
        slippage_base_bps=1.0,
        leverage=1.0,
        funding_rate=0.0001,
        liquidation_threshold=0.90,
        timeframe_minutes=60,
    )

    engine = BacktestEngine(config)

    # Generate signals and run
    signals = moving_average_signals(prices)
    result = engine.run(prices, signals)

    # Print report
    print(format_report(result))
    print(f"\n  Trades taken: {result.num_trades}")
    print(f"  Win rate: {result.win_rate:.1%}")


if __name__ == "__main__":
    main()
