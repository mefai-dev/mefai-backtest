"""Walk forward optimization example.

Demonstrates how to use the WalkForwardOptimizer to find
robust parameters while detecting overfitting.
"""

import numpy as np

from mefai_backtest import BacktestConfig
from mefai_backtest.walk_forward import WalkForwardOptimizer


def ma_crossover_signals(
    prices: np.ndarray, fast: int = 10, slow: int = 30
) -> np.ndarray:
    """Moving average crossover signal generator."""
    fast_ma = np.convolve(prices, np.ones(fast) / fast, mode="same")
    slow_ma = np.convolve(prices, np.ones(slow) / slow, mode="same")
    signals = np.where(fast_ma > slow_ma, 1.0, -1.0)
    signals[: slow + 5] = 0.0
    return signals


def main() -> None:
    """Run walk forward optimization on synthetic data."""
    rng = np.random.default_rng(123)
    n = 5000
    log_returns = rng.normal(0.0001, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))

    config = BacktestConfig(
        initial_capital=10_000.0,
        fee_rate=0.0004,
        funding_rate=0.0001,
        timeframe_minutes=60,
    )

    optimizer = WalkForwardOptimizer(
        n_folds=5,
        train_ratio=0.7,
        overfit_threshold=0.5,
        base_config=config,
    )

    param_grid = {
        "fast": [5, 10, 15, 20],
        "slow": [25, 30, 40, 50, 60],
    }

    result = optimizer.run(
        prices=prices,
        signal_fn=ma_crossover_signals,
        param_grid=param_grid,
    )

    print("=" * 60)
    print("  WALK FORWARD OPTIMIZATION RESULTS")
    print("=" * 60)
    print()

    for fold in result.folds:
        print(
            f"  Fold {fold.fold_index}: "
            f"Train Sharpe={fold.train_sharpe:+.3f}  "
            f"Test Sharpe={fold.test_sharpe:+.3f}  "
            f"Params={fold.best_params}"
        )

    print()
    print(f"  Avg Train Sharpe: {result.avg_train_sharpe:+.4f}")
    print(f"  Avg Test Sharpe:  {result.avg_test_sharpe:+.4f}")
    print(f"  Overfit Ratio:    {result.overfit_ratio:.4f}")
    print(f"  Is Overfit:       {result.is_overfit}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
