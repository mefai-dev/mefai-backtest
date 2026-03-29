"""Walk forward optimizer with overfit detection.

Splits a price series into sequential train/test folds and runs
parameter optimization on each training window. Then validates
on the out of sample test window. Compares in sample vs out of
sample performance to detect overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import structlog

from .engine import BacktestEngine
from .types import BacktestConfig, BacktestResult

logger = structlog.get_logger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk forward fold.

    Attributes:
        fold_index: Zero based fold number.
        train_sharpe: In sample Sharpe ratio.
        test_sharpe: Out of sample Sharpe ratio.
        train_result: Full backtest result on training data.
        test_result: Full backtest result on test data.
        best_params: Parameter dict that won the training phase.
    """

    fold_index: int = 0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None
    best_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk forward folds.

    Attributes:
        folds: List of per fold results.
        avg_train_sharpe: Mean in sample Sharpe.
        avg_test_sharpe: Mean out of sample Sharpe.
        overfit_ratio: Ratio indicating degree of overfitting.
            Values close to 1.0 indicate no overfitting.
            Values much less than 1.0 indicate severe overfitting.
        is_overfit: True if overfit_ratio is below the threshold.
    """

    folds: list[FoldResult] = field(default_factory=list)
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    overfit_ratio: float = 0.0
    is_overfit: bool = False


class WalkForwardOptimizer:
    """Walk forward optimizer for backtesting parameter selection.

    Divides the data into sequential non overlapping folds.
    For each fold the optimizer searches over a parameter grid
    on the training portion and validates on the test portion.

    This approach gives a realistic estimate of strategy performance
    because the test data is always strictly after the training data
    and each fold is independent.

    Usage::

        def signal_fn(prices, fast=10, slow=30):
            fast_ma = np.convolve(prices, np.ones(fast)/fast, 'same')
            slow_ma = np.convolve(prices, np.ones(slow)/slow, 'same')
            return np.where(fast_ma > slow_ma, 1, -1)

        optimizer = WalkForwardOptimizer(
            n_folds=5,
            train_ratio=0.7,
            overfit_threshold=0.5,
        )

        result = optimizer.run(
            prices=price_array,
            signal_fn=signal_fn,
            param_grid={"fast": [5, 10, 20], "slow": [20, 30, 50]},
        )

    Args:
        n_folds: Number of sequential folds.
        train_ratio: Fraction of each fold used for training.
        overfit_threshold: Overfit ratio below which is_overfit is True.
        base_config: Base BacktestConfig to use for all runs.
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.7,
        overfit_threshold: float = 0.5,
        base_config: BacktestConfig | None = None,
    ) -> None:
        """Initialize the walk forward optimizer.

        Args:
            n_folds: Number of sequential folds to create.
            train_ratio: Fraction of each fold for training.
            overfit_threshold: Ratio below which overfitting is flagged.
            base_config: Base config for backtests. Uses defaults if None.
        """
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.overfit_threshold = overfit_threshold
        self.base_config = base_config or BacktestConfig()

    def _make_folds(
        self, n: int
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Split data indices into train/test fold pairs.

        Args:
            n: Total number of data points.

        Returns:
            List of ((train_start, train_end), (test_start, test_end)) tuples.
        """
        fold_size = n // self.n_folds
        folds = []

        for i in range(self.n_folds):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size if i < self.n_folds - 1 else n
            split = fold_start + int((fold_end - fold_start) * self.train_ratio)

            train_range = (fold_start, split)
            test_range = (split, fold_end)
            folds.append((train_range, test_range))

        return folds

    def _grid_search(
        self,
        prices: np.ndarray,
        signal_fn: Callable,
        param_grid: dict[str, list],
    ) -> tuple[dict[str, Any], float, BacktestResult]:
        """Search over parameter grid to find best Sharpe.

        Args:
            prices: Price array for this segment.
            signal_fn: Function(prices, **params) -> signals.
            param_grid: Dict of parameter names to lists of values.

        Returns:
            Tuple of (best_params, best_sharpe, best_result).
        """
        keys = list(param_grid.keys())
        value_lists = list(param_grid.values())

        # Generate all combinations
        from itertools import product

        combinations = list(product(*value_lists))

        best_sharpe = float("-inf")
        best_params: dict[str, Any] = {}
        best_result: BacktestResult | None = None

        engine = BacktestEngine(self.base_config)

        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                signals = signal_fn(prices, **params)
                result = engine.run(prices, signals)
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_params = params
                    best_result = result
            except Exception as exc:
                logger.warning(
                    "walkforward.grid_search.error",
                    params=params,
                    error=str(exc),
                )

        if best_result is None:
            best_result = BacktestResult(config=self.base_config)

        return best_params, best_sharpe, best_result

    def run(
        self,
        prices: np.ndarray,
        signal_fn: Callable,
        param_grid: dict[str, list],
    ) -> WalkForwardResult:
        """Execute walk forward optimization across all folds.

        For each fold:
        1. Run grid search on training data to find best parameters
        2. Apply those parameters to test data
        3. Record in sample and out of sample Sharpe ratios

        After all folds the overfit ratio is computed as the ratio of
        mean test Sharpe to mean train Sharpe.

        Args:
            prices: Full price array.
            signal_fn: Function(prices, **params) -> signals array.
            param_grid: Dict of parameter names to lists of candidate values.

        Returns:
            WalkForwardResult with all fold results and overfit metrics.
        """
        prices = np.asarray(prices, dtype=np.float64)
        folds_indices = self._make_folds(len(prices))
        fold_results: list[FoldResult] = []

        logger.info(
            "walkforward.start",
            n_folds=self.n_folds,
            total_bars=len(prices),
        )

        engine = BacktestEngine(self.base_config)

        for idx, (train_range, test_range) in enumerate(folds_indices):
            train_prices = prices[train_range[0] : train_range[1]]
            test_prices = prices[test_range[0] : test_range[1]]

            if len(train_prices) < 10 or len(test_prices) < 5:
                logger.warning(
                    "walkforward.fold.skip",
                    fold=idx,
                    train_len=len(train_prices),
                    test_len=len(test_prices),
                )
                continue

            # Train: grid search
            best_params, train_sharpe, train_result = self._grid_search(
                train_prices, signal_fn, param_grid
            )

            # Test: apply best params out of sample
            test_signals = signal_fn(test_prices, **best_params)
            test_result = engine.run(test_prices, test_signals)

            fold_results.append(
                FoldResult(
                    fold_index=idx,
                    train_sharpe=train_sharpe,
                    test_sharpe=test_result.sharpe_ratio,
                    train_result=train_result,
                    test_result=test_result,
                    best_params=best_params,
                )
            )

            logger.info(
                "walkforward.fold.complete",
                fold=idx,
                train_sharpe=train_sharpe,
                test_sharpe=test_result.sharpe_ratio,
                params=best_params,
            )

        # Aggregate
        if fold_results:
            avg_train = np.mean([f.train_sharpe for f in fold_results])
            avg_test = np.mean([f.test_sharpe for f in fold_results])
            overfit_ratio = (
                avg_test / avg_train if avg_train != 0 else 0.0
            )
        else:
            avg_train = 0.0
            avg_test = 0.0
            overfit_ratio = 0.0

        wf_result = WalkForwardResult(
            folds=fold_results,
            avg_train_sharpe=float(avg_train),
            avg_test_sharpe=float(avg_test),
            overfit_ratio=float(overfit_ratio),
            is_overfit=overfit_ratio < self.overfit_threshold,
        )

        logger.info(
            "walkforward.complete",
            avg_train_sharpe=wf_result.avg_train_sharpe,
            avg_test_sharpe=wf_result.avg_test_sharpe,
            overfit_ratio=wf_result.overfit_ratio,
            is_overfit=wf_result.is_overfit,
        )

        return wf_result
