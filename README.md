# mefai-backtest

[![CI](https://github.com/mefai-dev/mefai-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/mefai-dev/mefai-backtest/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mefai-backtest.svg)](https://pypi.org/project/mefai-backtest/)

Production grade vectorized backtesting engine for crypto perpetual futures. Built for realistic simulation with proper fee modeling and compounding and liquidation detection and walk forward validation.

## Features

This engine includes ten key improvements over naive backtesting:

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Notional fee model** | Fees scale with position size (notional value) rather than just capital |
| 2 | **Compounding equity** | Position sizes grow and shrink with your equity over time |
| 3 | **Liquidation detection** | Forced close when equity drops below the configured threshold |
| 4 | **Timeframe aware Sharpe** | Annualization factor adapts to bar duration (1m to 1d and beyond) |
| 5 | **Funding only when open** | Funding costs only apply while a position is actually open |
| 6 | **Volatility scaled slippage** | Slippage increases proportionally with recent price volatility |
| 7 | **Trade duration tracking** | Every trade records how many bars it was held |
| 8 | **Full docstrings** | Every class and method is fully documented |
| 9 | **Structured logging** | All engine events logged via structlog for observability |
| 10 | **Walk forward ready** | Results integrate with the built in WalkForwardOptimizer |

## Installation

```bash
pip install mefai-backtest
```

Or install from source:

```bash
git clone https://github.com/mefai-dev/mefai-backtest.git
cd mefai-backtest
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from mefai_backtest import BacktestConfig, BacktestEngine
from mefai_backtest.stats import format_report

# Price data (use your own OHLCV close prices)
prices = np.array([100.0, 101.0, 103.0, 102.0, 105.0, 107.0])

# Signal array: 1=long  0=flat  -1=short
signals = np.array([1.0, 1.0, 1.0, -1.0, -1.0, 0.0])

config = BacktestConfig(
    initial_capital=10_000.0,
    fee_rate=0.0004,
    leverage=1.0,
    timeframe_minutes=60,
)

engine = BacktestEngine(config)
result = engine.run(prices, signals)

print(format_report(result))
```

## Walk Forward Optimization

The `WalkForwardOptimizer` splits your data into sequential train/test folds and searches for parameters that generalize well out of sample. It also computes an overfit ratio to flag strategies that only work on historical data.

```python
import numpy as np
from mefai_backtest import BacktestConfig
from mefai_backtest.walk_forward import WalkForwardOptimizer

def signal_fn(prices, fast=10, slow=30):
    fast_ma = np.convolve(prices, np.ones(fast) / fast, mode="same")
    slow_ma = np.convolve(prices, np.ones(slow) / slow, mode="same")
    signals = np.where(fast_ma > slow_ma, 1.0, -1.0)
    signals[:slow + 5] = 0.0
    return signals

optimizer = WalkForwardOptimizer(
    n_folds=5,
    train_ratio=0.7,
    overfit_threshold=0.5,
    base_config=BacktestConfig(fee_rate=0.0004),
)

result = optimizer.run(
    prices=price_array,
    signal_fn=signal_fn,
    param_grid={"fast": [5, 10, 20], "slow": [20, 30, 50]},
)

print(f"Overfit ratio: {result.overfit_ratio:.3f}")
print(f"Is overfit: {result.is_overfit}")
```

## API Reference

### BacktestConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 10000 | Starting equity in quote currency |
| `fee_rate` | float | 0.0004 | Fee as fraction of notional value |
| `slippage_base_bps` | float | 1.0 | Base slippage in basis points |
| `leverage` | float | 1.0 | Position leverage multiplier |
| `funding_rate` | float | 0.0001 | Per bar funding rate on open positions |
| `liquidation_threshold` | float | 0.90 | Drawdown fraction triggering liquidation |
| `timeframe_minutes` | int | 60 | Bar duration for Sharpe annualization |
| `volatility_lookback` | int | 20 | Bars for rolling volatility estimate |

### BacktestEngine

- `BacktestEngine(config=None)` ··· Create engine with optional config
- `engine.run(prices, signals)` ··· Run backtest and return `BacktestResult`

### BacktestResult

Contains `equity_curve` and `returns` (numpy arrays) plus `trades` (list of TradeRecord) and summary stats: `total_pnl` and `total_fees` and `sharpe_ratio` and `max_drawdown` and `win_rate` and `num_trades` and `liquidations` and `avg_trade_duration`.

### WalkForwardOptimizer

- `WalkForwardOptimizer(n_folds=5, train_ratio=0.7, overfit_threshold=0.5, base_config=None)`
- `optimizer.run(prices, signal_fn, param_grid)` ··· Returns `WalkForwardResult`

### Stats Module

- `annualized_sharpe(returns, timeframe_minutes)` ··· Timeframe aware Sharpe ratio
- `max_drawdown(equity_curve)` ··· Peak to trough drawdown fraction
- `profit_factor(trades)` ··· Gross profit over gross loss
- `format_report(result)` ··· Human readable performance summary

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
