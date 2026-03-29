"""MEFAI Backtest: Production grade vectorized backtesting engine.

Provides realistic simulation of crypto perpetual futures trading
with notional fees and compounding and liquidation and walk forward
validation.
"""

from .engine import BacktestEngine
from .types import BacktestConfig, BacktestResult
from .walk_forward import WalkForwardOptimizer

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "WalkForwardOptimizer",
]

__version__ = "0.1.0"
