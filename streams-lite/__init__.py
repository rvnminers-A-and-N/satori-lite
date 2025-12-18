"""
streams-lite: Oracle/Data Stream Module for Satori-Lite

Provides data oracle functionality - fetching real-world data and publishing
observations to the Satori network via P2P or HTTP.

Usage:
    from streams_lite import StreamManager

    manager = StreamManager()
    await manager.start()  # Starts all configured oracles

Components:
    - StreamManager: Orchestrates multiple oracle streams
    - BaseOracle: Base class for custom oracles
    - Built-in sources: FRED, crypto exchanges, etc.
"""

from .manager import StreamManager
from .base import BaseOracle, OracleConfig
from .publisher import P2PPublisher, get_networking_mode

__version__ = "1.0.0"
__all__ = [
    "StreamManager",
    "BaseOracle",
    "OracleConfig",
    "P2PPublisher",
    "get_networking_mode",
]
