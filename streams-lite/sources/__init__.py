"""
Built-in Oracle Sources for Streams-Lite

Available sources:
- FREDOracle: Federal Reserve Economic Data
- CryptoOracle: Cryptocurrency prices
- HTTPOracle: Generic HTTP/JSON endpoint
"""

from .fred import FREDOracle
from .crypto import CryptoOracle
from .http import HTTPOracle

__all__ = [
    "FREDOracle",
    "CryptoOracle",
    "HTTPOracle",
]
