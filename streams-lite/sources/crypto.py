"""
Cryptocurrency Price Oracle

Fetches crypto prices from public APIs (no API key required).

Supported sources:
- CoinGecko (default, free tier)
- Binance public API

Example pairs:
- bitcoin/usd
- ethereum/usd
- dogecoin/usd
"""

import logging
import asyncio
from typing import Optional, Dict

import requests

from ..base import BaseOracle, OracleConfig

# Try to import trio for compatibility
try:
    import trio
    HAS_TRIO = True
except ImportError:
    HAS_TRIO = False

logger = logging.getLogger(__name__)


class CryptoOracle(BaseOracle):
    """
    Oracle for cryptocurrency prices.

    Uses CoinGecko API by default (no API key required).
    """

    # CoinGecko ID mappings for common coins
    COINGECKO_IDS = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'doge': 'dogecoin',
        'xrp': 'ripple',
        'ada': 'cardano',
        'sol': 'solana',
        'dot': 'polkadot',
        'matic': 'polygon',
        'link': 'chainlink',
        'avax': 'avalanche-2',
        'evr': 'evrmore',
        'rvn': 'ravencoin',
    }

    def __init__(self, config: OracleConfig, publisher=None):
        super().__init__(config, publisher)

        # Parse coin and currency from config
        self.coin = config.extra.get('coin', 'bitcoin').lower()
        self.currency = config.extra.get('currency', 'usd').lower()
        self.source = config.extra.get('source', 'coingecko').lower()

        # Map common symbols to CoinGecko IDs
        self.coingecko_id = self.COINGECKO_IDS.get(self.coin, self.coin)

    async def fetch_value(self) -> Optional[float]:
        """Fetch current price from crypto API."""
        if self.source == 'binance':
            return await self._fetch_binance()
        else:
            return await self._fetch_coingecko()

    async def _fetch_coingecko(self) -> Optional[float]:
        """Fetch from CoinGecko API."""
        url = (
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={self.coingecko_id}"
            f"&vs_currencies={self.currency}"
        )

        try:
            # Use trio.to_thread if in Trio context, otherwise asyncio
            if HAS_TRIO:
                try:
                    trio.lowlevel.current_trio_token()
                    # In Trio context - use trio.to_thread
                    response = await trio.to_thread.run_sync(
                        lambda: requests.get(url, timeout=10)
                    )
                except RuntimeError:
                    # Not in Trio, use asyncio
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.get(url, timeout=10)
                    )
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(url, timeout=10)
                )

            response.raise_for_status()

            data = response.json()
            price = data.get(self.coingecko_id, {}).get(self.currency)

            if price is not None:
                return float(price)
            return None

        except requests.RequestException as e:
            logger.error(f"CoinGecko API error: {e}")
            raise

    async def _fetch_binance(self) -> Optional[float]:
        """Fetch from Binance API."""
        # Convert to Binance symbol format (e.g., BTCUSDT)
        symbol = f"{self.coin.upper()}{self.currency.upper()}"
        if self.currency == 'usd':
            symbol = f"{self.coin.upper()}USDT"

        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"

        try:
            # Use trio.to_thread if in Trio context, otherwise asyncio
            if HAS_TRIO:
                try:
                    trio.lowlevel.current_trio_token()
                    # In Trio context - use trio.to_thread
                    response = await trio.to_thread.run_sync(
                        lambda: requests.get(url, timeout=10)
                    )
                except RuntimeError:
                    # Not in Trio, use asyncio
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.get(url, timeout=10)
                    )
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(url, timeout=10)
                )

            response.raise_for_status()

            data = response.json()
            price = data.get('price')

            if price is not None:
                return float(price)
            return None

        except requests.RequestException as e:
            logger.error(f"Binance API error: {e}")
            raise

    def validate_value(self, value: float) -> bool:
        """Validate crypto price (must be positive)."""
        return value is not None and value > 0
