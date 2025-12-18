"""
FRED (Federal Reserve Economic Data) Oracle

Fetches economic indicators from the St. Louis Fed's FRED API.

Example series:
- DGS10: 10-Year Treasury Constant Maturity Rate
- DGS2: 2-Year Treasury Constant Maturity Rate
- FEDFUNDS: Federal Funds Effective Rate
- CPIAUCSL: Consumer Price Index
- UNRATE: Unemployment Rate
- GDP: Gross Domestic Product

Get an API key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import logging
import asyncio
from typing import Optional

import requests

from ..base import BaseOracle, OracleConfig

logger = logging.getLogger(__name__)


class FREDOracle(BaseOracle):
    """
    Oracle for FRED (Federal Reserve Economic Data) series.
    """

    # Default API key (rate limited, get your own for production)
    DEFAULT_API_KEY = "7ef44306675240d156b2b8786339b867"

    def __init__(self, config: OracleConfig, publisher=None):
        super().__init__(config, publisher)

        # Get series ID from config
        self.series_id = config.extra.get('series_id', 'DGS10')
        self.api_key = config.api_key or os.environ.get('FRED_API_KEY', self.DEFAULT_API_KEY)

        # Build API URL
        self.api_url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={self.series_id}"
            f"&api_key={self.api_key}"
            f"&file_type=json"
            f"&sort_order=desc"
            f"&limit=10"
        )

    async def setup(self):
        """Validate API key and series."""
        # Quick validation request
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.api_url, timeout=10)
            )
            if response.status_code == 400:
                logger.error(f"Invalid FRED series: {self.series_id}")
                raise ValueError(f"Invalid FRED series: {self.series_id}")
            elif response.status_code == 429:
                logger.warning("FRED API rate limited, will retry")
        except requests.RequestException as e:
            logger.warning(f"FRED API check failed: {e}")

    async def fetch_value(self) -> Optional[float]:
        """Fetch latest value from FRED API."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.api_url, timeout=30)
            )
            response.raise_for_status()

            data = response.json()
            observations = data.get('observations', [])

            # Find latest non-empty value
            for obs in observations:
                try:
                    value = float(obs['value'])
                    return value
                except (ValueError, KeyError):
                    continue

            return None

        except requests.RequestException as e:
            logger.error(f"FRED API error for {self.series_id}: {e}")
            raise

    def validate_value(self, value: float) -> bool:
        """Validate FRED value (must be reasonable number)."""
        if value is None:
            return False
        # FRED values can be negative (e.g., real rates) but should be bounded
        return -1000 < value < 1000000
