"""
Generic HTTP/JSON Oracle

Fetches values from any HTTP endpoint that returns JSON.
Use json_path to specify which field to extract.

Example config:
    type: http
    name: My API Data
    stream_id: myapi|satori|temperature|sensor1
    extra:
        url: https://api.example.com/data
        json_path: data.value
        headers:
            Authorization: Bearer xxx
"""

import logging
import asyncio
from typing import Optional, Dict, Any

import requests

from ..base import BaseOracle, OracleConfig

logger = logging.getLogger(__name__)


class HTTPOracle(BaseOracle):
    """
    Generic oracle for any HTTP/JSON endpoint.

    Configure with:
        - url: The endpoint URL
        - json_path: Dot-notation path to the value (e.g., "data.price")
        - headers: Optional HTTP headers
        - method: GET or POST (default: GET)
        - body: Request body for POST
    """

    def __init__(self, config: OracleConfig, publisher=None):
        super().__init__(config, publisher)

        self.url = config.extra.get('url', '')
        self.json_path = config.extra.get('json_path', 'value')
        self.headers = config.extra.get('headers', {})
        self.method = config.extra.get('method', 'GET').upper()
        self.body = config.extra.get('body')
        self.timeout = config.extra.get('timeout', 30)

        if not self.url:
            raise ValueError("HTTPOracle requires 'url' in extra config")

    async def setup(self):
        """Validate URL is reachable."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.head(self.url, timeout=10, headers=self.headers)
            )
            # Accept any response (even 405 for HEAD not allowed)
            if response.status_code >= 500:
                logger.warning(f"HTTP endpoint returned {response.status_code}")
        except requests.RequestException as e:
            logger.warning(f"HTTP endpoint check failed: {e}")

    async def fetch_value(self) -> Optional[float]:
        """Fetch value from HTTP endpoint."""
        try:
            loop = asyncio.get_event_loop()

            if self.method == 'POST':
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        self.url,
                        headers=self.headers,
                        json=self.body,
                        timeout=self.timeout
                    )
                )
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        self.url,
                        headers=self.headers,
                        timeout=self.timeout
                    )
                )

            response.raise_for_status()
            data = response.json()

            # Extract value using json_path
            value = self._extract_value(data, self.json_path)

            if value is not None:
                return float(value)
            return None

        except requests.RequestException as e:
            logger.error(f"HTTP fetch error for {self.url}: {e}")
            raise
        except (ValueError, TypeError) as e:
            logger.error(f"Value extraction error: {e}")
            raise

    def _extract_value(self, data: Any, path: str) -> Any:
        """
        Extract value from nested data using dot notation.

        Examples:
            - "value" -> data['value']
            - "data.price" -> data['data']['price']
            - "results.0.value" -> data['results'][0]['value']
        """
        if not path:
            return data

        keys = path.split('.')
        result = data

        for key in keys:
            if result is None:
                return None

            if isinstance(result, dict):
                result = result.get(key)
            elif isinstance(result, (list, tuple)) and key.isdigit():
                idx = int(key)
                result = result[idx] if idx < len(result) else None
            else:
                return None

        return result

    def validate_value(self, value: float) -> bool:
        """Validate extracted value is a valid number."""
        if value is None:
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
