"""
Unit Tests for streams-lite Oracle Sources

Tests for FRED, Crypto, and HTTP oracles with mocked external APIs.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import json
import requests

import sys
import os

# Import base classes from our test module
from tests.unit.test_streams_lite import OracleConfig, BaseOracle


# =============================================================================
# Oracle Implementations for Testing (mirror real implementations)
# =============================================================================

class FREDOracle(BaseOracle):
    """Test version of FRED Oracle."""

    DEFAULT_API_KEY = "7ef44306675240d156b2b8786339b867"

    def __init__(self, config, publisher=None):
        super().__init__(config, publisher)
        self.series_id = config.extra.get('series_id', 'DGS10')
        self.api_key = config.api_key or os.environ.get('FRED_API_KEY', self.DEFAULT_API_KEY)
        self.api_url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={self.series_id}"
            f"&api_key={self.api_key}"
            f"&file_type=json"
            f"&sort_order=desc"
            f"&limit=10"
        )

    async def fetch_value(self):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(self.api_url, timeout=30)
        )
        response.raise_for_status()
        data = response.json()
        observations = data.get('observations', [])
        for obs in observations:
            try:
                return float(obs['value'])
            except (ValueError, KeyError):
                continue
        return None

    def validate_value(self, value):
        if value is None:
            return False
        return -1000 < value < 1000000


class CryptoOracle(BaseOracle):
    """Test version of Crypto Oracle."""

    COINGECKO_IDS = {
        'btc': 'bitcoin', 'eth': 'ethereum', 'doge': 'dogecoin',
        'evr': 'evrmore', 'rvn': 'ravencoin',
    }

    def __init__(self, config, publisher=None):
        super().__init__(config, publisher)
        self.coin = config.extra.get('coin', 'bitcoin').lower()
        self.currency = config.extra.get('currency', 'usd').lower()
        self.source = config.extra.get('source', 'coingecko').lower()
        self.coingecko_id = self.COINGECKO_IDS.get(self.coin, self.coin)

    async def fetch_value(self):
        if self.source == 'binance':
            return await self._fetch_binance()
        return await self._fetch_coingecko()

    async def _fetch_coingecko(self):
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={self.coingecko_id}&vs_currencies={self.currency}"
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=10))
        response.raise_for_status()
        data = response.json()
        price = data.get(self.coingecko_id, {}).get(self.currency)
        return float(price) if price is not None else None

    async def _fetch_binance(self):
        symbol = f"{self.coin.upper()}{'USDT' if self.currency == 'usd' else self.currency.upper()}"
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=10))
        response.raise_for_status()
        data = response.json()
        price = data.get('price')
        return float(price) if price is not None else None

    def validate_value(self, value):
        return value is not None and value > 0


class HTTPOracle(BaseOracle):
    """Test version of HTTP Oracle."""

    def __init__(self, config, publisher=None):
        super().__init__(config, publisher)
        self.url = config.extra.get('url', '')
        self.json_path = config.extra.get('json_path', 'value')
        self.headers = config.extra.get('headers', {})
        self.timeout = config.extra.get('timeout', 30)

    async def fetch_value(self):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(self.url, headers=self.headers, timeout=self.timeout)
        )
        response.raise_for_status()
        data = response.json()
        value = self._extract_value(data, self.json_path)
        return float(value) if value is not None else None

    def _extract_value(self, data, path):
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


# =============================================================================
# FREDOracle Tests
# =============================================================================

class TestFREDOracle:
    """Tests for FRED Oracle."""

    @pytest.fixture
    def config(self):
        return OracleConfig(
            stream_id='fred|satori|DGS10|rate',
            name='10-Year Treasury',
            poll_interval=3600,
            extra={'series_id': 'DGS10'},
        )

    def test_init(self, config):
        """Test FRED oracle initialization."""
        oracle = FREDOracle(config)
        assert oracle.series_id == 'DGS10'
        assert 'DGS10' in oracle.api_url

    def test_init_custom_api_key(self):
        """Test with custom API key."""
        config = OracleConfig(
            stream_id='fred|test',
            name='Test',
            api_key='custom_key_123',
            extra={'series_id': 'FEDFUNDS'},
        )
        oracle = FREDOracle(config)
        assert 'custom_key_123' in oracle.api_url

    @pytest.mark.asyncio
    async def test_fetch_value_success(self, config):
        """Test successful FRED data fetch."""
        oracle = FREDOracle(config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'observations': [
                {'date': '2024-01-15', 'value': '4.25'},
            ]
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value == 4.25

    @pytest.mark.asyncio
    async def test_fetch_value_skips_invalid(self, config):
        """Test that FRED oracle skips invalid values."""
        oracle = FREDOracle(config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'observations': [
                {'date': '2024-01-15', 'value': '.'},
                {'date': '2024-01-14', 'value': '3.99'},
            ]
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value == 3.99

    def test_validate_value(self, config):
        """Test FRED value validation."""
        oracle = FREDOracle(config)

        assert oracle.validate_value(4.25) == True
        assert oracle.validate_value(0) == True
        assert oracle.validate_value(-0.5) == True
        assert oracle.validate_value(None) == False
        assert oracle.validate_value(1000001) == False


# =============================================================================
# CryptoOracle Tests
# =============================================================================

class TestCryptoOracle:
    """Tests for Cryptocurrency Price Oracle."""

    @pytest.fixture
    def btc_config(self):
        return OracleConfig(
            stream_id='crypto|satori|BTC|USD',
            name='Bitcoin Price',
            poll_interval=300,
            extra={
                'coin': 'bitcoin',
                'currency': 'usd',
                'source': 'coingecko',
            },
        )

    def test_init_coingecko(self, btc_config):
        """Test CoinGecko initialization."""
        oracle = CryptoOracle(btc_config)

        assert oracle.coin == 'bitcoin'
        assert oracle.currency == 'usd'
        assert oracle.source == 'coingecko'

    def test_coingecko_id_mapping(self):
        """Test common coin symbol to CoinGecko ID mapping."""
        config = OracleConfig(
            stream_id='crypto|test',
            name='Test',
            extra={'coin': 'btc', 'currency': 'usd'},
        )
        oracle = CryptoOracle(config)
        assert oracle.coingecko_id == 'bitcoin'

    @pytest.mark.asyncio
    async def test_fetch_coingecko_success(self, btc_config):
        """Test successful CoinGecko fetch."""
        oracle = CryptoOracle(btc_config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'bitcoin': {'usd': 43256.78}
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value == 43256.78

    @pytest.mark.asyncio
    async def test_fetch_binance_success(self):
        """Test successful Binance fetch."""
        config = OracleConfig(
            stream_id='crypto|test',
            name='Test',
            extra={'coin': 'ethereum', 'currency': 'usd', 'source': 'binance'},
        )
        oracle = CryptoOracle(config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'symbol': 'ETHUSDT',
            'price': '2345.67'
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value == 2345.67

    def test_validate_value(self, btc_config):
        """Test crypto price validation."""
        oracle = CryptoOracle(btc_config)

        assert oracle.validate_value(43000.0) == True
        assert oracle.validate_value(0.00001) == True
        assert oracle.validate_value(None) == False
        assert oracle.validate_value(0) == False
        assert oracle.validate_value(-100) == False


# =============================================================================
# HTTPOracle Tests
# =============================================================================

class TestHTTPOracle:
    """Tests for generic HTTP JSON Oracle."""

    @pytest.fixture
    def simple_config(self):
        return OracleConfig(
            stream_id='http|test|temp|sensor',
            name='Temperature Sensor',
            extra={
                'url': 'http://sensor.local/api/reading',
                'json_path': 'temperature',
            },
        )

    @pytest.fixture
    def nested_config(self):
        return OracleConfig(
            stream_id='http|test|weather|nyc',
            name='NYC Weather',
            extra={
                'url': 'https://api.weather.gov/forecast',
                'json_path': 'properties.periods.0.temperature',
                'headers': {'User-Agent': 'SatoriOracle/1.0'},
            },
        )

    def test_init_simple(self, simple_config):
        """Test simple HTTP oracle init."""
        oracle = HTTPOracle(simple_config)

        assert oracle.url == 'http://sensor.local/api/reading'
        assert oracle.json_path == 'temperature'
        assert oracle.headers == {}

    def test_init_with_headers(self, nested_config):
        """Test HTTP oracle with custom headers."""
        oracle = HTTPOracle(nested_config)
        assert oracle.headers == {'User-Agent': 'SatoriOracle/1.0'}

    @pytest.mark.asyncio
    async def test_fetch_simple_json(self, simple_config):
        """Test fetching from simple JSON structure."""
        oracle = HTTPOracle(simple_config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'temperature': 72.5,
            'humidity': 45,
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value == 72.5

    @pytest.mark.asyncio
    async def test_fetch_nested_json(self, nested_config):
        """Test fetching from nested JSON structure."""
        oracle = HTTPOracle(nested_config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'properties': {
                'periods': [
                    {'temperature': 68},
                    {'temperature': 65},
                ]
            }
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value == 68

    @pytest.mark.asyncio
    async def test_fetch_missing_path(self, simple_config):
        """Test handling missing JSON path."""
        oracle = HTTPOracle(simple_config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'humidity': 45,
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value is None

    @pytest.mark.asyncio
    async def test_fetch_array_index(self):
        """Test accessing array elements in JSON path."""
        config = OracleConfig(
            stream_id='http|test|array',
            name='Array Test',
            extra={
                'url': 'http://test.local/data',
                'json_path': 'values.2',
            },
        )
        oracle = HTTPOracle(config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'values': [10, 20, 30, 40]
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value == 30

    @pytest.mark.asyncio
    async def test_fetch_string_to_float(self):
        """Test converting string values to float."""
        config = OracleConfig(
            stream_id='http|test|string',
            name='String Value',
            extra={
                'url': 'http://test.local/data',
                'json_path': 'price',
            },
        )
        oracle = HTTPOracle(config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            'price': '123.45'
        })

        with patch('requests.get', return_value=mock_response):
            value = await oracle.fetch_value()

        assert value == 123.45
        assert isinstance(value, float)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestOracleErrorHandling:
    """Tests for oracle error handling."""

    @pytest.mark.asyncio
    async def test_fred_request_error(self):
        """Test FRED oracle handles request errors."""
        config = OracleConfig(
            stream_id='fred|test',
            name='Test',
            extra={'series_id': 'DGS10'},
        )
        oracle = FREDOracle(config)

        with patch('requests.get', side_effect=requests.RequestException("Network error")):
            with pytest.raises(requests.RequestException):
                await oracle.fetch_value()

    @pytest.mark.asyncio
    async def test_crypto_request_error(self):
        """Test Crypto oracle handles request errors."""
        config = OracleConfig(
            stream_id='crypto|test',
            name='Test',
            extra={'coin': 'bitcoin', 'currency': 'usd'},
        )
        oracle = CryptoOracle(config)

        with patch('requests.get', side_effect=requests.RequestException("Timeout")):
            with pytest.raises(requests.RequestException):
                await oracle.fetch_value()

    @pytest.mark.asyncio
    async def test_http_request_error(self):
        """Test HTTP oracle handles request errors."""
        config = OracleConfig(
            stream_id='http|test',
            name='Test',
            extra={'url': 'http://test.local', 'json_path': 'value'},
        )
        oracle = HTTPOracle(config)

        with patch('requests.get', side_effect=requests.RequestException("Connection refused")):
            with pytest.raises(requests.RequestException):
                await oracle.fetch_value()
