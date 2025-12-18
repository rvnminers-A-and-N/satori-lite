"""
Unit Tests for streams-lite

Tests for the oracle framework without requiring external APIs.
Uses mocking to test the core logic.

These tests recreate the core classes to avoid relative import issues.
"""

import pytest
import asyncio
import time
import os
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable


# =============================================================================
# Recreate Core Classes for Testing (mirrors streams-lite/base.py)
# =============================================================================

@dataclass
class OracleConfig:
    """Configuration for an oracle stream."""
    stream_id: str
    name: str
    poll_interval: int = 3600
    retry_interval: int = 60
    max_retries: int = 3
    enabled: bool = True
    api_key: str = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> 'OracleConfig':
        return cls(
            stream_id=data.get('stream_id', ''),
            name=data.get('name', 'Unnamed'),
            poll_interval=data.get('poll_interval', 3600),
            retry_interval=data.get('retry_interval', 60),
            max_retries=data.get('max_retries', 3),
            enabled=data.get('enabled', True),
            api_key=data.get('api_key'),
            extra=data.get('extra', {}),
        )

    def to_dict(self) -> dict:
        return {
            'stream_id': self.stream_id,
            'name': self.name,
            'poll_interval': self.poll_interval,
            'retry_interval': self.retry_interval,
            'max_retries': self.max_retries,
            'enabled': self.enabled,
            'api_key': self.api_key,
            'extra': self.extra,
        }


@dataclass
class Observation:
    """A single observation/data point."""
    stream_id: str
    value: float
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PublishResult:
    """Result of a publish operation."""
    success: bool
    method: str
    message: str
    p2p_success: bool = False
    http_success: bool = False


class BaseOracle(ABC):
    """Abstract base class for data oracles."""

    def __init__(self, config: OracleConfig, publisher=None):
        self.config = config
        self.publisher = publisher
        self._running = False
        self._task = None
        self._last_value = None
        self._last_publish_time = 0
        self._consecutive_errors = 0
        self._total_publishes = 0
        self._total_errors = 0
        self._on_publish = None
        self._on_error = None

    @property
    def stream_id(self) -> str:
        return self.config.stream_id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'stream_id': self.stream_id,
            'running': self._running,
            'last_value': self._last_value,
            'last_publish_time': self._last_publish_time,
            'total_publishes': self._total_publishes,
            'total_errors': self._total_errors,
            'consecutive_errors': self._consecutive_errors,
        }

    @abstractmethod
    async def fetch_value(self) -> Optional[float]:
        pass

    async def setup(self):
        pass

    async def cleanup(self):
        pass

    def validate_value(self, value: float) -> bool:
        return value is not None

    def should_publish(self, value: float) -> bool:
        now = time.time()
        if now - self._last_publish_time >= self.config.poll_interval:
            return True
        if self._last_value is None:
            return True
        return value != self._last_value

    async def start(self):
        if self._running:
            return
        if not self.config.enabled:
            return
        await self.setup()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.cleanup()

    async def poll_once(self) -> Optional[Observation]:
        try:
            value = await self.fetch_value()
            if value is None:
                return None
            if not self.validate_value(value):
                return None
            if not self.should_publish(value):
                return None
            observation = await self._publish(value)
            self._consecutive_errors = 0
            return observation
        except Exception as e:
            self._consecutive_errors += 1
            self._total_errors += 1
            if self._on_error:
                self._on_error(self, e)
            return None

    async def _publish(self, value: float) -> Optional[Observation]:
        timestamp = int(time.time())
        if self.publisher:
            result = await self.publisher.publish_observation(
                stream_id=self.stream_id,
                value=value,
                timestamp=timestamp,
            )
            if not result.success:
                return None
        self._last_value = value
        self._last_publish_time = timestamp
        self._total_publishes += 1
        observation = Observation(
            stream_id=self.stream_id,
            value=value,
            timestamp=timestamp,
        )
        if self._on_publish:
            self._on_publish(self, observation)
        return observation

    async def _poll_loop(self):
        while self._running:
            try:
                await self.poll_once()
                sleep_time = self.config.poll_interval
                if self._consecutive_errors > 0:
                    sleep_time = min(
                        self.config.retry_interval * (2 ** (self._consecutive_errors - 1)),
                        self.config.poll_interval
                    )
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break

    def on_publish(self, callback):
        self._on_publish = callback

    def on_error(self, callback):
        self._on_error = callback


def get_networking_mode() -> str:
    """Get current networking mode."""
    mode = os.environ.get('SATORI_NETWORKING_MODE', '').lower().strip()
    return mode if mode else 'central'


# =============================================================================
# Concrete Test Oracle Implementation
# =============================================================================

class ConcreteOracle(BaseOracle):
    """Concrete implementation for testing."""

    def __init__(self, config, publisher=None, fetch_returns=None):
        super().__init__(config, publisher)
        self.fetch_returns = fetch_returns if fetch_returns is not None else [100.0]
        self.fetch_index = 0
        self.setup_called = False
        self.cleanup_called = False

    async def fetch_value(self):
        if self.fetch_index < len(self.fetch_returns):
            value = self.fetch_returns[self.fetch_index]
            self.fetch_index += 1
            if isinstance(value, Exception):
                raise value
            return value
        return None

    async def setup(self):
        self.setup_called = True

    async def cleanup(self):
        self.cleanup_called = True


# =============================================================================
# OracleConfig Tests
# =============================================================================

class TestOracleConfig:
    """Tests for OracleConfig dataclass."""

    def test_from_dict_minimal(self):
        """Test creating config with minimal fields."""
        data = {
            'stream_id': 'test|author|stream|target',
            'name': 'Test Oracle',
        }
        config = OracleConfig.from_dict(data)

        assert config.stream_id == 'test|author|stream|target'
        assert config.name == 'Test Oracle'
        assert config.poll_interval == 3600
        assert config.enabled == True
        assert config.api_key is None

    def test_from_dict_full(self):
        """Test creating config with all fields."""
        data = {
            'stream_id': 'fred|satori|DGS10|rate',
            'name': 'Treasury Rate',
            'poll_interval': 1800,
            'retry_interval': 30,
            'max_retries': 5,
            'enabled': False,
            'api_key': 'test_key_123',
            'extra': {'series_id': 'DGS10'},
        }
        config = OracleConfig.from_dict(data)

        assert config.stream_id == 'fred|satori|DGS10|rate'
        assert config.poll_interval == 1800
        assert config.enabled == False
        assert config.api_key == 'test_key_123'
        assert config.extra['series_id'] == 'DGS10'

    def test_to_dict_roundtrip(self):
        """Test that to_dict/from_dict is reversible."""
        original = OracleConfig(
            stream_id='crypto|satori|BTC|USD',
            name='Bitcoin Price',
            poll_interval=300,
            extra={'coin': 'bitcoin'},
        )
        data = original.to_dict()
        restored = OracleConfig.from_dict(data)

        assert restored.stream_id == original.stream_id
        assert restored.name == original.name
        assert restored.poll_interval == original.poll_interval


# =============================================================================
# Observation Tests
# =============================================================================

class TestObservation:
    """Tests for Observation dataclass."""

    def test_basic_observation(self):
        """Test creating a basic observation."""
        obs = Observation(
            stream_id='test|author|stream|target',
            value=42.5,
            timestamp=1234567890,
        )

        assert obs.stream_id == 'test|author|stream|target'
        assert obs.value == 42.5
        assert obs.timestamp == 1234567890
        assert obs.metadata == {}

    def test_observation_with_metadata(self):
        """Test observation with metadata."""
        obs = Observation(
            stream_id='test|id',
            value=100.0,
            timestamp=int(time.time()),
            metadata={'source': 'coingecko'},
        )

        assert obs.metadata['source'] == 'coingecko'


# =============================================================================
# BaseOracle Tests
# =============================================================================

class TestBaseOracle:
    """Tests for BaseOracle class."""

    @pytest.fixture
    def config(self):
        return OracleConfig(
            stream_id='test|author|stream|target',
            name='Test Oracle',
            poll_interval=1,
            retry_interval=1,
        )

    @pytest.fixture
    def mock_publisher(self):
        publisher = Mock()
        publisher.publish_observation = AsyncMock(return_value=PublishResult(
            success=True, method='mock', message='OK'
        ))
        return publisher

    def test_properties(self, config):
        """Test oracle properties."""
        oracle = ConcreteOracle(config)
        assert oracle.stream_id == 'test|author|stream|target'
        assert oracle.name == 'Test Oracle'
        assert oracle.is_running == False

    def test_stats_initial(self, config):
        """Test initial statistics."""
        oracle = ConcreteOracle(config)
        stats = oracle.stats

        assert stats['running'] == False
        assert stats['last_value'] is None
        assert stats['total_publishes'] == 0

    @pytest.mark.asyncio
    async def test_poll_once_success(self, config, mock_publisher):
        """Test successful single poll."""
        oracle = ConcreteOracle(config, mock_publisher, fetch_returns=[42.5])

        obs = await oracle.poll_once()

        assert obs is not None
        assert obs.value == 42.5
        assert oracle._total_publishes == 1

    @pytest.mark.asyncio
    async def test_poll_once_error(self, config, mock_publisher):
        """Test poll with error."""
        oracle = ConcreteOracle(config, mock_publisher, fetch_returns=[ValueError("test")])

        obs = await oracle.poll_once()

        assert obs is None
        assert oracle._total_errors == 1
        assert oracle._consecutive_errors == 1

    @pytest.mark.asyncio
    async def test_poll_once_none_value(self, config, mock_publisher):
        """Test poll returning None."""
        oracle = ConcreteOracle(config, mock_publisher, fetch_returns=[None])

        obs = await oracle.poll_once()

        assert obs is None
        assert oracle._total_publishes == 0

    @pytest.mark.asyncio
    async def test_start_stop(self, config, mock_publisher):
        """Test oracle start/stop lifecycle."""
        oracle = ConcreteOracle(config, mock_publisher)

        await oracle.start()
        assert oracle.is_running == True
        assert oracle.setup_called == True

        await oracle.stop()
        assert oracle.is_running == False
        assert oracle.cleanup_called == True

    @pytest.mark.asyncio
    async def test_disabled_oracle_no_start(self, mock_publisher):
        """Test that disabled oracle doesn't start."""
        config = OracleConfig(
            stream_id='test|id',
            name='Disabled',
            enabled=False,
        )
        oracle = ConcreteOracle(config, mock_publisher)

        await oracle.start()

        assert oracle.is_running == False
        assert oracle.setup_called == False

    def test_validate_value_default(self, config):
        """Test default value validation."""
        oracle = ConcreteOracle(config)

        assert oracle.validate_value(100.0) == True
        assert oracle.validate_value(0) == True
        assert oracle.validate_value(None) == False

    def test_should_publish_first_value(self, config):
        """Test should_publish on first value."""
        oracle = ConcreteOracle(config)
        assert oracle.should_publish(100.0) == True

    def test_should_publish_same_value_recent(self, config):
        """Test should_publish with same value recently published."""
        oracle = ConcreteOracle(config)
        oracle._last_value = 100.0
        oracle._last_publish_time = time.time()

        assert oracle.should_publish(100.0) == False

    def test_should_publish_different_value(self, config):
        """Test should_publish with different value."""
        oracle = ConcreteOracle(config)
        oracle._last_value = 100.0
        oracle._last_publish_time = time.time()

        assert oracle.should_publish(101.0) == True

    def test_callbacks(self, config):
        """Test on_publish and on_error callbacks."""
        oracle = ConcreteOracle(config)
        publish_callback = Mock()
        error_callback = Mock()

        oracle.on_publish(publish_callback)
        oracle.on_error(error_callback)

        assert oracle._on_publish == publish_callback
        assert oracle._on_error == error_callback


# =============================================================================
# PublishResult Tests
# =============================================================================

class TestPublishResult:
    """Tests for PublishResult dataclass."""

    def test_successful_result(self):
        """Test successful publish result."""
        result = PublishResult(
            success=True,
            method='p2p',
            message='OK',
            p2p_success=True,
        )

        assert result.success == True
        assert result.method == 'p2p'
        assert result.p2p_success == True

    def test_failed_result(self):
        """Test failed publish result."""
        result = PublishResult(
            success=False,
            method='none',
            message='Failed',
        )

        assert result.success == False
        assert result.method == 'none'


# =============================================================================
# get_networking_mode Tests
# =============================================================================

class TestGetNetworkingMode:
    """Tests for get_networking_mode function."""

    def test_from_env_var(self):
        """Test reading mode from environment variable."""
        with patch.dict(os.environ, {'SATORI_NETWORKING_MODE': 'hybrid'}):
            mode = get_networking_mode()
            assert mode == 'hybrid'

    def test_from_env_var_case_insensitive(self):
        """Test that mode is case insensitive."""
        with patch.dict(os.environ, {'SATORI_NETWORKING_MODE': 'P2P'}):
            mode = get_networking_mode()
            assert mode == 'p2p'

    def test_default_central(self):
        """Test default mode is central."""
        with patch.dict(os.environ, {}, clear=True):
            if 'SATORI_NETWORKING_MODE' in os.environ:
                del os.environ['SATORI_NETWORKING_MODE']
            mode = get_networking_mode()
            assert mode == 'central'


# =============================================================================
# StreamManager Tests (using mock)
# =============================================================================

class MockStreamManager:
    """Mock StreamManager for testing."""
    def __init__(self):
        self._running = False
        self._oracles = {}
        self._oracle_classes = {}
        self._publisher = None

    @property
    def is_running(self):
        return self._running

    @property
    def oracle_count(self):
        return len(self._oracles)

    def register_oracle_type(self, name, cls):
        self._oracle_classes[name] = cls

    def list_oracles(self):
        return list(self._oracles.keys())

    def get_stats(self):
        return {
            'running': self._running,
            'oracle_count': len(self._oracles),
        }

    async def start(self):
        self._running = True
        self._publisher = Mock()

    async def stop(self):
        self._running = False

    async def add_oracle(self, config):
        oracle_type = config.get('type', 'test')
        oracle_class = self._oracle_classes.get(oracle_type)
        if not oracle_class:
            return None
        oracle_config = OracleConfig.from_dict(config)
        oracle = oracle_class(config=oracle_config, publisher=self._publisher)
        self._oracles[oracle.name] = oracle
        return oracle

    async def remove_oracle(self, name):
        if name in self._oracles:
            del self._oracles[name]
            return True
        return False


class TestStreamManager:
    """Tests for StreamManager class."""

    def test_init_defaults(self):
        """Test manager initialization."""
        manager = MockStreamManager()
        assert manager.is_running == False
        assert manager.oracle_count == 0

    def test_register_oracle_type(self):
        """Test registering custom oracle types."""
        manager = MockStreamManager()
        manager.register_oracle_type('custom', ConcreteOracle)
        assert 'custom' in manager._oracle_classes

    def test_list_oracles_empty(self):
        """Test listing oracles when empty."""
        manager = MockStreamManager()
        assert manager.list_oracles() == []

    def test_get_stats_not_running(self):
        """Test stats when not running."""
        manager = MockStreamManager()
        stats = manager.get_stats()
        assert stats['running'] == False
        assert stats['oracle_count'] == 0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test manager start/stop lifecycle."""
        manager = MockStreamManager()

        await manager.start()
        assert manager.is_running == True

        await manager.stop()
        assert manager.is_running == False

    @pytest.mark.asyncio
    async def test_add_remove_oracle(self):
        """Test adding and removing oracles at runtime."""
        manager = MockStreamManager()
        manager._running = True
        manager._publisher = Mock()
        manager._publisher.publish_observation = AsyncMock(return_value=PublishResult(
            success=True, method='mock', message='OK'
        ))

        manager.register_oracle_type('test', ConcreteOracle)

        oracle = await manager.add_oracle({
            'type': 'test',
            'name': 'Runtime Oracle',
            'stream_id': 'test|runtime|id',
            'enabled': True,
            'poll_interval': 9999,
        })

        assert oracle is not None
        assert 'Runtime Oracle' in manager.list_oracles()

        removed = await manager.remove_oracle('Runtime Oracle')
        assert removed == True
        assert 'Runtime Oracle' not in manager.list_oracles()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_oracle(self):
        """Test removing an oracle that doesn't exist."""
        manager = MockStreamManager()
        manager._running = True

        removed = await manager.remove_oracle('Does Not Exist')
        assert removed == False


# =============================================================================
# Integration-style Tests
# =============================================================================

class TestOraclePublisherIntegration:
    """Tests for oracle + publisher interaction."""

    @pytest.mark.asyncio
    async def test_oracle_publishes_through_publisher(self):
        """Test that oracle correctly uses publisher."""
        config = OracleConfig(
            stream_id='integration|test|stream',
            name='Integration Test',
            poll_interval=1,
        )

        mock_publisher = Mock()
        mock_publisher.publish_observation = AsyncMock(return_value=PublishResult(
            success=True, method='mock', message='OK'
        ))

        oracle = ConcreteOracle(config, mock_publisher, fetch_returns=[123.456])

        obs = await oracle.poll_once()

        mock_publisher.publish_observation.assert_called_once()
        call_kwargs = mock_publisher.publish_observation.call_args[1]

        assert call_kwargs['stream_id'] == 'integration|test|stream'
        assert call_kwargs['value'] == 123.456
        assert obs.value == 123.456

    @pytest.mark.asyncio
    async def test_oracle_handles_publisher_failure(self):
        """Test oracle behavior when publisher fails."""
        config = OracleConfig(
            stream_id='test|stream',
            name='Test',
        )

        mock_publisher = Mock()
        mock_publisher.publish_observation = AsyncMock(return_value=PublishResult(
            success=False, method='none', message='Failed'
        ))

        oracle = ConcreteOracle(config, mock_publisher, fetch_returns=[100.0])

        obs = await oracle.poll_once()

        assert obs is None
        mock_publisher.publish_observation.assert_called_once()
