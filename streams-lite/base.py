"""
Base Oracle Class for Streams-Lite

Provides the foundation for all data oracle implementations.
Oracles fetch data from external sources and publish observations.
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)


@dataclass
class OracleConfig:
    """Configuration for an oracle stream."""

    # Stream identification
    stream_id: str              # Format: "source|author|stream|target"
    name: str                   # Human-readable name

    # Polling settings
    poll_interval: int = 3600   # Seconds between polls (default: 1 hour)
    retry_interval: int = 60    # Seconds to wait after error
    max_retries: int = 3        # Max consecutive retries before backoff

    # Optional settings
    enabled: bool = True        # Whether this oracle is active
    api_key: str = None         # API key if required
    extra: Dict[str, Any] = field(default_factory=dict)  # Custom config

    @classmethod
    def from_dict(cls, data: dict) -> 'OracleConfig':
        """Create config from dictionary."""
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
        """Convert to dictionary."""
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


class BaseOracle(ABC):
    """
    Abstract base class for data oracles.

    Subclasses must implement:
        - fetch_value(): Get the current value from the data source

    Optional overrides:
        - setup(): One-time initialization
        - cleanup(): Cleanup on shutdown
        - validate_value(): Validate fetched values
    """

    def __init__(self, config: OracleConfig, publisher=None):
        """
        Initialize the oracle.

        Args:
            config: Oracle configuration
            publisher: P2PPublisher instance (shared across oracles)
        """
        self.config = config
        self.publisher = publisher

        # State tracking
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_value: Optional[float] = None
        self._last_publish_time: int = 0
        self._consecutive_errors: int = 0
        self._total_publishes: int = 0
        self._total_errors: int = 0

        # Callbacks
        self._on_publish: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

    @property
    def stream_id(self) -> str:
        """Get the stream ID."""
        return self.config.stream_id

    @property
    def name(self) -> str:
        """Get the oracle name."""
        return self.config.name

    @property
    def is_running(self) -> bool:
        """Check if oracle is running."""
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        """Get oracle statistics."""
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

    # =========================================================================
    # Abstract Methods (must implement)
    # =========================================================================

    @abstractmethod
    async def fetch_value(self) -> Optional[float]:
        """
        Fetch the current value from the data source.

        Returns:
            The observed value, or None if unavailable

        Raises:
            Any exception on error (will be caught and logged)
        """
        pass

    # =========================================================================
    # Optional Overrides
    # =========================================================================

    async def setup(self):
        """
        One-time setup before polling starts.

        Override to initialize connections, validate API keys, etc.
        """
        pass

    async def cleanup(self):
        """
        Cleanup when oracle stops.

        Override to close connections, save state, etc.
        """
        pass

    def validate_value(self, value: float) -> bool:
        """
        Validate a fetched value before publishing.

        Override to add custom validation logic.

        Args:
            value: The fetched value

        Returns:
            True if valid, False to skip publishing
        """
        return value is not None

    def should_publish(self, value: float) -> bool:
        """
        Determine if a value should be published.

        Default: publish if value changed or enough time passed.
        Override for custom logic.
        """
        now = time.time()
        time_since_last = now - self._last_publish_time

        # Always publish if enough time has passed
        if time_since_last >= self.config.poll_interval:
            return True

        # Publish if value changed significantly
        if self._last_value is None:
            return True

        # Simple change detection (override for custom logic)
        return value != self._last_value

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def start(self):
        """Start the oracle polling loop."""
        if self._running:
            logger.warning(f"Oracle {self.name} already running")
            return

        if not self.config.enabled:
            logger.info(f"Oracle {self.name} is disabled, not starting")
            return

        try:
            await self.setup()
        except Exception as e:
            logger.error(f"Oracle {self.name} setup failed: {e}")
            raise

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Oracle {self.name} started (interval: {self.config.poll_interval}s)")

    async def stop(self):
        """Stop the oracle."""
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

        try:
            await self.cleanup()
        except Exception as e:
            logger.warning(f"Oracle {self.name} cleanup error: {e}")

        logger.info(f"Oracle {self.name} stopped")

    async def poll_once(self) -> Optional[Observation]:
        """
        Perform a single poll and publish cycle.

        Returns:
            The observation if published, None otherwise
        """
        try:
            # Fetch value
            value = await self.fetch_value()

            if value is None:
                logger.debug(f"Oracle {self.name}: no value fetched")
                return None

            # Validate
            if not self.validate_value(value):
                logger.debug(f"Oracle {self.name}: value {value} failed validation")
                return None

            # Check if should publish
            if not self.should_publish(value):
                logger.debug(f"Oracle {self.name}: skipping publish (no change)")
                return None

            # Publish
            observation = await self._publish(value)

            # Reset error counter on success
            self._consecutive_errors = 0

            return observation

        except Exception as e:
            self._consecutive_errors += 1
            self._total_errors += 1
            logger.error(f"Oracle {self.name} poll error: {e}")

            if self._on_error:
                try:
                    self._on_error(self, e)
                except Exception:
                    pass

            return None

    async def _publish(self, value: float) -> Optional[Observation]:
        """Publish an observation."""
        timestamp = int(time.time())

        if self.publisher:
            result = await self.publisher.publish_observation(
                stream_id=self.stream_id,
                value=value,
                timestamp=timestamp,
            )

            if not result.success:
                logger.warning(f"Oracle {self.name}: publish failed - {result.message}")
                return None

        # Update state
        self._last_value = value
        self._last_publish_time = timestamp
        self._total_publishes += 1

        observation = Observation(
            stream_id=self.stream_id,
            value=value,
            timestamp=timestamp,
        )

        logger.info(f"Oracle {self.name}: published {value}")

        # Callback
        if self._on_publish:
            try:
                self._on_publish(self, observation)
            except Exception:
                pass

        return observation

    async def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                await self.poll_once()

                # Calculate sleep time (with backoff on errors)
                if self._consecutive_errors > 0:
                    backoff = min(
                        self.config.retry_interval * (2 ** (self._consecutive_errors - 1)),
                        self.config.poll_interval
                    )
                    sleep_time = backoff
                else:
                    sleep_time = self.config.poll_interval

                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Oracle {self.name} loop error: {e}")
                await asyncio.sleep(self.config.retry_interval)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_publish(self, callback: Callable[['BaseOracle', Observation], None]):
        """Set callback for successful publishes."""
        self._on_publish = callback

    def on_error(self, callback: Callable[['BaseOracle', Exception], None]):
        """Set callback for errors."""
        self._on_error = callback
