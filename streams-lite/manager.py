"""
Stream Manager for Streams-Lite

Orchestrates multiple oracle streams, managing their lifecycle,
configuration, and coordination with the main neuron.
"""

import os
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Type
from pathlib import Path

import yaml

from .base import BaseOracle, OracleConfig, Observation
from .publisher import P2PPublisher, get_networking_mode

# Try to import trio for compatibility with py-libp2p environment
try:
    import trio
    HAS_TRIO = True
except ImportError:
    HAS_TRIO = False

logger = logging.getLogger(__name__)


class StreamManager:
    """
    Manages multiple oracle streams.

    Responsibilities:
    - Load oracle configurations
    - Start/stop oracle streams
    - Share P2P publisher across oracles
    - Provide status/stats API
    - Handle WebSocket event emission
    """

    def __init__(
        self,
        config_path: str = None,
        peers=None,
        identity=None,
        send_to_ui=None,
        oracle_network=None,
    ):
        """
        Initialize the stream manager.

        Args:
            config_path: Path to streams config YAML
            peers: Shared Peers instance from neuron (optional)
            identity: Shared identity from neuron (optional)
            send_to_ui: Callback to emit WebSocket events (optional)
            oracle_network: Shared OracleNetwork instance for publishing (optional)
        """
        self.config_path = config_path or self._find_config()
        self._peers = peers
        self._identity = identity
        self._send_to_ui = send_to_ui
        self._oracle_network = oracle_network

        # Components
        self._publisher: Optional[P2PPublisher] = None
        self._oracles: Dict[str, BaseOracle] = {}
        self._oracle_classes: Dict[str, Type[BaseOracle]] = {}

        # State
        self._running = False
        self._config: Dict[str, Any] = {}

        # Register built-in oracle types
        self._register_builtin_oracles()

    def _find_config(self) -> str:
        """Find streams config file."""
        paths = [
            os.path.expanduser('~/.satori/streams.yaml'),
            '/satori/config/streams.yaml',
            'streams.yaml',
            os.path.join(os.path.dirname(__file__), 'config.yaml'),
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        return paths[0]  # Default to first path

    def _register_builtin_oracles(self):
        """Register built-in oracle types."""
        try:
            from .sources.fred import FREDOracle
            self._oracle_classes['fred'] = FREDOracle
        except ImportError:
            pass

        try:
            from .sources.crypto import CryptoOracle
            self._oracle_classes['crypto'] = CryptoOracle
        except ImportError:
            pass

        try:
            from .sources.http import HTTPOracle
            self._oracle_classes['http'] = HTTPOracle
        except ImportError:
            pass

    def register_oracle_type(self, name: str, oracle_class: Type[BaseOracle]):
        """Register a custom oracle type."""
        self._oracle_classes[name] = oracle_class

    async def start(self):
        """Start the stream manager and all enabled oracles."""
        if self._running:
            logger.warning("StreamManager already running")
            return

        logger.info(f"Starting StreamManager (mode: {get_networking_mode()})")

        # Initialize shared publisher
        self._publisher = P2PPublisher(
            identity=self._identity,
            peers=self._peers,
        )

        # Share peers instance if provided
        if self._peers:
            self._publisher.set_peers(self._peers)

        # Share oracle network if provided (use main network's instance)
        if self._oracle_network:
            self._publisher.set_oracle_network(self._oracle_network)

        await self._publisher.start()

        # Load configuration
        self._load_config()

        # Create and start oracles
        await self._start_oracles()

        self._running = True
        logger.info(f"StreamManager started with {len(self._oracles)} oracles")

    async def stop(self):
        """Stop all oracles and cleanup."""
        if not self._running:
            return

        logger.info("Stopping StreamManager...")

        # Stop all oracles
        for name, oracle in self._oracles.items():
            try:
                await oracle.stop()
                logger.debug(f"Stopped oracle: {name}")
            except Exception as e:
                logger.warning(f"Error stopping oracle {name}: {e}")

        self._oracles.clear()

        # Stop publisher
        if self._publisher:
            await self._publisher.stop()
            self._publisher = None

        self._running = False
        logger.info("StreamManager stopped")

    def _load_config(self):
        """Load oracle configurations from YAML."""
        if not os.path.exists(self.config_path):
            logger.info(f"No streams config at {self.config_path}, using defaults")
            self._config = self._default_config()
            return

        try:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded streams config from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            self._config = self._default_config()

    def _default_config(self) -> dict:
        """Default configuration with example oracles."""
        return {
            'enabled': True,
            'oracles': [
                # Example FRED oracle (disabled by default)
                {
                    'type': 'fred',
                    'name': '10-Year Treasury Rate',
                    'stream_id': 'fred|satori|DGS10|rate',
                    'enabled': False,
                    'poll_interval': 3600,
                    'extra': {
                        'series_id': 'DGS10',
                    }
                },
            ]
        }

    async def _start_oracles(self):
        """Create and start all configured oracles."""
        if not self._config.get('enabled', True):
            logger.info("Streams disabled in config")
            return

        oracles_config = self._config.get('oracles', [])

        # Check if we're in a Trio context
        in_trio = False
        if HAS_TRIO:
            try:
                trio.lowlevel.current_trio_token()
                in_trio = True
            except RuntimeError:
                pass

        for oracle_cfg in oracles_config:
            try:
                oracle = self._create_oracle(oracle_cfg)
                if oracle and oracle.config.enabled:
                    # Set up callbacks
                    oracle.on_publish(self._on_oracle_publish)
                    oracle.on_error(self._on_oracle_error)

                    await oracle.start()
                    self._oracles[oracle.name] = oracle

            except Exception as e:
                logger.error(f"Failed to create oracle: {e}")

        # If in Trio mode, start a background polling task
        if in_trio and self._oracles:
            self._trio_polling_task = True
            # We can't easily spawn a trio task from here, but we can
            # provide a method that the caller can use to run polling
            logger.info(f"Trio mode: {len(self._oracles)} oracles ready for polling")

    def _create_oracle(self, config: dict) -> Optional[BaseOracle]:
        """Create an oracle from configuration."""
        oracle_type = config.get('type', 'http')
        oracle_class = self._oracle_classes.get(oracle_type)

        if not oracle_class:
            logger.warning(f"Unknown oracle type: {oracle_type}")
            return None

        oracle_config = OracleConfig.from_dict(config)
        return oracle_class(config=oracle_config, publisher=self._publisher)

    def _on_oracle_publish(self, oracle: BaseOracle, observation: Observation):
        """Handle oracle publish events."""
        # Emit WebSocket event if callback provided
        if self._send_to_ui:
            try:
                self._send_to_ui('observation', {
                    'stream_id': observation.stream_id,
                    'value': observation.value,
                    'timestamp': observation.timestamp,
                    'oracle': oracle.name,
                })
            except Exception:
                pass

    def _on_oracle_error(self, oracle: BaseOracle, error: Exception):
        """Handle oracle error events."""
        logger.warning(f"Oracle {oracle.name} error: {error}")

    async def run_trio_polling(self):
        """
        Run the oracle polling loop in Trio context.

        Call this from a Trio nursery to enable oracle polling.
        This is needed because py-libp2p uses Trio, not asyncio.
        """
        import random

        if not HAS_TRIO:
            logger.warning("Trio not available, polling disabled")
            return

        logger.info(f"Starting Trio polling loop for {len(self._oracles)} oracles")

        # Add random initial delay (0-60s) to stagger containers hitting same APIs
        initial_jitter = random.uniform(0, 60)
        logger.info(f"Waiting {initial_jitter:.1f}s initial jitter before starting oracle polling")
        await trio.sleep(initial_jitter)

        while self._running:
            for name, oracle in list(self._oracles.items()):
                if not oracle._running:
                    continue

                try:
                    # Check if it's time to poll this oracle
                    now = time.time()
                    time_since_last = now - oracle._last_publish_time

                    # Add small random jitter (0-30s) to poll interval to prevent API stampede
                    jitter = random.uniform(0, 30) if oracle._last_publish_time > 0 else 0
                    effective_interval = oracle.config.poll_interval + jitter

                    if time_since_last >= effective_interval:
                        # Fetch and publish
                        try:
                            value = await oracle.fetch_value()
                            if value is not None and oracle.validate_value(value):
                                await oracle._publish(value)
                        except Exception as e:
                            oracle._consecutive_errors += 1
                            oracle._total_errors += 1
                            if oracle._on_error:
                                oracle._on_error(oracle, e)
                            logger.warning(f"Oracle {name} error: {e}")

                except Exception as e:
                    logger.debug(f"Oracle {name} poll error: {e}")

            # Sleep between poll cycles
            await trio.sleep(10)  # Check every 10 seconds

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    @property
    def oracle_count(self) -> int:
        """Get number of active oracles."""
        return len(self._oracles)

    def get_oracle(self, name: str) -> Optional[BaseOracle]:
        """Get an oracle by name."""
        return self._oracles.get(name)

    def list_oracles(self) -> List[str]:
        """List all oracle names."""
        return list(self._oracles.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all oracles."""
        return {
            'running': self._running,
            'mode': get_networking_mode(),
            'oracle_count': len(self._oracles),
            'oracles': {
                name: oracle.stats
                for name, oracle in self._oracles.items()
            }
        }

    async def add_oracle(self, config: dict) -> Optional[BaseOracle]:
        """
        Add a new oracle at runtime.

        Args:
            config: Oracle configuration dict

        Returns:
            The created oracle, or None on failure
        """
        try:
            oracle = self._create_oracle(config)
            if oracle:
                oracle.on_publish(self._on_oracle_publish)
                oracle.on_error(self._on_oracle_error)
                await oracle.start()
                self._oracles[oracle.name] = oracle
                return oracle
        except Exception as e:
            logger.error(f"Failed to add oracle: {e}")
        return None

    async def remove_oracle(self, name: str) -> bool:
        """
        Remove an oracle by name.

        Args:
            name: Oracle name

        Returns:
            True if removed, False if not found
        """
        oracle = self._oracles.pop(name, None)
        if oracle:
            await oracle.stop()
            return True
        return False

    async def reload_config(self):
        """Reload configuration and restart oracles."""
        # Stop current oracles
        for oracle in list(self._oracles.values()):
            await oracle.stop()
        self._oracles.clear()

        # Reload and restart
        self._load_config()
        await self._start_oracles()

        logger.info(f"Reloaded config, now running {len(self._oracles)} oracles")
