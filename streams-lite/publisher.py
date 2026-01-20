"""
P2P Publisher for Streams-Lite

Publishes observations to the Satori network via P2P and/or HTTP.
Supports central, hybrid, and p2p networking modes.
"""

import os
import time
import json
import asyncio
import logging
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PublishResult:
    """Result of a publish operation."""
    success: bool
    method: str  # 'p2p', 'http', 'both', 'none'
    message: str
    p2p_success: bool = False
    http_success: bool = False


def get_networking_mode() -> str:
    """Get the current networking mode from environment or config."""
    # First try environment variable
    mode = os.environ.get('SATORI_NETWORKING_MODE', '').lower().strip()
    if mode:
        return mode

    # Try to read from config file
    try:
        import yaml
        config_paths = [
            os.path.expanduser('~/.satori/config.yaml'),
            '/Satori/Neuron/config/config.yaml',  # Container path
            '/satori/config.yaml',
            'config.yaml',
        ]
        for path in config_paths:
            if os.path.exists(path):
                with open(path) as f:
                    config = yaml.safe_load(f)
                    if config and 'networking mode' in config:
                        return config['networking mode'].lower().strip()
    except Exception:
        pass

    return 'central'


class P2PPublisher:
    """
    Publisher that sends observations via P2P network and/or HTTP.

    Modes:
    - central: HTTP only (default/legacy)
    - hybrid: P2P first, HTTP fallback
    - p2p: P2P only
    """

    def __init__(
        self,
        wallet_path: str = None,
        neuron_url: str = 'http://localhost:24601',
        p2p_port: int = 24600,
        identity=None,
        peers=None,
    ):
        """
        Initialize the publisher.

        Args:
            wallet_path: Path to wallet.yaml for P2P identity
            neuron_url: URL of local Neuron for HTTP
            p2p_port: Port for P2P networking
            identity: Pre-initialized identity (optional)
            peers: Pre-initialized Peers instance (optional)
        """
        self.wallet_path = wallet_path or os.path.expanduser('~/.satori/wallet/wallet.yaml')
        self.neuron_url = neuron_url
        self.p2p_port = p2p_port
        self.mode = get_networking_mode()

        # P2P components (can be injected or lazy initialized)
        self._peers = peers
        self._oracle_network = None
        self._identity = identity
        self._owns_peers = False  # Whether we created the Peers instance
        self._started = False

    async def start(self):
        """Initialize P2P components if in hybrid/p2p mode."""
        if self._started:
            return

        if self.mode in ('hybrid', 'p2p', 'p2p_only'):
            try:
                await self._init_p2p()
                logger.info(f"P2P publisher started in {self.mode} mode")
            except ImportError as e:
                logger.warning(f"satorip2p not installed ({e}), using HTTP only")
                self.mode = 'central'
            except Exception as e:
                logger.warning(f"P2P init failed: {e}, using HTTP only")
                if self.mode == 'p2p':
                    raise  # In pure P2P mode, this is fatal
                self.mode = 'central'

        self._started = True

    async def _init_p2p(self):
        """Initialize P2P networking components."""
        # Only create new Peers if not injected
        if self._peers is None:
            from satorip2p import Peers
            from satorilib.wallet.evrmore.identity import EvrmoreIdentity

            # Load identity from wallet
            if self._identity is None:
                if os.path.exists(self.wallet_path):
                    self._identity = EvrmoreIdentity(self.wallet_path)
                else:
                    raise FileNotFoundError(f"Wallet not found: {self.wallet_path}")

            # Initialize P2P peers
            self._peers = Peers(
                identity=self._identity,
                listen_port=self.p2p_port,
            )
            await self._peers.start()
            self._owns_peers = True

        # Only create new OracleNetwork if not injected
        if self._oracle_network is None:
            from satorip2p.protocol.oracle_network import OracleNetwork
            self._oracle_network = OracleNetwork(self._peers)
            await self._oracle_network.start()
            logger.debug("Created new OracleNetwork for publisher")
        else:
            logger.debug("Using injected OracleNetwork for publisher")

        logger.debug("P2P publisher components initialized")

    async def stop(self):
        """Stop P2P components."""
        if self._oracle_network:
            try:
                await self._oracle_network.stop()
            except Exception:
                pass
            self._oracle_network = None

        # Only stop Peers if we created it
        if self._owns_peers and self._peers:
            try:
                await self._peers.stop()
            except Exception:
                pass
            self._peers = None

        self._started = False

    def set_peers(self, peers):
        """
        Inject a shared Peers instance.

        Call this before start() to share the Peers instance
        with the main neuron instead of creating a new one.
        """
        self._peers = peers
        self._owns_peers = False

    def set_oracle_network(self, oracle_network):
        """
        Inject a shared OracleNetwork instance.

        Call this before start() to share the OracleNetwork instance
        with the main neuron instead of creating a new one.
        """
        self._oracle_network = oracle_network

    async def publish_observation(
        self,
        stream_id: str,
        value: float,
        timestamp: int = None,
        metadata: Dict[str, Any] = None,
    ) -> PublishResult:
        """
        Publish an observation to the network.

        Args:
            stream_id: Stream identifier (format: "source|author|stream|target")
            value: Observed value
            timestamp: Unix timestamp (default: now)
            metadata: Optional metadata dict

        Returns:
            PublishResult with success status and method used
        """
        if not self._started:
            await self.start()

        timestamp = timestamp or int(time.time())
        p2p_success = False
        http_success = False
        messages = []

        # Try P2P first if in hybrid/p2p mode
        if self.mode in ('hybrid', 'p2p', 'p2p_only') and self._oracle_network:
            try:
                observation = await self._oracle_network.publish_observation(
                    stream_id=stream_id,
                    value=value,
                    timestamp=timestamp,
                )
                if observation:
                    p2p_success = True
                    messages.append("P2P OK")
                    logger.debug(f"Published via P2P: {stream_id} = {value}")
            except Exception as e:
                messages.append(f"P2P failed: {e}")
                logger.debug(f"P2P publish failed: {e}")

        # Try HTTP if in central/hybrid mode or P2P failed
        should_try_http = (
            self.mode == 'central' or
            (self.mode == 'hybrid' and not p2p_success)
        )

        if should_try_http:
            try:
                http_success = await self._publish_http(stream_id, value, timestamp, metadata)
                if http_success:
                    messages.append("HTTP OK")
            except Exception as e:
                messages.append(f"HTTP failed: {e}")
                logger.debug(f"HTTP publish failed: {e}")

        # Determine result
        success = p2p_success or http_success
        if p2p_success and http_success:
            method = 'both'
        elif p2p_success:
            method = 'p2p'
        elif http_success:
            method = 'http'
        else:
            method = 'none'

        return PublishResult(
            success=success,
            method=method,
            message='; '.join(messages),
            p2p_success=p2p_success,
            http_success=http_success,
        )

    async def _publish_http(
        self,
        stream_id: str,
        value: float,
        timestamp: int,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Publish observation via HTTP to local Neuron."""
        try:
            # Parse stream_id
            parts = stream_id.split('|')
            if len(parts) >= 4:
                source, author, stream, target = parts[0], parts[1], parts[2], parts[3]
            else:
                source, author, stream, target = 'satori', '', stream_id, ''

            # Prepare payload
            payload = {
                'source': source,
                'author': author,
                'stream': stream,
                'target': target,
                'value': str(value),
                'timestamp': str(timestamp),
            }
            if metadata:
                payload['metadata'] = json.dumps(metadata)

            # Post to Neuron (run in executor to not block async)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.neuron_url}/relay/submit",
                    data=payload,
                    timeout=10
                )
            )

            if response.ok:
                logger.debug(f"Published via HTTP: {stream_id} = {value}")
                return True
            else:
                logger.debug(f"HTTP returned {response.status_code}")
                return False

        except requests.RequestException as e:
            logger.debug(f"HTTP request failed: {e}")
            return False
