# Note: No monkey-patching here to maintain compatibility with trio (used by libp2p)
# Flask-SocketIO will use threading mode which works fine for our use case

from typing import Union, Callable
import os
import time
import json
import threading
import hashlib
from satorilib.concepts.structs import StreamId, Stream
from satorilib.concepts import constants
from satorilib.wallet import EvrmoreWallet
from satorilib.wallet.evrmore.identity import EvrmoreIdentity
from satorilib.server.api import CheckinDetails
from satorineuron import VERSION
from satorineuron import logging
from satorineuron import config
from satorineuron.init.wallet import WalletManager
from satorineuron.structs.start import RunMode, StartupDagStruct
# from satorilib.utils.ip import getPublicIpv4UsingCurl  # Removed - not needed
from satoriengine.veda.engine import Engine


def _get_server_client_class():
    """Get appropriate SatoriServerClient based on networking mode.

    Supports P2P networking via satorip2p when configured.
    Falls back to central server client if P2P not available.
    """
    # Check environment variable first, then config
    networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
    if networking_mode is None:
        try:
            networking_mode = config.get().get('networking mode', 'central')
        except Exception:
            networking_mode = 'central'

    networking_mode = networking_mode.lower().strip()

    if networking_mode in ('hybrid', 'p2p', 'p2p_only'):
        try:
            from satorip2p.integration import P2PSatoriServerClient
            logging.info(f"Using P2P networking mode: {networking_mode}", color="cyan")
            return P2PSatoriServerClient
        except ImportError:
            logging.warning(
                "satorip2p not installed, falling back to central server",
                color="yellow"
            )

    from satorilib.server import SatoriServerClient
    return SatoriServerClient


# Get the appropriate client class based on config
SatoriServerClient = _get_server_client_class()


def _get_networking_mode() -> str:
    """Get the current networking mode from config or environment.

    Priority order:
    1. Config file (allows runtime switching via UI)
    2. Environment variable (fallback for Docker override)
    3. Default: 'hybrid'
    """
    # Check config file first (allows UI changes to take effect)
    try:
        mode = config.get().get('networking mode')
        if mode:
            return mode.lower().strip()
    except Exception:
        pass

    # Fallback to environment variable
    mode = os.environ.get('SATORI_NETWORKING_MODE')
    if mode:
        return mode.lower().strip()

    return 'hybrid'


def _set_networking_mode(new_mode: str) -> bool:
    """
    Set the networking mode.

    Updates both the environment variable (for current process) and
    the config file (for persistence across restarts).

    Note: A full restart is recommended for the change to fully take effect,
    as P2P components may already be initialized.

    Args:
        new_mode: One of 'central', 'hybrid', or 'p2p'

    Returns:
        True if successful, False otherwise
    """
    import yaml

    new_mode = new_mode.lower().strip()
    if new_mode not in ('central', 'hybrid', 'p2p'):
        logging.warning(f"Invalid networking mode: {new_mode}")
        return False

    try:
        # Update environment variable for current process
        os.environ['SATORI_NETWORKING_MODE'] = new_mode

        # Save to config file for persistence
        config_path = config.root('config', 'config.yaml')
        cfg = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        cfg['networking mode'] = new_mode
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        logging.info(f"Networking mode set to: {new_mode}", color="cyan")
        return True
    except Exception as e:
        logging.error(f"Failed to set networking mode: {e}")
        return False


# P2P Module Imports (lazy-loaded for optional dependency)
def _get_p2p_modules():
    """
    Get all available P2P modules from satorip2p.
    Returns a dict of module references, or empty dict if satorip2p not installed.

    Available modules:
    - Peers: Core P2P networking
    - EvrmoreIdentityBridge: Wallet-to-P2P identity
    - UptimeTracker, Heartbeat: Node uptime and relay bonus tracking
    - ConsensusManager, ConsensusVote: Stake-weighted voting
    - SignerNode: Multi-sig signing (3-of-5)
    - DistributionTrigger: Reward distribution coordination
    - SatoriScorer, RewardCalculator: Local reward calculation
    - OracleNetwork: P2P observation publishing
    - PredictionProtocol: Commit-reveal predictions
    - PeerRegistry, StreamRegistry: P2P discovery
    """
    try:
        from satorip2p import (
            Peers,
            EvrmoreIdentityBridge,
            SatoriScorer,
            RewardCalculator,
            RoundDataStore,
            PredictionInput,
            ScoreBreakdown,
            NetworkingMode,
            get_networking_mode as p2p_get_networking_mode,
        )
        from satorip2p.protocol.uptime import (
            UptimeTracker,
            Heartbeat,
            RELAY_UPTIME_THRESHOLD,
            HEARTBEAT_INTERVAL,
        )
        from satorip2p.protocol.consensus import (
            ConsensusManager,
            ConsensusVote,
            ConsensusPhase,
        )
        from satorip2p.protocol.signer import (
            SignerNode,
            MULTISIG_THRESHOLD,
            AUTHORIZED_SIGNERS,
        )
        from satorip2p.protocol.distribution_trigger import (
            DistributionTrigger,
        )
        from satorip2p.protocol.peer_registry import PeerRegistry
        from satorip2p.protocol.stream_registry import StreamRegistry
        from satorip2p.protocol.oracle_network import OracleNetwork
        from satorip2p.protocol.prediction_protocol import PredictionProtocol
        from satorip2p.protocol.lending import LendingManager, PoolConfig, LendRegistration
        from satorip2p.protocol.delegation import DelegationManager, DelegationRecord, CharityUpdate
        from satorip2p.protocol.ping import (
            PingProtocol, PingRequest, PongResponse,
            PING_TOPIC, PONG_TOPIC, PING_TIMEOUT,
        )
        from satorip2p.protocol.identify import (
            IdentifyProtocol, PeerIdentity, IdentifyRequest,
            IDENTIFY_TOPIC, IDENTIFY_REQUEST_TOPIC,
        )
        from satorip2p.signing import (
            EvrmoreWallet as P2PEvrmoreWallet,
            sign_message,
            verify_message,
        )
        # New protocol features: versioning, storage, bandwidth
        from satorip2p.protocol.versioning import (
            ProtocolVersion, VersionNegotiator, PeerVersionTracker,
            PROTOCOL_VERSION, MIN_SUPPORTED_VERSION, get_current_version,
        )
        from satorip2p.protocol.storage import (
            StorageManager, DeferredRewardsStorage, AlertStorage,
            MemoryBackend, FileBackend, DHTBackend,
        )
        from satorip2p.protocol.bandwidth import (
            BandwidthTracker, QoSManager, QoSPolicy, MessagePriority,
            create_qos_manager, get_priority_for_message_type,
        )
        # Referral, Pricing, and Reward Address
        from satorip2p.protocol.referral import (
            ReferralManager, Referral, ReferrerStats,
            get_tier_for_count, get_bonus_for_tier,
        )
        from satorip2p.protocol.pricing import (
            SafeTradePriceProvider, PriceQuote, TickerData,
            get_price_provider, calculate_satori_reward,
        )
        from satorip2p.protocol.reward_address import (
            RewardAddressManager, RewardAddressRecord,
        )
        return {
            'available': True,
            'Peers': Peers,
            'EvrmoreIdentityBridge': EvrmoreIdentityBridge,
            'UptimeTracker': UptimeTracker,
            'Heartbeat': Heartbeat,
            'RELAY_UPTIME_THRESHOLD': RELAY_UPTIME_THRESHOLD,
            'HEARTBEAT_INTERVAL': HEARTBEAT_INTERVAL,
            'ConsensusManager': ConsensusManager,
            'ConsensusVote': ConsensusVote,
            'ConsensusPhase': ConsensusPhase,
            'SignerNode': SignerNode,
            'MULTISIG_THRESHOLD': MULTISIG_THRESHOLD,
            'AUTHORIZED_SIGNERS': AUTHORIZED_SIGNERS,
            'DistributionTrigger': DistributionTrigger,
            'SatoriScorer': SatoriScorer,
            'RewardCalculator': RewardCalculator,
            'RoundDataStore': RoundDataStore,
            'PredictionInput': PredictionInput,
            'ScoreBreakdown': ScoreBreakdown,
            'PeerRegistry': PeerRegistry,
            'StreamRegistry': StreamRegistry,
            'OracleNetwork': OracleNetwork,
            'PredictionProtocol': PredictionProtocol,
            'LendingManager': LendingManager,
            'PoolConfig': PoolConfig,
            'LendRegistration': LendRegistration,
            'DelegationManager': DelegationManager,
            'DelegationRecord': DelegationRecord,
            'CharityUpdate': CharityUpdate,
            'P2PEvrmoreWallet': P2PEvrmoreWallet,
            'sign_message': sign_message,
            'verify_message': verify_message,
            'NetworkingMode': NetworkingMode,
            # New protocol features
            'ProtocolVersion': ProtocolVersion,
            'VersionNegotiator': VersionNegotiator,
            'PeerVersionTracker': PeerVersionTracker,
            'PROTOCOL_VERSION': PROTOCOL_VERSION,
            'MIN_SUPPORTED_VERSION': MIN_SUPPORTED_VERSION,
            'get_current_version': get_current_version,
            'StorageManager': StorageManager,
            'DeferredRewardsStorage': DeferredRewardsStorage,
            'AlertStorage': AlertStorage,
            'MemoryBackend': MemoryBackend,
            'FileBackend': FileBackend,
            'DHTBackend': DHTBackend,
            'BandwidthTracker': BandwidthTracker,
            'QoSManager': QoSManager,
            'QoSPolicy': QoSPolicy,
            'MessagePriority': MessagePriority,
            'create_qos_manager': create_qos_manager,
            'get_priority_for_message_type': get_priority_for_message_type,
            # Referral, Pricing, Reward Address
            'ReferralManager': ReferralManager,
            'Referral': Referral,
            'ReferrerStats': ReferrerStats,
            'get_tier_for_count': get_tier_for_count,
            'get_bonus_for_tier': get_bonus_for_tier,
            'SafeTradePriceProvider': SafeTradePriceProvider,
            'PriceQuote': PriceQuote,
            'TickerData': TickerData,
            'get_price_provider': get_price_provider,
            'calculate_satori_reward': calculate_satori_reward,
            'RewardAddressManager': RewardAddressManager,
            'RewardAddressRecord': RewardAddressRecord,
            # Ping and Identify protocols
            'PingProtocol': PingProtocol,
            'PingRequest': PingRequest,
            'PongResponse': PongResponse,
            'PING_TOPIC': PING_TOPIC,
            'PONG_TOPIC': PONG_TOPIC,
            'PING_TIMEOUT': PING_TIMEOUT,
            'IdentifyProtocol': IdentifyProtocol,
            'PeerIdentity': PeerIdentity,
            'IdentifyRequest': IdentifyRequest,
            'IDENTIFY_TOPIC': IDENTIFY_TOPIC,
            'IDENTIFY_REQUEST_TOPIC': IDENTIFY_REQUEST_TOPIC,
        }
    except ImportError:
        return {'available': False}


# Cache the P2P modules (lazy loaded on first access)
_p2p_modules_cache = None


def get_p2p_module(name: str):
    """Get a specific P2P module by name, or None if not available."""
    global _p2p_modules_cache
    if _p2p_modules_cache is None:
        _p2p_modules_cache = _get_p2p_modules()
    return _p2p_modules_cache.get(name)


def is_p2p_available() -> bool:
    """Check if satorip2p is installed and available."""
    global _p2p_modules_cache
    if _p2p_modules_cache is None:
        _p2p_modules_cache = _get_p2p_modules()
    return _p2p_modules_cache.get('available', False)


# Global singleton instance - accessible across imports
_startup_instance = None


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        global _startup_instance
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
            _startup_instance = cls._instances[cls]
        return cls._instances[cls]


def getStart():
    """Returns StartupDag singleton if it exists, None otherwise.

    Note: This returns the actual running instance with all protocols
    initialized, not a new blank instance.
    """
    global _startup_instance
    return _startup_instance


class StartupDag(StartupDagStruct, metaclass=SingletonMeta):
    """a DAG of startup tasks."""


    @classmethod
    def create(
        cls,
        *args,
        env: str = 'prod',
        runMode: str = None,
        isDebug: bool = False,
    ) -> 'StartupDag':
        '''Factory method to create and initialize StartupDag'''
        startupDag = cls(
            *args,
            env=env,
            runMode=runMode,
            isDebug=isDebug)
        startupDag.startFunction()
        return startupDag

    def __init__(
        self,
        *args,
        env: str = 'dev',
        runMode: str = None,
        isDebug: bool = False,
    ):
        super(StartupDag, self).__init__(*args)
        self.env = env
        self.runMode = RunMode.choose(runMode or config.get().get('mode', None))
        self.uiPort = self.getUiPort()
        self.walletManager: WalletManager
        self.isDebug: bool = isDebug
        self.balances: dict = {}
        self.aiengine: Union[Engine, None] = None
        self.publications: list[Stream] = []  # Keep for engine
        self.subscriptions: list[Stream] = []  # Keep for engine
        self.identity: EvrmoreIdentity = EvrmoreIdentity(config.walletPath('wallet.yaml'))
        self.latestObservationTime: float = 0
        self.configRewardAddress: str = None
        self.setupWalletManager()
        # Health check thread: monitors observations and restarts if none received in 24 hours
        self.checkinCheckThread = threading.Thread(
            target=self.checkinCheck,
            daemon=True)
        self.checkinCheckThread.start()
        alreadySetup: bool = os.path.exists(config.walletPath("wallet.yaml"))
        if not alreadySetup:
            threading.Thread(target=self.delayedEngine).start()
        self.ranOnce = False
        self.startFunction = self.start
        if self.runMode == RunMode.normal:
            self.startFunction = self.start
        elif self.runMode == RunMode.worker:
            self.startFunction = self.startWorker
        elif self.runMode == RunMode.wallet:
            self.startFunction = self.startWalletOnly
        if not config.get().get("disable restart", False):
            self.restartThread = threading.Thread(
                target=self.restartEverythingPeriodic,
                daemon=True)
            self.restartThread.start()

    @staticmethod
    def getUiPort() -> int:
        """Get UI port with priority: config file > environment variable > default (24601)"""
        existing_port = config.get().get('uiport')
        if existing_port is not None:
            return int(existing_port)
        else:
            port = int(os.environ.get('SATORI_UI_PORT', '24601'))
            config.add(data={'uiport': port})
            return port

    @property
    def walletOnlyMode(self) -> bool:
        return self.runMode == RunMode.wallet

    @property
    def rewardAddress(self) -> str:
        return self.configRewardAddress

    @property
    def network(self) -> str:
        return 'main' if self.env in ['prod', 'local', 'testprod'] else 'test'

    @property
    def vault(self) -> EvrmoreWallet:
        return self.walletManager.vault

    @property
    def wallet(self) -> EvrmoreWallet:
        return self.walletManager.wallet

    @property
    def holdingBalance(self) -> float:
        if self.wallet.balance.amount > 0:
            self._holdingBalance = round(
                self.wallet.balance.amount
                + (self.vault.balance.amount if self.vault is not None else 0),
                8)
        else:
            self._holdingBalance = self.getBalance()
        return self._holdingBalance

    def refreshBalance(self, threaded: bool = True, forWallet: bool = True, forVault: bool = True):
        self.walletManager.connect()
        if forWallet and isinstance(self.wallet, EvrmoreWallet):
            if threaded:
                threading.Thread(target=self.wallet.get).start()
            else:
                self.wallet.get()
        if forVault and isinstance(self.vault, EvrmoreWallet):
            if threaded:
                threading.Thread(target=self.vault.get).start()
            else:
                self.vault.get()
        return self.holdingBalance

    def refreshUnspents(self, threaded: bool = True, forWallet: bool = True, forVault: bool = True):
        self.walletManager.connect()
        if forWallet and isinstance(self.wallet, EvrmoreWallet):
            if threaded:
                threading.Thread(target=self.wallet.getReadyToSend).start()
            else:
                self.wallet.getReadyToSend()
        if forVault and isinstance(self.vault, EvrmoreWallet):
            if threaded:
                threading.Thread(target=self.vault.getReadyToSend).start()
            else:
                self.vault.getReadyToSend()
        return self._holdingBalance

    @property
    def holdingBalanceBase(self) -> float:
        """Get Satori from Base with 5-minute interval cache"""
        # TEMPORARY DISABLE
        return 0

    @property
    def ethaddressforward(self) -> str:
        eth_address = self.vault.ethAddress
        if eth_address:
            return eth_address
        else:
            return ""

    def getVaultInfoFromFile(self) -> dict:
        """Read vault info (address and pubkey) from vault.yaml without decrypting.

        The address and pubkey are stored unencrypted in vault.yaml, so we can read them
        even when the vault is locked.

        Returns:
            dict: {'address': str, 'pubkey': str} or empty dict if file doesn't exist
        """
        try:
            import yaml
            vault_path = config.walletPath('vault.yaml')
            if not os.path.exists(vault_path):
                return {}

            with open(vault_path, 'r') as f:
                vault_data = yaml.safe_load(f)

            result = {}
            if vault_data:
                # Address is under evr: section
                if 'evr' in vault_data and 'address' in vault_data['evr']:
                    result['address'] = vault_data['evr']['address']
                # publicKey is at top level
                if 'publicKey' in vault_data:
                    result['pubkey'] = vault_data['publicKey']

            return result
        except Exception as e:
            logging.warning(f"Could not read vault info from file: {e}")
            return {}

    def setupWalletManager(self):
        # Never auto-decrypt the global vault - it should remain encrypted
        self.walletManager = WalletManager.create(useConfigPassword=False)

    def shutdownWallets(self):
        self.walletManager._electrumx = None
        self.walletManager._wallet = None
        self.walletManager._vault = None

    def closeVault(self):
        self.walletManager.closeVault()

    def openVault(self, password: Union[str, None] = None, create: bool = False):
        return self.walletManager.openVault(password=password, create=create)

    def getWallet(self, **kwargs):
        return self.walletManager.wallet

    def getVault(self, password: Union[str, None] = None, create: bool = False) -> Union[EvrmoreWallet, None]:
        return self.walletManager.openVault(password=password, create=create)

    def electrumxCheck(self):
        return self.walletManager.isConnected()

    def collectAndSubmitPredictions(self):
        """Collect predictions from all models and submit in batch."""
        try:
            if not hasattr(self, 'aiengine') or self.aiengine is None:
                logging.warning("AI Engine not initialized, skipping prediction collection", color='yellow')
                return

            # Collect predictions from all models
            predictions_collected = 0
            for stream_uuid, model in self.aiengine.streamModels.items():
                if hasattr(model, '_pending_prediction') and model._pending_prediction:
                    # Queue prediction in engine
                    pred = model._pending_prediction
                    self.aiengine.queuePrediction(
                        stream_uuid=pred['stream_uuid'],
                        stream_name=pred['stream_name'],
                        value=pred['value'],
                        observed_at=pred['observed_at'],
                        hash_val=pred['hash']
                    )
                    predictions_collected += 1
                    # Clear the pending prediction
                    model._pending_prediction = None

            if predictions_collected > 0:
                logging.info(f"Collected {predictions_collected} predictions from models", color='cyan')
                # Submit all queued predictions in batch
                result = self.aiengine.flushPredictionQueue()
                if result:
                    logging.info(f"✓ Batch predictions submitted: {result['successful']}/{result['total_submitted']}", color='green')
                else:
                    logging.warning("Failed to submit batch predictions", color='yellow')
            else:
                logging.debug("No predictions ready to submit")

        except Exception as e:
            logging.error(f"Error collecting and submitting predictions: {e}", color='red')

    def pollObservationsForever(self):
        """
        Poll for new observations - P2P first, central fallback.
        In P2P/hybrid mode, also subscribes to real-time P2P observations.
        Initial delay: random (0-11 hours) to distribute load.
        Subsequent polls: every 11 hours as backup for missed real-time events.
        """
        import pandas as pd
        import random

        def processObservation(observation: dict):
            """Process an observation from any source (P2P or central)."""
            if observation is None:
                return

            try:
                if not hasattr(self, 'aiengine') or self.aiengine is None:
                    logging.warning("AI Engine not initialized, skipping observation", color='yellow')
                    return

                # Convert observation to DataFrame for engine
                value = observation.get('value') or observation.get('bitcoin_price')
                hash_val = observation.get('hash') or observation.get('id') or observation.get('observation_id')
                df = pd.DataFrame([{
                    'ts': observation.get('observed_at') or observation.get('ts'),
                    'value': float(value) if value is not None else None,
                    'hash': str(hash_val) if hash_val is not None else None,
                }])

                # Pass to each stream in the engine
                for streamUuid, streamModel in self.aiengine.streamModels.items():
                    try:
                        logging.info(f"Passing observation to stream {streamUuid}", color='green')
                        streamModel.onDataReceived(df)
                    except Exception as e:
                        logging.error(f"Error passing observation to stream {streamUuid}: {e}", color='red')

            except Exception as e:
                logging.error(f"Error processing observation: {e}", color='red')

        def getObservationP2PFirst(stream: str = 'bitcoin') -> dict:
            """Get observation - P2P first, central fallback."""
            # Try P2P oracle network first
            if hasattr(self, '_oracle_network') and self._oracle_network:
                try:
                    obs = self._oracle_network.get_latest_observation(stream)
                    if obs:
                        logging.info(f"Got observation from P2P for {stream}", color='green')
                        return {
                            'observation_id': obs.hash if hasattr(obs, 'hash') else obs.get('hash'),
                            'value': obs.value if hasattr(obs, 'value') else obs.get('value'),
                            'observed_at': obs.timestamp if hasattr(obs, 'timestamp') else obs.get('timestamp'),
                            'ts': obs.timestamp if hasattr(obs, 'timestamp') else obs.get('ts'),
                            'hash': obs.hash if hasattr(obs, 'hash') else obs.get('hash'),
                            'source': 'p2p',
                        }
                except Exception as e:
                    logging.warning(f"P2P observation failed: {e}", color='yellow')

            # Fallback to central
            if hasattr(self, 'server') and self.server:
                try:
                    obs = self.server.getObservation(stream)
                    if obs:
                        obs['source'] = 'central'
                        logging.info(f"Got observation from central for {stream}", color='blue')
                        return obs
                except Exception as e:
                    logging.warning(f"Central observation failed: {e}", color='yellow')

            return None

        def subscribeToP2PObservations():
            """Check P2P observation subscription status.

            Note: P2P observations are now wired directly via the
            oracle_network.on_observation_received callback set during
            P2P initialization in _initializeP2PComponentsAsync().
            This function just logs the status.
            """
            if hasattr(self, '_oracle_network') and self._oracle_network:
                if self._oracle_network.on_observation_received is not None:
                    logging.info("P2P observations wired to Engine (callback active)", color='green')
                else:
                    logging.warning("P2P observation callback not set", color='yellow')

        def pollForever():
            # Subscribe to real-time P2P observations first
            subscribeToP2PObservations()

            # First poll: random delay between 1 and 11 hours to distribute load
            initial_delay = random.randint(60 * 60, 60 * 60 * 11)
            logging.info(f"First observation poll in {initial_delay / 3600:.1f} hours", color='blue')
            time.sleep(initial_delay)

            # Subsequent polls: every 11 hours as backup for P2P
            while True:
                try:
                    if not hasattr(self, 'server') or self.server is None:
                        logging.warning("Server not initialized, skipping observation poll", color='yellow')
                        time.sleep(60 * 60 * 11)
                        continue

                    if not hasattr(self, 'aiengine') or self.aiengine is None:
                        logging.warning("AI Engine not initialized, skipping observation poll", color='yellow')
                        time.sleep(60 * 60 * 11)
                        continue

                    # Try P2P first for single observation
                    observation = getObservationP2PFirst()
                    if observation:
                        processObservation(observation)

                    # Also get batch of observations from central-lite
                    # This includes Bitcoin, multi-crypto, and SafeTrade observations
                    storage = getattr(self.aiengine, 'storage', None)
                    observations = self.server.getObservationsBatch(storage=storage)

                    if observations is None or len(observations) == 0:
                        logging.info("No new observations available", color='blue')
                        time.sleep(60 * 60 * 11)
                        continue

                    logging.info(f"Received {len(observations)} observations from server", color='cyan')

                    # Update last observation time
                    self.latestObservationTime = time.time()

                    # Process each observation
                    observations_processed = 0
                    for observation in observations:
                        try:
                            # Extract values
                            value = observation.get('value')
                            hash_val = observation.get('hash') or observation.get('id')
                            stream_uuid = observation.get('stream_uuid')
                            stream = observation.get('stream')
                            stream_name = stream.get('name', 'unknown') if stream else 'unknown'

                            if value is None:
                                logging.warning(f"Skipping observation with no value (stream: {stream_name})", color='yellow')
                                continue

                            # Convert observation to DataFrame for engine
                            df = pd.DataFrame([{
                                'ts': observation.get('observed_at') or observation.get('ts'),
                                'value': float(value),
                                'hash': str(hash_val) if hash_val is not None else None,
                            }])

                            # Store using server-provided stream UUID
                            if stream_uuid:
                                observations_processed += 1

                                # Create stream model if it doesn't exist
                                if stream_uuid not in self.aiengine.streamModels:
                                    try:
                                        # Import required classes
                                        from satoriengine.veda.engine import StreamModel

                                        # Create StreamId objects for subscription and publication
                                        sub_id = StreamId(
                                            source='central-lite',
                                            author='satori',
                                            stream=stream_name,
                                            target=''
                                        )

                                        # Prediction stream uses "_pred" suffix
                                        pub_id = StreamId(
                                            source='central-lite',
                                            author='satori',
                                            stream=f"{stream_name}_pred",
                                            target=''
                                        )

                                        # Create Stream objects
                                        subscriptionStream = Stream(streamId=sub_id)
                                        publicationStream = Stream(streamId=pub_id, predicting=sub_id)

                                        # Create StreamModel using factory method
                                        self.aiengine.streamModels[stream_uuid] = StreamModel.createFromServer(
                                            streamUuid=stream_uuid,
                                            predictionStreamUuid=pub_id.uuid,
                                            server=self.server,
                                            wallet=self.wallet,
                                            subscriptionStream=subscriptionStream,
                                            publicationStream=publicationStream,
                                            pauseAll=self.aiengine.pause,
                                            resumeAll=self.aiengine.resume,
                                            storage=self.aiengine.storage
                                        )

                                        # Choose and initialize appropriate adapter
                                        self.aiengine.streamModels[stream_uuid].chooseAdapter(inplace=True)

                                        logging.info(f"✓ Created model for stream: {stream_name} (UUID: {stream_uuid[:8]}...)", color='magenta')
                                    except Exception as e:
                                        logging.error(f"Failed to create model for {stream_name}: {e}", color='red')
                                        import traceback
                                        logging.error(traceback.format_exc())

                                # Pass data to the model
                                if stream_uuid in self.aiengine.streamModels:
                                    try:
                                        self.aiengine.streamModels[stream_uuid].onDataReceived(df)
                                        logging.info(f"✓ Stored {stream_name}: ${float(value):,.2f} (UUID: {stream_uuid[:8]}...)", color='green')
                                    except Exception as e:
                                        logging.error(f"Error passing to engine for {stream_name}: {e}", color='red')
                            else:
                                logging.warning(f"Observation for {stream_name} missing stream_uuid", color='yellow')

                        except Exception as e:
                            logging.error(f"Error processing individual observation: {e}", color='red')

                    logging.info(f"✓ Processed and stored {observations_processed}/{len(observations)} observations", color='cyan')

                    # After processing all observations, collect predictions and submit in batch
                    self.collectAndSubmitPredictions()

                except Exception as e:
                    logging.error(f"Error polling observations: {e}", color='red')

                # Wait 11 hours before next poll
                time.sleep(60 * 60 * 11)

        self.pollObservationsThread = threading.Thread(
            target=pollForever,
            daemon=True)
        self.pollObservationsThread.start()

    def delayedEngine(self):
        time.sleep(60 * 60 * 6)
        self.buildEngine()

    def checkinCheck(self):
        while True:
            time.sleep(60 * 60 * 6)  # Check every 6 hours
            current_time = time.time()
            if self.latestObservationTime and (current_time - self.latestObservationTime > 60*60*24):
                logging.warning("No observations in 24 hours, restarting", print=True)
                self.triggerRestart()
            if hasattr(self, 'server') and hasattr(self.server, 'checkinCheck') and self.server.checkinCheck():
                logging.warning("Server check failed, restarting", print=True)
                self.triggerRestart()

    def networkIsTest(self, network: str = None) -> bool:
        return network.lower().strip() in ("testnet", "test", "ravencoin", "rvn")

    def start(self):
        """start the satori engine."""
        if self.ranOnce:
            time.sleep(60 * 60)
        self.ranOnce = True
        if self.env == 'prod' and self.serverConnectedRecently():
            last_checkin = config.get().get('server checkin')
            elapsed_minutes = (time.time() - last_checkin) / 60
            wait_minutes = max(0, 10 - elapsed_minutes)
            if wait_minutes > 0:
                logging.info(f"Server connected recently, waiting {wait_minutes:.1f} minutes")
                time.sleep(wait_minutes * 60)
        self.recordServerConnection()
        networking_mode = os.environ.get('SATORI_NETWORKING_MODE', config.get().get('networking mode', 'central')).lower().strip()
        if self.walletOnlyMode:
            self.createServerConn()
            if networking_mode != 'p2p':
                self.authWithCentral()  # Skip in pure P2P mode
            self.setRewardAddress(globally=True)  # Sync reward address with server
            self.announceToNetwork()  # P2P announcement (hybrid/p2p modes)
            self.initializeP2PComponents()  # Initialize consensus, rewards, etc.
            logging.info("in WALLETONLYMODE")
            startWebUI(self, port=self.uiPort)  # Start web UI after sync
            return
        self.setMiningMode()
        self.createServerConn()
        if networking_mode != 'p2p':
            self.authWithCentral()  # Skip in pure P2P mode
        self.setRewardAddress(globally=True)  # Sync reward address with server
        self.announceToNetwork()  # P2P announcement (hybrid/p2p modes)
        self.initializeP2PComponents()  # Initialize consensus, rewards, etc.
        # In P2P and hybrid modes, engine is created when streams are claimed
        # Only pure central mode auto-creates streams from server
        if networking_mode == 'central':
            self.setupDefaultStream()
            self.spawnEngine()
        else:
            # P2P and hybrid modes: engine created on first stream claim
            self.aiengine = None
            logging.info(f"{networking_mode.upper()} mode: Engine will be initialized when streams are claimed", color="cyan")
        startWebUI(self, port=self.uiPort)  # Start web UI after sync

    def startWalletOnly(self):
        """start the satori engine."""
        logging.info("running in walletOnly mode", color="blue")
        self.createServerConn()
        return

    def startWorker(self):
        """start the satori engine."""
        logging.info("running in worker mode", color="blue")
        networking_mode = config.get().get('networking mode', 'central').lower().strip()
        if self.env == 'prod' and self.serverConnectedRecently():
            last_checkin = config.get().get('server checkin')
            elapsed_minutes = (time.time() - last_checkin) / 60
            wait_minutes = max(0, 10 - elapsed_minutes)
            if wait_minutes > 0:
                logging.info(f"Server connected recently, waiting {wait_minutes:.1f} minutes")
                time.sleep(wait_minutes * 60)
        self.recordServerConnection()
        self.setMiningMode()
        self.createServerConn()
        if networking_mode != 'p2p':
            self.authWithCentral()  # Skip in pure P2P mode
        self.setRewardAddress(globally=True)  # Sync reward address with server
        self.announceToNetwork()  # P2P announcement (hybrid/p2p modes)
        self.initializeP2PComponents()  # Initialize consensus, rewards, etc.
        # In P2P and hybrid modes, engine is created when streams are claimed
        # Only pure central mode auto-creates streams from server
        if networking_mode == 'central':
            self.setupDefaultStream()
            self.spawnEngine()
        else:
            # P2P and hybrid modes: engine created on first stream claim
            self.aiengine = None
            logging.info(f"{networking_mode.upper()} mode: Engine will be initialized when streams are claimed", color="cyan")
        startWebUI(self, port=self.uiPort)  # Start web UI after sync
        threading.Event().wait()

    def serverConnectedRecently(self, threshold_minutes: int = 10) -> bool:
        """Check if server was connected to recently without side effects."""
        last_checkin = config.get().get('server checkin')
        if last_checkin is None:
            return False
        elapsed_seconds = time.time() - last_checkin
        return elapsed_seconds < (threshold_minutes * 60)

    def recordServerConnection(self) -> None:
        """Record the current time as the last server connection time."""
        config.add(data={'server checkin': time.time()})

    def createServerConn(self):
        # logging.debug(self.urlServer, color="teal")
        self.server = SatoriServerClient(self.wallet)

    def authWithCentral(self):
        """Register peer with central-lite server."""
        x = 30
        attempt = 0
        while True:
            attempt += 1
            try:
                # Get vault info from vault.yaml (available even when encrypted)
                vault_info = self.getVaultInfoFromFile()

                # Build vaultInfo dict for registration
                vaultInfo = None
                if vault_info.get('address') or vault_info.get('pubkey'):
                    vaultInfo = {
                        'vaultaddress': vault_info.get('address'),
                        'vaultpubkey': vault_info.get('pubkey')
                    }

                # Register peer with central server
                self.server.checkin(vaultInfo=vaultInfo)

                logging.info("authenticated with central-lite", color="green")
                break
            except Exception as e:
                logging.warning(f"connecting to central err: {e}")
            x = x * 1.5 if x < 60 * 60 * 6 else 60 * 60 * 6
            logging.warning(f"trying again in {x}")
            time.sleep(x)

    def announceToNetwork(self, capabilities: list = None):
        """
        Announce our presence to the P2P network.

        Called alongside authWithCentral() in hybrid mode.
        In central mode, this does nothing.

        Note: This method now only sets up the P2P reference - actual
        initialization happens in startP2PServices() which runs in a
        persistent Trio context.

        Args:
            capabilities: List of capabilities (e.g., ["predictor", "oracle"])
        """
        networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
        if networking_mode is None:
            try:
                networking_mode = config.get().get('networking mode', 'central')
            except Exception:
                networking_mode = 'central'

        networking_mode = networking_mode.lower().strip()

        # Skip in central mode
        if networking_mode == 'central':
            return

        # Store capabilities for use in startP2PServices
        self._p2p_capabilities = capabilities or ["predictor"]
        logging.debug("P2P announcement queued (will run in startP2PServices)")

    def initializeP2PComponents(self):
        """
        Initialize all P2P components for consensus, rewards, and distribution.

        This method starts the unified P2P services thread that runs in a
        single persistent Trio context. All P2P operations (announcement,
        component initialization, and ongoing network tasks) run within
        this same Trio event loop to avoid "internal error in Trio" and
        "Bad file descriptor" errors.

        Components initialized:
        - _uptime_tracker: Heartbeat-based uptime tracking for relay bonus
        - _reward_calculator: Local reward calculation and tracking
        - _consensus_manager: Stake-weighted voting coordination
        - _distribution_trigger: Automatic distribution on consensus
        - _signer_node: Multi-sig signing (if this node is an authorized signer)
        """
        networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
        if networking_mode is None:
            try:
                networking_mode = config.get().get('networking mode', 'central')
            except Exception:
                networking_mode = 'central'

        networking_mode = networking_mode.lower().strip()

        # Skip in central mode
        if networking_mode == 'central':
            return

        # Start the unified P2P services thread
        self._startP2PServicesThread()

    def _startP2PServicesThread(self):
        """
        Start the unified P2P services in a persistent Trio context.

        This runs ALL P2P operations in a single long-running Trio event loop:
        1. Creates and starts the Peers instance
        2. Announces to network
        3. Initializes all P2P protocol components
        4. Keeps running for ongoing P2P operations

        Using a single persistent Trio context prevents "internal error in Trio"
        and "Bad file descriptor" errors that occur when libp2p objects are
        used across multiple Trio event loops.
        """
        # Prevent double-start
        if hasattr(self, '_p2p_thread_started') and self._p2p_thread_started:
            logging.debug("P2P services thread already started")
            return

        self._p2p_thread_started = True
        self._p2p_ready = threading.Event()

        async def _run_p2p_services():
            """Main async function running in persistent Trio context."""
            try:
                import trio
                from satorip2p.protocol.peer_registry import PeerRegistry
                from satorip2p import Peers

                # Store trio token for cross-thread async calls (used by web routes)
                self._trio_token = trio.lowlevel.current_trio_token()

                # Phase 1: Create and start Peers
                logging.info("Starting P2P network services...", color="cyan")
                self._p2p_peers = Peers(
                    identity=self.identity,
                    listen_port=config.get().get('p2p port', 24600),
                )
                await self._p2p_peers.start()
                logging.info("P2P peers started", color="cyan")

                # Phase 2: Announce to network
                try:
                    self._peer_registry = PeerRegistry(self._p2p_peers)
                    await self._peer_registry.start()

                    caps = getattr(self, '_p2p_capabilities', ["predictor"])
                    announcement = await self._peer_registry.announce(capabilities=caps)

                    if announcement:
                        logging.info(
                            f"Announced to P2P network: {announcement.evrmore_address[:16]}...",
                            color="cyan"
                        )
                    else:
                        logging.debug("P2P announcement returned None (may succeed later)")
                except Exception as e:
                    logging.debug(f"P2P announcement phase: {e}")

                # Signal that P2P basic services are ready (peers started)
                self._p2p_ready.set()

                # Phase 3: Run forever - starts background tasks including:
                # - KadDHT service (peer routing, content routing)
                # - Pubsub message processing
                # - Mesh repair task (fixes py-libp2p GossipSub race conditions)
                # - Bootstrap connections to seed nodes
                # - Subscription re-advertisement
                # This keeps the Trio context alive for all P2P operations
                logging.info("P2P services running (starting background tasks...)", color="cyan")

                # Queue P2P protocol initialization to run after nursery is created
                # This ensures DHT is available for subscription announcements
                self._p2p_peers.spawn_background_task(self._initializeP2PComponentsAsync)

                # run_forever() never returns - runs until cancelled
                await self._p2p_peers.run_forever()

            except ImportError as e:
                logging.debug(f"satorip2p not available: {e}")
                self._p2p_ready.set()  # Signal ready even on failure
            except Exception as e:
                logging.warning(f"P2P services error: {e}")
                self._p2p_ready.set()  # Signal ready even on failure

        def run_p2p_thread():
            """Thread target that runs the persistent Trio event loop."""
            import trio
            try:
                trio.run(_run_p2p_services)
            except Exception as e:
                logging.warning(f"P2P services thread error: {e}")

        # Start the P2P thread
        p2p_thread = threading.Thread(target=run_p2p_thread, daemon=True, name="P2PServices")
        p2p_thread.start()

        # Wait for P2P to be ready (with timeout)
        if not self._p2p_ready.wait(timeout=30):
            logging.debug("P2P ready timeout - continuing anyway")

    async def _initializeP2PComponentsAsync(self):
        """
        Initialize all P2P protocol components (async version).

        This is called from within the persistent Trio context.
        """
        try:
            # Import P2P modules
            from satorip2p.protocol.uptime import UptimeTracker
            from satorip2p.protocol.rewards import RewardCalculator
            from satorip2p.protocol.consensus import ConsensusManager
            from satorip2p.protocol.distribution_trigger import DistributionTrigger
            from satorip2p.protocol.signer import SignerNode, AUTHORIZED_SIGNERS

            # 1. Initialize Uptime Tracker
            if not hasattr(self, '_uptime_tracker') or self._uptime_tracker is None:
                try:
                    self._uptime_tracker = UptimeTracker(
                        peers=self._p2p_peers,
                        wallet=self.identity,
                    )

                    # Set up role callback for dynamic role determination
                    def get_current_roles() -> list:
                        """Get current node roles based on oracle/claim/signer state."""
                        roles = []
                        # Check if we're a primary or secondary oracle (both are "oracle" role)
                        oracle_network = getattr(self, '_oracle_network', None)
                        if oracle_network:
                            my_registrations = getattr(oracle_network, '_my_registrations', {})
                            if my_registrations and 'oracle' not in roles:
                                # Both primary and secondary oracles show as "oracle"
                                roles.append('oracle')
                        # Check if we're a signer
                        signer = getattr(self, '_signer_node', None)
                        if signer and getattr(signer, '_is_authorized', False):
                            roles.append('signer')
                        # Check if we have claimed prediction slots
                        stream_registry = getattr(self, '_stream_registry', None)
                        if stream_registry:
                            my_claims = getattr(stream_registry, '_my_claims', {})
                            if my_claims:
                                roles.append('predictor')
                        # Default to node if no other roles
                        return roles if roles else ['node']

                    self._uptime_tracker.role_callback = get_current_roles

                    await self._uptime_tracker.start()
                    # Spawn heartbeat loop as background task (will start when run_forever creates nursery)
                    self._p2p_peers.spawn_background_task(self._uptime_tracker.run_heartbeat_loop)
                    logging.info("P2P uptime tracker initialized (heartbeat loop queued)", color="cyan")
                    # Wire to WebSocket bridge for real-time UI updates
                    try:
                        from web.p2p_bridge import get_bridge
                        get_bridge().wire_protocol('uptime_tracker', self._uptime_tracker)
                    except Exception:
                        pass  # Bridge may not be started yet
                except Exception as e:
                    logging.debug(f"Uptime tracker: {e}")

            # 2. Initialize Reward Calculator
            if not hasattr(self, '_reward_calculator') or self._reward_calculator is None:
                try:
                    self._reward_calculator = RewardCalculator()
                    logging.info("P2P reward calculator initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Reward calculator: {e}")

            # 3. Initialize Consensus Manager
            if not hasattr(self, '_consensus_manager') or self._consensus_manager is None:
                try:
                    self._consensus_manager = ConsensusManager(
                        peers=self._p2p_peers,
                        wallet=self.identity,
                    )
                    await self._consensus_manager.start()
                    logging.info("P2P consensus manager initialized", color="cyan")
                    # Wire to WebSocket bridge for real-time UI updates
                    try:
                        from web.p2p_bridge import get_bridge
                        get_bridge().wire_protocol('consensus_manager', self._consensus_manager)
                    except Exception:
                        pass  # Bridge may not be started yet
                except Exception as e:
                    logging.debug(f"Consensus manager: {e}")

            # 4. Initialize Distribution Trigger
            if not hasattr(self, '_distribution_trigger') or self._distribution_trigger is None:
                try:
                    self._distribution_trigger = DistributionTrigger(
                        peers=self._p2p_peers,
                        consensus_manager=getattr(self, '_consensus_manager', None),
                    )
                    await self._distribution_trigger.start()
                    # Alias for routes.py compatibility
                    self._distribution_manager = self._distribution_trigger
                    logging.info("P2P distribution trigger initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Distribution trigger: {e}")

            # 5. Initialize Signer Node (only if authorized)
            my_address = self.identity.address if hasattr(self.identity, 'address') else None
            is_authorized_signer = my_address and my_address in AUTHORIZED_SIGNERS

            if is_authorized_signer and (not hasattr(self, '_signer_node') or self._signer_node is None):
                try:
                    self._signer_node = SignerNode(
                        peers=self._p2p_peers,
                        wallet=self.identity,
                    )
                    await self._signer_node.start()
                    logging.info("P2P signer node initialized (authorized signer)", color="green")
                except Exception as e:
                    logging.debug(f"Signer node: {e}")

            # 6. Initialize Oracle Network (for P2P observations)
            if not hasattr(self, '_oracle_network') or self._oracle_network is None:
                try:
                    from satorip2p.protocol.oracle_network import OracleNetwork
                    self._oracle_network = OracleNetwork(self._p2p_peers)
                    await self._oracle_network.start()
                    # Wire up oracle network to peers for role detection
                    if hasattr(self._p2p_peers, 'set_oracle_network'):
                        self._p2p_peers.set_oracle_network(self._oracle_network)

                    # Wire observations to Engine for P2P-first predictions
                    def _feed_observation_to_engine(observation):
                        """Feed P2P observations to the AI Engine for predictions."""
                        try:
                            if not hasattr(self, 'aiengine') or self.aiengine is None:
                                return  # Engine not ready yet

                            import pandas as pd

                            # Extract observation data
                            value = observation.value if hasattr(observation, 'value') else observation.get('value')
                            timestamp = observation.timestamp if hasattr(observation, 'timestamp') else observation.get('timestamp')
                            hash_val = observation.hash if hasattr(observation, 'hash') else observation.get('hash')
                            stream_id = observation.stream_id if hasattr(observation, 'stream_id') else observation.get('stream_id')

                            if value is None:
                                return

                            # Convert to DataFrame for engine
                            df = pd.DataFrame([{
                                'ts': timestamp,
                                'value': float(value),
                                'hash': str(hash_val) if hash_val else None,
                            }])

                            # Pass to all stream models in the engine
                            for streamUuid, streamModel in self.aiengine.streamModels.items():
                                try:
                                    streamModel.onDataReceived(df)
                                except Exception as e:
                                    logging.debug(f"Error feeding observation to stream {streamUuid}: {e}")

                            logging.debug(f"Fed P2P observation to Engine: stream={stream_id}, value={value}")
                        except Exception as e:
                            logging.debug(f"Error feeding observation to engine: {e}")

                    # Set the callback on the oracle network
                    self._oracle_network.on_observation_received = _feed_observation_to_engine

                    logging.info("P2P oracle network initialized (wired to Engine)", color="cyan")
                except Exception as e:
                    logging.debug(f"Oracle network: {e}")

            # 7. Initialize Lending Manager (for P2P pool operations)
            if not hasattr(self, '_lending_manager') or self._lending_manager is None:
                try:
                    from satorip2p.protocol.lending import LendingManager
                    self._lending_manager = LendingManager(
                        peers=self._p2p_peers,
                        wallet=self.identity,
                    )
                    await self._lending_manager.start()
                    logging.info("P2P lending manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Lending manager: {e}")

            # 8. Initialize Delegation Manager (for P2P proxy/delegation)
            if not hasattr(self, '_delegation_manager') or self._delegation_manager is None:
                try:
                    from satorip2p.protocol.delegation import DelegationManager
                    self._delegation_manager = DelegationManager(
                        peers=self._p2p_peers,
                        wallet=self.identity,
                    )
                    await self._delegation_manager.start()
                    logging.info("P2P delegation manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Delegation manager: {e}")

            # 9. Initialize Stream Registry (for P2P stream discovery)
            if not hasattr(self, '_stream_registry') or self._stream_registry is None:
                try:
                    from satorip2p.protocol.stream_registry import StreamRegistry
                    self._stream_registry = StreamRegistry(self._p2p_peers)
                    await self._stream_registry.start()
                    # Wire stream registry to oracle network for activity tracking
                    if hasattr(self, '_oracle_network') and self._oracle_network is not None:
                        if hasattr(self._oracle_network, 'set_stream_registry'):
                            self._oracle_network.set_stream_registry(self._stream_registry)
                    logging.info("P2P stream registry initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Stream registry: {e}")

            # 10. Initialize Prediction Protocol (for P2P commit-reveal predictions)
            if not hasattr(self, '_prediction_protocol') or self._prediction_protocol is None:
                try:
                    from satorip2p.protocol.prediction_protocol import PredictionProtocol
                    self._prediction_protocol = PredictionProtocol(self._p2p_peers)
                    await self._prediction_protocol.start()

                    # Wire prediction protocol to Engine so it can publish predictions
                    if hasattr(self, 'aiengine') and self.aiengine is not None:
                        if hasattr(self.aiengine, 'setPredictionProtocol'):
                            self.aiengine.setPredictionProtocol(self._prediction_protocol)

                    logging.info("P2P prediction protocol initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Prediction protocol: {e}")

            # 11. Initialize Treasury Alert Manager
            if not hasattr(self, '_alert_manager') or self._alert_manager is None:
                try:
                    from satorip2p.protocol.alerts import TreasuryAlertManager
                    self._alert_manager = TreasuryAlertManager(
                        peers=self._p2p_peers,
                    )
                    await self._alert_manager.start()
                    logging.info("P2P treasury alert manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Alert manager: {e}")

            # 12. Initialize Deferred Rewards Manager
            if not hasattr(self, '_deferred_rewards_manager') or self._deferred_rewards_manager is None:
                try:
                    from satorip2p.protocol.deferred_rewards import DeferredRewardsManager
                    self._deferred_rewards_manager = DeferredRewardsManager()
                    logging.info("P2P deferred rewards manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Deferred rewards manager: {e}")

            # 13. Initialize Donation Manager
            if not hasattr(self, '_donation_manager') or self._donation_manager is None:
                try:
                    from satorip2p.protocol.donation import DonationManager
                    treasury_addr = getattr(self, 'treasury_address', None)
                    if self.identity and treasury_addr:
                        self._donation_manager = DonationManager(
                            wallet=self.identity,
                            treasury_address=treasury_addr,
                        )
                        await self._donation_manager.start()
                        logging.info("P2P donation manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Donation manager: {e}")

            # 14. Initialize Protocol Versioning
            if not hasattr(self, '_version_tracker') or self._version_tracker is None:
                try:
                    PeerVersionTracker = get_p2p_module('PeerVersionTracker')
                    VersionNegotiator = get_p2p_module('VersionNegotiator')
                    PROTOCOL_VERSION = get_p2p_module('PROTOCOL_VERSION')
                    if PeerVersionTracker and VersionNegotiator:
                        self._version_tracker = PeerVersionTracker()
                        self._version_negotiator = VersionNegotiator()
                        self._protocol_version = PROTOCOL_VERSION or '1.0.0'
                        logging.info(f"P2P protocol versioning initialized (v{self._protocol_version})", color="cyan")
                except Exception as e:
                    logging.debug(f"Version tracker: {e}")

            # 15. Initialize Storage Manager (redundant storage for deferred rewards & alerts)
            if not hasattr(self, '_storage_manager') or self._storage_manager is None:
                try:
                    StorageManager = get_p2p_module('StorageManager')
                    DeferredRewardsStorage = get_p2p_module('DeferredRewardsStorage')
                    AlertStorage = get_p2p_module('AlertStorage')
                    if StorageManager:
                        from pathlib import Path
                        storage_dir = Path.home() / '.satori' / 'storage'
                        self._storage_manager = StorageManager(storage_dir=storage_dir)
                        if DeferredRewardsStorage:
                            self._deferred_rewards_storage = DeferredRewardsStorage(
                                storage_dir=storage_dir,
                                peers=self._p2p_peers,
                            )
                        if AlertStorage:
                            self._alert_storage = AlertStorage(
                                storage_dir=storage_dir,
                                peers=self._p2p_peers,
                            )
                        logging.info("P2P storage manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Storage manager: {e}")

            # 16. Initialize Bandwidth Tracker & QoS Manager
            if not hasattr(self, '_bandwidth_tracker') or self._bandwidth_tracker is None:
                try:
                    create_qos_manager = get_p2p_module('create_qos_manager')
                    if create_qos_manager:
                        self._bandwidth_tracker, self._qos_manager = create_qos_manager()
                        if self._p2p_peers and hasattr(self._p2p_peers, 'set_bandwidth_tracker'):
                            self._p2p_peers.set_bandwidth_tracker(self._bandwidth_tracker)
                        logging.info("P2P bandwidth tracker and QoS manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Bandwidth tracker: {e}")

            # 17. Initialize Referral Manager
            if not hasattr(self, '_referral_manager') or self._referral_manager is None:
                try:
                    ReferralManager = get_p2p_module('ReferralManager')
                    if ReferralManager:
                        self._referral_manager = ReferralManager(
                            peers=self._p2p_peers,
                            wallet=self.identity,
                        )
                        await self._referral_manager.start()
                        logging.info("P2P referral manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Referral manager: {e}")

            # 18. Initialize Pricing Provider
            if not hasattr(self, '_price_provider') or self._price_provider is None:
                try:
                    get_price_provider = get_p2p_module('get_price_provider')
                    if get_price_provider:
                        self._price_provider = get_price_provider()
                        logging.info("P2P price provider initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Price provider: {e}")

            # 19. Initialize Reward Address Manager
            if not hasattr(self, '_reward_address_manager') or self._reward_address_manager is None:
                try:
                    RewardAddressManager = get_p2p_module('RewardAddressManager')
                    if RewardAddressManager:
                        self._reward_address_manager = RewardAddressManager(
                            peers=self._p2p_peers,
                            wallet=self.identity,
                        )
                        await self._reward_address_manager.start()
                        logging.info("P2P reward address manager initialized", color="cyan")
                except Exception as e:
                    logging.debug(f"Reward address manager: {e}")

            # 20. Wire managers to P2PSatoriServerClient if available
            if hasattr(self, 'server') and self.server is not None:
                try:
                    if hasattr(self.server, 'set_lending_manager'):
                        self.server.set_lending_manager(getattr(self, '_lending_manager', None))
                    if hasattr(self.server, 'set_delegation_manager'):
                        self.server.set_delegation_manager(getattr(self, '_delegation_manager', None))
                    if hasattr(self.server, 'set_oracle_network'):
                        self.server.set_oracle_network(getattr(self, '_oracle_network', None))
                    if hasattr(self.server, 'set_stream_registry'):
                        self.server.set_stream_registry(getattr(self, '_stream_registry', None))
                    if hasattr(self.server, 'set_alert_manager'):
                        self.server.set_alert_manager(getattr(self, '_alert_manager', None))
                    if hasattr(self.server, 'set_deferred_rewards_manager'):
                        self.server.set_deferred_rewards_manager(getattr(self, '_deferred_rewards_manager', None))
                    if hasattr(self.server, 'set_donation_manager'):
                        self.server.set_donation_manager(getattr(self, '_donation_manager', None))
                    if hasattr(self.server, 'set_version_tracker'):
                        self.server.set_version_tracker(getattr(self, '_version_tracker', None))
                    if hasattr(self.server, 'set_version_negotiator'):
                        self.server.set_version_negotiator(getattr(self, '_version_negotiator', None))
                    if hasattr(self.server, 'set_storage_manager'):
                        self.server.set_storage_manager(getattr(self, '_storage_manager', None))
                    if hasattr(self.server, 'set_bandwidth_tracker'):
                        self.server.set_bandwidth_tracker(getattr(self, '_bandwidth_tracker', None))
                    if hasattr(self.server, 'set_qos_manager'):
                        self.server.set_qos_manager(getattr(self, '_qos_manager', None))
                    if hasattr(self.server, 'set_referral_manager'):
                        self.server.set_referral_manager(getattr(self, '_referral_manager', None))
                    if hasattr(self.server, 'set_price_provider'):
                        self.server.set_price_provider(getattr(self, '_price_provider', None))
                    if hasattr(self.server, 'set_reward_address_manager'):
                        self.server.set_reward_address_manager(getattr(self, '_reward_address_manager', None))
                    logging.info("P2P managers wired to server client", color="cyan")
                except Exception as e:
                    logging.debug(f"Server wiring: {e}")

            logging.info("P2P components initialization complete", color="cyan")

            # 20.5. Sync oracle registrations to stream registry
            # This auto-populates the stream registry with streams that oracles are publishing
            await self._sync_oracle_streams_to_registry()

            # 21. Initialize StreamManager (oracle data streams)
            try:
                try:
                    from Streams import StreamManager
                except ImportError:
                    from streams_lite import StreamManager
                if not hasattr(self, '_stream_manager') or self._stream_manager is None:
                    self._stream_manager = StreamManager(
                        peers=getattr(self, '_p2p_peers', None),
                        identity=getattr(self, 'identity', None),
                        send_to_ui=getattr(self, 'sendToUI', None),
                        oracle_network=getattr(self, '_oracle_network', None),
                    )
                    await self._stream_manager.start()
                    logging.info(f"StreamManager initialized ({self._stream_manager.oracle_count} oracles)", color="cyan")

                    # Spawn Trio polling task if oracles are configured
                    if self._stream_manager.oracle_count > 0 and self._p2p_peers:
                        if hasattr(self._p2p_peers, 'spawn_background_task'):
                            self._p2p_peers.spawn_background_task(
                                self._stream_manager.run_trio_polling
                            )
                            logging.info("Oracle polling task spawned", color="cyan")

                        # Auto-subscribe to streams we're publishing to (receive others' observations)
                        if hasattr(self, '_oracle_network') and self._oracle_network:
                            async def auto_subscribe_oracle_streams():
                                """Subscribe to all streams we're running oracles for."""
                                try:
                                    # Get stream IDs from configured oracles
                                    for oracle in self._stream_manager._oracles.values():
                                        stream_id = getattr(oracle, 'stream_id', None)
                                        if not stream_id:
                                            stream_id = getattr(oracle.config, 'stream_id', None)
                                        if stream_id:
                                            # Define callback for received observations
                                            def make_callback(sid):
                                                def on_observation(observation):
                                                    logging.debug(f"Received observation for {sid}: {observation}")
                                                    # Emit to websocket bridge if available
                                                    try:
                                                        from web.p2p_bridge import get_bridge
                                                        bridge = get_bridge()
                                                        if bridge:
                                                            bridge._on_observation(observation)
                                                    except Exception:
                                                        pass
                                                return on_observation

                                            await self._oracle_network.subscribe_to_stream(
                                                stream_id, make_callback(stream_id)
                                            )
                                            logging.info(f"Auto-subscribed to stream: {stream_id}", color="cyan")
                                except Exception as e:
                                    logging.debug(f"Auto-subscribe failed: {e}")

                            self._p2p_peers.spawn_background_task(auto_subscribe_oracle_streams)
                            logging.info("Auto-subscribing to oracle streams...", color="cyan")
            except ImportError:
                logging.debug("streams-lite not available, oracle streams disabled")
            except Exception as e:
                logging.debug(f"StreamManager: {e}")

            # 22. Initialize Governance Protocol
            if not hasattr(self, '_governance') or self._governance is None:
                try:
                    from satorip2p.protocol.governance import GovernanceProtocol
                    self._governance = GovernanceProtocol(
                        peers=self._p2p_peers,
                    )
                    await self._governance.start()
                    # Wire uptime tracker for uptime-based voting power
                    if hasattr(self, '_uptime_tracker') and self._uptime_tracker is not None:
                        if hasattr(self._governance, 'set_uptime_tracker'):
                            self._governance.set_uptime_tracker(self._uptime_tracker)
                    # Wire wallet for stake-based voting power
                    if self.identity is not None:
                        if hasattr(self._governance, 'set_wallet'):
                            self._governance.set_wallet(self.identity)
                    # Wire activity storage for governance participation tracking
                    if hasattr(self, '_uptime_tracker') and self._uptime_tracker is not None:
                        if hasattr(self._uptime_tracker, '_activity_storage') and self._uptime_tracker._activity_storage:
                            if hasattr(self._governance, 'set_activity_storage'):
                                self._governance.set_activity_storage(self._uptime_tracker._activity_storage)
                    logging.info("P2P governance protocol initialized", color="cyan")
                    # Wire to WebSocket bridge for real-time UI updates
                    try:
                        from web.p2p_bridge import get_bridge
                        get_bridge().wire_protocol('governance_manager', self._governance)
                    except Exception:
                        pass  # Bridge may not be started yet
                except Exception as e:
                    logging.debug(f"Governance protocol: {e}")

            # 23. Spawn claim renewal background task
            if hasattr(self, '_stream_registry') and self._stream_registry is not None:
                async def claim_renewal_loop():
                    """Periodically renew stream claims to prevent expiration."""
                    import trio
                    RENEWAL_INTERVAL = 12 * 60 * 60  # 12 hours in seconds

                    # Wait a bit before first renewal to let system stabilize
                    await trio.sleep(60)

                    while True:
                        try:
                            my_claims = self._stream_registry.get_my_claims()
                            if my_claims:
                                logging.info(f"Auto-renewing {len(my_claims)} stream claims...", color="cyan")
                                result = await self._stream_registry.renew_claims()
                                logging.info(f"Claim renewal complete: {result['renewed']} renewed, {result['failed']} failed", color="cyan")
                        except Exception as e:
                            logging.warning(f"Claim renewal failed: {e}")

                        # Sleep until next renewal
                        await trio.sleep(RENEWAL_INTERVAL)

                if hasattr(self._p2p_peers, 'spawn_background_task'):
                    self._p2p_peers.spawn_background_task(claim_renewal_loop)
                    logging.info("Claim renewal background task spawned (every 12h)", color="cyan")

        except ImportError as e:
            logging.debug(f"satorip2p not available: {e}")
        except Exception as e:
            logging.debug(f"P2P component initialization: {e}")

    async def _sync_oracle_streams_to_registry(self):
        """
        Sync oracle registrations to the stream registry.

        This auto-populates the stream registry with streams that oracles
        are publishing, so users can discover and claim prediction slots.

        Oracle stream_id format: "source|stream|target" (e.g., "crypto|satori|BTC|USD")
        """
        try:
            oracle_network = getattr(self, '_oracle_network', None)
            stream_registry = getattr(self, '_stream_registry', None)

            if not oracle_network or not stream_registry:
                logging.debug("Cannot sync: oracle_network or stream_registry not available")
                return

            # Get all oracle registrations (our own + received from network)
            all_stream_ids = set()

            # Add our own registrations
            my_registrations = getattr(oracle_network, '_my_registrations', {})
            all_stream_ids.update(my_registrations.keys())

            # Add received registrations from other oracles
            oracle_registrations = getattr(oracle_network, '_oracle_registrations', {})
            all_stream_ids.update(oracle_registrations.keys())

            # Also check observation cache for streams we've seen data for
            observation_cache = getattr(oracle_network, '_observation_cache', {})
            all_stream_ids.update(observation_cache.keys())

            if not all_stream_ids:
                logging.debug("No oracle streams to sync to registry")
                return

            synced_count = 0
            for stream_id in all_stream_ids:
                try:
                    # Check if already in registry
                    existing = stream_registry._streams.get(stream_id)
                    if existing:
                        continue

                    # Parse stream_id: "source|stream|target" format
                    # e.g., "crypto|satori|BTC|USD" -> source="crypto|satori", stream="BTC", target="USD"
                    parts = stream_id.split('|')
                    if len(parts) >= 4:
                        # Format: source|provider|symbol|currency
                        source = f"{parts[0]}|{parts[1]}"  # e.g., "crypto|satori"
                        stream = parts[2]  # e.g., "BTC"
                        target = parts[3]  # e.g., "USD"
                    elif len(parts) == 3:
                        source = parts[0]
                        stream = parts[1]
                        target = parts[2]
                    else:
                        # Can't parse, use as-is
                        source = stream_id
                        stream = "data"
                        target = "value"

                    # Create stream definition directly with the oracle's stream_id
                    # (rather than generating a new hash-based ID)
                    from satorip2p.protocol.stream_registry import StreamDefinition
                    import time as time_module

                    definition = StreamDefinition(
                        stream_id=stream_id,  # Use oracle's stream_id directly
                        source=source,
                        stream=stream,
                        target=target,
                        datatype="price" if "USD" in target or "price" in stream_id.lower() else "numeric",
                        cadence=60,
                        predictor_slots=1000,
                        creator=getattr(stream_registry, 'evrmore_address', '') or '',
                        timestamp=int(time_module.time()),
                        description=f"Auto-discovered from oracle network",
                        tags=["oracle", "auto-discovered"],
                    )

                    # Store directly in registry's stream cache
                    stream_registry._streams[stream_id] = definition
                    synced_count += 1
                    logging.debug(f"Synced oracle stream to registry: {stream_id}")

                    # Auto-claim our own oracle streams for prediction
                    try:
                        async def _auto_claim():
                            claim = await stream_registry.claim_stream(stream_id)
                            if claim:
                                logging.info(f"Auto-claimed own oracle stream: {stream_id[:30]}... slot={claim.slot_index}")
                            return claim

                        import asyncio
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(_auto_claim())
                        finally:
                            loop.close()
                    except Exception as claim_error:
                        logging.debug(f"Auto-claim failed for {stream_id}: {claim_error}")

                except Exception as e:
                    logging.debug(f"Failed to sync stream {stream_id}: {e}")

            if synced_count > 0:
                logging.info(f"Synced {synced_count} oracle streams to registry", color="cyan")

        except Exception as e:
            logging.debug(f"Oracle stream sync failed: {e}")

    def discoverStreams(
        self,
        source: str = None,
        datatype: str = None,
        use_p2p: bool = True,
        use_central: bool = True
    ) -> list:
        """
        Discover available streams from P2P network and/or central server.

        In hybrid mode, queries both and merges results.
        In P2P mode, only queries P2P network.
        In central mode, only queries central server.

        Args:
            source: Filter by source (e.g., "exchange/binance")
            datatype: Filter by datatype (e.g., "price")
            use_p2p: Whether to query P2P network
            use_central: Whether to query central server

        Returns:
            List of stream definitions
        """
        import asyncio

        networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
        if networking_mode is None:
            try:
                networking_mode = config.get().get('networking mode', 'central')
            except Exception:
                networking_mode = 'central'
        networking_mode = networking_mode.lower().strip()

        streams = []

        async def _discover_p2p():
            """Async P2P discovery."""
            try:
                from satorip2p.protocol.stream_registry import StreamRegistry

                # Ensure P2P peers and registry are initialized
                if not hasattr(self, '_stream_registry') or self._stream_registry is None:
                    if hasattr(self, '_p2p_peers') and self._p2p_peers is not None:
                        self._stream_registry = StreamRegistry(self._p2p_peers)
                        await self._stream_registry.start()

                if hasattr(self, '_stream_registry') and self._stream_registry is not None:
                    p2p_streams = await self._stream_registry.discover_streams(
                        source=source,
                        datatype=datatype
                    )
                    # Convert to common format
                    result = []
                    for s in p2p_streams:
                        result.append({
                            'stream_id': s.stream_id,
                            'source': s.source,
                            'stream': s.stream,
                            'target': s.target,
                            'datatype': s.datatype,
                            'cadence': s.cadence,
                            'from_p2p': True,
                        })
                    logging.debug(f"Discovered {len(p2p_streams)} streams from P2P network")
                    return result

            except ImportError:
                logging.debug("satorip2p not available for stream discovery")
            except Exception as e:
                logging.warning(f"P2P stream discovery failed: {e}")
            return []

        # Query P2P network if enabled
        if use_p2p and networking_mode in ('hybrid', 'p2p'):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync context with running loop
                    pass
                else:
                    p2p_streams = loop.run_until_complete(_discover_p2p())
                    streams.extend(p2p_streams)
            except RuntimeError:
                p2p_streams = asyncio.run(_discover_p2p())
                streams.extend(p2p_streams)

        # Query central server if enabled
        if use_central and networking_mode in ('central', 'hybrid'):
            try:
                if hasattr(self, 'server') and self.server is not None:
                    # Central-lite has simpler stream queries
                    if hasattr(self.server, 'getStreams'):
                        central_streams = self.server.getStreams(source=source)
                        for s in central_streams:
                            stream_id = s.get('uuid', s.get('stream_id', ''))
                            if not any(existing.get('stream_id') == stream_id for existing in streams):
                                streams.append({
                                    **s,
                                    'from_p2p': False,
                                })
                        logging.debug(f"Discovered {len(central_streams)} streams from central server")
            except Exception as e:
                logging.warning(f"Central server stream discovery failed: {e}")

        return streams

    def claimStream(self, stream_id: str, slot_index: int = None) -> bool:
        """
        Claim a predictor slot on a stream (P2P mode).

        Args:
            stream_id: Stream to claim
            slot_index: Specific slot (None = first available)

        Returns:
            True if claimed successfully
        """
        import asyncio

        networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
        if networking_mode is None:
            try:
                networking_mode = config.get().get('networking mode', 'central')
            except Exception:
                networking_mode = 'central'
        networking_mode = networking_mode.lower().strip()

        if networking_mode == 'central':
            logging.debug("Stream claiming not available in central mode")
            return False

        async def _claim():
            try:
                from satorip2p.protocol.stream_registry import StreamRegistry

                # Ensure stream registry is initialized
                if not hasattr(self, '_stream_registry') or self._stream_registry is None:
                    if hasattr(self, '_p2p_peers') and self._p2p_peers is not None:
                        self._stream_registry = StreamRegistry(self._p2p_peers)
                        await self._stream_registry.start()

                if hasattr(self, '_stream_registry') and self._stream_registry is not None:
                    claim = await self._stream_registry.claim_stream(
                        stream_id=stream_id,
                        slot_index=slot_index
                    )
                    if claim:
                        logging.info(f"Claimed stream {stream_id[:16]}... slot {claim.slot_index}")
                        return True

            except ImportError:
                logging.debug("satorip2p not available for stream claiming")
            except Exception as e:
                logging.warning(f"Stream claiming failed: {e}")
            return False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return False  # Can't await in sync context
            else:
                return loop.run_until_complete(_claim())
        except RuntimeError:
            return asyncio.run(_claim())

    def getMyClaimedStreams(self) -> list:
        """
        Get list of streams we've claimed in P2P mode.

        Returns:
            List of StreamClaim objects
        """
        import asyncio

        async def _get_claims():
            if hasattr(self, '_stream_registry') and self._stream_registry is not None:
                try:
                    return await self._stream_registry.get_my_streams()
                except Exception as e:
                    logging.warning(f"Failed to get claimed streams: {e}")
            return []

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            else:
                return loop.run_until_complete(_get_claims())
        except RuntimeError:
            return asyncio.run(_get_claims())

    def subscribeToP2PData(self, stream_id: str, callback: callable) -> bool:
        """
        Subscribe to stream data via P2P network.

        Args:
            stream_id: Stream to subscribe to
            callback: Function called with each observation

        Returns:
            True if subscribed successfully
        """
        import asyncio

        networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
        if networking_mode is None:
            try:
                networking_mode = config.get().get('networking mode', 'central')
            except Exception:
                networking_mode = 'central'
        networking_mode = networking_mode.lower().strip()

        if networking_mode == 'central':
            logging.debug("P2P data subscription not available in central mode")
            return False

        async def _subscribe():
            try:
                from satorip2p.protocol.oracle_network import OracleNetwork

                # Ensure oracle network is initialized
                if not hasattr(self, '_oracle_network') or self._oracle_network is None:
                    if hasattr(self, '_p2p_peers') and self._p2p_peers is not None:
                        self._oracle_network = OracleNetwork(self._p2p_peers)
                        await self._oracle_network.start()
                        # Wire up oracle network to peers for role detection
                        if hasattr(self._p2p_peers, 'set_oracle_network'):
                            self._p2p_peers.set_oracle_network(self._oracle_network)

                if hasattr(self, '_oracle_network') and self._oracle_network is not None:
                    success = await self._oracle_network.subscribe_to_stream(
                        stream_id=stream_id,
                        callback=callback
                    )
                    if success:
                        logging.info(f"Subscribed to P2P data for stream {stream_id[:16]}...")
                    return success

            except ImportError:
                logging.debug("satorip2p not available for P2P data subscription")
            except Exception as e:
                logging.warning(f"P2P data subscription failed: {e}")
            return False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return False
            else:
                return loop.run_until_complete(_subscribe())
        except RuntimeError:
            return asyncio.run(_subscribe())

    def publishObservation(
        self,
        stream_id: str,
        value: float,
        timestamp: int = None,
        to_p2p: bool = True,
        to_central: bool = True
    ) -> bool:
        """
        Publish an observation to the network.

        Args:
            stream_id: Stream to publish to
            value: Observed value
            timestamp: Observation timestamp (default: now)
            to_p2p: Whether to publish to P2P network
            to_central: Whether to publish to central server

        Returns:
            True if published successfully
        """
        import asyncio

        networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
        if networking_mode is None:
            try:
                networking_mode = config.get().get('networking mode', 'central')
            except Exception:
                networking_mode = 'central'
        networking_mode = networking_mode.lower().strip()

        timestamp = timestamp or int(time.time())
        success = False

        async def _publish_p2p():
            try:
                from satorip2p.protocol.oracle_network import OracleNetwork

                # Ensure oracle network is initialized
                if not hasattr(self, '_oracle_network') or self._oracle_network is None:
                    if hasattr(self, '_p2p_peers') and self._p2p_peers is not None:
                        self._oracle_network = OracleNetwork(self._p2p_peers)
                        await self._oracle_network.start()
                        # Wire up oracle network to peers for role detection
                        if hasattr(self._p2p_peers, 'set_oracle_network'):
                            self._p2p_peers.set_oracle_network(self._oracle_network)

                if hasattr(self, '_oracle_network') and self._oracle_network is not None:
                    observation = await self._oracle_network.publish_observation(
                        stream_id=stream_id,
                        value=value,
                        timestamp=timestamp
                    )
                    if observation:
                        logging.debug(f"Published observation to P2P: {stream_id[:16]}... = {value}")
                        # Emit WebSocket event for our own observation
                        try:
                            from web.p2p_bridge import get_bridge
                            bridge = get_bridge()
                            if bridge:
                                bridge.emit_own_observation(
                                    stream_id=stream_id,
                                    value=value,
                                    peer_id=getattr(observation, 'peer_id', ''),
                                    oracle_address=getattr(observation, 'oracle', '')
                                )
                        except Exception as e:
                            logging.debug(f"Failed to emit own observation event: {e}")
                        return True

            except ImportError:
                logging.debug("satorip2p not available for P2P observation publishing")
            except Exception as e:
                logging.warning(f"P2P observation publishing failed: {e}")
            return False

        # Publish to P2P network
        if to_p2p and networking_mode in ('hybrid', 'p2p'):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pass  # Can't await in sync context
                else:
                    if loop.run_until_complete(_publish_p2p()):
                        success = True
            except RuntimeError:
                if asyncio.run(_publish_p2p()):
                    success = True

        # Publish to central server
        if to_central and networking_mode in ('central', 'hybrid'):
            try:
                if hasattr(self, 'server') and self.server is not None:
                    self.server.publish(
                        topic=stream_id,
                        data=str(value),
                        observationTime=str(timestamp),
                        observationHash="",
                        isPrediction=False
                    )
                    logging.debug(f"Published observation to central: {stream_id[:16]}... = {value}")
                    success = True
            except Exception as e:
                logging.warning(f"Central server observation publishing failed: {e}")

        return success

    def getP2PObservations(self, stream_id: str, limit: int = 100) -> list:
        """
        Get cached P2P observations for a stream.

        Args:
            stream_id: Stream to get observations for
            limit: Maximum observations to return

        Returns:
            List of Observation objects
        """
        if hasattr(self, '_oracle_network') and self._oracle_network is not None:
            try:
                return self._oracle_network.get_cached_observations(stream_id, limit)
            except Exception as e:
                logging.warning(f"Failed to get P2P observations: {e}")
        return []

    def publishP2PPrediction(
        self,
        stream_id: str,
        value: float,
        target_time: int,
        confidence: float = 0.0,
        to_p2p: bool = True,
        to_central: bool = True
    ) -> bool:
        """
        Publish a prediction to the network.

        Args:
            stream_id: Stream to predict
            value: Predicted value
            target_time: When this prediction is for
            confidence: Confidence level (0-1)
            to_p2p: Whether to publish to P2P network
            to_central: Whether to publish to central server

        Returns:
            True if published successfully
        """
        import asyncio

        networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
        if networking_mode is None:
            try:
                networking_mode = config.get().get('networking mode', 'central')
            except Exception:
                networking_mode = 'central'
        networking_mode = networking_mode.lower().strip()

        success = False

        async def _publish_p2p():
            try:
                from satorip2p.protocol.prediction_protocol import PredictionProtocol

                # Ensure prediction protocol is initialized
                if not hasattr(self, '_prediction_protocol') or self._prediction_protocol is None:
                    if hasattr(self, '_p2p_peers') and self._p2p_peers is not None:
                        self._prediction_protocol = PredictionProtocol(self._p2p_peers)
                        await self._prediction_protocol.start()

                if hasattr(self, '_prediction_protocol') and self._prediction_protocol is not None:
                    prediction = await self._prediction_protocol.publish_prediction(
                        stream_id=stream_id,
                        value=value,
                        target_time=target_time,
                        confidence=confidence
                    )
                    if prediction:
                        logging.debug(f"Published prediction to P2P: {stream_id[:16]}... = {value}")
                        return True

            except ImportError:
                logging.debug("satorip2p not available for P2P prediction publishing")
            except Exception as e:
                logging.warning(f"P2P prediction publishing failed: {e}")
            return False

        # Publish to P2P network
        if to_p2p and networking_mode in ('hybrid', 'p2p'):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pass  # Can't await in sync context
                else:
                    if loop.run_until_complete(_publish_p2p()):
                        success = True
            except RuntimeError:
                if asyncio.run(_publish_p2p()):
                    success = True

        return success

    def subscribeToP2PPredictions(self, stream_id: str, callback: callable) -> bool:
        """
        Subscribe to predictions from other predictors via P2P.

        Args:
            stream_id: Stream to subscribe to
            callback: Function called with each prediction

        Returns:
            True if subscribed successfully
        """
        import asyncio

        networking_mode = os.environ.get('SATORI_NETWORKING_MODE')
        if networking_mode is None:
            try:
                networking_mode = config.get().get('networking mode', 'central')
            except Exception:
                networking_mode = 'central'
        networking_mode = networking_mode.lower().strip()

        if networking_mode == 'central':
            logging.debug("P2P prediction subscription not available in central mode")
            return False

        async def _subscribe():
            try:
                from satorip2p.protocol.prediction_protocol import PredictionProtocol

                # Ensure prediction protocol is initialized
                if not hasattr(self, '_prediction_protocol') or self._prediction_protocol is None:
                    if hasattr(self, '_p2p_peers') and self._p2p_peers is not None:
                        self._prediction_protocol = PredictionProtocol(self._p2p_peers)
                        await self._prediction_protocol.start()

                if hasattr(self, '_prediction_protocol') and self._prediction_protocol is not None:
                    success = await self._prediction_protocol.subscribe_to_predictions(
                        stream_id=stream_id,
                        callback=callback
                    )
                    if success:
                        logging.info(f"Subscribed to P2P predictions for {stream_id[:16]}...")
                    return success

            except ImportError:
                logging.debug("satorip2p not available for P2P prediction subscription")
            except Exception as e:
                logging.warning(f"P2P prediction subscription failed: {e}")
            return False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return False
            else:
                return loop.run_until_complete(_subscribe())
        except RuntimeError:
            return asyncio.run(_subscribe())

    def getP2PPredictions(self, stream_id: str, limit: int = 100) -> list:
        """
        Get cached P2P predictions for a stream.

        Args:
            stream_id: Stream to get predictions for
            limit: Maximum predictions to return

        Returns:
            List of Prediction objects
        """
        if hasattr(self, '_prediction_protocol') and self._prediction_protocol is not None:
            try:
                return self._prediction_protocol.get_cached_predictions(stream_id, limit)
            except Exception as e:
                logging.warning(f"Failed to get P2P predictions: {e}")
        return []

    def getPredictorScore(self, predictor: str, stream_id: str = None) -> float:
        """
        Get average prediction score for a predictor.

        Args:
            predictor: Evrmore address of predictor
            stream_id: Optional stream filter

        Returns:
            Average score (0-1)
        """
        if hasattr(self, '_prediction_protocol') and self._prediction_protocol is not None:
            try:
                return self._prediction_protocol.get_predictor_average_score(predictor, stream_id)
            except Exception as e:
                logging.warning(f"Failed to get predictor score: {e}")
        return 0.0

    # ========== P2P Rewards (Phase 5/6) ==========

    def getPendingRewards(self, stream_id: str = None) -> dict:
        """
        Get pending (unclaimed) rewards for this predictor.

        Works in hybrid/p2p mode by querying the reward data store.
        In central mode, queries the central server.

        Args:
            stream_id: Optional stream filter

        Returns:
            Dict with pending rewards info:
            {
                'total_pending': float,
                'rounds': [
                    {'round_id': str, 'stream_id': str, 'amount': float, 'score': float},
                    ...
                ]
            }
        """
        mode = _get_networking_mode()

        if mode in ('hybrid', 'p2p', 'p2p_only'):
            # P2P mode: Query reward data store
            try:
                from satorip2p.protocol.rewards import RoundDataStore
                if hasattr(self, '_peers') and self._peers is not None:
                    store = RoundDataStore(self._peers)
                    # Get our address
                    our_address = ""
                    if hasattr(self, 'wallet') and self.wallet:
                        our_address = self.wallet.address
                    elif hasattr(self, '_peers') and self._peers._identity_bridge:
                        our_address = self._peers._identity_bridge.evrmore_address

                    # Query recent rounds from local cache
                    pending = {'total_pending': 0.0, 'rounds': []}

                    # Check local cache for recent round data
                    for key, summary in store._local_cache.items():
                        if stream_id and summary.stream_id != stream_id:
                            continue
                        # Find our reward in this round
                        for reward in summary.rewards:
                            if reward.address == our_address and reward.amount > 0:
                                pending['rounds'].append({
                                    'round_id': summary.round_id,
                                    'stream_id': summary.stream_id,
                                    'amount': reward.amount,
                                    'score': reward.score,
                                    'rank': reward.rank,
                                })
                                pending['total_pending'] += reward.amount

                    return pending
            except Exception as e:
                logging.warning(f"Failed to get pending rewards from P2P: {e}")

        # Central mode or fallback: Query server
        try:
            if hasattr(self, 'server') and self.server:
                success, data = self.server.getPendingRewards(stream_id=stream_id)
                if success:
                    return data
        except Exception as e:
            logging.warning(f"Failed to get pending rewards from server: {e}")

        return {'total_pending': 0.0, 'rounds': []}

    def getRewardHistory(
        self,
        stream_id: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> list:
        """
        Get historical reward claims for this predictor.

        Args:
            stream_id: Optional stream filter
            limit: Max records to return
            offset: Pagination offset

        Returns:
            List of reward history entries:
            [
                {
                    'round_id': str,
                    'stream_id': str,
                    'amount': float,
                    'score': float,
                    'rank': int,
                    'tx_hash': str,
                    'claimed_at': int,
                },
                ...
            ]
        """
        mode = _get_networking_mode()

        if mode in ('hybrid', 'p2p', 'p2p_only'):
            # P2P mode: Query DHT for historical round data
            try:
                from satorip2p.protocol.rewards import RoundDataStore
                if hasattr(self, '_peers') and self._peers is not None:
                    store = RoundDataStore(self._peers)
                    our_address = ""
                    if hasattr(self, 'wallet') and self.wallet:
                        our_address = self.wallet.address
                    elif hasattr(self, '_peers') and self._peers._identity_bridge:
                        our_address = self._peers._identity_bridge.evrmore_address

                    history = []
                    # Note: In production, would query DHT for historical rounds
                    # For now, return from local cache
                    for key, summary in list(store._local_cache.items())[:limit]:
                        if stream_id and summary.stream_id != stream_id:
                            continue
                        for reward in summary.rewards:
                            if reward.address == our_address:
                                history.append({
                                    'round_id': summary.round_id,
                                    'stream_id': summary.stream_id,
                                    'amount': reward.amount,
                                    'score': reward.score,
                                    'rank': reward.rank,
                                    'tx_hash': summary.evrmore_tx_hash,
                                    'claimed_at': summary.created_at,
                                })
                    return history[offset:offset + limit]
            except Exception as e:
                logging.warning(f"Failed to get reward history from P2P: {e}")

        # Central mode or fallback: Query server
        try:
            if hasattr(self, 'server') and self.server:
                success, data = self.server.getRewardHistory(
                    stream_id=stream_id,
                    limit=limit,
                    offset=offset
                )
                if success:
                    return data
        except Exception as e:
            logging.warning(f"Failed to get reward history from server: {e}")

        return []

    def claimRewards(
        self,
        round_ids: list = None,
        stream_id: str = None,
        claim_address: str = None
    ) -> dict:
        """
        Claim pending rewards.

        In P2P mode, rewards are distributed automatically at round end.
        This method is primarily for:
        - Querying claim status
        - Overriding claim address
        - Manual claiming in transition phase

        Args:
            round_ids: Specific rounds to claim (None = all pending)
            stream_id: Filter by stream
            claim_address: Override default reward address

        Returns:
            Dict with claim result:
            {
                'success': bool,
                'claimed_amount': float,
                'tx_hash': str or None,
                'message': str,
            }
        """
        mode = _get_networking_mode()

        # Determine claim address
        if not claim_address:
            # Try config reward address first
            claim_address = getattr(self, 'configRewardAddress', None)
            if not claim_address and hasattr(self, 'wallet') and self.wallet:
                claim_address = self.wallet.address

        if not claim_address:
            return {
                'success': False,
                'claimed_amount': 0.0,
                'tx_hash': None,
                'message': 'No claim address available'
            }

        if mode in ('hybrid', 'p2p', 'p2p_only'):
            # P2P mode: Rewards are auto-distributed
            # This queries status and can request manual claim if needed
            try:
                pending = self.getPendingRewards(stream_id=stream_id)

                if pending['total_pending'] == 0:
                    return {
                        'success': True,
                        'claimed_amount': 0.0,
                        'tx_hash': None,
                        'message': 'No pending rewards to claim'
                    }

                # In fully decentralized mode, rewards are auto-distributed
                # This is informational - rewards will arrive at claim_address
                return {
                    'success': True,
                    'claimed_amount': pending['total_pending'],
                    'tx_hash': None,
                    'message': f"Rewards ({pending['total_pending']:.8f} SATORI) will be distributed to {claim_address}"
                }
            except Exception as e:
                logging.warning(f"Failed to claim rewards via P2P: {e}")

        # Central mode or fallback: Request claim from server
        try:
            if hasattr(self, 'server') and self.server:
                success, data = self.server.claimRewards(
                    round_ids=round_ids,
                    stream_id=stream_id,
                    claim_address=claim_address,
                )
                if success:
                    return data
                return {
                    'success': False,
                    'claimed_amount': 0.0,
                    'tx_hash': None,
                    'message': data.get('message', 'Claim failed') if isinstance(data, dict) else 'Claim failed'
                }
        except Exception as e:
            logging.warning(f"Failed to claim rewards from server: {e}")

        return {
            'success': False,
            'claimed_amount': 0.0,
            'tx_hash': None,
            'message': 'Failed to process claim'
        }

    def getMyRewardScore(self, stream_id: str = None) -> dict:
        """
        Get this predictor's scoring statistics.

        Returns:
            Dict with scoring stats:
            {
                'average_score': float,
                'total_predictions': int,
                'total_rewards': float,
                'rank': int or None,
            }
        """
        our_address = ""
        if hasattr(self, 'wallet') and self.wallet:
            our_address = self.wallet.address
        elif hasattr(self, '_peers') and self._peers and self._peers._identity_bridge:
            our_address = self._peers._identity_bridge.evrmore_address

        if not our_address:
            return {
                'average_score': 0.0,
                'total_predictions': 0,
                'total_rewards': 0.0,
                'rank': None,
            }

        # Get score from prediction protocol
        avg_score = self.getPredictorScore(our_address, stream_id)

        # Get reward history for totals
        history = self.getRewardHistory(stream_id=stream_id, limit=1000)
        total_rewards = sum(h.get('amount', 0) for h in history)

        return {
            'average_score': avg_score,
            'total_predictions': len(history),
            'total_rewards': total_rewards,
            'rank': None,  # Would need leaderboard query
        }

    def subscribeToRewardNotifications(
        self,
        stream_id: str,
        callback: Callable = None
    ) -> bool:
        """
        Subscribe to reward distribution notifications for a stream.

        Receives notifications when rewards are distributed for rounds
        you participated in.

        Args:
            stream_id: Stream to subscribe to
            callback: Function called with notification dict:
                {
                    'type': 'round_complete',
                    'round_id': str,
                    'merkle_root': str,
                    'total_rewards': float,
                    'tx_hash': str,
                }

        Returns:
            True if subscribed successfully
        """
        mode = _get_networking_mode()

        if mode not in ('hybrid', 'p2p', 'p2p_only'):
            logging.debug("Reward notifications only available in P2P mode")
            return False

        try:
            from satorip2p.protocol.rewards import RoundDataStore
            if hasattr(self, '_peers') and self._peers is not None:
                store = RoundDataStore(self._peers)

                def notification_handler(data):
                    """Handle reward notification."""
                    if callback:
                        try:
                            callback(data)
                        except Exception as e:
                            logging.debug(f"Reward notification callback error: {e}")

                    # Log notification
                    logging.info(
                        f"Reward notification for {stream_id}: "
                        f"round={data.get('round_id')}, "
                        f"total={data.get('total_rewards', 0):.4f} SATORI"
                    )

                # Use sync wrapper for async subscribe
                if hasattr(self, '_run_async'):
                    return self._run_async(store.subscribe_to_rewards(stream_id, notification_handler))
                else:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(store.subscribe_to_rewards(stream_id, notification_handler))
        except Exception as e:
            logging.warning(f"Failed to subscribe to reward notifications: {e}")

        return False

    def getBalance(self, currency: str = 'currency') -> float:
        return self.balances.get(currency, 0)

    def setRewardAddress(
        self,
        address: Union[str, None] = None,
        globally: bool = False
    ) -> bool:
        """
        Set or sync reward address between local config and central server.

        Args:
            address: Reward address to set. If None, loads from config or syncs from server.
            globally: If True, also syncs with central server (requires production env).

        Returns:
            True if successfully set/synced, False otherwise.
        """
        # If address is provided, validate and save to config
        if EvrmoreWallet.addressIsValid(address):
            self.configRewardAddress = address
            config.add(data={'reward address': address})

            # If globally=True, check if server needs update
            if globally and self.env in ['prod', 'local', 'testprod', 'dev']:
                try:
                    serverAddress = self.server.mineToAddressStatus()
                    # Only send to server if addresses differ
                    if address != serverAddress:
                        self.server.setRewardAddress(address=address)
                        logging.info(f"Updated server reward address: {address[:8]}...", color="green")
                except Exception as e:
                    logging.debug(f"Could not sync reward address with server: {e}")
            return True
        else:
            # No address provided - load from config
            self.configRewardAddress: str = str(config.get().get('reward address', ''))

            # If we need to sync with server, check if addresses match
            if (
                hasattr(self, 'server') and
                self.server is not None and
                self.env in ['prod', 'local', 'testprod', 'dev']
            ):
                try:
                    serverAddress = self.server.mineToAddressStatus()

                    # If config is empty but server has address, fetch and save
                    if not self.configRewardAddress and serverAddress and EvrmoreWallet.addressIsValid(serverAddress):
                        self.configRewardAddress = serverAddress
                        config.add(data={'reward address': serverAddress})
                        logging.info(f"Synced reward address from server: {serverAddress[:8]}...", color="green")
                        return True

                    # If config has address and globally=True, check if server needs update
                    if (
                        globally and
                        EvrmoreWallet.addressIsValid(self.configRewardAddress) and
                        self.configRewardAddress != serverAddress
                    ):
                        # Only send to server if addresses differ
                        self.server.setRewardAddress(address=self.configRewardAddress)
                        logging.info(f"Updated server reward address: {self.configRewardAddress[:8]}...", color="green")
                        return True

                except Exception as e:
                    logging.debug(f"Could not sync reward address with server: {e}")

        return False

    @staticmethod
    def predictionStreams(streams: list[Stream]):
        """filter down to prediciton publications"""
        return [s for s in streams if s.predicting is not None]

    @staticmethod
    def oracleStreams(streams: list[Stream]):
        """filter down to prediciton publications"""
        return [s for s in streams if s.predicting is None]

    def removePair(self, pub: StreamId, sub: StreamId):
        self.publications = [p for p in self.publications if p.streamId != pub]
        self.subscriptions = [s for s in self.subscriptions if s.streamId != sub]

    def addToEngine(self, stream: Stream, publication: Stream):
        if self.aiengine is not None:
            self.aiengine.addStream(stream, publication)

    def getMatchingStream(self, streamId: StreamId) -> Union[StreamId, None]:
        for stream in self.publications:
            if stream.streamId == streamId:
                return stream.predicting
            if stream.predicting == streamId:
                return stream.streamId
        return None

    def setupDefaultStream(self):
        """Setup hard-coded default stream for central-lite.

        Central-lite has a single observation stream, so we create one
        subscription/publication pair for the engine to work with.
        """
        # Create subscription stream (input observations)
        sub_id = StreamId(
            source="central-lite",
            author="satori",
            stream="observations",
            target=""
        )
        subscription = Stream(streamId=sub_id)

        # Create publication stream (output predictions)
        pub_id = StreamId(
            source="central-lite",
            author="satori",
            stream="predictions",
            target=""
        )
        publication = Stream(streamId=pub_id, predicting=sub_id)

        # Assign to neuron
        self.subscriptions = [subscription]
        self.publications = [publication]

        logging.info(f"Default stream configured: {sub_id.uuid}", color="green")

    def spawnEngine(self):
        """Spawn the AI Engine with stream assignments from Neuron"""
        if not self.subscriptions or not self.publications:
            logging.warning("No stream assignments available, skipping Engine spawn")
            return

        # logging.info("Spawning AI Engine...", color="blue")
        try:
            self.aiengine = Engine.createFromNeuron(
                subscriptions=self.subscriptions,
                publications=self.publications,
                server=self.server,
                wallet=self.wallet)

            # Wire P2P peers to engine if available
            if hasattr(self, '_p2p_peers') and self._p2p_peers is not None:
                if hasattr(self.aiengine, 'setP2PPeers'):
                    self.aiengine.setP2PPeers(self._p2p_peers)

            def runEngine():
                try:
                    self.aiengine.initializeFromNeuron()
                    while True:
                        time.sleep(60)
                except Exception as e:
                    logging.error(f"Engine error: {e}")

            engineThread = threading.Thread(target=runEngine, daemon=True)
            engineThread.start()

            # Start polling for observations from central-lite
            self.pollObservationsForever()

            logging.info("AI Engine spawned successfully", color="green")
        except Exception as e:
            logging.error(f"Failed to spawn AI Engine: {e}")

    def delayedStart(self):
        alreadySetup: bool = os.path.exists(config.walletPath("wallet.yaml"))
        if alreadySetup:
            threading.Thread(target=self.delayedEngine).start()

    def triggerRestart(self, return_code=1):
        os._exit(return_code)

    def emergencyRestart(self):
        import time
        logging.warning("restarting in 10 minutes", print=True)
        time.sleep(60 * 10)
        self.triggerRestart()

    def restartEverythingPeriodic(self):
        import random
        restartTime = time.time() + config.get().get(
            "restartTime", random.randint(60 * 60 * 21, 60 * 60 * 24)
        )
        while True:
            if time.time() > restartTime:
                self.triggerRestart()
            time.sleep(random.randint(60 * 60, 60 * 60 * 4))

    def performStakeCheck(self):
        self.stakeStatus = self.server.stakeCheck()
        return self.stakeStatus

    def setMiningMode(self, miningMode: Union[bool, None] = None):
        miningMode = (
            miningMode
            if isinstance(miningMode, bool)
            else config.get().get('mining mode', True))
        self.miningMode = miningMode
        config.add(data={'mining mode': self.miningMode})
        if hasattr(self, 'server') and self.server is not None:
            self.server.setMiningMode(miningMode)
        return self.miningMode

    # Removed setInvitedBy - central-lite doesn't use referrer system

    def poolAccepting(self, status: bool):
        success, result = self.server.poolAccepting(status)
        if success:
            self.poolIsAccepting = status
        return success, result

    @property
    def stakeRequired(self) -> float:
        return constants.stakeRequired


def startP2PInternalAPI(startupDag: StartupDag, port: int = 24602):
    """Start the internal P2P IPC API server.

    This runs a lightweight HTTP API on 127.0.0.1 that the gunicorn web worker
    can call to perform P2P operations. This solves the subprocess isolation
    problem - gunicorn can't access main process memory, but it can make HTTP
    requests to this internal API.

    The API runs in-process so it has direct access to StartupDag and P2P state.
    """
    try:
        from flask import Flask, jsonify, request
        import trio

        ipc_app = Flask(__name__)

        @ipc_app.route('/p2p/status')
        def p2p_status():
            """Get P2P status including peer ID, connected peers, etc."""
            peers = getattr(startupDag, '_p2p_peers', None)
            identity = getattr(startupDag, 'identity', None)

            if not peers:
                return jsonify({'error': 'P2P not initialized', 'available': False}), 503

            try:
                # connected_peers is an int property in satorip2p
                connected_count = peers.connected_peers if hasattr(peers, 'connected_peers') else 0

                # Get actual peer IDs using satorip2p method
                connected_peer_ids = []
                if hasattr(peers, 'get_connected_peers'):
                    connected_peer_ids = peers.get_connected_peers()

                mesh_count = 0
                if hasattr(peers, 'get_pubsub_debug'):
                    try:
                        debug = peers.get_pubsub_debug()
                        # Calculate mesh peer count from mesh dict
                        # mesh is {topic: [peer_ids...], ...}
                        # Count unique peers across all topics in the mesh
                        mesh = debug.get('mesh', {})
                        unique_mesh_peers = set()
                        for topic_peers in mesh.values():
                            unique_mesh_peers.update(topic_peers)
                        mesh_count = len(unique_mesh_peers)
                    except:
                        pass

                return jsonify({
                    'available': True,
                    'peer_id': str(peers.peer_id) if hasattr(peers, 'peer_id') else None,
                    'identity_address': identity.address if identity and hasattr(identity, 'address') else None,
                    'connected_peers': connected_peer_ids,
                    'connected_count': connected_count,
                    'mesh_peer_count': mesh_count,
                    'listen_port': getattr(peers, 'listen_port', None),
                })
            except Exception as e:
                return jsonify({'error': str(e), 'available': False}), 500

        @ipc_app.route('/p2p/peers')
        def p2p_peers():
            """List all connected peers with details."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                peer_list = []
                # Use satorip2p's get_connected_peers method
                if hasattr(peers, 'get_connected_peers'):
                    connected = peers.get_connected_peers()
                    for peer_id in connected:
                        peer_list.append({
                            'peer_id': str(peer_id),
                            'info': None
                        })
                return jsonify({'peers': peer_list, 'count': len(peer_list)})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/ping/<peer_id>', methods=['POST'])
        def p2p_ping(peer_id):
            """Ping a peer to test connectivity."""
            peers = getattr(startupDag, '_p2p_peers', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            count = request.json.get('count', 3) if request.json else 3

            try:
                async def do_ping():
                    return await peers.ping_peer(peer_id, count=count)

                if trio_token:
                    latencies = trio.from_thread.run(do_ping, trio_token=trio_token)
                else:
                    latencies = trio.run(do_ping)

                if latencies:
                    return jsonify({
                        'success': True,
                        'peer_id': peer_id,
                        'latencies_ms': [l * 1000 for l in latencies],
                        'avg_ms': sum(latencies) / len(latencies) * 1000,
                        'count': len(latencies),
                    })
                else:
                    return jsonify({'success': False, 'peer_id': peer_id, 'error': 'No response'})
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @ipc_app.route('/p2p/connect', methods=['POST'])
        def p2p_connect():
            """Connect to a peer by multiaddr."""
            peers = getattr(startupDag, '_p2p_peers', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            multiaddr = request.json.get('multiaddr') if request.json else None
            if not multiaddr:
                return jsonify({'error': 'multiaddr required', 'success': False}), 400

            try:
                async def do_connect():
                    return await peers.connect_peer(multiaddr)

                if trio_token:
                    result = trio.from_thread.run(do_connect, trio_token=trio_token)
                else:
                    result = trio.run(do_connect)

                return jsonify({'success': bool(result), 'multiaddr': multiaddr})
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @ipc_app.route('/p2p/disconnect/<peer_id>', methods=['POST'])
        def p2p_disconnect(peer_id):
            """Disconnect from a peer."""
            peers = getattr(startupDag, '_p2p_peers', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            try:
                async def do_disconnect():
                    return await peers.disconnect_peer(peer_id)

                if trio_token:
                    result = trio.from_thread.run(do_disconnect, trio_token=trio_token)
                else:
                    result = trio.run(do_disconnect)

                return jsonify({'success': bool(result), 'peer_id': peer_id})
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @ipc_app.route('/p2p/identity')
        def p2p_identity():
            """Get identity information."""
            identity = getattr(startupDag, 'identity', None)
            peers = getattr(startupDag, '_p2p_peers', None)

            return jsonify({
                'address': identity.address if identity and hasattr(identity, 'address') else None,
                'peer_id': str(peers.peer_id) if peers and hasattr(peers, 'peer_id') else None,
            })

        @ipc_app.route('/reward-address', methods=['GET', 'POST'])
        def ipc_reward_address():
            """Get or set reward address via IPC."""
            if request.method == 'POST':
                data = request.get_json() or {}
                new_address = data.get('reward_address')
                if not new_address:
                    return jsonify({'error': 'Missing reward_address'}), 400
                try:
                    success = startupDag.setRewardAddress(address=new_address, globally=True)
                    if success:
                        return jsonify({'success': True, 'reward_address': new_address})
                    else:
                        return jsonify({'error': 'Failed to set reward address'}), 500
                except Exception as e:
                    return jsonify({'error': str(e)}), 500

            # GET - return current reward address
            reward_address = getattr(startupDag, 'configRewardAddress', None) or ''
            return jsonify({'reward_address': reward_address})

        @ipc_app.route('/engine/performance')
        def ipc_engine_performance():
            """Get engine performance metrics via IPC."""
            import pandas as pd

            empty_response = {
                'observations': [],
                'predictions': [],
                'accuracy': [],
                'stats': {},
                'available': False
            }

            try:
                if not hasattr(startupDag, 'aiengine') or startupDag.aiengine is None:
                    return jsonify(empty_response)

                if not startupDag.aiengine.streamModels:
                    empty_response['error'] = 'No streams configured'
                    return jsonify(empty_response)

                streamUuid = list(startupDag.aiengine.streamModels.keys())[0]
                streamModel = startupDag.aiengine.streamModels[streamUuid]

                if not hasattr(streamModel, 'storage') or streamModel.storage is None:
                    empty_response['error'] = 'Storage not initialized'
                    return jsonify(empty_response)

                # Get last 100 observations
                obs_df = streamModel.storage.getStreamData(streamModel.streamUuid)
                if obs_df.empty:
                    return jsonify({
                        'observations': [],
                        'predictions': [],
                        'accuracy': [],
                        'stats': {},
                        'available': True
                    })

                obs_df = obs_df.tail(100).reset_index()
                observations = [
                    {'ts': str(row['ts']), 'value': float(row['value'])}
                    for _, row in obs_df.iterrows()
                ]

                # Get last 100 predictions
                pred_df = streamModel.storage.getPredictions(streamModel.predictionStreamUuid)
                if pred_df.empty:
                    return jsonify({
                        'observations': observations,
                        'predictions': [],
                        'accuracy': [],
                        'stats': {},
                        'available': True
                    })

                pred_df = pred_df.tail(100).reset_index()
                predictions = [
                    {'ts': str(row['ts']), 'value': float(row['value'])}
                    for _, row in pred_df.iterrows()
                ]

                # Calculate accuracy
                accuracy_data = []
                for idx, pred_row in pred_df.iterrows():
                    pred_ts = pred_row['ts']
                    pred_value = float(pred_row['value'])

                    try:
                        if pd.api.types.is_datetime64_any_dtype(obs_df['ts']):
                            pred_ts_compare = pd.to_datetime(pred_ts)
                        elif pd.api.types.is_numeric_dtype(obs_df['ts']):
                            try:
                                pred_ts_compare = float(pred_ts)
                            except (ValueError, TypeError):
                                pred_ts_dt = pd.to_datetime(pred_ts)
                                pred_ts_compare = pred_ts_dt.timestamp()
                        else:
                            pred_ts_compare = pred_ts

                        next_obs = obs_df[obs_df['ts'] > pred_ts_compare]
                    except Exception:
                        continue

                    if not next_obs.empty:
                        obs_value = float(next_obs.iloc[0]['value'])
                        error = pred_value - obs_value
                        abs_error = abs(error)
                        accuracy_data.append({
                            'ts': str(pred_ts),
                            'error': error,
                            'abs_error': abs_error,
                            'predicted': pred_value,
                            'actual': obs_value
                        })

                # Calculate statistics
                stats = {}
                if accuracy_data:
                    errors = [d['error'] for d in accuracy_data]
                    abs_errors = [d['abs_error'] for d in accuracy_data]
                    actuals = [d['actual'] for d in accuracy_data]

                    avg_error = sum(errors) / len(errors)
                    avg_abs_error = sum(abs_errors) / len(abs_errors)
                    avg_actual = sum(actuals) / len(actuals) if actuals else 1

                    avg_pct_error = (avg_abs_error / avg_actual * 100) if avg_actual != 0 else 0
                    accuracy_pct = max(0, 100 - avg_pct_error)

                    stats = {
                        'avg_error': round(avg_error, 4),
                        'avg_abs_error': round(avg_abs_error, 4),
                        'accuracy_pct': round(accuracy_pct, 2)
                    }

                return jsonify({
                    'observations': observations,
                    'predictions': predictions,
                    'accuracy': accuracy_data,
                    'stats': stats,
                    'available': True
                })

            except Exception as e:
                empty_response['error'] = str(e)
                return jsonify(empty_response)

        @ipc_app.route('/p2p/multiaddrs')
        def p2p_multiaddrs():
            """Get listen multiaddresses."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                addrs = []
                # satorip2p uses _host (private)
                host = getattr(peers, '_host', None) or getattr(peers, 'host', None)
                if host:
                    addrs = [str(a) for a in host.get_addrs()]
                return jsonify({'multiaddrs': addrs})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/identify/known')
        def p2p_identify_known():
            """Get known peer identities from the Identify protocol."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                identities = {}
                if hasattr(peers, 'get_known_peer_identities'):
                    raw_identities = peers.get_known_peer_identities()
                    # Convert PeerIdentity objects to dicts
                    # Use field names that match what routes.py expects
                    for peer_id, identity in raw_identities.items():
                        # Handle both possible attribute names for addresses
                        listen_addrs = getattr(identity, 'listen_addresses', None) or getattr(identity, 'listen_addrs', [])
                        evrmore_addr = getattr(identity, 'evrmore_address', None) or getattr(identity, 'wallet_address', None) or ''

                        identities[str(peer_id)] = {
                            'peer_id': str(peer_id),
                            'protocol_version': getattr(identity, 'protocol_version', None),
                            'agent_version': getattr(identity, 'agent_version', None),
                            'listen_addresses': [str(a) for a in listen_addrs],
                            'protocols': list(getattr(identity, 'protocols', [])),
                            'observed_addr': str(getattr(identity, 'observed_addr', '')) if getattr(identity, 'observed_addr', None) else None,
                            'roles': list(getattr(identity, 'roles', [])),
                            'evrmore_address': evrmore_addr,
                            'capabilities': list(getattr(identity, 'capabilities', [])),
                            'timestamp': getattr(identity, 'timestamp', 0),
                        }
                return jsonify({'identities': identities, 'count': len(identities)})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/identify/announce', methods=['POST'])
        def p2p_identify_announce():
            """Announce our identity to the network."""
            peers = getattr(startupDag, '_p2p_peers', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            try:
                async def do_announce():
                    if hasattr(peers, 'announce_identity'):
                        await peers.announce_identity()
                        return True
                    return False

                if trio_token:
                    result = trio.from_thread.run(do_announce, trio_token=trio_token)
                else:
                    result = trio.run(do_announce)

                return jsonify({'success': result})
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @ipc_app.route('/p2p/full-status')
        def p2p_full_status():
            """Get comprehensive P2P status including all properties."""
            peers = getattr(startupDag, '_p2p_peers', None)
            identity = getattr(startupDag, 'identity', None)

            if not peers:
                return jsonify({'error': 'P2P not initialized', 'available': False}), 503

            try:
                result = {
                    'available': True,
                    'peer_id': str(peers.peer_id) if hasattr(peers, 'peer_id') else None,
                    'public_key': peers.public_key if hasattr(peers, 'public_key') else None,
                    'evrmore_address': peers.evrmore_address if hasattr(peers, 'evrmore_address') else None,
                    'nat_type': peers.nat_type if hasattr(peers, 'nat_type') else 'unknown',
                    'is_relay': peers.is_relay if hasattr(peers, 'is_relay') else False,
                    'is_connected': peers.is_connected if hasattr(peers, 'is_connected') else False,
                    'connected_count': peers.connected_peers if hasattr(peers, 'connected_peers') else 0,
                    # Protocol flags
                    'enable_pubsub': getattr(peers, 'enable_pubsub', True),
                    'enable_dht': getattr(peers, 'enable_dht', True),
                    'enable_ping': getattr(peers, 'enable_ping', True),
                    'enable_identify': getattr(peers, 'enable_identify', True),
                    'enable_relay': getattr(peers, 'enable_relay', True),
                    'enable_mdns': getattr(peers, 'enable_mdns', True),
                    'enable_rendezvous': getattr(peers, 'enable_rendezvous', False),
                    'enable_upnp': getattr(peers, 'enable_upnp', True),
                }
                # Add public addresses
                if hasattr(peers, 'public_addresses'):
                    result['public_addresses'] = [str(a) for a in peers.public_addresses]
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e), 'available': False}), 500

        @ipc_app.route('/p2p/latencies')
        def p2p_latencies():
            """Get peer latencies."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                latencies = {}
                avg_latency = None
                if hasattr(peers, 'get_all_peer_latencies'):
                    latencies = peers.get_all_peer_latencies()
                    # Convert peer IDs to strings
                    latencies = {str(k): v for k, v in latencies.items()}
                if hasattr(peers, 'get_network_avg_latency'):
                    avg_latency = peers.get_network_avg_latency()
                return jsonify({'latencies': latencies, 'avg_latency': avg_latency})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/pubsub/debug')
        def p2p_pubsub_debug():
            """Get pubsub debug info."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                if hasattr(peers, 'get_pubsub_debug'):
                    return jsonify(peers.get_pubsub_debug())
                return jsonify({})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/rendezvous')
        def p2p_rendezvous():
            """Get rendezvous/bootstrap status."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                if hasattr(peers, 'get_rendezvous_status'):
                    return jsonify(peers.get_rendezvous_status())
                return jsonify({})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/peers/by-role/<role>')
        def p2p_peers_by_role(role):
            """Get peers filtered by role."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                peer_list = []
                if hasattr(peers, 'get_peers_by_role'):
                    peer_list = [str(p) for p in peers.get_peers_by_role(role)]
                return jsonify({'peers': peer_list, 'role': role})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/forget/<peer_id>', methods=['POST'])
        def p2p_forget(peer_id):
            """Forget a peer."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            try:
                if hasattr(peers, 'forget_peer'):
                    result = peers.forget_peer(peer_id)
                    return jsonify({'success': bool(result), 'peer_id': peer_id})
                return jsonify({'success': False, 'error': 'forget_peer not available'})
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @ipc_app.route('/p2p/discover', methods=['POST'])
        def p2p_discover():
            """Discover peers on the network.

            This triggers active peer discovery via DHT/rendezvous.
            Optionally filter by stream_id to find publishers for a specific stream.
            """
            peers = getattr(startupDag, '_p2p_peers', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            stream_id = request.json.get('stream_id') if request.json else None

            try:
                async def do_discover():
                    if hasattr(peers, 'discover_peers'):
                        return await peers.discover_peers(stream_id=stream_id)
                    return []

                if trio_token:
                    discovered = trio.from_thread.run(do_discover, trio_token=trio_token)
                else:
                    discovered = trio.run(do_discover)

                return jsonify({
                    'success': True,
                    'discovered_peers': [str(p) for p in discovered],
                    'count': len(discovered),
                    'stream_id': stream_id,
                })
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @ipc_app.route('/p2p/network-map')
        def p2p_network_map():
            """Get the network topology map.

            Returns information about the network structure including
            connected peers, their relationships, and mesh topology.
            """
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                if hasattr(peers, 'get_network_map'):
                    network_map = peers.get_network_map()
                    return jsonify(network_map)
                return jsonify({})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/subscriptions')
        def p2p_subscriptions():
            """Get our pubsub subscriptions."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                subscriptions = []
                publications = []
                if hasattr(peers, 'get_my_subscriptions'):
                    subscriptions = peers.get_my_subscriptions()
                if hasattr(peers, 'get_my_publications'):
                    publications = peers.get_my_publications()
                return jsonify({
                    'subscriptions': subscriptions,
                    'publications': publications,
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/subscription-map')
        def p2p_subscription_map():
            """Get the subscription map showing who subscribes to what."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                if hasattr(peers, 'get_subscription_map'):
                    return jsonify(peers.get_subscription_map())
                return jsonify({})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/publishers/<stream_id>')
        def p2p_publishers(stream_id):
            """Get publishers for a specific stream."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                publishers = []
                if hasattr(peers, 'get_publishers'):
                    publishers = [str(p) for p in peers.get_publishers(stream_id)]
                return jsonify({'publishers': publishers, 'stream_id': stream_id})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/subscribers/<stream_id>')
        def p2p_subscribers(stream_id):
            """Get subscribers for a specific stream."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                subscribers = []
                if hasattr(peers, 'get_subscribers'):
                    subscribers = [str(p) for p in peers.get_subscribers(stream_id)]
                return jsonify({'subscribers': subscribers, 'stream_id': stream_id})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/peer-subscriptions/<peer_id>')
        def p2p_peer_subscriptions(peer_id):
            """Get what streams a specific peer subscribes to."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                subscriptions = []
                if hasattr(peers, 'get_peer_subscriptions'):
                    subscriptions = peers.get_peer_subscriptions(peer_id)
                return jsonify({'peer_id': peer_id, 'subscriptions': subscriptions})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/discover-publishers/<stream_id>', methods=['POST'])
        def p2p_discover_publishers(stream_id):
            """Discover publishers for a specific stream via DHT."""
            peers = getattr(startupDag, '_p2p_peers', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            try:
                async def do_discover():
                    if hasattr(peers, 'discover_publishers'):
                        return await peers.discover_publishers(stream_id)
                    return []

                if trio_token:
                    publishers = trio.from_thread.run(do_discover, trio_token=trio_token)
                else:
                    publishers = trio.run(do_discover)

                return jsonify({
                    'success': True,
                    'publishers': [str(p) for p in publishers],
                    'count': len(publishers),
                    'stream_id': stream_id,
                })
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @ipc_app.route('/p2p/connection-changes')
        def p2p_connection_changes():
            """Check for recent connection changes."""
            peers = getattr(startupDag, '_p2p_peers', None)
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            try:
                changes = []
                if hasattr(peers, 'check_connection_changes'):
                    raw_changes = peers.check_connection_changes()
                    for peer_id, connected in raw_changes:
                        changes.append({
                            'peer_id': str(peer_id),
                            'connected': connected,
                        })
                return jsonify({'changes': changes})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @ipc_app.route('/p2p/uptime')
        def p2p_uptime():
            """Get uptime tracker information."""
            uptime_tracker = getattr(startupDag, '_uptime_tracker', None)
            identity = getattr(startupDag, 'identity', None)

            result = {
                'success': True,
                'streak_days': 0,
                'heartbeats_sent': 0,
                'heartbeats_received': 0,
                'current_round': '--',
                'is_relay_qualified': False,
                'uptime_percentage': 0.0,
            }

            if not uptime_tracker:
                return jsonify(result)

            try:
                evrmore_address = identity.address if identity and hasattr(identity, 'address') else ''

                # Streak days
                if hasattr(uptime_tracker, 'get_uptime_streak_days'):
                    result['streak_days'] = uptime_tracker.get_uptime_streak_days(evrmore_address)

                # Heartbeat counts
                if hasattr(uptime_tracker, '_heartbeats_sent'):
                    result['heartbeats_sent'] = uptime_tracker._heartbeats_sent
                if hasattr(uptime_tracker, '_heartbeats_received'):
                    result['heartbeats_received'] = uptime_tracker._heartbeats_received

                # Current round
                if hasattr(uptime_tracker, '_current_round') and uptime_tracker._current_round:
                    result['current_round'] = uptime_tracker._current_round

                # Uptime percentage
                if hasattr(uptime_tracker, 'get_uptime_percentage'):
                    try:
                        # Try with address first (newer API), fallback to no args (older API)
                        result['uptime_percentage'] = uptime_tracker.get_uptime_percentage(evrmore_address)
                    except TypeError:
                        result['uptime_percentage'] = uptime_tracker.get_uptime_percentage()

                # Relay qualification (95% uptime threshold)
                if hasattr(uptime_tracker, 'is_relay_qualified'):
                    result['is_relay_qualified'] = uptime_tracker.is_relay_qualified(evrmore_address)

                # Active node count
                if hasattr(uptime_tracker, 'get_active_node_count'):
                    result['active_node_count'] = uptime_tracker.get_active_node_count()

                # Last status message from our own heartbeat
                if hasattr(uptime_tracker, '_last_status_message'):
                    result['last_status_message'] = uptime_tracker._last_status_message
                elif hasattr(uptime_tracker, 'last_status_message'):
                    result['last_status_message'] = uptime_tracker.last_status_message

                return jsonify(result)
            except Exception as e:
                result['error'] = str(e)
                return jsonify(result)

        @ipc_app.route('/p2p/heartbeats')
        def p2p_heartbeats():
            """Get recent heartbeats from the network."""
            uptime_tracker = getattr(startupDag, '_uptime_tracker', None)
            limit = request.args.get('limit', 20, type=int)

            heartbeats = []
            if uptime_tracker and hasattr(uptime_tracker, 'get_recent_heartbeats'):
                try:
                    raw = uptime_tracker.get_recent_heartbeats(limit=limit)
                    for hb in raw:
                        heartbeats.append({
                            'node_id': hb.node_id[:12] + '...' if len(hb.node_id) > 12 else hb.node_id,
                            'full_node_id': hb.node_id,
                            'address': getattr(hb, 'evrmore_address', '')[:12] + '...',
                            'full_address': getattr(hb, 'evrmore_address', ''),
                            'timestamp': hb.timestamp,
                            'roles': list(getattr(hb, 'roles', [])),
                            'status': getattr(hb, 'status', ''),
                        })
                except Exception as e:
                    return jsonify({'heartbeats': [], 'error': str(e)})

            return jsonify({'heartbeats': heartbeats})

        @ipc_app.route('/p2p/consensus/status')
        def p2p_consensus_status():
            """Get consensus manager status."""
            consensus = getattr(startupDag, '_consensus_manager', None)

            result = {
                'success': True,
                'current_round': None,
                'phase': 'inactive',
                'vote_count': 0,
                'my_vote_submitted': False,
                'round_start_time': None,
                'phase_deadline': None,
            }

            if not consensus:
                return jsonify(result)

            try:
                if hasattr(consensus, '_current_round'):
                    result['current_round'] = consensus._current_round
                if hasattr(consensus, '_phase'):
                    result['phase'] = consensus._phase.value if hasattr(consensus._phase, 'value') else str(consensus._phase)
                if hasattr(consensus, '_votes'):
                    result['vote_count'] = len(consensus._votes)
                if hasattr(consensus, '_my_vote_submitted'):
                    result['my_vote_submitted'] = consensus._my_vote_submitted
                if hasattr(consensus, '_round_start_time'):
                    result['round_start_time'] = consensus._round_start_time
                if hasattr(consensus, 'get_phase_deadline'):
                    result['phase_deadline'] = consensus.get_phase_deadline()
            except Exception as e:
                result['error'] = str(e)

            return jsonify(result)

        @ipc_app.route('/p2p/consensus/history')
        def p2p_consensus_history():
            """Get consensus round history."""
            consensus = getattr(startupDag, '_consensus_manager', None)
            limit = request.args.get('limit', 10, type=int)

            rounds = []
            if consensus and hasattr(consensus, 'get_round_history'):
                try:
                    history = consensus.get_round_history(limit=limit)
                    for r in history:
                        rounds.append({
                            'round_id': r.round_id,
                            'merkle_root': r.merkle_root,
                            'vote_count': r.vote_count,
                            'consensus_reached': r.consensus_reached,
                            'timestamp': r.timestamp,
                        })
                except Exception:
                    pass

            return jsonify({'rounds': rounds})

        @ipc_app.route('/p2p/activity-stats')
        def p2p_activity_stats():
            """Get live activity stats for the Live Data Streams display."""
            try:
                from web.p2p_bridge import get_bridge
                bridge = get_bridge()
                counts = bridge.get_counts()
                hourly = bridge.get_hourly_activity(hours=24)
                return jsonify({
                    'success': True,
                    'counts': counts,
                    'hourly': hourly,
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'counts': {
                        'predictions': 0,
                        'observations': 0,
                        'heartbeats': 0,
                        'consensus_votes': 0,
                        'governance': 0,
                    },
                    'hourly': {},
                })

        @ipc_app.route('/p2p/recent-events')
        def p2p_recent_events():
            """Get recent events for live display (polling fallback for WebSocket)."""
            try:
                from web.p2p_bridge import get_bridge
                bridge = get_bridge()
                # Get events since last poll (returns recent events list)
                events = bridge.get_recent_events(limit=50)
                return jsonify({
                    'success': True,
                    'events': events,
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'events': [],
                })

        @ipc_app.route('/p2p/bandwidth')
        def p2p_bandwidth():
            """Get bandwidth usage statistics and QoS status."""
            result = {
                'success': True,
                'status': 'unavailable',
                'global': {
                    'bytes_sent': 0,
                    'bytes_received': 0,
                    'messages_sent': 0,
                    'messages_received': 0,
                    'bytes_per_second': 0.0,
                    'messages_per_second': 0.0,
                },
                'topics': {},
                'qos': {
                    'enabled': False,
                    'drops_low_priority': 0,
                    'drops_rate_limited': 0,
                    'policy': 'none',
                },
                'peers': {},
            }

            # Get bandwidth tracker stats
            tracker = getattr(startupDag, '_bandwidth_tracker', None)
            if tracker:
                result['status'] = 'active'

                # Global metrics
                if hasattr(tracker, 'get_global_metrics'):
                    global_metrics = tracker.get_global_metrics()
                    if isinstance(global_metrics, dict):
                        # Flatten the nested structure for the UI
                        cumulative = global_metrics.get('cumulative', {})
                        rates = global_metrics.get('rates', {})
                        result['global'] = {
                            'bytes_sent': cumulative.get('bytes_sent', 0),
                            'bytes_received': cumulative.get('bytes_received', 0),
                            'messages_sent': cumulative.get('messages_sent', 0),
                            'messages_received': cumulative.get('messages_received', 0),
                            'bytes_per_second': rates.get('bytes_out_1s', 0) + rates.get('bytes_in_1s', 0),
                            'messages_per_second': rates.get('msgs_out_1s', 0) + rates.get('msgs_in_1s', 0),
                        }

                # Per-topic metrics (get all topic rates)
                if hasattr(tracker, 'get_all_topic_rates'):
                    topic_rates = tracker.get_all_topic_rates()
                    for topic, rate in topic_rates.items():
                        # Get detailed metrics for each topic
                        if hasattr(tracker, 'get_topic_metrics'):
                            result['topics'][topic] = tracker.get_topic_metrics(topic)
                            result['topics'][topic]['rate_bytes_sec'] = rate

                # Per-peer metrics (summarized)
                if hasattr(tracker, 'get_top_peers'):
                    top_peers = tracker.get_top_peers(20)
                    result['peers'] = {
                        'count': len(top_peers),
                        'top_by_bandwidth': [
                            {'peer_id': peer_id[:16] + '...', 'bytes_per_second': rate}
                            for peer_id, rate in top_peers
                        ],
                    }

            # Get QoS manager stats
            qos = getattr(startupDag, '_qos_manager', None)
            if qos:
                result['qos']['enabled'] = True
                if hasattr(qos, 'get_stats'):
                    qos_stats = qos.get_stats()
                    result['qos']['drops_low_priority'] = qos_stats.get('drops_low_priority', 0)
                    result['qos']['drops_rate_limited'] = qos_stats.get('drops_rate_limited', 0)
                    result['qos']['policy'] = qos_stats.get('policy', 'default')

            return jsonify(result)

        @ipc_app.route('/p2p/bandwidth/history')
        def p2p_bandwidth_history():
            """Get bandwidth usage history for charting."""
            result = {
                'success': True,
                'history': [],
                'interval_seconds': 60,
                'points': 60,
            }

            tracker = getattr(startupDag, '_bandwidth_tracker', None)
            if tracker and hasattr(tracker, 'get_history'):
                result['history'] = tracker.get_history(points=60)

            return jsonify(result)

        @ipc_app.route('/p2p/storage')
        def p2p_storage():
            """Get storage redundancy status."""
            import os

            result = {
                'success': True,
                'status': 'unavailable',
                'disk_usage': {
                    'used_bytes': 0,
                    'used_mb': 0.0,
                    'storage_dir': '~/.satori/storage',
                },
                'backends': {
                    'memory': {'enabled': False, 'items': 0},
                    'file': {'enabled': False, 'items': 0},
                    'dht': {'enabled': False, 'items': 0},
                },
                'deferred_rewards': {
                    'stored_count': 0,
                    'pending_sync': 0,
                },
                'alerts': {
                    'stored_count': 0,
                    'pending_sync': 0,
                },
            }

            # Calculate actual disk usage
            storage_dir = os.path.expanduser('~/.satori/storage')
            if os.path.exists(storage_dir):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(storage_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                result['disk_usage']['used_bytes'] = total_size
                result['disk_usage']['used_mb'] = round(total_size / (1024 * 1024), 2)
                result['disk_usage']['storage_dir'] = storage_dir

            # Get storage manager status
            manager = getattr(startupDag, '_storage_manager', None)
            if manager:
                result['status'] = 'active'
                if hasattr(manager, 'get_status'):
                    status = manager.get_status()
                    result['backends'] = status.get('backends', result['backends'])

            # Get deferred rewards storage status
            deferred_storage = getattr(startupDag, '_deferred_rewards_storage', None)
            if deferred_storage:
                result['backends']['file']['enabled'] = True
                if hasattr(deferred_storage, 'count'):
                    result['deferred_rewards']['stored_count'] = deferred_storage.count()
                if hasattr(deferred_storage, 'pending_sync_count'):
                    result['deferred_rewards']['pending_sync'] = deferred_storage.pending_sync_count()

            # Get alert storage status
            alert_storage = getattr(startupDag, '_alert_storage', None)
            if alert_storage:
                if hasattr(alert_storage, 'count'):
                    result['alerts']['stored_count'] = alert_storage.count()
                if hasattr(alert_storage, 'pending_sync_count'):
                    result['alerts']['pending_sync'] = alert_storage.pending_sync_count()

            return jsonify(result)

        @ipc_app.route('/p2p/version')
        def p2p_version():
            """Get protocol version and enabled features."""
            peers = getattr(startupDag, '_p2p_peers', None)

            result = {
                'success': True,
                'current_version': '1.0.0',
                'features': [],
            }

            if peers:
                # Build feature list from enabled protocols
                features = []
                if getattr(peers, 'enable_pubsub', False):
                    features.append({'name': 'pubsub_gossipsub', 'enabled': True})
                if getattr(peers, 'enable_dht', False):
                    features.append({'name': 'dht_kademlia', 'enabled': True})
                if getattr(peers, 'enable_ping', False):
                    features.append({'name': 'ping_protocol', 'enabled': True})
                if getattr(peers, 'enable_identify', False):
                    features.append({'name': 'identify_protocol', 'enabled': True})
                if getattr(peers, 'enable_relay', False):
                    features.append({'name': 'circuit_relay', 'enabled': True})
                if getattr(peers, 'enable_mdns', False):
                    features.append({'name': 'mdns_discovery', 'enabled': True})
                if getattr(peers, 'enable_rendezvous', False):
                    features.append({'name': 'rendezvous', 'enabled': True})
                if getattr(peers, 'enable_upnp', False):
                    features.append({'name': 'upnp_nat', 'enabled': True})

                # Add satori-specific protocols
                if getattr(startupDag, '_uptime_tracker', None):
                    features.append({'name': 'heartbeat_uptime', 'enabled': True})
                if getattr(startupDag, '_consensus_manager', None):
                    features.append({'name': 'consensus', 'enabled': True})
                if getattr(startupDag, '_lending_manager', None):
                    features.append({'name': 'lending_protocol', 'enabled': True})
                if getattr(startupDag, '_delegation_manager', None):
                    features.append({'name': 'delegation', 'enabled': True})
                if getattr(startupDag, '_governance_manager', None):
                    features.append({'name': 'governance', 'enabled': True})

                result['features'] = features

            return jsonify(result)

        # =====================================================================
        # Oracle Network IPC Endpoints
        # =====================================================================

        @ipc_app.route('/p2p/oracle/stats')
        def p2p_oracle_stats():
            """Get oracle network statistics."""
            oracle = getattr(startupDag, '_oracle_network', None)

            result = {
                'started': False,
                'subscribed_streams': 0,
                'my_oracle_registrations': 0,
                'known_oracles': 0,
                'cached_observations': 0,
            }

            if oracle:
                result['started'] = True
                if hasattr(oracle, 'get_stats'):
                    stats = oracle.get_stats()
                    result.update(stats)
                else:
                    result['subscribed_streams'] = len(getattr(oracle, '_subscriptions', {}))
                    result['my_oracle_registrations'] = len(getattr(oracle, '_my_registrations', {}))
                    result['known_oracles'] = len(getattr(oracle, '_known_oracles', {}))
                    result['cached_observations'] = len(getattr(oracle, '_observation_cache', {}))

            return jsonify(result)

        @ipc_app.route('/p2p/oracle/observations')
        def p2p_oracle_observations():
            """Get recent observations."""
            oracle = getattr(startupDag, '_oracle_network', None)
            limit = request.args.get('limit', 20, type=int)
            include_own = request.args.get('include_own', 'true').lower() == 'true'

            result = {'observations': [], 'count': 0}

            if oracle and hasattr(oracle, 'get_recent_observations'):
                observations = oracle.get_recent_observations(limit=limit, include_own=include_own)
                result['observations'] = [
                    {
                        'stream_id': o.stream_id if hasattr(o, 'stream_id') else str(o.get('stream_id', '')),
                        'value': o.value if hasattr(o, 'value') else o.get('value'),
                        'timestamp': o.timestamp if hasattr(o, 'timestamp') else o.get('timestamp', 0),
                        'oracle_address': getattr(o, 'oracle', '') or o.get('oracle', ''),
                        'peer_id': getattr(o, 'peer_id', '') or o.get('peer_id', ''),
                    }
                    for o in observations
                ]
                result['count'] = len(result['observations'])

            return jsonify(result)

        @ipc_app.route('/p2p/oracle/observations/my')
        def p2p_oracle_observations_my():
            """Get our own published observations with pagination."""
            oracle = getattr(startupDag, '_oracle_network', None)
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)

            result = {'observations': [], 'total': 0, 'page': page, 'per_page': per_page}

            if oracle and hasattr(oracle, 'get_my_published_observations'):
                # Get all observations (no limit in the method)
                all_observations = oracle.get_my_published_observations()
                total = len(all_observations)

                # Paginate
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                paginated = all_observations[start_idx:end_idx]

                result['observations'] = [
                    {
                        'stream_id': o.stream_id if hasattr(o, 'stream_id') else str(o.get('stream_id', '')),
                        'value': o.value if hasattr(o, 'value') else o.get('value'),
                        'timestamp': o.timestamp if hasattr(o, 'timestamp') else o.get('timestamp', 0),
                        'oracle_address': getattr(o, 'oracle', '') or o.get('oracle', ''),
                        'peer_id': getattr(o, 'peer_id', '') or o.get('peer_id', ''),
                        'signature': getattr(o, 'signature', '') or o.get('signature', ''),
                    }
                    for o in paginated
                ]
                result['total'] = total
                result['total_pages'] = (total + per_page - 1) // per_page

            return jsonify(result)

        @ipc_app.route('/p2p/oracle/publish-observation', methods=['POST'])
        def p2p_oracle_publish_observation():
            """Manually publish an observation (for testing). Must be registered as oracle."""
            oracle = getattr(startupDag, '_oracle_network', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not oracle:
                return jsonify({'success': False, 'error': 'Oracle network not initialized'}), 503

            data = request.get_json() or {}
            stream_id = data.get('stream_id')
            value = data.get('value')
            timestamp = data.get('timestamp')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400
            if value is None:
                return jsonify({'success': False, 'error': 'value is required'}), 400

            try:
                import trio
                import time

                timestamp = timestamp or int(time.time())

                async def do_publish():
                    return await oracle.publish_observation(
                        stream_id=stream_id,
                        value=float(value),
                        timestamp=timestamp
                    )

                if trio_token:
                    observation = trio.from_thread.run(do_publish, trio_token=trio_token)
                else:
                    return jsonify({'success': False, 'error': 'Trio not initialized'}), 503

                if observation:
                    return jsonify({
                        'success': True,
                        'stream_id': stream_id,
                        'value': value,
                        'timestamp': timestamp,
                        'observation': observation.to_dict() if hasattr(observation, 'to_dict') else str(observation)
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to publish (not registered as primary oracle?)'}), 400

            except Exception as e:
                logging.warning(f"Manual observation publish failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/oracle/subscribe', methods=['POST'])
        def p2p_oracle_subscribe():
            """Subscribe to observations for a stream."""
            oracle = getattr(startupDag, '_oracle_network', None)

            if not oracle:
                return jsonify({'success': False, 'error': 'Oracle network not initialized'}), 503

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            # Define a callback that emits to websocket via bridge
            def on_observation(observation):
                try:
                    from web.p2p_bridge import get_bridge
                    bridge = get_bridge()
                    if bridge:
                        bridge._on_observation(observation)
                except Exception as e:
                    logging.debug(f"Failed to emit observation: {e}")

            trio_token = getattr(startupDag, '_trio_token', None)

            async def do_subscribe():
                return await oracle.subscribe_to_stream(stream_id, on_observation)

            try:
                import trio
                if trio_token:
                    success = trio.from_thread.run(do_subscribe, trio_token=trio_token)
                else:
                    logging.warning("No trio token available for subscription")
                    return jsonify({'success': False, 'error': 'Trio not initialized'}), 503
            except Exception as e:
                logging.warning(f"Failed to subscribe via trio: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

            if success:
                return jsonify({
                    'success': True,
                    'stream_id': stream_id,
                    'message': f'Subscribed to observations for {stream_id}'
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to subscribe'})

        @ipc_app.route('/p2p/oracle/unsubscribe', methods=['POST'])
        def p2p_oracle_unsubscribe():
            """Unsubscribe from observations for a stream."""
            oracle = getattr(startupDag, '_oracle_network', None)

            if not oracle:
                return jsonify({'success': False, 'error': 'Oracle network not initialized'}), 503

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            trio_token = getattr(startupDag, '_trio_token', None)

            async def do_unsubscribe():
                return await oracle.unsubscribe_from_stream(stream_id)

            try:
                import trio
                if trio_token:
                    success = trio.from_thread.run(do_unsubscribe, trio_token=trio_token)
                else:
                    logging.warning("No trio token available for unsubscription")
                    return jsonify({'success': False, 'error': 'Trio not initialized'}), 503
            except Exception as e:
                logging.warning(f"Failed to unsubscribe via trio: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

            if success:
                return jsonify({
                    'success': True,
                    'stream_id': stream_id,
                    'message': f'Unsubscribed from observations for {stream_id}'
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to unsubscribe or not subscribed'})

        @ipc_app.route('/p2p/oracle/known')
        def p2p_oracle_known():
            """Get list of known oracles (discovered via registry)."""
            oracle = getattr(startupDag, '_oracle_network', None)

            result = {'oracles': [], 'count': 0}

            if oracle and hasattr(oracle, '_oracle_registrations'):
                oracles_list = []
                for stream_id, registrations in oracle._oracle_registrations.items():
                    for oracle_addr, reg in registrations.items():
                        oracles_list.append({
                            'stream_id': stream_id,
                            'oracle_address': oracle_addr,
                            'peer_id': getattr(reg, 'peer_id', ''),
                            'reputation': getattr(reg, 'reputation', 1.0),
                            'is_primary': getattr(reg, 'is_primary', False),
                            'timestamp': getattr(reg, 'timestamp', 0),
                        })
                result['oracles'] = oracles_list
                result['count'] = len(oracles_list)

            return jsonify(result)

        @ipc_app.route('/p2p/oracle/my_registrations')
        def p2p_oracle_my_registrations():
            """Get list of streams we're registered as oracle for."""
            oracle = getattr(startupDag, '_oracle_network', None)

            result = {'registrations': [], 'count': 0}

            if oracle and hasattr(oracle, '_my_registrations'):
                registrations = []
                for stream_id, reg in oracle._my_registrations.items():
                    registrations.append({
                        'stream_id': stream_id,
                        'oracle_address': getattr(reg, 'oracle', ''),
                        'timestamp': getattr(reg, 'timestamp', 0),
                    })
                result['registrations'] = registrations
                result['count'] = len(registrations)

            return jsonify(result)

        @ipc_app.route('/p2p/oracle/subscriptions')
        def p2p_oracle_subscriptions():
            """Get list of streams we're subscribed to for observations."""
            oracle = getattr(startupDag, '_oracle_network', None)

            result = {'subscriptions': [], 'count': 0}

            if oracle and hasattr(oracle, '_subscribed_streams'):
                result['subscriptions'] = list(oracle._subscribed_streams.keys())
                result['count'] = len(result['subscriptions'])

            return jsonify(result)

        @ipc_app.route('/p2p/oracle/summary')
        def p2p_oracle_summary():
            """Get summary of our oracle role (primary/secondary registrations)."""
            oracle = getattr(startupDag, '_oracle_network', None)

            if not oracle:
                return jsonify({
                    'primary_count': 0,
                    'secondary_count': 0,
                    'total_count': 0,
                    'primary_streams': [],
                    'secondary_streams': [],
                })

            if hasattr(oracle, 'get_my_oracle_summary'):
                return jsonify(oracle.get_my_oracle_summary())

            return jsonify({
                'primary_count': 0,
                'secondary_count': 0,
                'total_count': 0,
                'primary_streams': [],
                'secondary_streams': [],
            })

        @ipc_app.route('/p2p/oracle/role/<stream_id>')
        def p2p_oracle_role(stream_id: str):
            """Get our oracle role for a specific stream."""
            oracle = getattr(startupDag, '_oracle_network', None)

            if not oracle:
                return jsonify({'stream_id': stream_id, 'role': 'none'})

            role = 'none'
            if hasattr(oracle, 'get_oracle_role'):
                role = oracle.get_oracle_role(stream_id)

            return jsonify({'stream_id': stream_id, 'role': role})

        @ipc_app.route('/p2p/oracle/register-secondary', methods=['POST'])
        def p2p_oracle_register_secondary():
            """Register as a secondary oracle for a stream."""
            oracle = getattr(startupDag, '_oracle_network', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not oracle:
                return jsonify({'success': False, 'error': 'Oracle network not initialized'}), 503

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            try:
                async def do_register():
                    return await oracle.register_as_secondary_oracle(stream_id)

                if trio_token:
                    registration = trio.from_thread.run(do_register, trio_token=trio_token)
                else:
                    registration = trio.run(do_register)

                if registration:
                    return jsonify({
                        'success': True,
                        'stream_id': stream_id,
                        'role': 'secondary',
                        'registration': registration.to_dict(),
                    })
                else:
                    return jsonify({'success': False, 'error': 'Registration failed'}), 500

            except Exception as e:
                logging.warning(f"Secondary oracle registration failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/oracle/templates')
        def p2p_oracle_templates():
            """Get available oracle data source templates."""
            try:
                from satorip2p.protocol.oracle_network import list_templates
                templates = list_templates()
                return jsonify({'success': True, 'templates': templates})
            except ImportError:
                return jsonify({'success': False, 'error': 'Oracle templates not available'}), 503
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/oracle/register-primary', methods=['POST'])
        def p2p_oracle_register_primary():
            """Register as a primary oracle for a stream with data source config."""
            oracle = getattr(startupDag, '_oracle_network', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not oracle:
                return jsonify({'success': False, 'error': 'Oracle network not initialized'}), 503

            data = request.get_json() or {}
            stream_id = data.get('stream_id')
            data_source_config = data.get('data_source')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            try:
                from satorip2p.protocol.oracle_network import OracleDataSource

                # Parse data source config
                data_source = None
                if data_source_config:
                    data_source = OracleDataSource.from_dict(data_source_config)

                async def do_register():
                    return await oracle.register_as_oracle(
                        stream_id,
                        is_primary=True,
                        data_source=data_source
                    )

                if trio_token:
                    registration = trio.from_thread.run(do_register, trio_token=trio_token)
                else:
                    registration = trio.run(do_register)

                if registration:
                    return jsonify({
                        'success': True,
                        'stream_id': stream_id,
                        'role': 'primary',
                        'registration': registration.to_dict(),
                    })
                else:
                    return jsonify({'success': False, 'error': 'Registration failed'}), 500

            except Exception as e:
                logging.warning(f"Primary oracle registration failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/oracle/stream-info/<stream_id>')
        def p2p_oracle_stream_info(stream_id: str):
            """Get oracle info for a specific stream (primary, secondaries)."""
            oracle = getattr(startupDag, '_oracle_network', None)

            result = {
                'stream_id': stream_id,
                'primary': None,
                'secondaries': [],
                'my_role': 'none',
            }

            if not oracle:
                return jsonify(result)

            # Get primary oracle
            if hasattr(oracle, 'get_primary_oracle'):
                primary = oracle.get_primary_oracle(stream_id)
                if primary:
                    result['primary'] = {
                        'oracle': primary.oracle,
                        'peer_id': primary.peer_id,
                        'timestamp': primary.timestamp,
                        'reputation': primary.reputation,
                    }

            # Get secondary oracles
            if hasattr(oracle, 'get_secondary_oracles'):
                secondaries = oracle.get_secondary_oracles(stream_id)
                result['secondaries'] = [
                    {
                        'oracle': s.oracle,
                        'peer_id': s.peer_id,
                        'timestamp': s.timestamp,
                        'reputation': s.reputation,
                    }
                    for s in secondaries
                ]

            # Get our role
            if hasattr(oracle, 'get_oracle_role'):
                result['my_role'] = oracle.get_oracle_role(stream_id)

            return jsonify(result)

        # =====================================================================
        # Stream Registry IPC Endpoints
        # =====================================================================

        @ipc_app.route('/p2p/streams/sync', methods=['POST'])
        def p2p_streams_sync():
            """Sync oracle streams to the stream registry."""
            import asyncio

            try:
                # Run the sync function
                if hasattr(startupDag, '_sync_oracle_streams_to_registry'):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(startupDag._sync_oracle_streams_to_registry())
                    finally:
                        loop.close()

                # Return updated stats
                registry = getattr(startupDag, '_stream_registry', None)
                known_streams = len(getattr(registry, '_streams', {})) if registry else 0
                return jsonify({'success': True, 'synced_streams': known_streams})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/streams/stats')
        def p2p_streams_stats():
            """Get stream registry statistics, auto-syncing from oracle if empty."""
            import asyncio

            registry = getattr(startupDag, '_stream_registry', None)

            result = {
                'started': False,
                'known_streams': 0,
                'my_claims': 0,
                'total_claims': 0,
            }

            if registry:
                result['started'] = True

                # Auto-sync from oracle if registry is empty
                known_streams = len(getattr(registry, '_streams', {}))
                if known_streams == 0:
                    try:
                        if hasattr(startupDag, '_sync_oracle_streams_to_registry'):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                loop.run_until_complete(startupDag._sync_oracle_streams_to_registry())
                            finally:
                                loop.close()
                    except Exception:
                        pass

                if hasattr(registry, 'get_stats'):
                    stats = registry.get_stats()
                    result.update(stats)
                else:
                    result['known_streams'] = len(getattr(registry, '_streams', {}))
                    result['my_claims'] = len(getattr(registry, '_my_claims', {}))
                    result['total_claims'] = sum(
                        len(claims) for claims in getattr(registry, '_claims', {}).values()
                    )

            return jsonify(result)

        @ipc_app.route('/p2p/streams/discover')
        def p2p_streams_discover():
            """Discover available streams, auto-syncing from oracle if empty."""
            import asyncio

            registry = getattr(startupDag, '_stream_registry', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            result = {'streams': [], 'total': 0}

            # Auto-sync from oracle if registry is empty
            if registry and len(getattr(registry, '_streams', {})) == 0:
                try:
                    if hasattr(startupDag, '_sync_oracle_streams_to_registry'):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(startupDag._sync_oracle_streams_to_registry())
                        finally:
                            loop.close()
                except Exception:
                    pass

            if registry and hasattr(registry, 'discover_streams'):
                source = request.args.get('source')
                datatype = request.args.get('datatype')
                tags = request.args.getlist('tag')
                limit = request.args.get('limit', 100, type=int)
                # Filter for active streams only (with recent oracle activity)
                active_only = request.args.get('active_only', 'false').lower() == 'true'
                max_age = request.args.get('max_age', 900, type=int)  # Default 15 minutes

                try:
                    async def do_discover():
                        return await registry.discover_streams(
                            source=source,
                            datatype=datatype,
                            tags=tags if tags else None,
                            limit=limit
                        )

                    if trio_token:
                        streams = trio.from_thread.run(do_discover, trio_token=trio_token)
                    else:
                        streams = trio.run(do_discover)

                    # Filter for active streams if requested
                    if active_only:
                        import time
                        current_time = time.time()
                        streams = [
                            s for s in streams
                            if hasattr(s, 'last_observation_time') and
                               s.last_observation_time > 0 and
                               (current_time - s.last_observation_time) < max_age
                        ]

                    # Get claim counts per stream (registry uses _claims, not _stream_claims)
                    stream_claims = getattr(registry, '_claims', {})

                    # Get oracle network for oracle peer ID lookups
                    oracle_network = getattr(startupDag, '_oracle_network', None)
                    # Get my peer ID to mark "is_me"
                    my_peer_id = None
                    if hasattr(startupDag, '_p2p_peers') and startupDag._p2p_peers:
                        my_peer_id = getattr(startupDag._p2p_peers, 'peer_id', None)

                    def get_oracle_peer_ids(stream_id):
                        """Get list of oracle peer IDs for a stream, including self."""
                        if not oracle_network:
                            return []

                        oracle_list = []
                        seen_peers = set()

                        # Get network-discovered oracles
                        if hasattr(oracle_network, 'get_registered_oracles'):
                            for o in oracle_network.get_registered_oracles(stream_id):
                                if o.peer_id not in seen_peers:
                                    oracle_list.append({
                                        'peer_id': o.peer_id,
                                        'is_primary': o.is_primary,
                                        'is_me': o.peer_id == my_peer_id
                                    })
                                    seen_peers.add(o.peer_id)

                        # Also include our own registration if not already in list
                        if hasattr(oracle_network, '_my_registrations'):
                            my_reg = oracle_network._my_registrations.get(stream_id)
                            if my_reg and my_peer_id and my_peer_id not in seen_peers:
                                oracle_list.append({
                                    'peer_id': my_peer_id,
                                    'is_primary': getattr(my_reg, 'is_primary', False),
                                    'is_me': True
                                })

                        return oracle_list

                    result['streams'] = [
                        {
                            'stream_id': getattr(s, 'stream_id', ''),
                            'source': getattr(s, 'source', ''),
                            'stream': getattr(s, 'stream', ''),
                            'target': getattr(s, 'target', ''),
                            'datatype': getattr(s, 'datatype', ''),
                            'cadence': getattr(s, 'cadence', 0),
                            'predictor_slots': getattr(s, 'predictor_slots', 0),
                            'claimed_count': len(stream_claims.get(getattr(s, 'stream_id', ''), {})),
                            'creator': getattr(s, 'creator', ''),
                            'description': getattr(s, 'description', ''),
                            'tags': getattr(s, 'tags', []),
                            'last_observation_time': getattr(s, 'last_observation_time', 0),
                            'is_active': hasattr(s, 'is_active') and s.is_active(max_age),
                            'oracles': get_oracle_peer_ids(getattr(s, 'stream_id', '')),
                        }
                        for s in streams
                    ]
                    result['total'] = len(result['streams'])
                except Exception as e:
                    result['error'] = str(e)

            return jsonify(result)

        @ipc_app.route('/p2p/streams/list')
        def p2p_streams_list():
            """Get list of known streams."""
            registry = getattr(startupDag, '_stream_registry', None)

            result = {'streams': [], 'count': 0}

            if registry and hasattr(registry, 'get_all_streams'):
                streams = registry.get_all_streams()
                result['streams'] = [
                    {
                        'stream_id': s.stream_id if hasattr(s, 'stream_id') else s.get('stream_id', ''),
                        'source': s.source if hasattr(s, 'source') else s.get('source', ''),
                        'author': s.author if hasattr(s, 'author') else s.get('author', ''),
                        'claim_count': getattr(s, 'claim_count', 0) or s.get('claim_count', 0),
                    }
                    for s in streams
                ]
                result['count'] = len(result['streams'])

            return jsonify(result)

        @ipc_app.route('/p2p/streams/claim', methods=['POST'])
        def p2p_streams_claim():
            """Claim a predictor slot on a stream."""
            registry = getattr(startupDag, '_stream_registry', None)

            if not registry:
                return jsonify({'success': False, 'error': 'Stream registry not initialized'}), 503

            data = request.get_json() or {}
            stream_id = data.get('stream_id')
            slot_index = data.get('slot_index')
            ttl = data.get('ttl')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            async def do_claim():
                return await registry.claim_stream(
                    stream_id=stream_id,
                    slot_index=int(slot_index) if slot_index is not None else None,
                    ttl=int(ttl) if ttl else None
                )

            import asyncio
            try:
                loop = asyncio.new_event_loop()
                claim = loop.run_until_complete(do_claim())
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
            finally:
                loop.close()

            if claim:
                # After claiming, wire to Engine to start predicting
                engine_wired = False

                # In P2P mode, engine may not exist yet - create it on first claim
                if not hasattr(startupDag, 'aiengine') or startupDag.aiengine is None:
                    try:
                        from engine import Engine
                        startupDag.aiengine = Engine()
                        # Wire P2P components if available
                        if hasattr(startupDag, '_p2p_peers') and startupDag._p2p_peers is not None:
                            startupDag.aiengine.setP2PPeers(startupDag._p2p_peers)
                        if hasattr(startupDag, '_prediction_protocol') and startupDag._prediction_protocol is not None:
                            startupDag.aiengine.setPredictionProtocol(startupDag._prediction_protocol)
                        if hasattr(startupDag, '_oracle_network') and startupDag._oracle_network is not None:
                            startupDag.aiengine.setOracleNetwork(startupDag._oracle_network)
                        logging.info("Engine created for P2P stream claims", color='green')
                    except Exception as e:
                        logging.warning(f"Failed to create Engine for P2P claims: {e}")

                if hasattr(startupDag, 'aiengine') and startupDag.aiengine is not None:
                    try:
                        engine_wired = startupDag.aiengine.addStreamFromClaim(stream_id)
                        if engine_wired:
                            logging.info(f"Stream {stream_id} claimed and wired to Engine", color='green')
                        else:
                            logging.warning(f"Stream {stream_id} claimed but Engine wiring failed")
                    except Exception as e:
                        logging.warning(f"Failed to wire claim to Engine: {e}")

                # Also subscribe to observations via oracle network (using trio)
                oracle_network = getattr(startupDag, '_oracle_network', None)
                if oracle_network is not None and hasattr(startupDag, 'aiengine') and startupDag.aiengine is not None:
                    try:
                        engine = startupDag.aiengine

                        def observation_callback(observation):
                            """Handle observation and pass to engine."""
                            # Find the stream UUID for this stream_id
                            for uuid, model in engine.streamModels.items():
                                if getattr(model, 'stream_name', None) == stream_id:
                                    engine._handleP2PObservation(uuid, observation, stream_id)
                                    return
                            # Fallback if no match found by stream_name
                            engine._handleP2PObservation(stream_id, observation, stream_id)

                        async def do_subscribe():
                            return await oracle_network.subscribe_to_stream(stream_id, observation_callback)

                        sub_loop = asyncio.new_event_loop()
                        obs_subscribed = sub_loop.run_until_complete(do_subscribe())
                        sub_loop.close()
                        if obs_subscribed:
                            logging.info(f"Subscribed to observations for {stream_id}", color='green')
                    except Exception as e:
                        logging.warning(f"Failed to subscribe to observations: {e}")

                # Auto-subscribe to predictions for this stream
                prediction_subscribed = False
                prediction_protocol = getattr(startupDag, '_prediction_protocol', None)
                if prediction_protocol is not None:
                    try:
                        async def do_subscribe():
                            def on_prediction_received(prediction):
                                # Predictions are cached by the protocol, log for now
                                logging.debug(f"Received prediction for {stream_id[:16]}...")
                            return await prediction_protocol.subscribe_to_predictions(
                                stream_id=stream_id,
                                callback=on_prediction_received
                            )
                        sub_loop = asyncio.new_event_loop()
                        prediction_subscribed = sub_loop.run_until_complete(do_subscribe())
                        sub_loop.close()
                        if prediction_subscribed:
                            logging.info(f"Auto-subscribed to predictions for {stream_id[:16]}...", color='green')
                    except Exception as e:
                        logging.warning(f"Failed to auto-subscribe to predictions: {e}")

                return jsonify({
                    'success': True,
                    'claim': {
                        'stream_id': claim.stream_id,
                        'slot_index': claim.slot_index,
                        'predictor': claim.predictor,
                        'timestamp': claim.timestamp,
                        'expires': claim.expires,
                    },
                    'engine_wired': engine_wired,
                    'prediction_subscribed': prediction_subscribed
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to claim stream'})

        @ipc_app.route('/p2p/streams/release', methods=['POST'])
        def p2p_streams_release():
            """Release claim on a stream."""
            registry = getattr(startupDag, '_stream_registry', None)

            if not registry:
                return jsonify({'success': False, 'error': 'Stream registry not initialized'}), 503

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            async def do_release():
                return await registry.release_claim(stream_id)

            import asyncio
            try:
                loop = asyncio.new_event_loop()
                success = loop.run_until_complete(do_release())
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
            finally:
                loop.close()

            # After releasing, remove StreamModel from Engine
            engine_removed = False
            if success and hasattr(startupDag, 'aiengine') and startupDag.aiengine is not None:
                try:
                    engine_removed = startupDag.aiengine.removeStreamFromClaim(stream_id)
                    if engine_removed:
                        logging.info(f"Stream {stream_id} released and removed from Engine", color='yellow')
                except Exception as e:
                    logging.warning(f"Failed to remove stream from Engine: {e}")

            return jsonify({'success': success, 'engine_removed': engine_removed})

        @ipc_app.route('/p2p/streams/my-claims')
        def p2p_streams_my_claims():
            """Get my claimed streams."""
            registry = getattr(startupDag, '_stream_registry', None)

            result = {'claims': [], 'count': 0}

            if registry and hasattr(registry, 'get_my_streams'):
                async def get_claims():
                    return await registry.get_my_streams()

                import asyncio
                try:
                    loop = asyncio.new_event_loop()
                    claims = loop.run_until_complete(get_claims())
                except Exception:
                    claims = []
                finally:
                    loop.close()

                result['claims'] = [
                    {
                        'stream_id': c.stream_id if hasattr(c, 'stream_id') else c.get('stream_id', ''),
                        'slot_index': c.slot_index if hasattr(c, 'slot_index') else c.get('slot_index', 0),
                        'predictor': c.predictor if hasattr(c, 'predictor') else c.get('predictor', ''),
                        'timestamp': c.timestamp if hasattr(c, 'timestamp') else c.get('timestamp', 0),
                        'expires': c.expires if hasattr(c, 'expires') else c.get('expires', 0),
                    }
                    for c in claims
                ]
                result['count'] = len(result['claims'])

            return jsonify(result)

        @ipc_app.route('/p2p/streams/renew-claims', methods=['POST'])
        def p2p_streams_renew_claims():
            """Manually trigger renewal of all stream claims."""
            registry = getattr(startupDag, '_stream_registry', None)
            trio_token = getattr(startupDag, '_trio_token', None)

            if not registry:
                return jsonify({'success': False, 'error': 'Stream registry not initialized'}), 503

            try:
                async def do_renew():
                    return await registry.renew_claims()

                if trio_token:
                    result = trio.from_thread.run(do_renew, trio_token=trio_token)
                else:
                    result = trio.run(do_renew)

                return jsonify({
                    'success': True,
                    'renewed': result.get('renewed', 0),
                    'failed': result.get('failed', 0),
                })
            except Exception as e:
                logging.warning(f"Claim renewal failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/streams/engine-status')
        def p2p_streams_engine_status():
            """Get Engine stream model status for dashboard with per-stream details."""
            result = {
                'active_models': 0,
                'stream_uuids': [],
                'engine_ready': False,
                'stream_details': {}  # Per-stream status
            }

            if hasattr(startupDag, 'aiengine') and startupDag.aiengine is not None:
                engine = startupDag.aiengine
                result['engine_ready'] = True
                result['active_models'] = engine.getStreamModelCount()
                result['stream_uuids'] = engine.getClaimedStreamIds()

                # Get stream names from stream registry or claims
                stream_registry = getattr(startupDag, '_stream_registry', None)
                stream_names = {}
                if stream_registry:
                    # Try to get stream names from my claims
                    my_claims = getattr(stream_registry, '_my_claims', {})
                    for stream_id, claim in my_claims.items():
                        # stream_id is the full name like "crypto|satori|BTC|USD"
                        stream_names[stream_id] = stream_id
                    # Also check registered oracle streams
                    oracle_network = getattr(startupDag, '_oracle_network', None)
                    if oracle_network:
                        my_regs = getattr(oracle_network, '_my_registrations', {})
                        for stream_id in my_regs.keys():
                            stream_names[stream_id] = stream_id

                # Get detailed status for each stream model
                for stream_uuid in result['stream_uuids']:
                    stream_model = engine.streamModels.get(stream_uuid)
                    if stream_model:
                        # Determine training status
                        thread_running = (
                            hasattr(stream_model, 'thread') and
                            stream_model.thread is not None and
                            stream_model.thread.is_alive()
                        )
                        has_stable_model = (
                            hasattr(stream_model, 'stable') and
                            stream_model.stable is not None
                        )
                        observation_count = len(stream_model.data) if hasattr(stream_model, 'data') else 0
                        is_paused = getattr(stream_model, 'paused', False)

                        # Determine readiness stage
                        # Stage 1: Claimed (has StreamModel) - always true here
                        # Stage 2: Has enough data (>= 10 observations for basic training)
                        # Stage 3: Model trained (has stable model)
                        # Stage 4: Actively predicting (thread running, not paused)
                        min_observations = 10  # Minimum for training
                        has_enough_data = observation_count >= min_observations
                        is_ready_to_predict = has_stable_model and has_enough_data and not is_paused

                        # Find stream name - first check StreamModel, then claims/registrations
                        stream_name = getattr(stream_model, 'stream_name', None)
                        if not stream_name:
                            for name, uuid_or_id in stream_names.items():
                                # Match by UUID or by name pattern
                                if stream_uuid in name or name in stream_uuid:
                                    stream_name = name
                                    break

                        # Get oracle peer IDs for this stream
                        oracle_peer_ids = []
                        oracle_network = getattr(startupDag, '_oracle_network', None)
                        my_peer_id = None
                        if hasattr(startupDag, '_p2p_peers') and startupDag._p2p_peers:
                            my_peer_id = getattr(startupDag._p2p_peers, 'peer_id', None)

                        if oracle_network and stream_name:
                            seen_peers = set()
                            # Get network-discovered oracles
                            if hasattr(oracle_network, 'get_registered_oracles'):
                                for o in oracle_network.get_registered_oracles(stream_name):
                                    if o.peer_id not in seen_peers:
                                        oracle_peer_ids.append({
                                            'peer_id': o.peer_id,
                                            'is_primary': o.is_primary,
                                            'is_me': o.peer_id == my_peer_id
                                        })
                                        seen_peers.add(o.peer_id)
                            # Include our own registration if not already in list
                            if hasattr(oracle_network, '_my_registrations'):
                                my_reg = oracle_network._my_registrations.get(stream_name)
                                if my_reg and my_peer_id and my_peer_id not in seen_peers:
                                    oracle_peer_ids.append({
                                        'peer_id': my_peer_id,
                                        'is_primary': getattr(my_reg, 'is_primary', False),
                                        'is_me': True
                                    })

                        result['stream_details'][stream_uuid] = {
                            'stream_name': stream_name,  # Full stream ID like "crypto|satori|BTC|USD"
                            'observation_count': observation_count,
                            'has_enough_data': has_enough_data,
                            'min_observations': min_observations,
                            'training_active': thread_running,
                            'model_ready': has_stable_model,
                            'paused': is_paused,
                            'ready_to_predict': is_ready_to_predict,
                            'adapter': stream_model.adapter.__name__ if hasattr(stream_model, 'adapter') and stream_model.adapter else None,
                            'pending_p2p_commits': len(getattr(stream_model, '_pending_commits', {})),
                            'oracles': oracle_peer_ids,
                        }

            return jsonify(result)

        # =====================================================================
        # Prediction Protocol IPC Endpoints
        # =====================================================================

        @ipc_app.route('/p2p/predictions/stats')
        def p2p_predictions_stats():
            """Get prediction protocol statistics."""
            prediction = getattr(startupDag, '_prediction_protocol', None)

            result = {
                'started': False,
                'subscribed_streams': 0,
                'my_predictions': 0,
                'cached_predictions': 0,
                'cached_scores': 0,
                'my_average_score': 0.0,
            }

            if prediction:
                result['started'] = True
                if hasattr(prediction, 'get_stats'):
                    stats = prediction.get_stats()
                    result['subscribed_streams'] = stats.get('subscribed_streams', 0)
                    result['my_predictions'] = stats.get('my_predictions', 0)
                    result['cached_predictions'] = stats.get('cached_predictions', 0)
                    result['cached_scores'] = stats.get('cached_scores', 0)
                    result['my_average_score'] = stats.get('my_average_score', 0.0)
                else:
                    result['my_predictions'] = len(getattr(prediction, '_my_predictions', []))
                    result['cached_scores'] = len(getattr(prediction, '_scores', []))
                    result['subscribed_streams'] = len(getattr(prediction, '_subscriptions', {}))
                    # Calculate average score if possible
                    if hasattr(prediction, 'evrmore_address') and hasattr(prediction, 'get_predictor_average_score'):
                        try:
                            result['my_average_score'] = prediction.get_predictor_average_score(prediction.evrmore_address)
                        except:
                            pass

            return jsonify(result)

        @ipc_app.route('/p2p/predictions/recent')
        def p2p_predictions_recent():
            """Get recent predictions."""
            prediction = getattr(startupDag, '_prediction_protocol', None)
            limit = request.args.get('limit', 20, type=int)

            result = {'predictions': [], 'count': 0}

            if prediction and hasattr(prediction, 'get_recent_predictions'):
                predictions = prediction.get_recent_predictions(limit=limit)
                result['predictions'] = [
                    {
                        'stream_id': p.stream_id if hasattr(p, 'stream_id') else p.get('stream_id', ''),
                        'predicted_value': p.predicted_value if hasattr(p, 'predicted_value') else p.get('predicted_value'),
                        'timestamp': p.timestamp if hasattr(p, 'timestamp') else p.get('timestamp', 0),
                        'score': getattr(p, 'score', None) or p.get('score'),
                    }
                    for p in predictions
                ]
                result['count'] = len(result['predictions'])

            return jsonify(result)

        @ipc_app.route('/p2p/predictions/my')
        def p2p_predictions_my():
            """Get my predictions (predictions I've made)."""
            prediction = getattr(startupDag, '_prediction_protocol', None)
            stream_id = request.args.get('stream_id')  # Optional filter
            limit = request.args.get('limit', 50, type=int)

            result = {'predictions': [], 'total': 0}

            if prediction and hasattr(prediction, 'get_my_predictions'):
                try:
                    predictions = prediction.get_my_predictions(stream_id)
                    # Limit results
                    if limit and len(predictions) > limit:
                        predictions = predictions[-limit:]
                    result['predictions'] = [
                        {
                            'hash': getattr(p, 'hash', ''),
                            'stream_id': getattr(p, 'stream_id', ''),
                            'value': getattr(p, 'value', None),
                            'target_time': getattr(p, 'target_time', 0),
                            'predictor': getattr(p, 'predictor', ''),
                            'created_at': getattr(p, 'created_at', 0),
                            'confidence': getattr(p, 'confidence', 0.5),
                        }
                        for p in predictions
                    ]
                    # Sort by created_at descending
                    result['predictions'].sort(key=lambda x: x.get('created_at', 0), reverse=True)
                    result['total'] = len(result['predictions'])
                except Exception as e:
                    logging.warning(f"Failed to get my predictions: {e}")

            return jsonify(result)

        # ========== GOVERNANCE PROTOCOL IPC ENDPOINTS ==========

        @ipc_app.route('/p2p/governance/status')
        def ipc_governance_status():
            """Get governance protocol status and statistics."""
            governance = getattr(startupDag, '_governance', None)

            result = {
                'started': governance is not None,
                'total_proposals': 0,
                'active_proposals': 0,
                'passed_proposals': 0,
                'rejected_proposals': 0,
                'my_stake': 0,
                'my_voting_power': 0,
                'can_propose': False,
                'can_vote': False,
            }

            if governance and hasattr(governance, 'get_stats'):
                result.update(governance.get_stats())

            return jsonify(result)

        @ipc_app.route('/p2p/governance/proposals')
        def ipc_governance_proposals():
            """Get all governance proposals."""
            governance = getattr(startupDag, '_governance', None)

            result = {'proposals': [], 'active': []}

            if governance:
                if hasattr(governance, 'get_all_proposals'):
                    proposals = governance.get_all_proposals()
                    result['proposals'] = [p.to_dict() for p in proposals.values()]

                if hasattr(governance, 'get_active_proposals'):
                    active = governance.get_active_proposals()
                    result['active'] = [p.to_dict() for p in active]

            return jsonify(result)

        @ipc_app.route('/p2p/governance/proposal/<proposal_id>')
        def ipc_governance_proposal(proposal_id):
            """Get a specific proposal with tally and user's vote."""
            governance = getattr(startupDag, '_governance', None)

            result = {'proposal': None, 'tally': None, 'my_vote': None}

            if governance:
                if hasattr(governance, 'get_proposal'):
                    proposal = governance.get_proposal(proposal_id)
                    if proposal:
                        result['proposal'] = proposal.to_dict()

                if hasattr(governance, 'get_proposal_tally'):
                    tally = governance.get_proposal_tally(proposal_id)
                    if tally:
                        result['tally'] = tally.to_dict() if hasattr(tally, 'to_dict') else tally

                if hasattr(governance, 'get_my_vote'):
                    my_vote = governance.get_my_vote(proposal_id)
                    if my_vote:
                        result['my_vote'] = my_vote.to_dict() if hasattr(my_vote, 'to_dict') else my_vote

            return jsonify(result)

        @ipc_app.route('/p2p/governance/propose', methods=['POST'])
        def ipc_governance_propose():
            """Create a new governance proposal."""
            governance = getattr(startupDag, '_governance', None)

            if not governance:
                return jsonify({'success': False, 'error': 'Governance protocol not started'}), 503

            data = request.get_json() or {}
            title = data.get('title', '')
            description = data.get('description', '')
            proposal_type = data.get('proposal_type', 'community')
            voting_period_days = data.get('voting_period_days', 7)

            if not title or not description:
                return jsonify({'success': False, 'error': 'Title and description required'}), 400

            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    proposal = loop.run_until_complete(
                        governance.create_proposal(
                            title=title,
                            description=description,
                            proposal_type=proposal_type,
                            voting_period_days=voting_period_days
                        )
                    )
                    return jsonify({
                        'success': True,
                        'proposal_id': proposal.proposal_id if hasattr(proposal, 'proposal_id') else str(proposal)
                    })
                finally:
                    loop.close()
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/governance/vote', methods=['POST'])
        def ipc_governance_vote():
            """Submit a vote on a proposal."""
            governance = getattr(startupDag, '_governance', None)

            if not governance:
                return jsonify({'success': False, 'error': 'Governance protocol not started'}), 503

            data = request.get_json() or {}
            proposal_id = data.get('proposal_id', '')
            choice = data.get('choice', '')

            if not proposal_id or not choice:
                return jsonify({'success': False, 'error': 'proposal_id and choice required'}), 400

            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(
                        governance.vote(proposal_id=proposal_id, choice=choice)
                    )
                    return jsonify({'success': success})
                finally:
                    loop.close()
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/governance/comment', methods=['POST'])
        def ipc_governance_comment():
            """Add a comment to a proposal."""
            governance = getattr(startupDag, '_governance', None)

            if not governance:
                return jsonify({'success': False, 'error': 'Governance protocol not started'}), 503

            data = request.get_json() or {}
            proposal_id = data.get('proposal_id', '')
            content = data.get('content', '')

            if not proposal_id or not content:
                return jsonify({'success': False, 'error': 'proposal_id and content required'}), 400

            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    comment = loop.run_until_complete(
                        governance.add_comment(proposal_id=proposal_id, content=content)
                    )
                    return jsonify({
                        'success': True,
                        'comment_id': comment.comment_id if hasattr(comment, 'comment_id') else str(comment)
                    })
                finally:
                    loop.close()
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/governance/comments/<proposal_id>')
        def ipc_governance_comments(proposal_id):
            """Get comments for a proposal."""
            governance = getattr(startupDag, '_governance', None)

            result = {'comments': []}

            if governance and hasattr(governance, 'get_comments'):
                comments = governance.get_comments(proposal_id)
                result['comments'] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in comments]

            return jsonify(result)

        @ipc_app.route('/p2p/governance/pin', methods=['POST'])
        def ipc_governance_pin():
            """Pin or unpin a proposal (signer only)."""
            governance = getattr(startupDag, '_governance', None)

            if not governance:
                return jsonify({'success': False, 'error': 'Governance protocol not started'}), 503

            data = request.get_json() or {}
            proposal_id = data.get('proposal_id', '')
            pinned = data.get('pinned', True)

            if not proposal_id:
                return jsonify({'success': False, 'error': 'proposal_id required'}), 400

            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(
                        governance.pin_proposal(proposal_id=proposal_id, pinned=pinned)
                    )
                    return jsonify({'success': success})
                finally:
                    loop.close()
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/governance/emergency-cancel', methods=['POST'])
        def ipc_governance_emergency_cancel():
            """Emergency cancel a proposal (signer only)."""
            governance = getattr(startupDag, '_governance', None)

            if not governance:
                return jsonify({'success': False, 'error': 'Governance protocol not started'}), 503

            data = request.get_json() or {}
            proposal_id = data.get('proposal_id', '')

            if not proposal_id:
                return jsonify({'success': False, 'error': 'proposal_id required'}), 400

            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(
                        governance.emergency_cancel_vote(proposal_id=proposal_id)
                    )
                    return jsonify({'success': success})
                finally:
                    loop.close()
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/governance/execute', methods=['POST'])
        def ipc_governance_execute():
            """Mark a proposal as executed (signer only)."""
            governance = getattr(startupDag, '_governance', None)

            if not governance:
                return jsonify({'success': False, 'error': 'Governance protocol not started'}), 503

            data = request.get_json() or {}
            proposal_id = data.get('proposal_id', '')

            if not proposal_id:
                return jsonify({'success': False, 'error': 'proposal_id required'}), 400

            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if hasattr(governance, 'mark_executed'):
                        success = loop.run_until_complete(
                            governance.mark_executed(proposal_id=proposal_id)
                        )
                        return jsonify({'success': success})
                    else:
                        return jsonify({'success': False, 'error': 'mark_executed not available'}), 501
                finally:
                    loop.close()
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @ipc_app.route('/p2p/governance/pinned')
        def ipc_governance_pinned():
            """Get pinned proposals."""
            governance = getattr(startupDag, '_governance', None)

            result = {'pinned': []}

            if governance and hasattr(governance, 'get_pinned_proposals'):
                pinned = governance.get_pinned_proposals()
                result['pinned'] = [p.to_dict() for p in pinned]

            return jsonify(result)

        @ipc_app.route('/p2p/governance/voting-power')
        def ipc_governance_voting_power():
            """Get detailed voting power breakdown for the current user."""
            try:
                from satorip2p.protocol.governance import (
                    STAKE_WEIGHT, UPTIME_BONUS_90_DAYS, SIGNER_BONUS
                )
                from satorip2p.protocol.signer import is_authorized_signer
            except ImportError:
                STAKE_WEIGHT = 1.0
                UPTIME_BONUS_90_DAYS = 0.10
                SIGNER_BONUS = 0.25
                def is_authorized_signer(addr): return False

            result = {
                'success': True,
                'base_stake': 0.0,
                'stake_weight': STAKE_WEIGHT,
                'uptime_days': 0,
                'uptime_bonus_pct': 0,
                'is_signer': False,
                'signer_bonus_pct': 0,
                'total_voting_power': 0.0,
                'network_total_power': 0.0,
                'active_voters': 0,
            }

            evrmore_address = ""
            identity_bridge = getattr(startupDag, '_identity_bridge', None)
            if identity_bridge:
                evrmore_address = getattr(identity_bridge, 'evrmore_address', '') or ""

            # Get base stake (default 0 if no balance found)
            base_stake = 0.0
            identity = getattr(startupDag, 'identity', None)
            if identity:
                try:
                    balances = identity.getBalances()
                    if balances and 'SATORI' in balances:
                        base_stake = float(balances['SATORI'])
                except Exception:
                    pass
            result['base_stake'] = base_stake

            # Get uptime days
            uptime_days = 0
            uptime_tracker = getattr(startupDag, '_uptime_tracker', None)
            if uptime_tracker and hasattr(uptime_tracker, 'get_uptime_streak_days'):
                uptime_days = uptime_tracker.get_uptime_streak_days(evrmore_address)
            result['uptime_days'] = uptime_days
            result['uptime_bonus_pct'] = int(UPTIME_BONUS_90_DAYS * 100) if uptime_days >= 90 else 0

            # Check if signer
            is_signer_flag = is_authorized_signer(evrmore_address)
            result['is_signer'] = is_signer_flag
            result['signer_bonus_pct'] = int(SIGNER_BONUS * 100) if is_signer_flag else 0

            # Calculate total voting power
            power = base_stake * STAKE_WEIGHT
            if uptime_days >= 90:
                power *= (1 + UPTIME_BONUS_90_DAYS)
            if is_signer_flag:
                power *= (1 + SIGNER_BONUS)
            result['total_voting_power'] = power

            # Get network total from governance
            governance = getattr(startupDag, '_governance', None)
            if governance and hasattr(governance, '_get_total_voting_power'):
                result['network_total_power'] = governance._get_total_voting_power()
            if uptime_tracker and hasattr(uptime_tracker, 'get_active_node_count'):
                result['active_voters'] = uptime_tracker.get_active_node_count()

            return jsonify(result)

        def run_ipc_server():
            from waitress import serve
            logging.info(f"Starting P2P IPC API on 127.0.0.1:{port}", color="cyan")
            serve(ipc_app, host='127.0.0.1', port=port, threads=2, _quiet=True)

        ipc_thread = threading.Thread(target=run_ipc_server, daemon=True)
        ipc_thread.start()
        logging.info(f"P2P IPC API started on 127.0.0.1:{port}", color="green")
        return ipc_thread

    except Exception as e:
        logging.warning(f"Failed to start P2P IPC API: {e}")
        return None


def startWebUI(startupDag: StartupDag, host: str = '0.0.0.0', port: int = 24601):
    """Start the Flask web UI in a background thread with WebSocket support."""
    try:
        from web.app import create_app, get_socketio, sendToUI
        from web.routes import set_vault, set_startup

        app = create_app()
        socketio = get_socketio()

        # Connect vault and startup to web routes
        set_vault(startupDag.walletManager)
        set_startup(startupDag)  # Set startup immediately - initialization is complete

        # Start P2P IPC API for web worker to access P2P state
        startP2PInternalAPI(startupDag, port=24602)

        # Start P2P WebSocket bridge for real-time UI updates
        try:
            from web.p2p_bridge import start_bridge
            start_bridge(
                peers=getattr(startupDag, '_p2p_peers', None),
                prediction_protocol=getattr(startupDag, '_prediction_protocol', None),
                oracle_network=getattr(startupDag, '_oracle_network', None),
                consensus_manager=getattr(startupDag, '_consensus_manager', None),
                uptime_tracker=getattr(startupDag, '_uptime_tracker', None),
                lending_manager=getattr(startupDag, '_lending_manager', None),
                delegation_manager=getattr(startupDag, '_delegation_manager', None),
            )
        except ImportError:
            logging.debug("P2P WebSocket bridge not available")
        except Exception as e:
            logging.warning(f"Failed to start P2P WebSocket bridge: {e}")

        def run_flask():
            import subprocess
            import sys

            # Use gunicorn for production-ready WSGI serving with WebSocket support
            # The gunicorn subprocess can't access P2P state directly, but it can
            # call the P2P IPC API on 127.0.0.1:24602 for P2P operations.
            cmd = [
                sys.executable, '-m', 'gunicorn',
                '--bind', f'{host}:{port}',
                '--workers', '1',
                '--threads', '4',
                '--worker-class', 'gthread',
                '--timeout', '120',
                '--log-level', 'warning',
                '--no-sendfile',  # Disable sendfile - incompatible with gthread worker
                'web.wsgi:app'
            ]
            logging.info(f"Starting Web UI with gunicorn at http://{host}:{port}", color="cyan")
            subprocess.run(cmd, check=True)

        web_thread = threading.Thread(target=run_flask, daemon=True)
        web_thread.start()
        logging.info(f"Web UI started at http://{host}:{port} (WebSocket enabled)", color="green")
        return web_thread
    except ImportError as e:
        logging.warning(f"Web UI not available: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to start Web UI: {e}")
        return None


if __name__ == "__main__":
    logging.info("Starting Satori Neuron", color="green")

    # Web UI will be started after initialization completes
    # (called from start() or startWorker() methods after reward address sync)
    startup = StartupDag.create(env=os.environ.get('SATORI_ENV', 'prod'), runMode='worker')

    threading.Event().wait()
