from typing import Union
import os
import time
import json
import threading
from satorilib.concepts.structs import StreamId, Stream
from satorilib.concepts import constants
from satorilib.wallet import EvrmoreWallet
from satorilib.wallet.evrmore.identity import EvrmoreIdentity
from satorilib.server.api import CheckinDetails
from satorineuron import VERSION
from satorineuron import logging
from satorineuron import config
from satorineuron.init.tag import LatestTag, Version
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
    """Get the current networking mode from environment or config."""
    mode = os.environ.get('SATORI_NETWORKING_MODE')
    if mode is None:
        try:
            mode = config.get().get('networking mode', 'central')
        except Exception:
            mode = 'central'
    return mode.lower().strip()


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
            IDENTIFY_TOPIC, IDENTIFY_REQUEST_TOPIC, IDENTITY_CACHE_TTL,
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
            'IDENTITY_CACHE_TTL': IDENTITY_CACHE_TTL,
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


def getStart():
    """returns StartupDag singleton"""
    return StartupDag()


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class StartupDag(StartupDagStruct, metaclass=SingletonMeta):
    """a DAG of startup tasks."""

    _holdingBalanceBase_cache = None
    _holdingBalanceBase_timestamp = 0

    @classmethod
    def create(
        cls,
        *args,
        env: str = 'dev',
        runMode: str = None,
        urlServer: str = None,
        urlMundo: str = None,
        isDebug: bool = False,
    ) -> 'StartupDag':
        '''Factory method to create and initialize StartupDag'''
        startupDag = cls(
            *args,
            env=env,
            runMode=runMode,
            urlServer=urlServer,
            urlMundo=urlMundo,
            isDebug=isDebug)
        startupDag.startFunction()
        return startupDag

    def __init__(
        self,
        *args,
        env: str = 'dev',
        runMode: str = None,
        urlServer: str = None,
        urlMundo: str = None,
        isDebug: bool = False,
    ):
        super(StartupDag, self).__init__(*args)
        self.needsRestart: Union[str, None] = None
        self.version = Version(VERSION)
        self.env = env
        self.runMode = RunMode.choose(runMode or config.get().get('mode', None))
        # logging.debug(f'mode: {self.runMode.name}', print=True)
        # Read UI port from environment and save to config
        self.uiPort = int(os.environ.get('SATORI_UI_PORT', '24601'))
        config.add(data={'uiport': self.uiPort})
        self.userInteraction = time.time()
        self.walletManager: WalletManager
        self.isDebug: bool = isDebug
        self.urlServer: str = urlServer
        self.urlMundo: str = urlMundo
        self.paused: bool = False
        self.pauseThread: Union[threading.Thread, None] = None
        self.balances: dict = {}
        # Central-lite only needs basic fields - no subscriptions/publications/keys
        self.aiengine: Union[Engine, None] = None
        self.publications: list[Stream] = []  # Keep for engine
        self.subscriptions: list[Stream] = []  # Keep for engine
        self.identity: EvrmoreIdentity = EvrmoreIdentity(config.walletPath('wallet.yaml'))
        self.latestObservationTime: float = 0
        self.configRewardAddress: str = None
        self.setRewardAddress()
        self.setupWalletManager()
        # Disabled: checkinCheck thread makes unnecessary health check requests every 6 hours
        # self.checkinCheckThread = threading.Thread(
        #     target=self.checkinCheck,
        #     daemon=True)
        # self.checkinCheckThread.start()
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

    def watchForVersionUpdates(self):
        """
        if we notice the code version has updated, download code restart
        in order to restart we have to kill the main thread manually.
        """

        def getPidByName(name: str) -> Union[int, None]:
            import psutil
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if name in " ".join(proc.info["cmdline"]):
                        return proc.info["pid"]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return None

        def terminatePid(pid: int):
            import signal
            os.kill(pid, signal.SIGTERM)

        def watchForever():
            latestTag = LatestTag(self.version, serverURL=self.urlServer)
            while True:
                time.sleep(60 * 60 * 24)
                if latestTag.mustUpdate():
                    terminatePid(getPidByName("satori.py"))

        self.watchVersionThread = threading.Thread(
            target=watchForever,
            daemon=True)
        self.watchVersionThread.start()

    def pollObservationsForever(self):
        """
        Poll for new observations - P2P first, central fallback.
        In P2P/hybrid mode, also subscribes to real-time P2P observations.
        Polling every 11 hours as backup for missed real-time events.
        """
        import pandas as pd

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
            """Subscribe to real-time P2P observations if available."""
            if hasattr(self, '_oracle_network') and self._oracle_network:
                try:
                    # Subscribe to observation events
                    if hasattr(self._oracle_network, 'subscribe'):
                        self._oracle_network.subscribe(
                            callback=lambda obs: processObservation(obs)
                        )
                        logging.info("Subscribed to P2P observations", color='green')
                except Exception as e:
                    logging.warning(f"Failed to subscribe to P2P observations: {e}", color='yellow')

        def pollForever():
            # Subscribe to real-time P2P observations
            subscribeToP2PObservations()

            while True:
                time.sleep(60 * 60 * 11)  # 11 hours backup poll
                try:
                    if not hasattr(self, 'server') or self.server is None:
                        logging.warning("Server not initialized, skipping observation poll", color='yellow')
                        continue

                    if not hasattr(self, 'aiengine') or self.aiengine is None:
                        logging.warning("AI Engine not initialized, skipping observation poll", color='yellow')
                        continue

                    # Get latest observation - P2P first, central fallback
                    observation = getObservationP2PFirst()

                    if observation is None:
                        logging.info("No new observations available", color='blue')
                        continue

                    processObservation(observation)

                except Exception as e:
                    logging.error(f"Error polling observations: {e}", color='red')

        self.pollObservationsThread = threading.Thread(
            target=pollForever,
            daemon=True)
        self.pollObservationsThread.start()

    def delayedEngine(self):
        time.sleep(60 * 60 * 6)
        self.buildEngine()

    def checkinCheck(self):
        while True:
            time.sleep(60 * 60 * 6)
            current_time = time.time()
            if self.latestObservationTime and (current_time - self.latestObservationTime > 60*60*6):
                logging.warning("No observations in 6 hours, restarting")
                self.triggerRestart()
            if hasattr(self, 'server') and hasattr(self.server, 'checkinCheck') and self.server.checkinCheck():
                logging.warning("Server check failed, restarting")
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
            self.announceToNetwork()  # P2P announcement (hybrid/p2p modes)
            self.initializeP2PComponents()  # Initialize consensus, rewards, etc.
            logging.info("in WALLETONLYMODE")
            return
        self.setMiningMode()
        self.createServerConn()
        if networking_mode != 'p2p':
            self.authWithCentral()  # Skip in pure P2P mode
        self.announceToNetwork()  # P2P announcement (hybrid/p2p modes)
        self.initializeP2PComponents()  # Initialize consensus, rewards, etc.
        self.setupDefaultStream()
        self.spawnEngine()

    def startWalletOnly(self):
        """start the satori engine."""
        logging.info("running in walletOnly mode", color="blue")
        self.createServerConn()
        return

    def startWorker(self):
        """start the satori engine."""
        logging.info("running in worker mode", color="blue")
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
        self.authWithCentral()
        self.announceToNetwork()  # P2P announcement (hybrid/p2p modes)
        self.initializeP2PComponents()  # Initialize consensus, rewards, etc.
        self.setupDefaultStream()
        self.spawnEngine()
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
        self.server = SatoriServerClient(
            self.wallet, url=self.urlServer, sendingUrl=self.urlMundo
        )

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

        Args:
            capabilities: List of capabilities (e.g., ["predictor", "oracle"])
        """
        import asyncio

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

        async def _announce():
            try:
                from satorip2p.protocol.peer_registry import PeerRegistry
                from satorip2p import Peers

                # Get or create P2P peers instance
                if not hasattr(self, '_p2p_peers') or self._p2p_peers is None:
                    self._p2p_peers = Peers(
                        identity=self.identity,
                        listen_port=config.get().get('p2p port', 24600),
                    )
                    await self._p2p_peers.start()

                # Create registry and announce
                if not hasattr(self, '_peer_registry') or self._peer_registry is None:
                    self._peer_registry = PeerRegistry(self._p2p_peers)
                    await self._peer_registry.start()

                # Announce with capabilities
                caps = capabilities or ["predictor"]
                announcement = await self._peer_registry.announce(capabilities=caps)

                if announcement:
                    logging.info(
                        f"Announced to P2P network: {announcement.evrmore_address[:16]}...",
                        color="cyan"
                    )
                else:
                    logging.warning("Failed to announce to P2P network")

            except ImportError:
                logging.debug("satorip2p not available, skipping P2P announcement")
            except Exception as e:
                logging.warning(f"P2P announcement failed: {e}")

        # Run async announcement
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_announce())
            else:
                loop.run_until_complete(_announce())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(_announce())

    def initializeP2PComponents(self):
        """
        Initialize all P2P components for consensus, rewards, and distribution.

        Called after announceToNetwork() has created the _p2p_peers instance.
        Only runs in hybrid/p2p mode.

        Components initialized:
        - _uptime_tracker: Heartbeat-based uptime tracking for relay bonus
        - _reward_calculator: Local reward calculation and tracking
        - _consensus_manager: Stake-weighted voting coordination
        - _distribution_trigger: Automatic distribution on consensus
        - _signer_node: Multi-sig signing (if this node is an authorized signer)
        """
        import asyncio

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

        # Need P2P peers to be initialized first
        if not hasattr(self, '_p2p_peers') or self._p2p_peers is None:
            logging.debug("P2P peers not available, skipping component initialization")
            return

        async def _initialize():
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
                        await self._uptime_tracker.start()
                        logging.info("P2P uptime tracker initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize uptime tracker: {e}")

                # 2. Initialize Reward Calculator
                if not hasattr(self, '_reward_calculator') or self._reward_calculator is None:
                    try:
                        self._reward_calculator = RewardCalculator()
                        logging.info("P2P reward calculator initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize reward calculator: {e}")

                # 3. Initialize Consensus Manager
                if not hasattr(self, '_consensus_manager') or self._consensus_manager is None:
                    try:
                        self._consensus_manager = ConsensusManager(
                            peers=self._p2p_peers,
                            wallet=self.identity,
                        )
                        await self._consensus_manager.start()
                        logging.info("P2P consensus manager initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize consensus manager: {e}")

                # 4. Initialize Distribution Trigger
                if not hasattr(self, '_distribution_trigger') or self._distribution_trigger is None:
                    try:
                        self._distribution_trigger = DistributionTrigger(
                            peers=self._p2p_peers,
                            wallet=self.identity,
                            consensus_manager=getattr(self, '_consensus_manager', None),
                        )
                        await self._distribution_trigger.start()
                        logging.info("P2P distribution trigger initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize distribution trigger: {e}")

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
                        logging.warning(f"Failed to initialize signer node: {e}")

                # 6. Initialize Oracle Network (for P2P observations)
                if not hasattr(self, '_oracle_network') or self._oracle_network is None:
                    try:
                        from satorip2p.protocol.oracle_network import OracleNetwork
                        self._oracle_network = OracleNetwork(self._p2p_peers)
                        await self._oracle_network.start()
                        logging.info("P2P oracle network initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize oracle network: {e}")

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
                        logging.warning(f"Failed to initialize lending manager: {e}")

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
                        logging.warning(f"Failed to initialize delegation manager: {e}")

                # 9. Initialize Stream Registry (for P2P stream discovery)
                if not hasattr(self, '_stream_registry') or self._stream_registry is None:
                    try:
                        from satorip2p.protocol.stream_registry import StreamRegistry
                        self._stream_registry = StreamRegistry(self._p2p_peers)
                        await self._stream_registry.start()
                        logging.info("P2P stream registry initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize stream registry: {e}")

                # 10. Initialize Prediction Protocol (for P2P commit-reveal predictions)
                if not hasattr(self, '_prediction_protocol') or self._prediction_protocol is None:
                    try:
                        from satorip2p.protocol.prediction_protocol import PredictionProtocol
                        self._prediction_protocol = PredictionProtocol(self._p2p_peers)
                        await self._prediction_protocol.start()
                        logging.info("P2P prediction protocol initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize prediction protocol: {e}")

                # 11. Initialize Treasury Alert Manager
                if not hasattr(self, '_alert_manager') or self._alert_manager is None:
                    try:
                        from satorip2p.protocol.alerts import TreasuryAlertManager
                        self._alert_manager = TreasuryAlertManager(
                            peers=self._p2p_peers,
                            wallet=self.identity,
                        )
                        await self._alert_manager.start()
                        logging.info("P2P treasury alert manager initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize alert manager: {e}")

                # 12. Initialize Deferred Rewards Manager
                if not hasattr(self, '_deferred_rewards_manager') or self._deferred_rewards_manager is None:
                    try:
                        from satorip2p.protocol.deferred_rewards import DeferredRewardsManager
                        self._deferred_rewards_manager = DeferredRewardsManager()
                        logging.info("P2P deferred rewards manager initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize deferred rewards manager: {e}")

                # 13. Initialize Donation Manager
                if not hasattr(self, '_donation_manager') or self._donation_manager is None:
                    try:
                        from satorip2p.protocol.donation import DonationManager
                        self._donation_manager = DonationManager(
                            peers=self._p2p_peers,
                            wallet=self.identity,
                        )
                        await self._donation_manager.start()
                        logging.info("P2P donation manager initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize donation manager: {e}")

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
                        logging.warning(f"Failed to initialize version tracker: {e}")

                # 15. Initialize Storage Manager (redundant storage for deferred rewards & alerts)
                if not hasattr(self, '_storage_manager') or self._storage_manager is None:
                    try:
                        StorageManager = get_p2p_module('StorageManager')
                        DeferredRewardsStorage = get_p2p_module('DeferredRewardsStorage')
                        AlertStorage = get_p2p_module('AlertStorage')
                        if StorageManager:
                            import os
                            storage_dir = os.path.expanduser('~/.satori/storage')
                            self._storage_manager = StorageManager(storage_dir=storage_dir)
                            # Create specialized storage for deferred rewards and alerts
                            if DeferredRewardsStorage:
                                self._deferred_rewards_storage = DeferredRewardsStorage(
                                    storage_dir=storage_dir,
                                    dht_client=getattr(self._p2p_peers, 'dht', None) if self._p2p_peers else None,
                                )
                            if AlertStorage:
                                self._alert_storage = AlertStorage(
                                    storage_dir=storage_dir,
                                    dht_client=getattr(self._p2p_peers, 'dht', None) if self._p2p_peers else None,
                                )
                            logging.info("P2P storage manager initialized (redundant storage enabled)", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize storage manager: {e}")

                # 16. Initialize Bandwidth Tracker & QoS Manager
                if not hasattr(self, '_bandwidth_tracker') or self._bandwidth_tracker is None:
                    try:
                        BandwidthTracker = get_p2p_module('BandwidthTracker')
                        create_qos_manager = get_p2p_module('create_qos_manager')
                        if BandwidthTracker:
                            self._bandwidth_tracker = BandwidthTracker()
                            if create_qos_manager:
                                self._qos_manager = create_qos_manager(self._bandwidth_tracker)
                            logging.info("P2P bandwidth tracker and QoS manager initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize bandwidth tracker: {e}")

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
                        logging.warning(f"Failed to initialize referral manager: {e}")

                # 18. Initialize Pricing Provider
                if not hasattr(self, '_price_provider') or self._price_provider is None:
                    try:
                        get_price_provider = get_p2p_module('get_price_provider')
                        if get_price_provider:
                            self._price_provider = get_price_provider()
                            logging.info("P2P price provider initialized", color="cyan")
                    except Exception as e:
                        logging.warning(f"Failed to initialize price provider: {e}")

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
                        logging.warning(f"Failed to initialize reward address manager: {e}")

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
                        # Wire new protocol feature managers
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
                        logging.warning(f"Failed to wire P2P managers to server: {e}")

                logging.info("P2P components initialization complete", color="cyan")

                # 15. Initialize StreamManager (oracle data streams)
                try:
                    from streams_lite import StreamManager
                    if not hasattr(self, '_stream_manager') or self._stream_manager is None:
                        self._stream_manager = StreamManager(
                            peers=getattr(self, '_p2p_peers', None),
                            identity=getattr(self, 'identity', None),
                            send_to_ui=getattr(self, 'sendToUI', None),
                        )
                        await self._stream_manager.start()
                        logging.info(f"StreamManager initialized ({self._stream_manager.oracle_count} oracles)", color="cyan")
                except ImportError:
                    logging.debug("streams-lite not available, oracle streams disabled")
                except Exception as e:
                    logging.warning(f"StreamManager initialization failed: {e}")

            except ImportError as e:
                logging.debug(f"satorip2p not available: {e}")
            except Exception as e:
                logging.warning(f"P2P component initialization failed: {e}")

        # Run async initialization
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_initialize())
            else:
                loop.run_until_complete(_initialize())
        except RuntimeError:
            asyncio.run(_initialize())

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

                if hasattr(self, '_oracle_network') and self._oracle_network is not None:
                    observation = await self._oracle_network.publish_observation(
                        stream_id=stream_id,
                        value=value,
                        timestamp=timestamp
                    )
                    if observation:
                        logging.debug(f"Published observation to P2P: {stream_id[:16]}... = {value}")
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
        if EvrmoreWallet.addressIsValid(address):
            self.configRewardAddress = address
            config.add(data={'reward address': address})
            if not globally:
                return True
        else:
            self.configRewardAddress: str = str(config.get().get('reward address', ''))
        if (
            globally and
            self.env in ['prod', 'local', 'testprod'] and
            EvrmoreWallet.addressIsValid(self.configRewardAddress)
        ):
            self.server.setRewardAddress(address=self.configRewardAddress)
            return True
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

    # def findMatchingPubSubStream(self, uuid: str, sub: bool = True) -> Stream:
    #         if sub:
    #             for sub in self.subscriptions:
    #                 if sub.streamId.uuid == uuid:
    #                     return sub
    #         else:
    #             for pub in self.publications:
    #                 if pub.streamId.uuid == uuid:
    #                     return pub

    def pause(self, timeout: int = 60):
        """pause the engine."""
        self.paused = True
        self.pauseTimer = threading.Timer(timeout, self.unpause)
        self.pauseTimer.daemon = True
        self.pauseTimer.start()
        logging.info("AI engine paused", color="green")

    def unpause(self):
        """unpause the engine."""
        self.paused = False
        if hasattr(self, 'pauseTimer') and self.pauseTimer is not None:
            self.pauseTimer.cancel()
        self.pauseTimer = None
        logging.info("AI engine unpaused", color="green")

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


def startWebUI(startupDag: StartupDag, host: str = '0.0.0.0', port: int = 24601):
    """Start the Flask web UI in a background thread with WebSocket support."""
    try:
        from web.app import create_app, get_socketio, sendToUI
        from web.routes import set_vault

        app = create_app()
        socketio = get_socketio()

        # Connect vault to web routes
        set_vault(startupDag.walletManager)

        # Wire P2P event callbacks to WebSocket emission
        _wire_p2p_events_to_websocket(startupDag, sendToUI)

        def run_flask():
            # Suppress Flask/werkzeug logging
            import logging as stdlib_logging
            werkzeug_logger = stdlib_logging.getLogger('werkzeug')
            werkzeug_logger.setLevel(stdlib_logging.ERROR)

            # Use SocketIO server if available (for real-time updates)
            if socketio:
                socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
            else:
                app.run(host=host, port=port, debug=False, use_reloader=False)

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


def _wire_p2p_events_to_websocket(startupDag: StartupDag, sendToUI):
    """
    Wire P2P component callbacks to WebSocket event emission.

    This enables real-time updates in the web dashboard for:
    - Heartbeats from network peers
    - Predictions published/received
    - Observations from oracles
    - Consensus phase changes
    - Pool/lending updates
    - Delegation changes
    """
    import asyncio

    # Heartbeat callback
    def on_heartbeat(heartbeat):
        """Called when a heartbeat is received from the network."""
        try:
            sendToUI('heartbeat', {
                'node_id': heartbeat.node_id[:16] + '...' if len(heartbeat.node_id) > 16 else heartbeat.node_id,
                'address': getattr(heartbeat, 'evrmore_address', '')[:12] + '...',
                'timestamp': heartbeat.timestamp,
                'roles': getattr(heartbeat, 'roles', []),
            })
        except Exception as e:
            logging.debug(f"Failed to emit heartbeat: {e}")

    # Prediction callback
    def on_prediction(prediction):
        """Called when a prediction is received."""
        try:
            sendToUI('prediction', {
                'stream_id': prediction.stream_id[:16] + '...' if len(prediction.stream_id) > 16 else prediction.stream_id,
                'predictor': getattr(prediction, 'predictor', '')[:12] + '...',
                'value': getattr(prediction, 'value', None),
                'timestamp': getattr(prediction, 'timestamp', None),
                'target_time': getattr(prediction, 'target_time', None),
            })
        except Exception as e:
            logging.debug(f"Failed to emit prediction: {e}")

    # Observation callback
    def on_observation(observation):
        """Called when an observation is received from an oracle."""
        try:
            sendToUI('observation', {
                'stream_id': observation.stream_id[:16] + '...' if len(observation.stream_id) > 16 else observation.stream_id,
                'oracle': getattr(observation, 'oracle', '')[:12] + '...',
                'value': getattr(observation, 'value', None),
                'timestamp': getattr(observation, 'timestamp', None),
                'hash': getattr(observation, 'hash', None),
            })
        except Exception as e:
            logging.debug(f"Failed to emit observation: {e}")

    # Consensus callback
    def on_consensus_change(phase, round_id=None):
        """Called when consensus phase changes."""
        try:
            sendToUI('consensus', {
                'phase': str(phase),
                'round_id': round_id,
                'timestamp': int(time.time()),
            })
        except Exception as e:
            logging.debug(f"Failed to emit consensus: {e}")

    # Pool/lending callback
    def on_pool_update(update_type, data):
        """Called when pool/lending status changes."""
        try:
            sendToUI('pool_update', {
                'type': update_type,
                'data': data,
                'timestamp': int(time.time()),
            })
        except Exception as e:
            logging.debug(f"Failed to emit pool update: {e}")

    # Delegation callback
    def on_delegation_update(update_type, data):
        """Called when delegation status changes."""
        try:
            sendToUI('delegation_update', {
                'type': update_type,
                'data': data,
                'timestamp': int(time.time()),
            })
        except Exception as e:
            logging.debug(f"Failed to emit delegation update: {e}")

    # Wire callbacks to P2P components
    async def _wire_callbacks():
        try:
            # Wire uptime tracker for heartbeats
            if hasattr(startupDag, '_uptime_tracker') and startupDag._uptime_tracker:
                tracker = startupDag._uptime_tracker
                if hasattr(tracker, 'on_heartbeat_received'):
                    tracker.on_heartbeat_received = on_heartbeat
                elif hasattr(tracker, 'set_callback'):
                    tracker.set_callback('heartbeat', on_heartbeat)

            # Wire prediction protocol
            if hasattr(startupDag, '_prediction_protocol') and startupDag._prediction_protocol:
                proto = startupDag._prediction_protocol
                if hasattr(proto, 'on_prediction_received'):
                    proto.on_prediction_received = on_prediction
                elif hasattr(proto, 'set_callback'):
                    proto.set_callback('prediction', on_prediction)

            # Wire oracle network
            if hasattr(startupDag, '_oracle_network') and startupDag._oracle_network:
                oracle = startupDag._oracle_network
                if hasattr(oracle, 'on_observation_received'):
                    oracle.on_observation_received = on_observation
                elif hasattr(oracle, 'set_callback'):
                    oracle.set_callback('observation', on_observation)

            # Wire consensus manager
            if hasattr(startupDag, '_consensus_manager') and startupDag._consensus_manager:
                consensus = startupDag._consensus_manager
                if hasattr(consensus, 'on_phase_change'):
                    consensus.on_phase_change = on_consensus_change
                elif hasattr(consensus, 'set_callback'):
                    consensus.set_callback('phase_change', on_consensus_change)

            # Wire lending manager
            if hasattr(startupDag, '_lending_manager') and startupDag._lending_manager:
                lending = startupDag._lending_manager
                if hasattr(lending, 'on_update'):
                    lending.on_update = lambda data: on_pool_update('lending', data)
                elif hasattr(lending, 'set_callback'):
                    lending.set_callback('update', lambda data: on_pool_update('lending', data))

            # Wire delegation manager
            if hasattr(startupDag, '_delegation_manager') and startupDag._delegation_manager:
                delegation = startupDag._delegation_manager
                if hasattr(delegation, 'on_update'):
                    delegation.on_update = lambda data: on_delegation_update('delegation', data)
                elif hasattr(delegation, 'set_callback'):
                    delegation.set_callback('update', lambda data: on_delegation_update('delegation', data))

            logging.debug("P2P event callbacks wired to WebSocket")

        except Exception as e:
            logging.warning(f"Failed to wire P2P callbacks: {e}")

    # Run callback wiring (with delay to ensure P2P components are initialized)
    def wire_with_delay():
        time.sleep(5)  # Wait for P2P components to initialize
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_wire_callbacks())
            loop.close()
        except Exception as e:
            logging.debug(f"Async callback wiring failed: {e}")

    threading.Thread(target=wire_with_delay, daemon=True).start()


if __name__ == "__main__":
    logging.info("Starting Satori Neuron", color="green")

    # Start web UI early (before blocking server connection)
    def start_web_early():
        """Start web UI after a brief delay to let StartupDag initialize."""
        import time
        time.sleep(2)  # Wait for StartupDag to be created
        try:
            startup = getStart()
            startWebUI(startup, port=startup.uiPort)
        except Exception as e:
            logging.warning(f"Early web UI start failed: {e}")

    web_early_thread = threading.Thread(target=start_web_early, daemon=True)
    web_early_thread.start()

    startup = StartupDag.create(env=os.environ.get('SATORI_ENV', 'prod'), runMode='worker')

    threading.Event().wait()
