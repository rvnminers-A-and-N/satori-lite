from typing import Union, Optional, Callable
import os
import time
import json
import copy
import asyncio
import warnings
import threading
import hashlib
import numpy as np
import pandas as pd
from satorilib.concepts import Observation, Stream
from satorilib.logging import INFO, setup, debug, info, warning, error
from satorilib.utils.system import getProcessorCount
from satorilib.utils.time import datetimeToTimestamp, datetimeToUnixTimestamp, now
from satorilib.datamanager import DataClient, DataServerApi, DataClientApi, PeerInfo, Message, Subscription
from satorilib.wallet.evrmore.identity import EvrmoreIdentity
from satorilib.server import SatoriServerClient
# TODO: Commented out for engine-neuron integration
# from satorilib.centrifugo import (
#     create_centrifugo_client,
#     create_subscription_handler,
#     subscribe_to_stream
# )
from satoriengine.veda import config
from satoriengine.veda.data import StreamForecast, validate_single_entry
from satoriengine.veda.adapters import ModelAdapter, StarterAdapter, XgbAdapter, XgbChronosAdapter
from satoriengine.veda.storage import EngineStorageManager

warnings.filterwarnings('ignore')
setup(level=INFO)


def _get_networking_mode() -> str:
    """Get the current networking mode from environment or config."""
    mode = os.environ.get('SATORI_NETWORKING_MODE')
    if mode is None:
        try:
            # Try neuron config first (has the actual config.yaml access)
            from satorineuron import config as neuron_config
            mode = neuron_config.get().get('networking mode', 'central')
        except ImportError:
            # Fallback to engine config (which may not have get())
            try:
                mode = config.get().get('networking mode', 'central')
            except Exception:
                mode = 'central'
        except Exception:
            mode = 'central'
    return mode.lower().strip()


# P2P Module Imports (lazy-loaded for optional dependency)
def _get_p2p_modules():
    """
    Get all available P2P modules from satorip2p.
    Returns a dict of module references, or empty dict if satorip2p not installed.
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
        from satorip2p.protocol.oracle_network import OracleNetwork
        from satorip2p.protocol.prediction_protocol import PredictionProtocol
        from satorip2p.signing import (
            EvrmoreWallet as P2PEvrmoreWallet,
            sign_message,
            verify_message,
        )
        # Bandwidth and QoS modules
        try:
            from satorip2p.protocol.bandwidth import BandwidthTracker, QoSManager
            from satorip2p.protocol.versioning import VersionNegotiator, PeerVersionTracker
            from satorip2p.protocol.storage import StorageManager, RedundantStorage
        except ImportError:
            BandwidthTracker = None
            QoSManager = None
            VersionNegotiator = None
            PeerVersionTracker = None
            StorageManager = None
            RedundantStorage = None
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
            'SatoriScorer': SatoriScorer,
            'RewardCalculator': RewardCalculator,
            'RoundDataStore': RoundDataStore,
            'PredictionInput': PredictionInput,
            'ScoreBreakdown': ScoreBreakdown,
            'OracleNetwork': OracleNetwork,
            'PredictionProtocol': PredictionProtocol,
            'P2PEvrmoreWallet': P2PEvrmoreWallet,
            'sign_message': sign_message,
            'verify_message': verify_message,
            'NetworkingMode': NetworkingMode,
            'BandwidthTracker': BandwidthTracker,
            'QoSManager': QoSManager,
            'VersionNegotiator': VersionNegotiator,
            'PeerVersionTracker': PeerVersionTracker,
            'StorageManager': StorageManager,
            'RedundantStorage': RedundantStorage,
        }
    except ImportError:
        return {'available': False}


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


class Engine:

    @classmethod
    async def create(cls) -> 'Engine':
        engine = cls()
        await engine.initialize()
        return engine

    @classmethod
    def createFromNeuron(
        cls,
        subscriptions: list,
        publications: list,
        server: SatoriServerClient,
        wallet: any,
    ) -> 'Engine':
        """
        Factory method for Neuron to spawn Engine with stream assignments.

        Args:
            subscriptions: List of Stream objects to subscribe to
            publications: List of Stream objects for predictions
            server: SatoriServerClient instance from Neuron
            wallet: Wallet instance from Neuron for signing
        """
        engine = cls()
        engine.server = server
        engine.wallet = wallet
        engine.subscriptionStreams = subscriptions
        engine.publicationStreams = publications
        # Build subscription/publication dicts from Stream objects
        for sub in subscriptions:
            engine.subscriptions[sub.streamId.uuid] = PeerInfo([], [])
        for pub in publications:
            engine.publications[pub.streamId.uuid] = PeerInfo([], [])
        return engine

    def __init__(self):
        self.streamModels: dict[str, StreamModel] = {}
        self.subscriptions: dict[str, PeerInfo] = {}
        self.publications: dict[str, PeerInfo] = {}
        self.subscriptionStreams: list = []
        self.publicationStreams: list = []
        self.server: Union[SatoriServerClient, None] = None
        self.wallet: any = None
        self.dataServerIp: str = ''
        self.dataServerPort: Union[int, None] = None
        self.dataClient: Union[DataClient, None] = None
        self.paused: bool = False
        self.threads: list[threading.Thread] = []
        self.identity: EvrmoreIdentity = EvrmoreIdentity('/Satori/Neuron/wallet/wallet.yaml')
        self.transferProtocol: Union[str, None] = None
        # SQLite storage manager for local data persistence
        self.storage: EngineStorageManager = EngineStorageManager.getInstance()
        # P2P Integration
        self._p2p_peers = None
        self._oracle_network = None
        self._trio_token = None  # For trio.from_thread.run() calls
        self._prediction_protocol = None
        self._bandwidth_qos = None
        self._version_manager = None
        self._storage_redundancy = None
        self._p2p_observation_subscriptions: dict[str, bool] = {}
        # Prediction queue for batch submission (upstream feature)
        self.predictionQueue: list[dict] = []
        self.predictionQueueLock: threading.Lock = threading.Lock()
        # TODO: Commented out for engine-neuron integration
        # self.centrifugo = None
        # self.centrifugoSubscriptions: list = []



    # TODO: Commented out for engine-neuron integration
    # async def centrifugoConnect(self, centrifugoPayload: dict):
    #     """establish a centrifugo connection for subscribing"""
    #     if not centrifugoPayload:
    #         error("No centrifugo payload provided")
    #         return
    #
    #     token = centrifugoPayload.get('token')
    #     ws_url = centrifugoPayload.get('ws_url')
    #
    #     if not token or not ws_url:
    #         error("Missing token or ws_url in centrifugo payload")
    #         return
    #
    #     try:
    #         self.centrifugo = await create_centrifugo_client(
    #             ws_url=ws_url,
    #             token=token,
    #             on_connected_callback=lambda x: info("Centrifugo connected"),
    #             on_disconnected_callback=lambda x: info("Centrifugo disconnected"))
    #
    #         await self.centrifugo.connect()
    #     except Exception as e:
    #         error(f"Failed to connect to Centrifugo: {e}")
    #
    # async def handleCentrifugoMessage(self, ctx, streamUuid: str):
    #     """Handle messages from Centrifugo"""
    #     try:
    #         raw_data = ctx.pub.data
    #
    #         # Parse the JSON string to get the actual data
    #         if isinstance(raw_data, str):
    #             data = json.loads(raw_data)
    #         else:
    #             data = raw_data
    #
    #         # Format data for Observation parsing
    #         formatted_data = {
    #             "topic": json.dumps({"uuid": streamUuid}),
    #             "data": data.get('value'),
    #             "time": data.get('time'),
    #             "hash": data.get('hash')
    #         }
    #         obs = Observation.parse(json.dumps(formatted_data))
    #         streamModel = self.streamModels.get(streamUuid)
    #         if isinstance(streamModel, StreamModel) and getattr(streamModel, 'usePubSub', True):
    #             await streamModel.handleSubscriptionMessage(
    #                 "Subscription",
    #                 message=Message({
    #                     'data': obs,
    #                     'status': 'stream/observation'}),
    #                 pubSubFlag=True)
    #     except Exception as e:
    #         error(f"Error handling Centrifugo message: {e}")

    def initialize(self):
        self.connectToDataServer()
        threading.Thread(target=self.stayConnectedForever, daemon=True).start()
        self.startService()
        # Initialize P2P if in hybrid/p2p mode
        networking_mode = _get_networking_mode()
        if networking_mode in ('hybrid', 'p2p'):
            self.initializeP2P()

    def initializeP2P(self):
        """Initialize P2P oracle network and prediction protocol.

        Note: Engine should NOT create its own Peers instance or start protocols.
        P2P protocols are already started by start.py in the persistent Trio context.
        This method just logs that P2P peers have been wired.
        """
        # P2P peers must be provided externally via setP2PPeers()
        # Engine does not create its own Peers instance
        if self._p2p_peers is None:
            debug("P2P peers not set - waiting for external wiring from start.py")
            return

        # P2P protocols (OracleNetwork, PredictionProtocol, etc.) are already
        # initialized and started by start.py in the persistent Trio context.
        # Engine just receives references to the already-running protocols.
        info("Engine P2P ready (using shared Peers from start.py)", color="green")

    def setP2PPeers(self, peers) -> None:
        """Set the shared P2P Peers instance from start.py.

        Args:
            peers: The Peers instance initialized by start.py
        """
        self._p2p_peers = peers
        # Propagate to all existing streamModels
        for streamUuid, streamModel in self.streamModels.items():
            streamModel._p2p_peers = peers
        info("P2P peers wired to Engine", color="cyan")
        # Now initialize P2P protocols that depend on peers
        self.initializeP2P()

    def setPredictionProtocol(self, protocol) -> None:
        """Set the shared PredictionProtocol instance from start.py.

        Args:
            protocol: The PredictionProtocol instance initialized by start.py
        """
        self._prediction_protocol = protocol
        # Propagate to all existing streamModels (both protocol and peers for spawn_background_task)
        for streamUuid, streamModel in self.streamModels.items():
            streamModel._prediction_protocol = protocol
            if self._p2p_peers is not None:
                streamModel._p2p_peers = self._p2p_peers
            info(f"P2P prediction protocol wired to stream {streamUuid}", color="cyan")
        info("P2P prediction protocol wired to Engine", color="cyan")

    def setOracleNetwork(self, oracle_network, trio_token=None) -> None:
        """Set the shared OracleNetwork instance from start.py.

        Args:
            oracle_network: The OracleNetwork instance initialized by start.py
            trio_token: Optional trio token for async operations from sync context
        """
        self._oracle_network = oracle_network
        if trio_token:
            self._trio_token = trio_token
        # Propagate to all existing streamModels for oracle status checks
        for streamUuid, streamModel in self.streamModels.items():
            streamModel._oracle_network = oracle_network
        info("Oracle network wired to Engine for observation subscriptions", color="cyan")

    def subscribeToP2PObservations(self, stream_id: str, streamUuid: str = None) -> bool:
        """Subscribe to P2P observations for a stream.

        Args:
            stream_id: Full stream identifier like 'coingecko|SATORIUSD|price'
            streamUuid: Stream UUID hash (optional, derived from stream_id if not provided)
        """
        if self._oracle_network is None:
            return False

        # Use stream_id for subscription key
        if stream_id in self._p2p_observation_subscriptions:
            return True  # Already subscribed

        # If streamUuid not provided, use stream_id as lookup key
        lookup_uuid = streamUuid or stream_id

        try:
            import trio

            def on_p2p_observation(observation):
                """Handle P2P observation and pass to stream model."""
                # Try to find stream model by UUID or by stream_id
                self._handleP2PObservation(lookup_uuid, observation, stream_id)

            async def do_subscribe():
                return await self._oracle_network.subscribe_to_stream(
                    stream_id=stream_id,  # Use full stream_id for pubsub topic
                    callback=on_p2p_observation
                )

            # Use trio.from_thread if we have a trio token, otherwise fall back to trio.run
            if self._trio_token:
                success = trio.from_thread.run(do_subscribe, trio_token=self._trio_token)
            else:
                success = trio.run(do_subscribe)

            if success:
                self._p2p_observation_subscriptions[stream_id] = True
                info(f"Subscribed to P2P observations for {stream_id}")
                return True

        except Exception as e:
            warning(f"Failed to subscribe to P2P observations for {streamUuid}: {e}")

        return False

    def _handleP2PObservation(self, streamUuid: str, observation, stream_id: str = None):
        """Handle a P2P observation and pass it to the stream model."""
        try:
            streamModel = self.streamModels.get(streamUuid)
            # Fallback: try to find by stream_name if UUID lookup fails
            if streamModel is None and stream_id:
                for uuid, model in self.streamModels.items():
                    if getattr(model, 'stream_name', None) == stream_id:
                        streamModel = model
                        break
            if streamModel is None:
                debug(f"No stream model found for {streamUuid} / {stream_id}")
                return

            # Track bandwidth if QoS manager is available
            if self._bandwidth_qos is not None:
                try:
                    # Estimate message size (JSON overhead + data)
                    estimated_size = len(str(observation.value)) + 100
                    asyncio.run(self._bandwidth_qos.account_receive(
                        topic=f"p2p/observation/{streamUuid[:8]}",
                        byte_size=estimated_size
                    ))
                except Exception:
                    pass  # Don't fail observation handling for tracking errors

            # Convert P2P Observation to DataFrame format for onDataReceived
            data = pd.DataFrame({
                'date_time': [observation.timestamp],
                'value': [float(observation.value)],
                'id': [observation.hash]
            })

            # Pass to stream model for processing
            streamModel.onDataReceived(data)

            debug(f"P2P observation received for {streamUuid} value={observation.value}")

        except Exception as e:
            error(f"Error handling P2P observation: {e}")

    def initializeFromNeuron(self):
        """Initialize engine when spawned from Neuron (no DataServer needed)"""
        info("Engine initializing from Neuron...", color='blue')
        self.initializeModelsFromNeuron()
        # Initialize P2P if in hybrid/p2p mode
        networking_mode = _get_networking_mode()
        if networking_mode in ('hybrid', 'p2p'):
            self.initializeP2P()
            # Subscribe to P2P observations for all streams
            for streamUuid, streamModel in self.streamModels.items():
                # Get stream_name if available, otherwise use UUID
                stream_name = getattr(streamModel, 'stream_name', None) or streamUuid
                self.subscribeToP2PObservations(stream_name, streamUuid)
        info("Engine initialized successfully", color='green')

    def startService(self):
        self.getPubSubInfo()
        self.initializeModels()

    def addStream(self, stream: Stream, pubStream: Stream):
        ''' add streams to a running engine '''
        # don't duplicate effort
        if stream.streamId.uuid in [s.streamId.uuid for s in self.streams]:
            return
        self.streams.append(stream)
        self.pubstreams.append(pubStream)
        self.streamModels[stream.streamId] = StreamModel(
            streamId=stream.streamId,
            predictionStreamId=pubStream.streamId,
            predictionProduced=self.predictionProduced)
        self.streamModels[stream.streamId].chooseAdapter(inplace=True)
        # Propagate P2P protocol if already set
        if self._prediction_protocol is not None:
            self.streamModels[stream.streamId]._prediction_protocol = self._prediction_protocol
        if self._p2p_peers is not None:
            self.streamModels[stream.streamId]._p2p_peers = self._p2p_peers
        self.streamModels[stream.streamId].run_forever()

    def addStreamFromClaim(self, stream_id: str) -> bool:
        """
        Create a StreamModel for a claimed stream (P2P mode).

        Args:
            stream_id: Stream identifier like 'crypto|satori|BTC|USD'
                      Format: source|author|stream|target

        Returns:
            True if created successfully, False otherwise
        """
        try:
            # Parse stream_id to get components
            parts = stream_id.split('|')
            if len(parts) < 3:
                warning(f"Invalid stream_id format: {stream_id} (expected source|author|stream|target)")
                return False

            source = parts[0]
            author = parts[1]
            stream = parts[2]
            target = parts[3] if len(parts) > 3 else ''

            # Create StreamId to get UUID
            from satorilib.concepts.structs import StreamId as SatoriStreamId
            streamId = SatoriStreamId(source=source, author=author, stream=stream, target=target)
            streamUuid = streamId.uuid

            # Don't duplicate
            if streamUuid in self.streamModels:
                info(f"StreamModel already exists for {stream_id} (hash: {streamUuid})")
                return True

            # Create a minimal StreamModel for P2P predictions
            # In P2P mode, we receive observations via callback and make predictions
            # We use createFromServer-like initialization but without actual server
            streamModel = StreamModel.__new__(StreamModel)
            streamModel.cpu = getProcessorCount()
            streamModel.pauseAll = self.pause
            streamModel.resumeAll = self.resume
            streamModel.preferredAdapters = [XgbAdapter, StarterAdapter]
            streamModel.defaultAdapters = [XgbAdapter, XgbAdapter, StarterAdapter]
            streamModel.failedAdapters = []
            streamModel.thread = None
            streamModel.streamUuid = streamUuid
            streamModel.stream_name = stream_id  # Full stream name for logs
            streamModel.predictionStreamUuid = streamUuid  # Same for P2P
            streamModel.subscriptionStream = None  # No server subscription
            streamModel.publicationStream = None
            streamModel.server = self.server
            streamModel.wallet = self.wallet
            streamModel.rng = np.random.default_rng(37)
            streamModel.publisherHost = None
            streamModel.transferProtocol = 'p2p'
            streamModel.usePubSub = True
            streamModel.internal = False
            streamModel.useServer = False  # P2P mode
            streamModel.peerInfo = PeerInfo([], [])
            streamModel.dataClientOfIntServer = None
            streamModel.dataClientOfExtServer = None
            streamModel.identity = self.identity
            streamModel.storage = self.storage
            # P2P commit-reveal support
            streamModel._prediction_protocol = self._prediction_protocol
            streamModel._p2p_peers = self._p2p_peers
            streamModel._oracle_network = self._oracle_network
            streamModel._current_round_id = 0
            streamModel._pending_commits = {}
            streamModel.trainingDelay = streamModel._loadTrainingDelay()

            # Initialize the model
            streamModel.initializeForP2P()

            self.streamModels[streamUuid] = streamModel
            streamModel.chooseAdapter(inplace=True)
            streamModel.run_forever()

            # Subscribe to P2P observations for this stream
            # Note: subscribe_to_stream uses stream_id (like 'coingecko|SATORIUSD|price'), not UUID
            if self._oracle_network is not None:
                self.subscribeToP2PObservations(stream_id, streamUuid)
            else:
                debug(f"Oracle network not available for {stream_id} (hash: {streamUuid})")

            info(f"Created StreamModel for claimed stream: {stream_id} (hash: {streamUuid})", color='green')
            return True

        except Exception as e:
            error(f"Failed to create StreamModel for {stream_id}: {e}")
            return False

    def removeStreamFromClaim(self, stream_id: str) -> bool:
        """
        Remove a StreamModel when releasing a claim.

        Args:
            stream_id: Stream identifier like 'crypto|satori|BTC|USD'

        Returns:
            True if removed successfully, False otherwise
        """
        try:
            # Parse stream_id to get UUID
            parts = stream_id.split('|')
            if len(parts) < 3:
                warning(f"Invalid stream_id format: {stream_id}")
                return False

            source = parts[0]
            author = parts[1]
            stream = parts[2]
            target = parts[3] if len(parts) > 3 else ''

            from satorilib.concepts.structs import StreamId as SatoriStreamId
            streamId = SatoriStreamId(source=source, author=author, stream=stream, target=target)
            streamUuid = streamId.uuid

            if streamUuid not in self.streamModels:
                info(f"No StreamModel found for {stream_id} (hash: {streamUuid})")
                return True  # Already removed

            # Stop the model
            streamModel = self.streamModels[streamUuid]
            try:
                streamModel.pause()
            except Exception:
                pass

            # Remove from dict
            del self.streamModels[streamUuid]

            info(f"Removed StreamModel for released stream: {stream_id} (hash: {streamUuid})", color='yellow')
            return True

        except Exception as e:
            error(f"Failed to remove StreamModel for {stream_id}: {e}")
            return False

    def getStreamModelCount(self) -> int:
        """Get the number of active StreamModels."""
        return len(self.streamModels)

    def getClaimedStreamIds(self) -> list:
        """Get list of stream UUIDs that have active StreamModels."""
        return list(self.streamModels.keys())

    def pause(self, force: bool = False):
        if force:
            self.paused = True
        for streamModel in self.streamModels.values():
            streamModel.pause()

    def resume(self, force: bool = False):
        if force:
            self.paused = False
        if not self.paused:
            for streamModel in self.streamModels.values():
                streamModel.resume()

    @property
    def isConnectedToServer(self):
        if hasattr(self, 'dataClient') and self.dataClient is not None:
            return self.dataClient.isConnected()
        return False

    def connectToDataServer(self):
        ''' connect to server, retry if failed '''

        def authenticate() -> bool:
            response = asyncio.run(self.dataClient.authenticate(islocal='engine'))
            if response.status == DataServerApi.statusSuccess.value:
                info("Local Engine successfully connected to Server Ip at :", self.dataServerIp, color="green")
                return True
            return False

        def initiateServerConnection() -> bool:
            ''' local engine client authorization '''
            self.dataClient = DataClient(self.dataServerIp, self.dataServerPort, identity=self.identity)
            return authenticate()

        waitingPeriod = 10
        while not self.isConnectedToServer:
            try:
                self.dataServerIp = config.get().get('server ip', '0.0.0.0')
                self.dataServerPort = int(config.get().get('server port', 24600))
                if initiateServerConnection():
                    return True
            except Exception as e:
                warning(f'Failed to find a valid Server Ip, retrying in {waitingPeriod}')
                time.sleep(waitingPeriod)

    def getPubSubInfo(self):
        ''' gets the relation info between pub-sub streams '''
        waitingPeriod = 10
        while not self.subscriptions and self.isConnectedToServer:
            try:
                pubSubResponse: Message = asyncio.run(self.dataClient.getPubsubMap())
                self.transferProtocol = pubSubResponse.streamInfo.get('transferProtocol')
                transferProtocolPayload = pubSubResponse.streamInfo.get('transferProtocolPayload')
                transferProtocolKey = pubSubResponse.streamInfo.get('transferProtocolKey')
                pubSubMapping = pubSubResponse.streamInfo.get('pubSubMapping')
                if pubSubResponse.status == DataServerApi.statusSuccess.value and pubSubMapping:
                    for sub_uuid, data in pubSubMapping.items():
                        self.subscriptions[sub_uuid] = PeerInfo(data['dataStreamSubscribers'], data['dataStreamPublishers'])
                        self.publications[data['publicationUuid']] = PeerInfo(data['predictiveStreamSubscribers'], data['predictiveStreamPublishers'])
                    if self.subscriptions:
                        info(pubSubResponse.senderMsg, color='green')
                else:
                    raise Exception
                return
            except Exception:
                warning(f"Failed to fetch pub-sub info, waiting for {waitingPeriod} seconds")
                time.sleep(waitingPeriod)

    def stayConnectedForever(self):
        ''' runs in thread to maintain connection '''
        while True:
            time.sleep(5)
            self.cleanupThreads()
            if not self.isConnectedToServer:
                import sys
                sys.exit(1)

    def initializeModels(self):
        info(f'Transfer protocol : {self.transferProtocol}', color='green')
        for subUuid, pubUuid in zip(self.subscriptions.keys(), self.publications.keys()):
            peers = self.subscriptions[subUuid]
            try:
                self.streamModels[subUuid] = StreamModel.create(
                    streamUuid=subUuid,
                    predictionStreamUuid=pubUuid,
                    peerInfo=peers,
                    dataClient=self.dataClient,
                    identity=self.identity,
                    pauseAll=self.pause,
                    resumeAll=self.resume,
                    transferProtocol=self.transferProtocol)
            except Exception as e:
                error(e)
            # TODO: Commented out for engine-neuron integration
            # if self.centrifugo is not None:
            #     self.streamModels[subUuid].usePubSub = True
            self.streamModels[subUuid].chooseAdapter(inplace=True)
            # Propagate P2P protocol if already set
            if self._prediction_protocol is not None:
                self.streamModels[subUuid]._prediction_protocol = self._prediction_protocol
            if self._p2p_peers is not None:
                self.streamModels[subUuid]._p2p_peers = self._p2p_peers
            self.streamModels[subUuid].run_forever()

    def initializeModelsFromNeuron(self):
        """Initialize models when spawned from Neuron - uses server directly"""
        info("Initializing models from Neuron stream assignments...", color='green')

        # Match subscriptions to publications by predicting relationship
        for sub in self.subscriptionStreams:
            # Find matching publication (prediction stream)
            matchingPub = None
            for pub in self.publicationStreams:
                if pub.predicting == sub.streamId:
                    matchingPub = pub
                    break

            if matchingPub is None:
                warning(f"No matching publication for subscription {sub.streamId.uuid}")
                continue

            subUuid = sub.streamId.uuid
            pubUuid = matchingPub.streamId.uuid

            try:
                self.streamModels[subUuid] = StreamModel.createFromServer(
                    streamUuid=subUuid,
                    predictionStreamUuid=pubUuid,
                    server=self.server,
                    wallet=self.wallet,
                    subscriptionStream=sub,
                    publicationStream=matchingPub,
                    pauseAll=self.pause,
                    resumeAll=self.resume,
                    storage=self.storage)
                self.streamModels[subUuid].chooseAdapter(inplace=True)
                # Propagate P2P protocol if already set
                if self._prediction_protocol is not None:
                    self.streamModels[subUuid]._prediction_protocol = self._prediction_protocol
                if self._p2p_peers is not None:
                    self.streamModels[subUuid]._p2p_peers = self._p2p_peers
                self.streamModels[subUuid].run_forever()
                info(f"Model initialized for stream {subUuid}", color='green')
            except Exception as e:
                error(f"Failed to initialize model for stream {subUuid}: {e}")

        # TODO: Commented out for engine-neuron integration
        # if self.centrifugo is not None:
        #     for subUuid in self.subscriptions.keys():
        #         streamModel = self.streamModels.get(subUuid)
        #         if streamModel:
        #             try:
        #                 def create_callback(stream_uuid):
        #                     async def callback(ctx):
        #                         await self.handleCentrifugoMessage(ctx, stream_uuid)
        #                     return callback
        #
        #                 sub = await subscribe_to_stream(
        #                     client=self.centrifugo,
        #                     stream_uuid=subUuid,
        #                     events=create_subscription_handler(
        #                         stream_uuid=subUuid,
        #                         on_publication_callback=create_callback(subUuid)))
        #                 self.centrifugoSubscriptions.append(sub)
        #                 info(f"Subscribed to Centrifugo stream {subUuid}")
        #             except Exception as e:
        #                 error(f"Failed to subscribe to Centrifugo stream {subUuid}: {e}")

    def cleanupThreads(self):
        for thread in self.threads:
            if not thread.is_alive():
                self.threads.remove(thread)
        debug(f'prediction thread count: {len(self.threads)}')

    def queuePrediction(self, stream_uuid: str, stream_name: str, value: str, observed_at: str, hash_val: str):
        """Add a prediction to the queue for batch submission."""
        with self.predictionQueueLock:
            self.predictionQueue.append({
                'stream_uuid': stream_uuid,
                'stream_name': stream_name,
                'value': value,
                'observed_at': observed_at,
                'hash': hash_val
            })
            debug(f"Queued prediction for {stream_name} (queue size: {len(self.predictionQueue)})")

    def flushPredictionQueue(self) -> Union[dict, None]:
        """Submit all queued predictions in a batch and clear the queue."""
        with self.predictionQueueLock:
            if not self.predictionQueue:
                return None

            predictions_to_submit = self.predictionQueue.copy()
            queue_size = len(predictions_to_submit)

        # Submit batch outside of lock
        if self.server is not None:
            info(f"Submitting batch of {queue_size} predictions to server...", color='cyan')
            result = self.server.publishPredictionsBatch(predictions_to_submit)

            if result and result.get('successful', 0) > 0:
                # Clear queue only if submission was successful
                with self.predictionQueueLock:
                    self.predictionQueue = []
                info(f"✓ Batch submitted: {result['successful']}/{result['total_submitted']} successful", color='green')
                return result
            else:
                warning(f"Batch prediction submission failed, keeping {queue_size} predictions in queue for retry")
                return None
        else:
            warning("Server not initialized, cannot submit batch predictions")
            return None

    # TODO: Commented out for engine-neuron integration
    # async def cleanupCentrifugo(self):
    #     """Clean up Centrifugo connections and subscriptions"""
    #     try:
    #         if hasattr(self, 'centrifugoSubscriptions') and self.centrifugoSubscriptions:
    #             for subscription in self.centrifugoSubscriptions:
    #                 try:
    #                     await subscription.unsubscribe()
    #                 except Exception as e:
    #                     error(f"Failed to unsubscribe from Centrifugo stream: {e}")
    #             self.centrifugoSubscriptions = []
    #
    #         if hasattr(self, 'centrifugo') and self.centrifugo:
    #             try:
    #                 await self.centrifugo.disconnect()
    #                 info("Centrifugo client disconnected")
    #             except Exception as e:
    #                 error(f"Failed to disconnect Centrifugo client: {e}")
    #             finally:
    #                 self.centrifugo = None
    #     except Exception as e:
    #         error(f"Error during Centrifugo cleanup: {e}")


class StreamModel:

    @classmethod
    def create(
        cls,
        streamUuid: str,
        predictionStreamUuid: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        identity: EvrmoreIdentity,
        pauseAll:callable,
        resumeAll:callable,
        transferProtocol: str
    ):
        streamModel = cls(
            streamUuid,
            predictionStreamUuid,
            peerInfo,
            dataClient,
            identity,
            pauseAll,
            resumeAll,
            transferProtocol
        )
        streamModel.initialize()
        return streamModel

    @classmethod
    def createFromServer(
        cls,
        streamUuid: str,
        predictionStreamUuid: str,
        server: SatoriServerClient,
        wallet: any,
        subscriptionStream: Stream,
        publicationStream: Stream,
        pauseAll: callable,
        resumeAll: callable,
        storage: EngineStorageManager = None,
    ):
        """Factory method for creating StreamModel that uses Central Server directly"""
        streamModel = cls.__new__(cls)
        streamModel.cpu = getProcessorCount()
        streamModel.pauseAll = pauseAll
        streamModel.resumeAll = resumeAll
        streamModel.preferredAdapters = [XgbAdapter, StarterAdapter]
        streamModel.defaultAdapters = [XgbAdapter, XgbAdapter, StarterAdapter]
        streamModel.failedAdapters = []
        streamModel.thread = None
        streamModel.streamUuid = streamUuid
        streamModel.predictionStreamUuid = predictionStreamUuid
        streamModel.subscriptionStream = subscriptionStream
        streamModel.publicationStream = publicationStream
        streamModel.server = server
        streamModel.wallet = wallet
        streamModel.rng = np.random.default_rng(37)
        streamModel.publisherHost = None
        streamModel.transferProtocol = 'central'
        streamModel.usePubSub = False
        streamModel.internal = True
        streamModel.useServer = True  # Flag to use server instead of DataClient
        streamModel.peerInfo = PeerInfo([], [])
        streamModel.dataClientOfIntServer = None
        streamModel.dataClientOfExtServer = None
        streamModel.identity = None
        # SQLite storage manager for local persistence
        streamModel.storage = storage or EngineStorageManager.getInstance()
        # P2P commit-reveal support
        streamModel._prediction_protocol = None
        streamModel._oracle_network = None  # For checking if we're an oracle
        streamModel._current_round_id: int = 0
        streamModel._pending_commits: dict[int, float] = {}  # round_id -> predicted_value
        # Model training customization (from team)
        streamModel.trainingDelay = streamModel._loadTrainingDelay()  # Load from config
        streamModel.initializeFromServer()
        return streamModel

    def __init__(
        self,
        streamUuid: str,
        predictionStreamUuid: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        identity: EvrmoreIdentity,
        pauseAll:callable,
        resumeAll:callable,
        transferProtocol: str
    ):
        self.cpu = getProcessorCount()
        self.pauseAll = pauseAll
        self.resumeAll = resumeAll
        # self.preferredAdapters: list[ModelAdapter] = [XgbChronosAdapter, XgbAdapter, StarterAdapter ]# SKAdapter #model[0] issue
        self.preferredAdapters: list[ModelAdapter] = [ XgbAdapter, StarterAdapter ]# SKAdapter #model[0] issue
        self.defaultAdapters: list[ModelAdapter] = [XgbAdapter, XgbAdapter, StarterAdapter]
        self.failedAdapters = []
        self.thread: threading.Thread = None
        self.streamUuid: str = streamUuid
        self.predictionStreamUuid: str = predictionStreamUuid
        self.peerInfo: PeerInfo = peerInfo
        self.dataClientOfIntServer: DataClient = dataClient
        self.identity: EvrmoreIdentity = identity
        self.rng = np.random.default_rng(37)
        self.publisherHost = None
        self.transferProtocol: str = transferProtocol
        self.usePubSub: bool = False
        self.internal: bool = False
        self.useServer: bool = False  # Default: use DataClient
        # P2P commit-reveal support
        self._prediction_protocol = None
        self._oracle_network = None  # For checking if we're an oracle
        self._current_round_id: int = 0
        self._pending_commits: dict[int, float] = {}  # round_id -> predicted_value
        # Model training customization (from team)
        self.trainingDelay: int = self._loadTrainingDelay()  # Load from config

    def initialize(self):
        self.data: pd.DataFrame = self.loadData()
        self.adapter: ModelAdapter = self.chooseAdapter()
        self.pilot: ModelAdapter = self.adapter(uid=self.streamUuid)
        self.pilot.load(self.modelPath())
        self.stable: ModelAdapter = copy.deepcopy(self.pilot)
        self.paused: bool = False
        self.dataClientOfExtServer: Union[DataClient, None] = DataClient(self.dataClientOfIntServer.serverHostPort[0], self.dataClientOfIntServer.serverPort, identity=self.identity)
        debug(f'AI Engine: stream id {self.streamUuid} using {self.adapter.__name__}', color='teal')

    def initializeFromServer(self):
        """Initialize model when using Central Server directly"""
        self.data: pd.DataFrame = self.loadDataFromServer()
        self.adapter: ModelAdapter = self.chooseAdapter()
        self.pilot: ModelAdapter = self.adapter(uid=self.streamUuid)
        self.pilot.load(self.modelPath())
        self.stable: ModelAdapter = copy.deepcopy(self.pilot)
        self.paused: bool = False
        debug(f'AI Engine: stream id {self.streamUuid} using {self.adapter.__name__} (Central Server mode)', color='teal')

    def initializeForP2P(self):
        """Initialize model for P2P mode (receiving observations via callback)."""
        self.data: pd.DataFrame = self.loadDataFromStorage()
        self.adapter: ModelAdapter = self.chooseAdapter()
        self.pilot: ModelAdapter = self.adapter(uid=self.streamUuid)
        self.pilot.load(self.modelPath())
        self.stable: ModelAdapter = copy.deepcopy(self.pilot)
        self.paused: bool = False
        debug(f'AI Engine: stream id {self.streamUuid} using {self.adapter.__name__} (P2P mode)', color='teal')

    def loadDataFromStorage(self) -> pd.DataFrame:
        """Load historical data from local SQLite storage (P2P mode)."""
        localData = self.storage.getStreamDataForEngine(self.streamUuid)
        if not localData.empty:
            info(f"Loaded {len(localData)} rows from SQLite for stream {self.streamUuid}", color='green')
            return localData
        info(f"No local data for stream {self.streamUuid}, starting fresh", color='yellow')
        return pd.DataFrame(columns=["date_time", "value", "id"])

    def _loadTrainingDelay(self) -> int:
        """Load training delay from config file.

        Returns:
            int: Delay in seconds (default 600 = 10 minutes)
        """
        try:
            from satorineuron import config
            delay = config.get().get('training_delay', 600)
            return int(delay)
        except Exception as e:
            warning(f"Failed to load training delay from config: {e}")
            return 600  # Default to 10 minutes

    def loadDataFromServer(self) -> pd.DataFrame:
        """Load historical data - currently loads from local SQLite only."""
        # TODO: Implement fetching from Central Server when API is ready
        # For now, load from local SQLite if available
        localData = self.storage.getStreamDataForEngine(self.streamUuid)
        if not localData.empty:
            info(f"Loaded {len(localData)} rows from local SQLite for stream {self.streamUuid}", color='green')
            return localData
        info(f"No local data for stream {self.streamUuid}, starting fresh", color='yellow')
        return pd.DataFrame(columns=["date_time", "value", "id"])

    def onDataReceived(self, data: pd.DataFrame):
        """
        Called when new data is received from Central Server.
        Stores data in SQLite and passes to engine for predictions.

        Args:
            data: DataFrame with columns: ts (or date_time), value, hash (or id)
                  Index should be timestamp or 'ts' column should contain timestamps.

        TODO: Call this method when data arrives from Central Server.
        """
        try:
            if data.empty:
                return

            # Normalize column names for storage
            storageDf = data.copy()
            if 'date_time' in storageDf.columns:
                storageDf = storageDf.set_index('date_time')
            elif 'ts' in storageDf.columns:
                storageDf = storageDf.set_index('ts')

            # Store in SQLite (table name = streamUuid)
            insertedRows = self.storage.storeStreamData(
                self.streamUuid,
                storageDf,
                provider='central'
            )
            if insertedRows > 0:
                info(f"Stored {insertedRows} new rows in SQLite for stream {self.streamUuid}", color='green')

            # Also update in-memory data for predictions
            engineDf = data.copy()
            if 'ts' in engineDf.columns:
                engineDf = engineDf.rename(columns={'ts': 'date_time', 'hash': 'id'})
            if 'provider' in engineDf.columns:
                del engineDf['provider']

            self.data = pd.concat([self.data, engineDf], ignore_index=True)
            self.data = self.data.drop_duplicates(subset=['date_time'], keep='last')

            # Trigger prediction when new observation arrives
            if insertedRows > 0:
                info(f"New observation received, triggering prediction for stream {self.streamUuid}", color='blue')
                self.producePrediction()

        except Exception as e:
            error(f"Error storing received data: {e}")  

    # TODO: Commented out for engine-neuron integration - P2P removed
    # async def p2pInit(self):
    #     await self.connectToPeer()
    #     asyncio.create_task(self.monitorPublisherConnection())
    #     await self.startStreamService()
    #
    # async def startStreamService(self):
    #     await self.syncData()
    #     await self.makeSubscription()

    def updateDataClient(self, dataClient):
        ''' Update the internal server data client reference '''
        self.dataClientOfIntServer = dataClient

    # TODO: Commented out for engine-neuron integration - P2P removed
    # def returnPeerIp(self, peer: Union[str, None] = None) -> str:
    #     if peer is not None:
    #         return peer.split(':')[0]
    #     return self.publisherHost.split(':')[0]
    #
    # def returnPeerPort(self, peer: Union[str, None] = None) -> int:
    #     if peer is not None:
    #         return int(peer.split(':')[1])
    #     return int(self.publisherHost.split(':')[1])
    #
    # @property
    # def isConnectedToPublisher(self):
    #     if self.internal:
    #         if self.publisherHost is None:
    #             return False
    #         return True
    #     if hasattr(self, 'dataClientOfExtServer') and self.dataClientOfExtServer is not None and self.publisherHost is not None:
    #         return self.dataClientOfExtServer.isConnected(self.returnPeerIp(), self.returnPeerPort())
    #     return False
    #
    # async def monitorPublisherConnection(self):
    #     """Combined method that monitors connection status and stream activity"""
    #     while True:
    #         if self.internal:
    #             await asyncio.sleep(30)
    #             continue
    #         for _ in range(30):
    #             if not self.isConnectedToPublisher:
    #                 self.publisherHost = None
    #                 await self.dataClientOfIntServer.streamInactive(self.streamUuid)
    #                 await self.connectToPeer()
    #                 await self.startStreamService()
    #                 break
    #             await asyncio.sleep(9)
    #
    #         if self.publisherHost is not None and self.isConnectedToPublisher:
    #             if not await self._isPublisherActive():
    #                 await self.dataClientOfIntServer.streamInactive(self.streamUuid)
    #                 await self.connectToPeer()
    #                 await self.startStreamService()
    #
    # async def _isPublisherActive(self, publisher: str = None) -> bool:
    #     ''' confirms if the publisher has the subscription stream in its available stream '''
    #     try:
    #         response = await self.dataClientOfExtServer.isStreamActive(
    #                     peerHost=self.returnPeerIp(publisher) if publisher is not None else self.returnPeerIp(),
    #                     peerPort=self.returnPeerPort(publisher) if publisher is not None else self.returnPeerPort(),
    #                     uuid=self.streamUuid)
    #         if response.status == DataServerApi.statusSuccess.value:
    #             return True
    #         else:
    #             raise Exception
    #     except Exception as e:
    #         return False
    #
    # async def connectToPeer(self) -> bool:
    #     ''' Connects to a peer to receive subscription if it has an active subscription to the stream '''
    #
    #     async def check_peer_active(ip):
    #         """Only check if peer is active without establishing connection"""
    #         try:
    #             response = await self.dataClientOfExtServer.isStreamActive(
    #                 peerHost=self.returnPeerIp(ip),
    #                 peerPort=self.returnPeerPort(ip),
    #                 uuid=self.streamUuid
    #             )
    #             return response.status == DataServerApi.statusSuccess.value
    #         except Exception:
    #             return False
    #
    #     async def establish_connection(ip):
    #         """Actually establish the connection"""
    #         try:
    #             if await self._isPublisherActive(ip):
    #                 await self.dataClientOfIntServer.addActiveStream(uuid=self.streamUuid)
    #                 return True
    #         except Exception:
    #             pass
    #         return False
    #
    #     while not self.isConnectedToPublisher:
    #         if self.peerInfo.publishersIp is not None and len(self.peerInfo.publishersIp) > 0:
    #             candidate_publisher = self.peerInfo.publishersIp[0]
    #             if await establish_connection(candidate_publisher):
    #                 self.publisherHost = candidate_publisher
    #                 self.usePubSub = False
    #                 return True
    #
    #         response = await self.dataClientOfIntServer.isStreamActive(uuid=self.streamUuid)
    #         if response.status == DataServerApi.statusSuccess.value:
    #             info("Connected to Local Data Server", self.streamUuid)
    #             self.publisherHost = f"{self.dataClientOfIntServer.serverHostPort[0]}:{self.dataClientOfIntServer.serverPort}"
    #             self.internal = True
    #             self.usePubSub = False
    #             return True
    #
    #         subscriber_ips = [ip for ip in self.peerInfo.subscribersIp]
    #         self.rng.shuffle(subscriber_ips)
    #
    #         check_tasks = [(ip, asyncio.create_task(check_peer_active(ip))) for ip in subscriber_ips]
    #         active_peers = []
    #         for ip, task in check_tasks:
    #             try:
    #                 if await task:
    #                     active_peers.append(ip)
    #             except Exception as e:
    #                 error(f"Error checking peer {ip}: {str(e)}")
    #         if active_peers:
    #             selected_peer = active_peers[0]
    #             if await establish_connection(selected_peer):
    #                 self.publisherHost = selected_peer
    #                 self.usePubSub = False
    #                 return True
    #
    #         self.publisherHost = None
    #         warning('Failed to connect to Peers, switching to PubSub', self.streamUuid, print=True)
    #         self.usePubSub = True
    #         await asyncio.sleep(60*60)

    # TODO: Commented out for engine-neuron integration - P2P removed
    # async def syncData(self):
    #     '''
    #     - this can be highly optimized. but for now we do the simple version
    #     - just ask for their entire dataset every time
    #         - if it's different than the df we got from our own dataserver,
    #           then tell dataserver to save this instead
    #         - replace what we have
    #     '''
    #     try:
    #         if self.internal:
    #             return
    #         else:
    #             DataResponse = await self.dataClientOfExtServer.getRemoteStreamData(
    #                 peerHost=self.returnPeerIp(),
    #                 peerPort=self.returnPeerPort(),
    #                 uuid=self.streamUuid)
    #         if DataResponse.status == DataServerApi.statusSuccess.value:
    #             externalDf = DataResponse.data
    #             if not externalDf.equals(self.data) and len(externalDf) > 0:
    #                 response = await self.dataClientOfIntServer.insertStreamData(
    #                                 uuid=self.streamUuid,
    #                                 data=externalDf,
    #                                 replace=True
    #                             )
    #                 if response.status == DataServerApi.statusSuccess.value:
    #                     info("Data updated in server", color='green')
    #                     externalDf = externalDf.reset_index().rename(columns={
    #                                 'ts': 'date_time',
    #                                 'hash': 'id'
    #                             }).drop(columns=['provider'])
    #                     self.data = externalDf
    #                 else:
    #                     raise Exception(DataResponse.senderMsg)
    #             else:
    #                 raise Exception(DataResponse.senderMsg)
    #     except Exception as e:
    #         error("Failed to sync data, ", e)
    #         self.publisherHost = None
    #
    # async def makeSubscription(self):
    #     '''
    #     - and subscribe to the stream so we get the information
    #         - whenever we get an observation on this stream, pass to the DataServer
    #     - continually generate predictions for prediction publication streams and pass that to
    #     '''
    #     if self.internal:
    #         await self.dataClientOfIntServer.subscribe(
    #             uuid=self.streamUuid,
    #             publicationUuid=self.predictionStreamUuid,
    #             callback=self.handleSubscriptionMessage,
    #             engineSubscribed=True)
    #     else:
    #         await self.dataClientOfExtServer.subscribe(
    #             uuid=self.streamUuid,
    #             peerHost=self.returnPeerIp(),
    #             peerPort=self.returnPeerPort(),
    #             publicationUuid=self.predictionStreamUuid,
    #             callback=self.handleSubscriptionMessage,
    #             engineSubscribed=True)
    #
    # async def handleSubscriptionMessage(self, subscription: any,  message: Message, pubSubFlag: bool = False):
    #     debug(f"Stream {self.streamUuid} received subscription message (pubsub: {pubSubFlag})")
    #     if message.status == DataClientApi.streamInactive.value:
    #         warning("Stream Inactive")
    #         await self.closePeerConnection()
    #         self.publisherHost = None
    #     else:
    #         await self.appendNewData(message.data, pubSubFlag)
    #         self.pauseAll(force=True)
    #         await self.producePrediction()
    #         self.resumeAll(force=True)
    #
    # async def closePeerConnection(self):
    #     """Close the connection to the current publisher peer"""
    #     if self.internal:
    #         self.publisherHost = None
    #         return
    #     if self.publisherHost is not None and hasattr(self, 'dataClientOfExtServer') and self.dataClientOfExtServer is not None:
    #         try:
    #             peer = self.dataClientOfExtServer.peers.get((self.returnPeerIp(), self.returnPeerPort()))
    #             if peer is not None:
    #                 await self.dataClientOfExtServer.disconnect(peer)
    #                 info(f"Closed connection to peer {(self.returnPeerIp(), self.returnPeerPort())}")
    #             self.publisherHost = None
    #         except Exception as e:
    #             error(f"Error closing peer connection: {e}")
    #             self.publisherHost = None

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    async def appendNewData(self, observation: Union[pd.DataFrame, dict], pubSubFlag: bool):
        """extract the data and save it to self.data"""
        try:
            if pubSubFlag:
                parsedData = json.loads(observation.raw)
                debug(f"Stream {self.streamUuid} appending new data: {parsedData['data']} at {parsedData['time']}")
                if validate_single_entry(parsedData["time"], parsedData["data"]):
                        await self.dataClientOfIntServer.insertStreamData(
                                uuid=self.streamUuid,
                                data=pd.DataFrame({ 'value': [float(parsedData["data"])]
                                            }, index=[str(parsedData["time"])]),
                                isSub=True
                            )
                        self.data = pd.concat(
                            [
                                self.data,
                                pd.DataFrame({
                                    "date_time": [str(parsedData["time"])],
                                    "value": [float(parsedData["data"])],
                                    "id": [str(parsedData["hash"])]})
                            ],
                            ignore_index=True)
                else:
                    error("Row not added due to corrupt observation")
            else:
                observation_id = observation['hash'].values[0]
                # Check if self.data is not empty and if the ID already exists
                if not self.data.empty and observation_id in self.data['id'].values:
                    error("Row not added because observation with same ID already exists")
                elif validate_single_entry(observation.index[0], observation["value"].values[0]):
                    if not self.internal:
                        response = await self.dataClientOfIntServer.insertStreamData(
                                uuid=self.streamUuid,
                                data=observation,
                                isSub=True
                            )
                        if response.status == DataServerApi.statusSuccess.value:
                            info(response.senderMsg, color='green')
                        else:
                            raise Exception("Raw ", response.senderMsg)
                    observationDf = observation.reset_index().rename(columns={
                                'index': 'date_time',
                                'hash': 'id'
                            }).drop(columns=['provider'])
                    self.data = pd.concat([self.data, observationDf], ignore_index=True)
                else:
                    error("Row not added due to corrupt observation")
        except Exception as e:
            error("Subscription data not added", e)

    def passPredictionData(self, forecast: pd.DataFrame, passToCentralServer: bool = False):
        try:
            # If using Central Server mode, publish directly to Central
            if getattr(self, 'useServer', False) and self.server is not None:
                self.publishPredictionToServer(forecast)
                return

            # Original DataClient mode
            if not hasattr(self, 'activatePredictionStream') and passToCentralServer:
                asyncio.run(self.dataClientOfIntServer.addActiveStream(uuid=self.predictionStreamUuid))
                self.activatePredictionStream = True
            response = asyncio.run(self.dataClientOfIntServer.insertStreamData(
                            uuid=self.predictionStreamUuid,
                            data=forecast,
                            isSub=True,
                            replace=False if passToCentralServer else True,
                        ))
            if response.status == DataServerApi.statusSuccess.value:
                info("Prediction", response.senderMsg, color='green')
            else:
                raise Exception(response.senderMsg)
        except Exception as e:
            error('Failed to send Prediction to server : ', e)

    def publishPredictionToServer(self, forecast: pd.DataFrame, useBatch: bool = True):
        """Publish prediction - either queue for batch submission or send immediately.

        Args:
            forecast: DataFrame with prediction
            useBatch: If True, queue for batch submission. If False, publish immediately (legacy)
        """
        try:
            predictionValue = str(forecast['value'].iloc[0])
            observationTime = str(forecast.index[0])
            # Generate hash for the prediction
            import hashlib
            observationHash = hashlib.sha256(
                f"{predictionValue}{observationTime}".encode()
            ).hexdigest()[:16]

            # Store prediction locally in SQLite (table name = predictionStreamUuid)
            if hasattr(self, 'storage') and self.storage is not None:
                stored = self.storage.storePrediction(
                    predictionStreamUuid=self.predictionStreamUuid,
                    timestamp=observationTime,
                    value=float(predictionValue),
                    hash_val=observationHash,
                    provider='engine'
                )
                if stored:
                    debug(f"Prediction stored locally for stream {self.predictionStreamUuid}")

            # Get stream name for logging
            stream_name = getattr(self.subscriptionStream.streamId, 'stream', 'unknown')

            if useBatch:
                # Queue prediction for batch submission
                # Get engine instance from parent
                engine = None
                # Try to find engine through model references
                for attr_name in dir(self):
                    if attr_name.startswith('_'):
                        continue
                    attr = getattr(self, attr_name, None)
                    if isinstance(attr, Engine):
                        engine = attr
                        break

                # If no direct reference, try to get from globals/parent context
                if engine is None and hasattr(self, '__dict__'):
                    # The engine creates StreamModel, so we need to access it differently
                    # For now, we'll use the server to access the parent neuron's engine
                    # This is a bit hacky but works for the neuron-engine integration
                    pass

                # For now, queue via server which will be picked up by neuron
                # The neuron will call engine.queuePrediction() after observation processing
                debug(f"Prediction ready for batching: {stream_name} = {predictionValue}")
                # Store in instance variable for neuron to collect
                if not hasattr(self, '_pending_prediction'):
                    self._pending_prediction = {
                        'stream_uuid': self.streamUuid,
                        'stream_name': stream_name,
                        'value': predictionValue,
                        'observed_at': observationTime,
                        'hash': observationHash
                    }
            else:
                # Legacy: immediate publish
                topic = self.publicationStream.streamId.jsonId

                isSuccess = self.server.publish(
                    topic=topic,
                    data=predictionValue,
                    observationTime=observationTime,
                    observationHash=observationHash,
                    isPrediction=True,
                    useAuthorizedCall=True)

                if isSuccess:
                    info(f"Prediction published to Central Server: {predictionValue} at {observationTime}", color='green')
                elif isSuccess is False:
                    # False means actual failure (not rate limiting which returns None)
                    warning(f"Failed to publish prediction to Central Server")
                # None means rate limited - already logged in server.publish()
        except Exception as e:
            error(f"Error publishing prediction to Central Server: {e}")

    def _createAugmentedData(self, firstForecast: pd.DataFrame) -> pd.DataFrame:
        try:
            firstValue = StreamForecast.firstPredictionOf(firstForecast)
            if 'date_time' in firstForecast.columns:
                timestamp = firstForecast['date_time'].iloc[0]
            elif 'ds' in firstForecast.columns:
                timestamp = firstForecast['ds'].iloc[0]
            else:
                timestamp = now()

            tempHash = hashlib.sha256(
                f"{firstValue}{timestamp}".encode()
            ).hexdigest()[:16]

            tempRow = pd.DataFrame({
                'date_time': [timestamp],
                'value': [firstValue],
                'id': [tempHash]
            })
            return pd.concat([self.data, tempRow], ignore_index=True)

        except Exception as e:
            error(f"Error creating augmented data for autoregression: {e}")
            return self.data

    def producePrediction(self, updatedModel=None):
        """
        triggered by
            - model replaced with a better one
            - new observation on the stream
        Supports P2P commit-reveal in hybrid/p2p mode.
        """
        try:
            model = updatedModel or self.stable
            if model is not None:
                firstForecast = model.predict(data=self.data)

                # Only do autoregression if first prediction is valid
                if isinstance(firstForecast, pd.DataFrame):
                    firstValue = StreamForecast.firstPredictionOf(firstForecast)
                    debug(f"[AUTOREGRESSION] First prediction: {firstValue}", color='cyan')

                    augmentedData = self._createAugmentedData(firstForecast)
                    debug(f"[AUTOREGRESSION] Augmented data size: {len(augmentedData)} rows (original: {len(self.data)})", color='cyan')

                    secondForecast = model.predict(data=augmentedData)
                    if isinstance(secondForecast, pd.DataFrame):
                        secondValue = StreamForecast.firstPredictionOf(secondForecast)
                        debug(f"[AUTOREGRESSION] Second prediction (queued for batch): {secondValue}", color='cyan')
                    else:
                        secondForecast = None

                    forecast = secondForecast if secondForecast is not None else firstForecast
                else:
                    # First prediction failed, skip autoregression
                    debug("[AUTOREGRESSION] Skipping - first prediction failed", color='yellow')
                    forecast = firstForecast

                if isinstance(forecast, pd.DataFrame):
                    # Use Unix timestamp for consistency with observation storage
                    predictionDf = pd.DataFrame({ 'value': [StreamForecast.firstPredictionOf(forecast)]
                                    }, index=[datetimeToUnixTimestamp(now())])
                    debug(predictionDf, print=True)

                    # Check if we should use P2P commit-reveal
                    networking_mode = _get_networking_mode()
                    if networking_mode in ('hybrid', 'p2p'):
                        # Commit prediction to P2P network
                        predicted_value = float(predictionDf['value'].iloc[0])
                        self._commitP2PPrediction(predicted_value)

                    # Also publish to central server if in hybrid or central mode
                    if networking_mode in ('central', 'hybrid'):
                        if updatedModel is not None:
                            self.passPredictionData(predictionDf)
                        else:
                            self.passPredictionData(predictionDf, True)
                else:
                    raise Exception('Forecast not in dataframe format')
        except Exception as e:
            error(e)
            self.fallback_prediction()

    def _commitP2PPrediction(self, predicted_value: float):
        """Commit prediction to P2P network using commit-reveal protocol."""
        try:
            # Calculate target time (e.g., next observation period)
            target_time = int(time.time()) + 3600  # 1 hour ahead (configurable)

            # Increment round ID
            self._current_round_id += 1

            # Store for potential reveal later
            self._pending_commits[self._current_round_id] = predicted_value

            debug(f"P2P prediction committed for {self.streamUuid} round={self._current_round_id} value={predicted_value}")

            # If we have a prediction protocol instance, publish the prediction
            # Try to get it from various sources if not set on StreamModel
            prediction_protocol = self._prediction_protocol
            if prediction_protocol is None:
                # Try to get from startupDag as fallback
                try:
                    from satorineuron.init import start
                    startup = start.getStart() if hasattr(start, 'getStart') else None
                    if startup and hasattr(startup, '_prediction_protocol'):
                        prediction_protocol = startup._prediction_protocol
                        # Cache it for future use
                        if prediction_protocol is not None:
                            self._prediction_protocol = prediction_protocol
                except Exception:
                    pass

            if prediction_protocol is not None:
                try:
                    # Check if we're an oracle for this stream
                    is_oracle = False
                    stream_name = getattr(self, 'stream_name', None) or self.streamUuid
                    if self._oracle_network is not None:
                        try:
                            oracle_role = self._oracle_network.get_oracle_role(stream_name)
                            is_oracle = oracle_role in ('primary', 'secondary')
                        except Exception:
                            pass

                    # Use the Peers' spawn_background_task to run async in Trio context
                    # This avoids conflicts with asyncio.run() and Trio event loops
                    p2p_peers = self._p2p_peers
                    if p2p_peers is None:
                        # Try to get from startupDag as fallback
                        try:
                            from satorineuron.init import start
                            startup = start.getStart() if hasattr(start, 'getStart') else None
                            if startup and hasattr(startup, '_p2p_peers'):
                                p2p_peers = startup._p2p_peers
                                if p2p_peers is not None:
                                    self._p2p_peers = p2p_peers
                        except Exception:
                            pass

                    if p2p_peers is not None and hasattr(p2p_peers, 'spawn_background_task'):
                        async def _publish():
                            try:
                                await prediction_protocol.publish_prediction(
                                    stream_id=self.streamUuid,
                                    value=predicted_value,
                                    target_time=target_time,
                                    confidence=0.5,
                                    is_oracle=is_oracle
                                )
                                oracle_tag = " [oracle]" if is_oracle else ""
                                info(f"Published P2P prediction{oracle_tag}: stream={self.streamUuid} value={predicted_value}", color="green")
                            except Exception as e:
                                warning(f"Failed to publish P2P prediction: {e}")
                        p2p_peers.spawn_background_task(_publish)
                    else:
                        # Fallback to asyncio.run (may fail in Trio context)
                        asyncio.run(prediction_protocol.publish_prediction(
                            stream_id=self.streamUuid,
                            value=predicted_value,
                            target_time=target_time,
                            confidence=0.5,
                            is_oracle=is_oracle
                        ))
                except Exception as e:
                    warning(f"Exception publishing P2P prediction: {e}")

        except ImportError:
            debug("satorip2p not available for P2P prediction commit")
        except Exception as e:
            warning(f"Failed to commit P2P prediction: {e}")

    def revealP2PPrediction(self, round_id: int):
        """Reveal a previously committed prediction (called after observation arrives)."""
        if round_id not in self._pending_commits:
            return

        try:
            if self._prediction_protocol is not None:
                asyncio.run(self._prediction_protocol.reveal_prediction(
                    stream_id=self.streamUuid,
                    round_id=round_id
                ))
                debug(f"P2P prediction revealed for round {round_id}")

            # Clean up
            del self._pending_commits[round_id]

        except Exception as e:
            warning(f"Failed to reveal P2P prediction: {e}")

    def fallback_prediction(self):
        if os.path.isfile(self.modelPath()):
            try:
                os.remove(self.modelPath())
                debug("Deleted failed model file:", self.modelPath(), color="teal")
            except Exception as e:
                error(f'Failed to delete model file: {str(e)}')
        backupModel = self.defaultAdapters[-1]()
        try:
            trainingResult = backupModel.fit(data=self.data)
            if abs(trainingResult.status) == 1:
                self.producePrediction(backupModel)
        except Exception as e:
            error(f"Error training new model: {str(e)}")

    def loadData(self) -> pd.DataFrame:
        try:
            response = asyncio.run(self.dataClientOfIntServer.getLocalStreamData(uuid=self.streamUuid))
            if response.status == DataServerApi.statusSuccess.value:
                conformedData = response.data.reset_index().rename(columns={
                    'ts': 'date_time',
                    'hash': 'id'
                })
                del conformedData['provider']
                return conformedData
            else:
                raise Exception(response.senderMsg)
        except Exception as e:
            debug(e)
            return pd.DataFrame(columns=["date_time", "value", "id"])

    def modelPath(self) -> str:
        return (
            '/Satori/Neuron/models/veda/'
            f'{self.predictionStreamUuid}/'
            f'{self.adapter.__name__}.joblib')

    def chooseAdapter(self, inplace: bool = False) -> ModelAdapter:
        """
        everything can try to handle some cases
        Engine
            - low resources available - SKAdapter
            - few observations - SKAdapter
            - (mapping of cases to suitable adapters)
        examples: StartPipeline, SKAdapter, XGBoostPipeline, ChronosPipeline, DNNPipeline
        """
        # TODO: this needs to be aultered. I think the logic is not right. we
        #       should gather a list of adapters that can be used in the
        #       current condition we're in. if we're already using one in that
        #       list, we should continue using it until it starts to make bad
        #       predictions. if not, we should then choose the best one from the
        #       list - we should optimize after we gather acceptable options.

        if False: # for testing specific adapters
            adapter = XgbChronosAdapter
        else:
            import psutil
            availableRamGigs = psutil.virtual_memory().available / 1e9
            availableSwapGigs = psutil.swap_memory().free / 1e9
            totalAvailableRamGigs = availableRamGigs + availableSwapGigs
            adapter = None
            for p in self.preferredAdapters:
                if p in self.failedAdapters:
                    continue
                if p.condition(data=self.data, cpu=self.cpu, availableRamGigs=totalAvailableRamGigs) == 1:
                    adapter = p
                    break
            if adapter is None:
                for adapter in self.defaultAdapters:
                    if adapter not in self.failedAdapters:
                        break
                if adapter is None:
                    adapter = self.defaultAdapters[-1]
        if (
            inplace and (
                not hasattr(self, 'pilot') or
                not isinstance(self.pilot, adapter))
        ):
            info(
                f'AI Engine: stream id {self.streamUuid} '
                f'switching from {self.adapter.__name__} '
                f'to {adapter.__name__} on {self.streamUuid}',
                color='teal')
            self.adapter = adapter
            self.pilot = adapter(uid=self.streamUuid)
            self.pilot.load(self.modelPath())
        return adapter

    def run(self):
        """
        Main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        """
        while True:
            # Wait for data if we don't have any yet
            if len(self.data) == 0:
                time.sleep(10)
                continue
            if self.paused:
                time.sleep(10)
                continue
            self.chooseAdapter(inplace=True)
            try:
                trainingResult = self.pilot.fit(data=self.data, stable=self.stable)
                if trainingResult.status == 1:
                    if self.pilot.compare(self.stable):
                        if self.pilot.save(self.modelPath()):
                            self.stable = copy.deepcopy(self.pilot)
                            info("stable model updated for stream:", self.streamUuid)
                else:
                    debug(f'model training failed on {self.streamUuid} waiting 10 minutes to retry')
                    self.failedAdapters.append(self.pilot)
                    time.sleep(600)
            except Exception as e:
                import traceback
                traceback.print_exc()
                error(e)
                try:
                    debug(self.pilot.dataset)
                except Exception as e:
                    pass

            # Sleep between training iterations based on user setting
            if self.trainingDelay > 0:
                debug(f"Sleeping {self.trainingDelay}s before next training iteration for stream {self.streamUuid}")
                time.sleep(self.trainingDelay)


    def run_forever(self):
        '''Creates separate threads for running the model training loop'''

        if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
            warning(f"Thread for model {self.streamUuid} already running. Not creating another.")
            return

        def training_loop_thread():
            try:
                self.run()
            except Exception as e:
                error(f"Error in training loop thread: {e}")
                import traceback
                traceback.print_exc()

        self.thread = threading.Thread(target=training_loop_thread, daemon=True)
        self.thread.start()


def main():
    engine = Engine()
    engine.initialize()
    # Keep running
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()