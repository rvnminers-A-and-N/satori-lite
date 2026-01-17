"""
P2P to WebSocket Bridge.

This module bridges P2P network events to the WebSocket layer,
enabling real-time updates in the UI for:
- Predictions
- Observations
- Consensus votes
- Heartbeats
- Pool/lending updates
- Delegation updates
- Peer connection changes

It also tracks hourly activity for the Network Activity (24h) chart.

Usage:
    from web.p2p_bridge import start_bridge

    # Call after P2P protocols are initialized
    start_bridge(
        peers=peers,
        prediction_protocol=prediction_protocol,
        oracle_network=oracle_network,
        consensus_manager=consensus_manager,
        uptime_tracker=uptime_tracker,
        lending_manager=lending_manager,
        delegation_manager=delegation_manager,
    )
"""

import logging
import time
import threading
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from satorip2p.peers import Peers

logger = logging.getLogger(__name__)


class P2PWebSocketBridge:
    """
    Bridge between P2P network events and WebSocket.

    Wires callbacks to P2P protocols and emits events to the
    web UI via sendToUI(). Also tracks activity counts for charts.
    """

    def __init__(self):
        self._started = False

        # Track counts for the session
        self._counts: Dict[str, int] = {
            'predictions': 0,
            'observations': 0,
            'heartbeats': 0,
            'consensus_votes': 0,
            'governance': 0,
        }

        # Track hourly activity for charts
        self._hourly_activity: Dict[int, Dict[str, int]] = {}

        # Store recent events for polling (when WebSocket is unavailable)
        self._recent_events: list = []
        self._max_recent_events = 100

        # Reference to uptime tracker for loading persisted stats
        self._uptime_tracker = None

        # Flag to track if we've loaded persisted stats
        self._loaded_persisted_stats = False

    def start(
        self,
        peers=None,
        prediction_protocol=None,
        oracle_network=None,
        consensus_manager=None,
        uptime_tracker=None,
        lending_manager=None,
        delegation_manager=None,
        governance_manager=None,
    ) -> None:
        """
        Start the bridge by wiring callbacks to P2P protocols.

        Args:
            peers: Peers instance for peer connection events
            prediction_protocol: For prediction events
            oracle_network: For observation events
            consensus_manager: For consensus vote/phase events
            uptime_tracker: For heartbeat events
            lending_manager: For pool/lending events
            delegation_manager: For delegation events
            governance_manager: For governance/voting events
        """
        if self._started:
            return

        # Wire all callbacks
        self._wire_uptime_tracker(uptime_tracker)
        self._wire_prediction_protocol(prediction_protocol)
        self._wire_oracle_network(oracle_network)
        self._wire_consensus_manager(consensus_manager)
        self._wire_lending_manager(lending_manager)
        self._wire_delegation_manager(delegation_manager)
        self._wire_governance_manager(governance_manager)
        self._wire_peer_connections(peers)

        self._started = True
        logger.info(f"P2P WebSocket bridge started (uptime_tracker={uptime_tracker is not None}, consensus={consensus_manager is not None})")

    def stop(self) -> None:
        """Stop the bridge."""
        self._started = False
        logger.info("P2P WebSocket bridge stopped")

    def wire_protocol(self, protocol_name: str, protocol) -> bool:
        """
        Wire a single protocol after bridge is started.

        This allows late-binding of protocols that aren't ready at startup.

        Args:
            protocol_name: One of 'uptime_tracker', 'consensus_manager', 'prediction_protocol',
                          'oracle_network', 'lending_manager', 'delegation_manager', 'governance_manager', 'peers'
            protocol: The protocol instance to wire

        Returns:
            True if successfully wired
        """
        if not protocol:
            return False

        try:
            if protocol_name == 'uptime_tracker':
                self._wire_uptime_tracker(protocol)
            elif protocol_name == 'consensus_manager':
                self._wire_consensus_manager(protocol)
            elif protocol_name == 'prediction_protocol':
                self._wire_prediction_protocol(protocol)
            elif protocol_name == 'oracle_network':
                self._wire_oracle_network(protocol)
            elif protocol_name == 'lending_manager':
                self._wire_lending_manager(protocol)
            elif protocol_name == 'delegation_manager':
                self._wire_delegation_manager(protocol)
            elif protocol_name == 'governance_manager':
                self._wire_governance_manager(protocol)
            elif protocol_name == 'peers':
                self._wire_peer_connections(protocol)
            else:
                logger.warning(f"Unknown protocol: {protocol_name}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Failed to wire {protocol_name}: {e}")
            return False

    # ========================================================================
    # WIRING METHODS
    # ========================================================================

    def _wire_uptime_tracker(self, tracker) -> None:
        """Wire heartbeat events from uptime tracker."""
        if not tracker:
            return

        try:
            # Wire heartbeat received callback (from network)
            if hasattr(tracker, 'on_heartbeat_received'):
                tracker.on_heartbeat_received = lambda hb: self._on_heartbeat(hb, is_own=False)
                logger.info("Wired uptime tracker: on_heartbeat_received")

            # Wire heartbeat sent callback (our own broadcasts)
            if hasattr(tracker, 'on_heartbeat_sent'):
                tracker.on_heartbeat_sent = lambda hb: self._on_heartbeat(hb, is_own=True)
                logger.info("Wired uptime tracker: on_heartbeat_sent")

            # Store reference for loading persisted stats
            self._uptime_tracker = tracker
        except Exception as e:
            logger.warning(f"Failed to wire uptime tracker: {e}")

    def _wire_prediction_protocol(self, protocol) -> None:
        """Wire prediction events from prediction protocol."""
        if not protocol:
            return

        try:
            if hasattr(protocol, 'on_prediction_received'):
                protocol.on_prediction_received = self._on_prediction
            elif hasattr(protocol, 'set_callback'):
                protocol.set_callback('prediction', self._on_prediction)
            logger.debug("Wired prediction protocol")
        except Exception as e:
            logger.warning(f"Failed to wire prediction protocol: {e}")

    def _wire_oracle_network(self, oracle) -> None:
        """Wire observation events from oracle network."""
        if not oracle:
            return

        try:
            if hasattr(oracle, 'on_observation_received'):
                oracle.on_observation_received = self._on_observation
            elif hasattr(oracle, 'set_callback'):
                oracle.set_callback('observation', self._on_observation)
            logger.debug("Wired oracle network for observations")
        except Exception as e:
            logger.warning(f"Failed to wire oracle network: {e}")

    def _wire_consensus_manager(self, consensus) -> None:
        """Wire consensus vote and phase events."""
        if not consensus:
            return

        try:
            # Wire vote received callback
            if hasattr(consensus, 'on_vote_received'):
                consensus.on_vote_received = self._on_consensus_vote
                logger.info("Wired consensus manager: on_vote_received")

            # Wire phase change callback
            if hasattr(consensus, 'on_phase_change'):
                consensus.on_phase_change = self._on_consensus_phase
                logger.info("Wired consensus manager: on_phase_change")
        except Exception as e:
            logger.warning(f"Failed to wire consensus manager: {e}")

    def _wire_lending_manager(self, lending) -> None:
        """Wire pool/lending events."""
        if not lending:
            return

        try:
            if hasattr(lending, 'on_update'):
                lending.on_update = lambda data: self._on_pool_update('lending', data)
            elif hasattr(lending, 'set_callback'):
                lending.set_callback('update', lambda data: self._on_pool_update('lending', data))
            logger.debug("Wired lending manager")
        except Exception as e:
            logger.warning(f"Failed to wire lending manager: {e}")

    def _wire_delegation_manager(self, delegation) -> None:
        """Wire delegation events."""
        if not delegation:
            return

        try:
            if hasattr(delegation, 'on_update'):
                delegation.on_update = lambda data: self._on_delegation_update('delegation', data)
            elif hasattr(delegation, 'set_callback'):
                delegation.set_callback('update', lambda data: self._on_delegation_update('delegation', data))
            logger.debug("Wired delegation manager")
        except Exception as e:
            logger.warning(f"Failed to wire delegation manager: {e}")

    def _wire_governance_manager(self, governance) -> None:
        """Wire governance/voting events."""
        if not governance:
            return

        try:
            # Wire vote received callback
            if hasattr(governance, 'on_vote_received'):
                governance.on_vote_received = self._on_governance_vote
                logger.info("Wired governance manager: on_vote_received")

            # Wire proposal received callback
            if hasattr(governance, 'on_proposal_received'):
                governance.on_proposal_received = self._on_governance_proposal
                logger.info("Wired governance manager: on_proposal_received")

            # Generic update callback
            if hasattr(governance, 'on_update'):
                governance.on_update = self._on_governance_update
            elif hasattr(governance, 'set_callback'):
                governance.set_callback('update', self._on_governance_update)
            logger.debug("Wired governance manager")
        except Exception as e:
            logger.warning(f"Failed to wire governance manager: {e}")

    def _wire_peer_connections(self, peers) -> None:
        """Wire peer connection change events."""
        if not peers:
            return

        try:
            if hasattr(peers, 'on_connection_change'):
                peers.on_connection_change(self._on_peer_connection)
                logger.debug("Wired peer connection callbacks")
        except Exception as e:
            logger.warning(f"Failed to wire peer connections: {e}")

    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================

    def _on_heartbeat(self, heartbeat, is_own: bool = False) -> None:
        """Handle heartbeat event.

        Args:
            heartbeat: The heartbeat object
            is_own: True if this is our own broadcast, False if received from network
        """
        try:
            # Get peer_id (libp2p ID) - this is what we show in the UI
            peer_id = getattr(heartbeat, 'peer_id', '') or ''
            # Get node_id (usually evrmore address)
            node_id = heartbeat.node_id if hasattr(heartbeat, 'node_id') else str(heartbeat)
            timestamp = heartbeat.timestamp if hasattr(heartbeat, 'timestamp') else int(time.time())

            action = "sent" if is_own else "received"
            logger.info(f"Bridge {action} heartbeat {'(own)' if is_own else 'from peer'} {peer_id[:20]}...")

            self._emit('network.heartbeat', {
                'peer_id': peer_id,  # libp2p peer ID for display
                'node_id': node_id,  # evrmore address
                'address': getattr(heartbeat, 'evrmore_address', ''),
                'timestamp': timestamp,
                'roles': getattr(heartbeat, 'roles', []),
                'status_message': getattr(heartbeat, 'status_message', ''),
                'type': 'heartbeat',
                'is_own': is_own,  # True if this is our own broadcast
            })

            self._counts['heartbeats'] += 1
            self._record_hourly_activity('heartbeats')
        except Exception as e:
            logger.warning(f"Failed to handle heartbeat: {e}")

    def _on_prediction(self, prediction) -> None:
        """Handle prediction event."""
        try:
            stream_id = prediction.stream_id if hasattr(prediction, 'stream_id') else str(prediction)

            self._emit('network.prediction', {
                'stream_id': stream_id,
                'node_id': getattr(prediction, 'predictor', 'unknown'),
                'value': getattr(prediction, 'value', None),
                'timestamp': getattr(prediction, 'created_at', getattr(prediction, 'timestamp', int(time.time()))),
                'target_time': getattr(prediction, 'target_time', None),
                'type': 'prediction',
            })

            self._counts['predictions'] += 1
            self._record_hourly_activity('predictions')
        except Exception as e:
            logger.debug(f"Failed to handle prediction: {e}")

    def _on_observation(self, observation) -> None:
        """Handle observation event."""
        try:
            stream_id = observation.stream_id if hasattr(observation, 'stream_id') else str(observation)

            self._emit('network.observation', {
                'stream_id': stream_id,
                'node_id': getattr(observation, 'oracle_address', getattr(observation, 'oracle', 'unknown')),
                'value': getattr(observation, 'value', None),
                'timestamp': getattr(observation, 'timestamp', int(time.time())),
                'type': 'observation',
            })

            self._counts['observations'] += 1
            self._record_hourly_activity('observations')
        except Exception as e:
            logger.debug(f"Failed to handle observation: {e}")

    def _on_consensus_vote(self, vote) -> None:
        """Handle consensus vote event."""
        try:
            node_id = vote.get('node_id', 'unknown') if isinstance(vote, dict) else getattr(vote, 'node_id', 'unknown')
            round_id = vote.get('round_id') if isinstance(vote, dict) else getattr(vote, 'round_id', None)
            merkle_root = vote.get('merkle_root', '') if isinstance(vote, dict) else getattr(vote, 'merkle_root', '')

            self._emit('network.consensus.vote', {
                'node_id': node_id,
                'round_id': round_id,
                'merkle_root': merkle_root[:16] + '...' if merkle_root else '',
                'timestamp': int(time.time()),
                'type': 'consensus_vote',
            })

            self._counts['consensus_votes'] += 1
            self._record_hourly_activity('consensus')
        except Exception as e:
            logger.debug(f"Failed to handle consensus vote: {e}")

    def _on_consensus_phase(self, phase, round_id=None) -> None:
        """Handle consensus phase change event."""
        try:
            self._emit('consensus.phase', {
                'phase': str(phase),
                'round_id': round_id,
                'timestamp': int(time.time()),
            })
        except Exception as e:
            logger.debug(f"Failed to handle consensus phase: {e}")

    def _on_pool_update(self, update_type: str, data: Any) -> None:
        """Handle pool/lending update event."""
        try:
            self._emit('pool_update', {
                'type': update_type,
                'data': data,
                'timestamp': int(time.time()),
            })
        except Exception as e:
            logger.debug(f"Failed to handle pool update: {e}")

    def _on_delegation_update(self, update_type: str, data: Any) -> None:
        """Handle delegation update event."""
        try:
            self._emit('delegation_update', {
                'type': update_type,
                'data': data,
                'timestamp': int(time.time()),
            })
        except Exception as e:
            logger.debug(f"Failed to handle delegation update: {e}")

    def _on_governance_vote(self, vote) -> None:
        """Handle governance vote event."""
        try:
            peer_id = vote.get('peer_id', '') if isinstance(vote, dict) else getattr(vote, 'peer_id', '')
            voter = vote.get('voter', '') if isinstance(vote, dict) else getattr(vote, 'voter', '')
            proposal_id = vote.get('proposal_id', '') if isinstance(vote, dict) else getattr(vote, 'proposal_id', '')
            vote_value = vote.get('vote', '') if isinstance(vote, dict) else getattr(vote, 'vote', '')

            self._emit('network.governance.vote', {
                'peer_id': peer_id or voter,
                'voter': voter,
                'proposal_id': proposal_id,
                'vote': vote_value,
                'timestamp': int(time.time()),
                'type': 'vote',
            })

            self._counts['governance'] += 1
            self._record_hourly_activity('governance')
        except Exception as e:
            logger.debug(f"Failed to handle governance vote: {e}")

    def _on_governance_proposal(self, proposal) -> None:
        """Handle governance proposal event."""
        try:
            peer_id = proposal.get('peer_id', '') if isinstance(proposal, dict) else getattr(proposal, 'peer_id', '')
            proposer = proposal.get('proposer', '') if isinstance(proposal, dict) else getattr(proposal, 'proposer', '')
            proposal_id = proposal.get('proposal_id', '') if isinstance(proposal, dict) else getattr(proposal, 'proposal_id', '')
            title = proposal.get('title', '') if isinstance(proposal, dict) else getattr(proposal, 'title', '')

            self._emit('network.governance.proposal', {
                'peer_id': peer_id or proposer,
                'proposer': proposer,
                'proposal_id': proposal_id,
                'title': title,
                'timestamp': int(time.time()),
                'type': 'proposal',
            })

            self._counts['governance'] += 1
            self._record_hourly_activity('governance')
        except Exception as e:
            logger.debug(f"Failed to handle governance proposal: {e}")

    def _on_governance_update(self, data: Any) -> None:
        """Handle generic governance update event."""
        try:
            peer_id = data.get('peer_id', '') if isinstance(data, dict) else getattr(data, 'peer_id', '')

            self._emit('network.governance', {
                'peer_id': peer_id,
                'data': data if isinstance(data, dict) else str(data),
                'timestamp': int(time.time()),
                'type': 'update',
            })

            self._counts['governance'] += 1
            self._record_hourly_activity('governance')
        except Exception as e:
            logger.debug(f"Failed to handle governance update: {e}")

    def _on_peer_connection(self, peer_id: str, connected: bool) -> None:
        """Handle peer connection change event."""
        try:
            event = 'peer.connect' if connected else 'peer.disconnect'
            self._emit(event, {
                'peer_id': peer_id,
                'timestamp': int(time.time()),
            })
        except Exception as e:
            logger.debug(f"Failed to handle peer connection: {e}")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _emit(self, event: str, data: dict) -> None:
        """Emit event to WebSocket and store for polling."""
        # Store event for polling (IPC fallback)
        self._store_event(event, data)

        try:
            from web.app import sendToUI
            sendToUI(event, data)
            # Log at INFO level so we can see it in container logs
            logger.info(f"Emitted WebSocket event: {event}")
        except ImportError:
            logger.warning(f"Cannot emit {event}: web.app not available")
        except Exception as e:
            logger.warning(f"Failed to emit {event}: {e}")

    def _store_event(self, event: str, data: dict) -> None:
        """Store event for polling retrieval."""
        event_record = {
            'event': event,
            'data': data,
            'timestamp': time.time(),
        }
        self._recent_events.append(event_record)
        # Keep only the most recent events
        if len(self._recent_events) > self._max_recent_events:
            self._recent_events = self._recent_events[-self._max_recent_events:]

    def get_recent_events(self, limit: int = 50, since: float = None) -> list:
        """Get recent events for polling.

        Args:
            limit: Maximum number of events to return
            since: Only return events after this timestamp

        Returns:
            List of recent events, newest first
        """
        events = self._recent_events
        if since:
            events = [e for e in events if e['timestamp'] > since]
        # Return newest first, limited
        return list(reversed(events[-limit:]))

    def _record_hourly_activity(self, event_type: str) -> None:
        """Record activity for hourly chart data."""
        hour = int(time.time()) // 3600

        if hour not in self._hourly_activity:
            self._hourly_activity[hour] = {
                'predictions': 0,
                'observations': 0,
                'heartbeats': 0,
                'consensus': 0,
                'governance': 0,
            }

        self._hourly_activity[hour][event_type] = self._hourly_activity[hour].get(event_type, 0) + 1

        # Clean up old data (keep last 25 hours)
        cutoff = hour - 25
        self._hourly_activity = {
            h: v for h, v in self._hourly_activity.items()
            if h > cutoff
        }

    def get_hourly_activity(self, hours: int = 24) -> Dict[str, list]:
        """
        Get hourly activity data for charts.

        Args:
            hours: Number of hours to return

        Returns:
            Dict with 'labels', 'predictions', 'observations', etc.
        """
        now_hour = int(time.time()) // 3600

        labels = []
        predictions = []
        observations = []
        heartbeats = []
        consensus = []
        governance = []

        for i in range(hours - 1, -1, -1):
            hour = now_hour - i
            hour_data = self._hourly_activity.get(hour, {})

            from datetime import datetime
            dt = datetime.fromtimestamp(hour * 3600)
            labels.append(f"{dt.hour:02d}:00")

            predictions.append(hour_data.get('predictions', 0))
            observations.append(hour_data.get('observations', 0))
            heartbeats.append(hour_data.get('heartbeats', 0))
            consensus.append(hour_data.get('consensus', 0))
            governance.append(hour_data.get('governance', 0))

        return {
            'labels': labels,
            'predictions': predictions,
            'observations': observations,
            'heartbeats': heartbeats,
            'consensus': consensus,
            'governance': governance,
        }

    def get_counts(self) -> Dict[str, int]:
        """Get current event counts."""
        return dict(self._counts)

    async def load_persisted_stats(self) -> bool:
        """
        Load persisted activity stats from storage.

        Call this on startup to restore counts that survived restart.
        Adds persisted counts to current session counts.

        Returns:
            True if stats were loaded successfully
        """
        if self._loaded_persisted_stats:
            logger.debug("Persisted stats already loaded")
            return True

        if not self._uptime_tracker:
            logger.debug("No uptime tracker available for persisted stats")
            return False

        try:
            # Check if uptime tracker has load_persisted_stats method
            if hasattr(self._uptime_tracker, 'load_persisted_stats'):
                persisted = await self._uptime_tracker.load_persisted_stats()

                # Add persisted counts to current session
                if persisted:
                    # Map storage field names to bridge count names
                    self._counts['heartbeats'] += persisted.get('heartbeats_sent', 0)
                    self._counts['heartbeats'] += persisted.get('heartbeats_received', 0)
                    self._counts['predictions'] += persisted.get('predictions', 0)
                    self._counts['observations'] += persisted.get('observations', 0)
                    self._counts['consensus_votes'] += persisted.get('consensus_votes', 0)
                    self._counts['governance'] += persisted.get('governance_votes', 0)

                    logger.info(f"Loaded persisted stats: heartbeats={self._counts['heartbeats']}, "
                               f"predictions={self._counts['predictions']}, "
                               f"observations={self._counts['observations']}")

                self._loaded_persisted_stats = True
                return True

        except Exception as e:
            logger.warning(f"Failed to load persisted stats: {e}")

        return False

    def load_persisted_stats_sync(self) -> bool:
        """
        Synchronous wrapper to load persisted stats.

        Spawns the async load as a background task if trio is available.
        """
        if self._loaded_persisted_stats:
            return True

        if not self._uptime_tracker:
            return False

        try:
            # Try to get storage directly and load synchronously
            if hasattr(self._uptime_tracker, 'get_activity_storage'):
                storage = self._uptime_tracker.get_activity_storage()
                if storage and hasattr(storage, '_disk'):
                    # Load from disk backend directly (sync)
                    import asyncio
                    loop = asyncio.new_event_loop()
                    try:
                        persisted = loop.run_until_complete(
                            self._uptime_tracker.load_persisted_stats()
                        )
                        if persisted:
                            self._counts['heartbeats'] += persisted.get('heartbeats_sent', 0)
                            self._counts['heartbeats'] += persisted.get('heartbeats_received', 0)
                            self._counts['predictions'] += persisted.get('predictions', 0)
                            self._counts['observations'] += persisted.get('observations', 0)
                            self._counts['consensus_votes'] += persisted.get('consensus_votes', 0)
                            self._counts['governance'] += persisted.get('governance_votes', 0)
                            self._loaded_persisted_stats = True
                            logger.info(f"Loaded persisted stats (sync): {self._counts}")
                            return True
                    finally:
                        loop.close()
        except Exception as e:
            logger.warning(f"Failed to load persisted stats (sync): {e}")

        return False

    def reset_loaded_flag(self) -> None:
        """Reset the loaded flag to allow reloading stats (e.g., after logout)."""
        self._loaded_persisted_stats = False


# Global bridge instance
_bridge: Optional[P2PWebSocketBridge] = None


def get_bridge() -> P2PWebSocketBridge:
    """Get or create the global bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = P2PWebSocketBridge()
    return _bridge


def start_bridge(
    peers=None,
    prediction_protocol=None,
    oracle_network=None,
    consensus_manager=None,
    uptime_tracker=None,
    lending_manager=None,
    delegation_manager=None,
    governance_manager=None,
) -> P2PWebSocketBridge:
    """
    Start the P2P WebSocket bridge.

    Call this after P2P protocols are initialized in start.py.
    """
    bridge = get_bridge()
    bridge.start(
        peers=peers,
        prediction_protocol=prediction_protocol,
        oracle_network=oracle_network,
        consensus_manager=consensus_manager,
        uptime_tracker=uptime_tracker,
        lending_manager=lending_manager,
        delegation_manager=delegation_manager,
        governance_manager=governance_manager,
    )
    return bridge
