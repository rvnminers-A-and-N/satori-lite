"""
WSGI entry point for gunicorn production deployment.

This module creates the Flask app and provides helper functions for web routes
to access P2P functionality via the internal IPC API.

The gunicorn subprocess cannot access the main process's memory, so P2P
operations are performed by calling the IPC API running on 127.0.0.1:24602.

Architecture:
- Main process: P2P networking + IPC API server (port 24602)
- Gunicorn subprocess: Web UI (port 24601) - calls IPC API for P2P operations
"""
import logging
import requests

from web.app import create_app, get_socketio

logger = logging.getLogger(__name__)

# Create the Flask application
app = create_app()

# Get the SocketIO instance
socketio = get_socketio()

# IPC API base URL (internal, same container)
IPC_API_URL = 'http://127.0.0.1:24602'
IPC_TIMEOUT = 5  # seconds


def _ipc_get(endpoint: str):
    """Make a GET request to the IPC API."""
    try:
        resp = requests.get(f'{IPC_API_URL}{endpoint}', timeout=IPC_TIMEOUT)
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.debug(f"IPC API request failed: {e}")
        return None


def _ipc_post(endpoint: str, json_data: dict = None):
    """Make a POST request to the IPC API."""
    try:
        resp = requests.post(
            f'{IPC_API_URL}{endpoint}',
            json=json_data or {},
            timeout=IPC_TIMEOUT
        )
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.debug(f"IPC API request failed: {e}")
        return None


class PeerIdentity:
    """Wrapper class for peer identity data from IPC API.

    Makes dict data accessible via attribute access for compatibility
    with routes.py that uses getattr(ident, 'field', default).
    """
    def __init__(self, data: dict):
        self.peer_id = data.get('peer_id', '')
        self.evrmore_address = data.get('evrmore_address', '')
        self.roles = data.get('roles', [])
        self.protocols = data.get('protocols', [])
        self.agent_version = data.get('agent_version', '')
        self.timestamp = data.get('timestamp', 0)
        self.listen_addresses = data.get('listen_addresses', [])
        self.protocol_version = data.get('protocol_version', None)
        self.observed_addr = data.get('observed_addr', None)
        self.capabilities = data.get('capabilities', [])


def get_web_identity():
    """Get identity information via IPC API."""
    result = _ipc_get('/p2p/identity')
    if result:
        # Return a simple object with address attribute
        class Identity:
            def __init__(self, data):
                self.address = data.get('address')
                self.peer_id = data.get('peer_id')
        return Identity(result)
    return None


def get_web_p2p_peers():
    """Get P2P peers instance proxy via IPC API.

    Returns a proxy object that provides P2P-like interface
    but actually calls the IPC API.
    """
    status = _ipc_get('/p2p/status')
    if status and status.get('available'):
        return P2PProxy()
    return None


def wait_for_p2p_init(timeout=5):
    """Check if P2P is initialized via IPC API."""
    try:
        status = _ipc_get('/p2p/status')
        return status is not None and status.get('available', False)
    except:
        return False


def get_web_uptime():
    """Get uptime information via IPC API.

    Returns dict with uptime stats that can be used directly by routes.py.
    """
    result = _ipc_get('/p2p/uptime')
    return result if result else {
        'success': False,
        'streak_days': 0,
        'heartbeats_sent': 0,
        'heartbeats_received': 0,
        'current_round': '--',
        'is_relay_qualified': False,
        'uptime_percentage': 0.0,
    }


class P2PProxy:
    """Proxy class that provides P2P-like interface via IPC API.

    This allows web routes to call methods like ping_peer() without
    knowing they're going through HTTP to the main process.
    """

    def __init__(self):
        self._status = None
        self._full_status = None
        self._refresh_status()

    def _refresh_status(self):
        self._status = _ipc_get('/p2p/status') or {}
        self._full_status = _ipc_get('/p2p/full-status') or {}

    @property
    def peer_id(self):
        if not self._status:
            self._refresh_status()
        return self._status.get('peer_id')

    @property
    def connected_peers(self):
        """Return count of connected peers (int, matching satorip2p)."""
        result = _ipc_get('/p2p/status')
        return result.get('connected_count', 0) if result else 0

    @property
    def public_key(self):
        """Get node's public key."""
        result = _ipc_get('/p2p/full-status')
        return result.get('public_key') if result else None

    @property
    def public_addresses(self):
        """Get public multiaddresses."""
        result = _ipc_get('/p2p/multiaddrs')
        return result.get('multiaddrs', []) if result else []

    @property
    def nat_type(self):
        """Get NAT type."""
        result = _ipc_get('/p2p/full-status')
        return result.get('nat_type', 'unknown') if result else 'unknown'

    @property
    def is_connected(self):
        """Check if connected to any peers."""
        result = _ipc_get('/p2p/status')
        return result.get('connected_count', 0) > 0 if result else False

    @property
    def is_relay(self):
        """Check if acting as relay."""
        result = _ipc_get('/p2p/full-status')
        return result.get('is_relay', False) if result else False

    @property
    def evrmore_address(self):
        """Get Evrmore address."""
        result = _ipc_get('/p2p/identity')
        return result.get('address') if result else None

    def get_status(self):
        """Get full P2P status."""
        return _ipc_get('/p2p/status')

    def get_multiaddrs(self):
        """Get listen multiaddresses."""
        result = _ipc_get('/p2p/multiaddrs')
        return result.get('multiaddrs', []) if result else []

    def get_peer_count(self):
        """Get number of connected peers."""
        result = _ipc_get('/p2p/status')
        return result.get('connected_count', 0) if result else 0

    def get_connected_peers(self):
        """Get list of connected peer IDs."""
        result = _ipc_get('/p2p/peers')
        if result and 'peers' in result:
            return [p['peer_id'] for p in result['peers']]
        return []

    def get_peers(self):
        """Get list of connected peers."""
        return self.get_connected_peers()

    def get_all_peer_latencies(self):
        """Get latencies for all peers."""
        result = _ipc_get('/p2p/latencies')
        return result.get('latencies', {}) if result else {}

    def get_network_avg_latency(self):
        """Get average network latency."""
        result = _ipc_get('/p2p/latencies')
        return result.get('avg_latency') if result else None

    def get_pubsub_debug(self):
        """Get pubsub debug info."""
        result = _ipc_get('/p2p/pubsub/debug')
        return result if result else {}

    def get_rendezvous_status(self):
        """Get rendezvous/bootstrap status."""
        result = _ipc_get('/p2p/rendezvous')
        return result if result else {}

    def get_peers_by_role(self, role):
        """Get peers filtered by role."""
        result = _ipc_get(f'/p2p/peers/by-role/{role}')
        return result.get('peers', []) if result else []

    def get_connected_signers(self):
        """Get connected signers."""
        return self.get_peers_by_role('signer')

    def forget_peer(self, peer_id):
        """Forget a peer."""
        result = _ipc_post(f'/p2p/forget/{peer_id}')
        return result.get('success', False) if result else False

    def ping_peer(self, peer_id, count=3):
        """Ping a peer (synchronous wrapper)."""
        result = _ipc_post(f'/p2p/ping/{peer_id}', {'count': count})
        if result and result.get('success'):
            # Convert ms back to seconds for compatibility
            return [l / 1000 for l in result.get('latencies_ms', [])]
        return None

    def connect(self, multiaddr):
        """Connect to a peer by multiaddr (synchronous wrapper)."""
        result = _ipc_post('/p2p/connect', {'multiaddr': multiaddr})
        return result.get('success', False) if result else False

    async def connect_peer(self, multiaddr, timeout=None):
        """Connect to a peer by multiaddr (async for compatibility)."""
        data = {'multiaddr': multiaddr}
        if timeout:
            data['timeout'] = timeout
        result = _ipc_post('/p2p/connect', data)
        return result.get('success', False) if result else False

    def disconnect(self, peer_id):
        """Disconnect from a peer (synchronous wrapper)."""
        result = _ipc_post(f'/p2p/disconnect/{peer_id}')
        return result.get('success', False) if result else False

    async def disconnect_peer(self, peer_id):
        """Disconnect from a peer (async for compatibility)."""
        result = _ipc_post(f'/p2p/disconnect/{peer_id}')
        return result.get('success', False) if result else False

    def get_known_peer_identities(self):
        """Get known peer identities from the Identify protocol.

        Returns:
            Dict mapping peer_id to PeerIdentity objects (compatible with getattr())
        """
        result = _ipc_get('/p2p/identify/known')
        if result and 'identities' in result:
            # Wrap each identity dict in a PeerIdentity object for attribute access
            return {
                peer_id: PeerIdentity(data)
                for peer_id, data in result['identities'].items()
            }
        return {}

    def announce_identity_sync(self):
        """Announce our identity to the network (synchronous)."""
        result = _ipc_post('/p2p/identify/announce')
        return result.get('success', False) if result else False

    async def announce_identity(self):
        """Announce our identity to the network (async wrapper for compatibility)."""
        return self.announce_identity_sync()

    def discover_peers(self, stream_id=None):
        """Discover peers on the network (synchronous wrapper).

        Args:
            stream_id: Optional stream ID to filter by publishers
        Returns:
            List of discovered peer IDs
        """
        data = {}
        if stream_id:
            data['stream_id'] = stream_id
        result = _ipc_post('/p2p/discover', data)
        if result and result.get('success'):
            return result.get('discovered_peers', [])
        return []

    async def discover_peers_async(self, stream_id=None):
        """Discover peers on the network (async wrapper for compatibility)."""
        return self.discover_peers(stream_id)

    def get_network_map(self):
        """Get the network topology map."""
        result = _ipc_get('/p2p/network-map')
        return result if result else {}

    def get_my_subscriptions(self):
        """Get our pubsub subscriptions."""
        result = _ipc_get('/p2p/subscriptions')
        return result.get('subscriptions', []) if result else []

    def get_my_publications(self):
        """Get our pubsub publications."""
        result = _ipc_get('/p2p/subscriptions')
        return result.get('publications', []) if result else []

    def get_subscription_map(self):
        """Get the subscription map showing who subscribes to what."""
        result = _ipc_get('/p2p/subscription-map')
        return result if result else {}

    def get_publishers(self, stream_id):
        """Get publishers for a specific stream."""
        result = _ipc_get(f'/p2p/publishers/{stream_id}')
        return result.get('publishers', []) if result else []

    def get_subscribers(self, stream_id):
        """Get subscribers for a specific stream."""
        result = _ipc_get(f'/p2p/subscribers/{stream_id}')
        return result.get('subscribers', []) if result else []

    def get_peer_subscriptions(self, peer_id):
        """Get what streams a specific peer subscribes to."""
        result = _ipc_get(f'/p2p/peer-subscriptions/{peer_id}')
        return result.get('subscriptions', []) if result else []

    def discover_publishers(self, stream_id):
        """Discover publishers for a specific stream via DHT (synchronous wrapper)."""
        result = _ipc_post(f'/p2p/discover-publishers/{stream_id}')
        if result and result.get('success'):
            return result.get('publishers', [])
        return []

    async def discover_publishers_async(self, stream_id):
        """Discover publishers for a specific stream (async wrapper for compatibility)."""
        return self.discover_publishers(stream_id)

    def check_connection_changes(self):
        """Check for recent connection changes."""
        result = _ipc_get('/p2p/connection-changes')
        if result and 'changes' in result:
            return [(c['peer_id'], c['connected']) for c in result['changes']]
        return []

    def get_uptime(self):
        """Get uptime tracker information via IPC API."""
        result = _ipc_get('/p2p/uptime')
        return result if result else {
            'success': False,
            'streak_days': 0,
            'heartbeats_sent': 0,
            'heartbeats_received': 0,
            'current_round': '--',
            'is_relay_qualified': False,
            'uptime_percentage': 0.0,
        }

    # Protocol enable flags (read from full status)
    @property
    def enable_pubsub(self):
        result = _ipc_get('/p2p/full-status')
        return result.get('enable_pubsub', True) if result else True

    @property
    def enable_dht(self):
        result = _ipc_get('/p2p/full-status')
        return result.get('enable_dht', True) if result else True

    @property
    def enable_ping(self):
        result = _ipc_get('/p2p/full-status')
        return result.get('enable_ping', True) if result else True

    @property
    def enable_identify(self):
        result = _ipc_get('/p2p/full-status')
        return result.get('enable_identify', True) if result else True

    @property
    def enable_relay(self):
        result = _ipc_get('/p2p/full-status')
        return result.get('enable_relay', True) if result else True

    @property
    def enable_mdns(self):
        result = _ipc_get('/p2p/full-status')
        return result.get('enable_mdns', True) if result else True

    @property
    def enable_rendezvous(self):
        result = _ipc_get('/p2p/full-status')
        return result.get('enable_rendezvous', False) if result else False

    @property
    def enable_upnp(self):
        result = _ipc_get('/p2p/full-status')
        return result.get('enable_upnp', True) if result else True


def run_in_p2p_context(async_fn, *args, **kwargs):
    """Run async function via IPC - not supported.

    The IPC API handles async operations internally.
    Web routes should use the P2PProxy methods instead.
    """
    raise RuntimeError(
        "run_in_p2p_context not available with IPC architecture. "
        "Use P2PProxy methods (get_web_p2p_peers()) instead."
    )
