"""
WSGI entry point for gunicorn production deployment.

This module creates the Flask app and SocketIO instance for use with gunicorn.

Run with threading mode (compatible with trio/P2P):
    gunicorn -k gthread -w 1 --threads 4 web.wsgi:app

The threading mode with simple-websocket provides WebSocket support
without requiring gevent/eventlet, which would conflict with trio
(used by libp2p for P2P networking).

P2P is initialized directly in this worker process (not through StartupDag)
so web routes have immediate access to identity and P2P peers.
"""
import os
import threading
import logging

from web.app import create_app, get_socketio

logger = logging.getLogger(__name__)

# Create the Flask application
app = create_app()

# Get the SocketIO instance
socketio = get_socketio()

# Module-level P2P state for web routes to access
_web_identity = None
_web_p2p_peers = None
_p2p_init_complete = threading.Event()
_p2p_init_error = None
_trio_token = None  # Token for cross-thread async calls


def get_web_identity():
    """Get the Evrmore identity for web routes."""
    return _web_identity


def get_web_p2p_peers():
    """Get the P2P peers instance for web routes."""
    logger.debug(f"get_web_p2p_peers called, returning: {_web_p2p_peers is not None}")
    return _web_p2p_peers


def wait_for_p2p_init(timeout=5):
    """Wait for P2P initialization to complete.

    Args:
        timeout: Max seconds to wait

    Returns:
        True if P2P initialized successfully
    """
    return _p2p_init_complete.wait(timeout=timeout)


def get_trio_token():
    """Get the trio token for cross-thread async calls."""
    return _trio_token


def run_in_p2p_context(async_fn, *args, **kwargs):
    """Run an async function in the P2P trio context.

    This allows Flask routes to call async P2P methods safely.

    Args:
        async_fn: Async function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the async function

    Raises:
        RuntimeError: If P2P context is not available
    """
    import trio

    if _trio_token is None:
        raise RuntimeError("P2P trio context not available")

    return trio.from_thread.run(async_fn, *args, **kwargs, trio_token=_trio_token)


def _initialize_p2p_for_web():
    """Initialize P2P directly in the gunicorn worker.

    Creates identity and P2P peers directly without relying on StartupDag
    state, which may not be available in the subprocess.

    Runs trio in a persistent loop to keep P2P alive.
    """
    global _web_identity, _web_p2p_peers, _p2p_init_error

    try:
        from satorineuron import config
        from satorineuron.init import start

        # Get networking mode
        mode = start._get_networking_mode() if hasattr(start, '_get_networking_mode') else 'central'
        logger.info(f"Web worker: Networking mode is '{mode}'")

        if mode == 'central':
            logger.info("Web worker: Central mode, skipping P2P initialization")
            _p2p_init_complete.set()
            return

        # Create identity directly from wallet file
        logger.info("Web worker: Creating identity from wallet file...")
        from satorilib.wallet.evrmore.identity import EvrmoreIdentity
        wallet_path = config.walletPath('wallet.yaml')
        _web_identity = EvrmoreIdentity(wallet_path)

        if _web_identity and hasattr(_web_identity, 'address'):
            logger.info(f"Web worker: Identity loaded - {_web_identity.address[:16]}...")
        else:
            logger.warning("Web worker: Identity loaded but no address available")
            _p2p_init_complete.set()
            return

        # Initialize P2P peers directly using trio - keep loop running
        logger.info("Web worker: Starting P2P peers...")

        import trio
        from satorip2p import Peers

        async def _run_p2p():
            """Run P2P in a persistent trio loop."""
            global _web_p2p_peers, _trio_token

            try:
                # Store the trio token for cross-thread async calls
                _trio_token = trio.lowlevel.current_trio_token()
                logger.info("Web worker: Trio token stored for cross-thread calls")

                _web_p2p_peers = Peers(
                    identity=_web_identity,
                    listen_port=config.get().get('p2p port', 24600),
                )
                await _web_p2p_peers.start()
                logger.info(f"Web worker: P2P peers started - peer_id={_web_p2p_peers.peer_id}")
                logger.info(f"Web worker: _web_p2p_peers global set to {_web_p2p_peers}")

                # Initialize UptimeTracker for the web worker's P2P peers
                # (main process has its own, but gunicorn is a separate process)
                _web_uptime_tracker = None
                try:
                    from satorip2p.protocol.uptime import UptimeTracker
                    _web_uptime_tracker = UptimeTracker(
                        peers=_web_p2p_peers,
                        wallet=_web_identity,
                    )
                    await _web_uptime_tracker.start()
                    _web_p2p_peers.spawn_background_task(_web_uptime_tracker.run_heartbeat_loop)
                    logger.info("Web worker: Uptime tracker initialized with heartbeat loop")

                    # Wire uptime tracker to P2P bridge for real-time UI updates
                    try:
                        from web.p2p_bridge import get_bridge
                        bridge = get_bridge()
                        bridge.wire_protocol('uptime_tracker', _web_uptime_tracker)
                        logger.info("Web worker: Uptime tracker wired to P2P bridge")
                    except Exception as e:
                        logger.warning(f"Web worker: Failed to wire uptime tracker to bridge: {e}")
                except Exception as e:
                    logger.warning(f"Web worker: Failed to initialize uptime tracker: {e}")

                # Also update StartupDag singleton so other code can access it
                try:
                    startup = start.getStart() if hasattr(start, 'getStart') else None
                    if startup:
                        startup._p2p_peers = _web_p2p_peers
                        startup._uptime_tracker = _web_uptime_tracker
                        if not hasattr(startup, 'identity') or startup.identity is None:
                            startup.identity = _web_identity
                        logger.info("Web worker: Updated StartupDag with P2P state")

                        # Initialize Storage Manager for web worker
                        try:
                            from satorip2p.protocol.storage import StorageManager, DeferredRewardsStorage, AlertStorage
                            from pathlib import Path
                            storage_dir = Path.home() / '.satori' / 'storage'
                            startup._storage_manager = StorageManager(storage_dir=storage_dir)
                            startup._deferred_rewards_storage = DeferredRewardsStorage(
                                storage_dir=storage_dir,
                                peers=_web_p2p_peers,
                            )
                            startup._alert_storage = AlertStorage(
                                storage_dir=storage_dir,
                                peers=_web_p2p_peers,
                            )
                            logger.info("Web worker: Storage manager initialized")
                        except Exception as e:
                            logger.warning(f"Web worker: Failed to initialize storage manager: {e}")

                        # Initialize Bandwidth Tracker for web worker
                        try:
                            from satorip2p.protocol.bandwidth import create_qos_manager
                            startup._bandwidth_tracker, startup._qos_manager = create_qos_manager()
                            if hasattr(_web_p2p_peers, 'set_bandwidth_tracker'):
                                _web_p2p_peers.set_bandwidth_tracker(startup._bandwidth_tracker)
                            logger.info("Web worker: Bandwidth tracker initialized")
                        except Exception as e:
                            logger.warning(f"Web worker: Failed to initialize bandwidth tracker: {e}")

                except Exception as e:
                    logger.warning(f"Web worker: Could not update StartupDag: {e}")

                # Signal that P2P is ready
                _p2p_init_complete.set()
                logger.info("Web worker: P2P initialization complete")

                # Run the P2P network (starts protocols like Ping, Identify, etc.)
                # This blocks indefinitely and keeps the P2P network alive
                await _web_p2p_peers.run_forever()

            except Exception as e:
                logger.error(f"Web worker: P2P error: {e}")
                _p2p_init_complete.set()
                raise

        # Run trio in this thread - it will block indefinitely
        try:
            trio.run(_run_p2p)
        except Exception as e:
            logger.error(f"Web worker: Trio loop exited: {e}")

    except ImportError as e:
        _p2p_init_error = f"Import error: {e}"
        logger.warning(f"Web worker: P2P not available - {e}")
        _p2p_init_complete.set()
    except Exception as e:
        _p2p_init_error = str(e)
        logger.error(f"Web worker: P2P initialization failed: {e}")
        _p2p_init_complete.set()


# Initialize P2P in a background thread when gunicorn worker starts
# This runs once per worker process - the trio loop runs indefinitely
_p2p_init_thread = threading.Thread(target=_initialize_p2p_for_web, daemon=True)
_p2p_init_thread.start()
