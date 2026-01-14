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
    """Initialize P2P state for web worker.

    NOTE: The web worker no longer creates its own P2P Peers instance.
    The main neuron process (start.py) manages P2P networking in a
    persistent Trio context. Creating a second Peers instance in the
    web worker causes race conditions and "Pubsub _manager" errors.

    Instead, the web worker:
    1. Loads identity from wallet (for display purposes)
    2. Signals ready immediately (no P2P initialization)
    3. Web routes that need P2P data should use the main process's
       P2P state via StartupDag singleton or internal APIs

    This avoids duplicate P2P networking, reduces resource usage,
    and eliminates async/trio conflicts between processes.
    """
    global _web_identity, _web_p2p_peers, _p2p_init_error

    try:
        from satorineuron import config
        from satorineuron.init import start

        # Get networking mode
        mode = start._get_networking_mode() if hasattr(start, '_get_networking_mode') else 'central'
        logger.info(f"Web worker: Networking mode is '{mode}'")

        if mode == 'central':
            logger.info("Web worker: Central mode, no P2P needed")
            _p2p_init_complete.set()
            return

        # Load identity for display purposes (no P2P initialization)
        logger.info("Web worker: Loading identity from wallet file...")
        try:
            from satorilib.wallet.evrmore.identity import EvrmoreIdentity
            wallet_path = config.walletPath('wallet.yaml')
            _web_identity = EvrmoreIdentity(wallet_path)

            if _web_identity and hasattr(_web_identity, 'address'):
                logger.info(f"Web worker: Identity loaded - {_web_identity.address[:16]}...")
            else:
                logger.info("Web worker: Identity loaded (no address)")
        except Exception as e:
            logger.debug(f"Web worker: Could not load identity: {e}")

        # Try to get P2P peers reference from main process's StartupDag
        # (This works because gunicorn with threads shares memory within worker)
        try:
            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_p2p_peers') and startup._p2p_peers:
                _web_p2p_peers = startup._p2p_peers
                logger.info(f"Web worker: Using main process P2P peers - {_web_p2p_peers.peer_id if hasattr(_web_p2p_peers, 'peer_id') else 'unknown'}")
            else:
                logger.info("Web worker: Main process P2P peers not available yet (will be set later)")
        except Exception as e:
            logger.debug(f"Web worker: Could not get StartupDag P2P peers: {e}")

        # Signal ready - web worker doesn't run its own P2P
        _p2p_init_complete.set()
        logger.info("Web worker: Ready (P2P managed by main process)")

    except ImportError as e:
        _p2p_init_error = f"Import error: {e}"
        logger.debug(f"Web worker: P2P modules not available - {e}")
        _p2p_init_complete.set()
    except Exception as e:
        _p2p_init_error = str(e)
        logger.debug(f"Web worker: Initialization note: {e}")
        _p2p_init_complete.set()


# Initialize identity in a background thread when gunicorn worker starts
# NOTE: No longer starts P2P - main process handles that
_p2p_init_thread = threading.Thread(target=_initialize_p2p_for_web, daemon=True)
_p2p_init_thread.start()
