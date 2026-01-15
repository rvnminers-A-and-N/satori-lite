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

    Since gunicorn runs as a subprocess, it cannot access the main process's
    P2P state. We must initialize our own Peers instance here.

    The previous "Pubsub _manager" race condition was caused by accessing
    Pubsub before Peers.start() fully completed. We now wait for start()
    to complete before signaling ready, and DON'T call run_forever()
    (which caused the race).
    """
    global _web_identity, _web_p2p_peers, _p2p_init_error, _trio_token

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

        # Load identity from wallet
        logger.info("Web worker: Loading identity from wallet file...")
        try:
            from satorilib.wallet.evrmore.identity import EvrmoreIdentity
            wallet_path = config.walletPath('wallet.yaml')
            _web_identity = EvrmoreIdentity(wallet_path)

            if _web_identity and hasattr(_web_identity, 'address'):
                logger.info(f"Web worker: Identity loaded - {_web_identity.address[:16]}...")
            else:
                logger.info("Web worker: Identity loaded (no address)")
                _p2p_init_complete.set()
                return
        except Exception as e:
            logger.warning(f"Web worker: Could not load identity: {e}")
            _p2p_init_complete.set()
            return

        # Initialize P2P peers in a persistent trio context
        logger.info("Web worker: Starting P2P peers...")

        import trio
        from satorip2p import Peers

        async def _run_p2p():
            """Run P2P - start and keep alive."""
            global _web_p2p_peers, _trio_token

            try:
                # Store trio token for cross-thread async calls
                _trio_token = trio.lowlevel.current_trio_token()
                logger.info("Web worker: Trio token stored")

                # Create and start Peers
                _web_p2p_peers = Peers(
                    identity=_web_identity,
                    listen_port=config.get().get('p2p port', 24600),
                )

                # Start and wait for full initialization
                await _web_p2p_peers.start()
                logger.info(f"Web worker: P2P started - peer_id={_web_p2p_peers.peer_id}")

                # Signal ready AFTER start completes (avoids race condition)
                _p2p_init_complete.set()
                logger.info("Web worker: P2P ready")

                # Keep trio context alive (but don't call run_forever which
                # starts additional protocols that may race)
                while True:
                    await trio.sleep(60)

            except Exception as e:
                logger.warning(f"Web worker: P2P error: {e}")
                _p2p_init_complete.set()

        def run_trio():
            try:
                trio.run(_run_p2p)
            except Exception as e:
                logger.warning(f"Web worker: Trio exited: {e}")

        # Run trio in background thread
        trio_thread = threading.Thread(target=run_trio, daemon=True)
        trio_thread.start()

        # Wait for P2P to be ready (with timeout)
        if not _p2p_init_complete.wait(timeout=60):
            logger.warning("Web worker: P2P init timeout")

    except ImportError as e:
        _p2p_init_error = f"Import error: {e}"
        logger.debug(f"Web worker: P2P not available - {e}")
        _p2p_init_complete.set()
    except Exception as e:
        _p2p_init_error = str(e)
        logger.warning(f"Web worker: P2P init failed: {e}")
        _p2p_init_complete.set()


# Initialize P2P in background thread when gunicorn worker starts
_p2p_init_thread = threading.Thread(target=_initialize_p2p_for_web, daemon=True)
_p2p_init_thread.start()
