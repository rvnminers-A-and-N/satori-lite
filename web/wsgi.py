"""
WSGI entry point for Flask application.

This module is kept for backward compatibility and provides helper functions
for web routes to access P2P functionality.

With waitress (in-process WSGI server), web routes can directly access
the StartupDag singleton and its P2P state. The functions here provide
convenient wrappers for common operations.
"""
import logging

from web.app import create_app, get_socketio

logger = logging.getLogger(__name__)

# Create the Flask application
app = create_app()

# Get the SocketIO instance
socketio = get_socketio()


def get_web_identity():
    """Get the Evrmore identity from StartupDag."""
    try:
        from satorineuron.init import start
        startup = start.getStart() if hasattr(start, 'getStart') else None
        if startup:
            return getattr(startup, 'identity', None)
    except Exception:
        pass
    return None


def get_web_p2p_peers():
    """Get P2P peers instance from StartupDag."""
    try:
        from satorineuron.init import start
        startup = start.getStart() if hasattr(start, 'getStart') else None
        if startup:
            return getattr(startup, '_p2p_peers', None)
    except Exception:
        pass
    return None


def wait_for_p2p_init(timeout=5):
    """Wait for P2P initialization to complete.

    Args:
        timeout: Max seconds to wait

    Returns:
        True if P2P is ready
    """
    try:
        from satorineuron.init import start
        startup = start.getStart() if hasattr(start, 'getStart') else None
        if startup and hasattr(startup, '_p2p_ready'):
            return startup._p2p_ready.wait(timeout=timeout)
    except Exception:
        pass
    return False


def get_trio_token():
    """Get the trio token for cross-thread async calls."""
    try:
        from satorineuron.init import start
        startup = start.getStart() if hasattr(start, 'getStart') else None
        if startup:
            return getattr(startup, '_trio_token', None)
    except Exception:
        pass
    return None


def run_in_p2p_context(async_fn, *args, **kwargs):
    """Run an async function in the P2P trio context.

    This allows Flask routes to call async P2P methods safely by
    running them in the persistent trio thread.

    Args:
        async_fn: Async function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the async function

    Raises:
        RuntimeError: If P2P trio context is not available
    """
    import trio

    token = get_trio_token()
    if token is None:
        raise RuntimeError("P2P trio context not available")

    return trio.from_thread.run(async_fn, *args, **kwargs, trio_token=token)
