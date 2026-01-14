"""
Satori-Lite Web UI Flask Application.

Minimal web interface for satori-lite functionality:
- Vault unlock/login
- Dashboard with balance, rewards, lending, and pool management
- Real-time P2P network updates via WebSocket
"""
import os
import logging
from datetime import timedelta
from typing import Any, Dict, Optional, Union
from flask import Flask
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

# Global SocketIO instance for use throughout the app
socketio: Optional[SocketIO] = None


def create_app(testing=False):
    """
    Create and configure the Flask application.

    Args:
        testing: If True, configure for testing mode

    Returns:
        Flask application instance
    """
    global socketio

    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['TESTING'] = testing

    # Session configuration - sessions expire after 24 hours
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

    # FIX: Set unique session cookie name per port to prevent cookie collision
    # between multiple instances running on localhost:24601, localhost:24602, etc.
    # Without this, browsers share the same session cookie across all ports,
    # causing JWT tokens and session data to be overwritten between instances.
    ui_port = os.environ.get('SATORI_UI_PORT', '24601')
    app.config['SESSION_COOKIE_NAME'] = f'satori_session_{ui_port}'

    # Server API URL (for proxying requests)
    from satorilib.config import get_api_url
    app.config['SATORI_API_URL'] = get_api_url()

    # Initialize SocketIO for real-time updates
    # Using threading mode for compatibility with trio (used by libp2p for P2P networking)
    # For production deployment, use gunicorn with gevent workers
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=False,
        engineio_logger=False
    )

    # Register socket event handlers
    register_socket_events(socketio)

    # Register routes
    from web.routes import register_routes
    register_routes(app)

    return app


def get_socketio() -> Optional[SocketIO]:
    """Get the global SocketIO instance."""
    return socketio


def sendToUI(
    event: str,
    data: Union[Dict, str, Any] = None,
    **kwargs
):
    """
    Send an event to all connected UI clients via WebSocket.

    This is the primary way to push real-time updates to the frontend.

    Args:
        event: Event name (e.g., 'heartbeat', 'prediction', 'observation')
        data: Data to send with the event
        **kwargs: Additional SocketIO emit options (room, namespace, etc.)

    Example:
        sendToUI('heartbeat', {'node_id': '...', 'timestamp': 1234567890})
        sendToUI('prediction', {'stream': 'BTC/USD', 'value': 42000})
    """
    global socketio
    if socketio is None:
        return

    try:
        # Convert data to JSON-safe format
        if data is not None and isinstance(data, str):
            import json
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = {'raw': data}

        socketio.emit(event, data, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to emit {event}: {e}")


def register_socket_events(sio: SocketIO):
    """Register WebSocket event handlers."""

    @sio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.debug("WebSocket client connected")
        emit('connected', {'status': 'ok', 'message': 'Connected to Satori-Lite'})

    @sio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.debug("WebSocket client disconnected")

    @sio.on('subscribe')
    def handle_subscribe(data):
        """
        Handle subscription requests from clients.

        Clients can subscribe to specific event types:
        - 'heartbeats': Network heartbeat events
        - 'predictions': Prediction events
        - 'observations': Observation events
        - 'consensus': Consensus phase changes
        - 'pool': Pool/lending updates
        - 'delegation': Delegation updates
        """
        event_type = data.get('type') if isinstance(data, dict) else data
        logger.debug(f"Client subscribed to: {event_type}")
        emit('subscribed', {'type': event_type, 'status': 'ok'})

    @sio.on('ping')
    def handle_ping():
        """Handle ping requests for connection testing."""
        emit('pong', {'timestamp': int(__import__('time').time())})


# For running directly
if __name__ == '__main__':
    app = create_app()
    if socketio:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
