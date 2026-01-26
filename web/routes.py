"""
Satori-Lite Web UI Routes.

Handles all web routes for the minimal UI:
- Login/vault unlock
- Dashboard
- API proxy endpoints
"""
from functools import wraps
import time
import logging
import base64
import os
import json
import requests
import uuid
from threading import Lock
from cryptography.fernet import Fernet
from satorilib.config import get_api_url
from flask import (
    render_template,
    redirect,
    url_for,
    request,
    session,
    flash,
    jsonify,
    current_app
)

logger = logging.getLogger(__name__)

# Global vault reference (will be set by the application) - used by background processes
_startup_vault = None

# Global startup instance reference (will be set by the application)
_startup_instance = None

# Session-specific vaults - each browser session gets its own WalletManager
_session_vaults = {}
_vault_lock = Lock()

# JWT authentication lock to prevent concurrent login attempts
_auth_lock = Lock()

# Peer cache for Option D - show cached peers immediately on startup
_peer_cache = {}
_peer_cache_lock = Lock()
_PEER_CACHE_FILE = None  # Set during init


def _get_peer_cache_path():
    """Get the path to the peer cache file."""
    global _PEER_CACHE_FILE
    if _PEER_CACHE_FILE:
        return _PEER_CACHE_FILE
    try:
        from satorineuron import config
        import os
        cache_dir = os.path.join(os.path.expanduser('~'), '.satori', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        _PEER_CACHE_FILE = os.path.join(cache_dir, 'known_peers.json')
        return _PEER_CACHE_FILE
    except Exception:
        return None


def _load_peer_cache():
    """Load cached peers from disk."""
    global _peer_cache
    try:
        cache_path = _get_peer_cache_path()
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
                with _peer_cache_lock:
                    _peer_cache = data
                logger.debug(f"Loaded {len(data.get('known_peers', []))} cached peers")
                return data
    except Exception as e:
        logger.debug(f"Could not load peer cache: {e}")
    return {}


def _save_peer_cache(known_peers: list):
    """Save peers to cache file."""
    global _peer_cache
    try:
        cache_path = _get_peer_cache_path()
        if not cache_path:
            return

        data = {
            'timestamp': int(time.time()),
            'known_peers': known_peers,
        }

        with open(cache_path, 'w') as f:
            json.dump(data, f)

        with _peer_cache_lock:
            _peer_cache = data

        logger.debug(f"Saved {len(known_peers)} peers to cache")
    except Exception as e:
        logger.debug(f"Could not save peer cache: {e}")


def _get_cached_peers():
    """Get cached peers (from memory or disk)."""
    global _peer_cache
    with _peer_cache_lock:
        if _peer_cache:
            return _peer_cache
    return _load_peer_cache()


def check_vault_file_exists():
    """Check if vault.yaml file exists.

    Returns:
        bool: True if vault.yaml file exists, False otherwise
    """
    try:
        from satorineuron import config
        import os
        vault_path = config.walletPath('vault.yaml')
        return os.path.exists(vault_path)
    except Exception as e:
        logger.warning(f"Failed to check vault file: {e}")
        return False


def set_vault(vault):
    """Set the startup vault instance (used by background processes)."""
    global _startup_vault
    _startup_vault = vault


def get_vault():
    """Get the startup vault instance (for backward compatibility)."""
    return _startup_vault


def set_startup(startup):
    """Set the startup instance (used by web routes to access AI engine, etc)."""
    global _startup_instance
    _startup_instance = startup


def get_startup():
    """Get the startup instance."""
    return _startup_instance


def get_or_create_session_vault():
    """Get or create WalletManager for current session.

    Each browser session gets its own WalletManager instance.
    This provides true session isolation - one user's logout doesn't
    affect another user's session.

    Returns:
        WalletManager instance for this session
    """
    # Get or generate session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session.permanent = True  # Use PERMANENT_SESSION_LIFETIME (24 hours)
        logger.info(f"Generated new session ID: {session_id}")

    # Get or create vault for this session
    with _vault_lock:
        if session_id not in _session_vaults:
            try:
                from satorineuron.init.wallet import WalletManager
                from satorineuron import config

                # Create NEW WalletManager instance for this session
                # Don't auto-decrypt vault - user must provide password via login
                logger.info(f"Creating new WalletManager for session: {session_id}")
                _session_vaults[session_id] = WalletManager.create(
                    walletPath=config.walletPath('wallet.yaml'),
                    vaultPath=config.walletPath('vault.yaml'),
                    useConfigPassword=False  # Don't auto-decrypt with config password
                    # cachePath is optional and defaults to None
                )
            except Exception as e:
                logger.error(f"Failed to create session vault: {e}")
                return None

    return _session_vaults[session_id]


def cleanup_session_vault(session_id):
    """Clean up vault when session ends.

    Args:
        session_id: The session ID to cleanup
    """
    with _vault_lock:
        if session_id in _session_vaults:
            try:
                logger.info(f"Cleaning up session vault: {session_id}")
                _session_vaults[session_id].closeVault()
            except Exception as e:
                logger.warning(f"Error closing vault for session {session_id}: {e}")

            try:
                del _session_vaults[session_id]
            except Exception as e:
                logger.warning(f"Error deleting session vault {session_id}: {e}")


def encrypt_vault_password(password):
    """Encrypt vault password for storage in session.

    Args:
        password: Plaintext password

    Returns:
        tuple: (encrypted_password, session_key) both as strings
    """
    session_key = Fernet.generate_key()
    f = Fernet(session_key)
    encrypted_pw = f.encrypt(password.encode())
    return encrypted_pw.decode(), session_key.decode()


def get_p2p_state():
    """Get P2P identity and peers via IPC API.

    The gunicorn web worker runs as a subprocess and cannot access the main
    process's memory. Instead, we use the IPC API (127.0.0.1:24602) to get
    P2P state and perform P2P operations.

    Returns:
        tuple: (identity, peers) - peers is a P2PProxy object that calls IPC API
    """
    try:
        from web.wsgi import get_web_identity, get_web_p2p_peers, wait_for_p2p_init

        # Wait briefly for P2P IPC API to be available
        if not wait_for_p2p_init(timeout=2):
            logger.debug("get_p2p_state: IPC API not available")
            return None, None

        identity = get_web_identity()
        peers = get_web_p2p_peers()

        logger.debug(f"get_p2p_state: identity={identity is not None}, peers={peers is not None}")
        return identity, peers
    except Exception as e:
        logger.debug(f"get_p2p_state: failed: {e}")
        return None, None




def decrypt_vault_password_from_session():
    """Decrypt vault password from current session.

    Returns:
        str: Decrypted password, or None if not available
    """
    try:
        encrypted_pw = session.get('encrypted_vault_password')
        session_key = session.get('session_key')

        if not encrypted_pw or not session_key:
            return None

        f = Fernet(session_key.encode())
        password = f.decrypt(encrypted_pw.encode()).decode()
        return password
    except Exception as e:
        logger.error(f"Failed to decrypt vault password: {e}")
        return None


def ensure_peer_registered(app, wallet_manager, max_retries=3):
    """Ensure the peer is registered with the API server.

    This function should be called after vault is successfully unlocked.
    It will:
    1. Try to login (check if peer exists)
    2. If not found, register the peer
    3. Retry with delay if registration fails

    Args:
        app: Flask app instance for config access
        wallet_manager: The wallet manager instance with wallet and vault (both must be decrypted)
        max_retries: Maximum number of retry attempts

    Returns:
        dict with peer info if successful, None if failed
    """
    api_url = app.config.get('SATORI_API_URL', get_api_url())

    # Validate wallet exists
    if not wallet_manager.wallet:
        logger.error("Wallet not initialized - cannot register peer")
        return None

    # Validate vault exists and is decrypted
    if not wallet_manager.vault:
        logger.error("Vault not initialized - cannot register peer")
        return None

    if not hasattr(wallet_manager.vault, 'isDecrypted') or not wallet_manager.vault.isDecrypted:
        logger.error("Vault is not decrypted - cannot register peer")
        return None

    # Get wallet pubkey (identity) - REQUIRED
    wallet_pubkey = None
    if hasattr(wallet_manager.wallet, 'pubkey'):
        wallet_pubkey = wallet_manager.wallet.pubkey

    # Get vault pubkey - REQUIRED (not optional)
    vault_pubkey = None
    if hasattr(wallet_manager.vault, 'pubkey'):
        vault_pubkey = wallet_manager.vault.pubkey

    # Validate both pubkeys are present
    if not wallet_pubkey:
        logger.error("Wallet pubkey not available - cannot register peer")
        return None

    if not vault_pubkey:
        logger.error("Vault pubkey not available - cannot register peer")
        return None

    for attempt in range(max_retries):
        try:
            # Step 1: Try login first
            headers = {'wallet-pubkey': wallet_pubkey}
            resp = requests.get(
                f"{api_url}/api/v1/peer/login",
                headers=headers,
                timeout=10
            )

            if resp.status_code == 200:
                data = resp.json()
                if data.get('exists'):
                    logger.info(f"Peer already registered: peer_id={data.get('peer_id')}")
                    return data

            # Step 2: Peer not found, register
            # POST /api/v1/peer/register
            # Required headers:
            #   - wallet-pubkey: The identity wallet's public key (REQUIRED)
            #   - vault-pubkey: The vault wallet's public key (REQUIRED - must be decrypted)
            headers = {
                'wallet-pubkey': wallet_pubkey,
                'vault-pubkey': vault_pubkey
            }

            resp = requests.post(
                f"{api_url}/api/v1/peer/register",
                headers=headers,
                timeout=10
            )

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success'):
                    logger.info(f"Peer registered: peer_id={data.get('peer_id')}")
                    return data

            logger.warning(f"Peer registration attempt {attempt + 1} failed: {resp.text}")

        except requests.RequestException as e:
            logger.warning(f"Peer registration attempt {attempt + 1} error: {e}")

        # Wait before retry (except on last attempt)
        if attempt < max_retries - 1:
            time.sleep(5)

    logger.error("Failed to register peer after all retries")
    return None


def login_required(f):
    """Decorator to require login for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Helper to return appropriate response based on request type
        def auth_error_response(message="Authentication required"):
            # For API endpoints, return JSON error instead of redirect
            if request.path.startswith('/api/'):
                return jsonify({'error': message, 'authenticated': False}), 401
            # For regular pages, redirect to login
            if not check_vault_file_exists():
                return redirect(url_for('vault_setup'))
            return redirect(url_for('login'))

        # Ensure session ID exists for tracking
        if not session.get('session_id'):
            session['session_id'] = str(uuid.uuid4())
            session.permanent = True  # Use PERMANENT_SESSION_LIFETIME (24 hours)

        # Check if user is logged in via session flag
        if not session.get('vault_open'):
            return auth_error_response("Not logged in")

        # Validate that vault actually exists and is open (handles container restart)
        # Check _session_vaults directly - don't call get_or_create (which creates new vault)
        session_id = session.get('session_id')
        if session_id and session_id not in _session_vaults:
            # Vault doesn't exist in memory (container/worker restarted)
            # Try to automatically recover using encrypted password from session
            logger.info(f"Session vault missing for {session_id} - attempting auto-recovery")
            password = decrypt_vault_password_from_session()
            if password:
                try:
                    # Import WalletManager locally (same as get_or_create_session_vault)
                    from satorineuron.init.wallet import WalletManager
                    from satorineuron import config

                    # Re-create wallet manager and unlock vault
                    wallet_manager = WalletManager.create(
                        config.walletPath('wallet.yaml'),
                        config.walletPath('vault.yaml'),
                    )
                    if wallet_manager and wallet_manager.openVault(password=password):
                        _session_vaults[session_id] = wallet_manager
                        logger.info(f"Session vault auto-recovered for {session_id}")
                    else:
                        logger.warning(f"Failed to unlock vault during auto-recovery for {session_id}")
                        session.pop('vault_open', None)
                        return auth_error_response("Session expired - please log in again")
                except Exception as e:
                    logger.warning(f"Auto-recovery failed for {session_id}: {e}")
                    session.pop('vault_open', None)
                    return auth_error_response("Session expired - please log in again")
            else:
                # No encrypted password in session - require full re-login
                logger.info(f"No encrypted password for {session_id} - forcing re-login")
                session.pop('vault_open', None)
                session['logged_out'] = True  # Prevent auto-login
                return auth_error_response("Session expired")

        # Also validate vault is actually decrypted
        if session_id and session_id in _session_vaults:
            wallet_manager = _session_vaults[session_id]
            if not wallet_manager or not wallet_manager.vault or not wallet_manager.vault.isDecrypted:
                logger.info(f"Session vault not decrypted for {session_id} - forcing re-login")
                session.pop('vault_open', None)
                session['logged_out'] = True  # Prevent auto-login
                return auth_error_response("Vault locked")

        return f(*args, **kwargs)
    return decorated_function


def register_routes(app):
    """Register all routes with the Flask app."""

    @app.route('/')
    def index():
        """Redirect to dashboard if logged in, vault setup if needed, otherwise to login."""
        # Check if vault file exists first
        if not check_vault_file_exists():
            return redirect(url_for('vault_setup'))

        if session.get('vault_open'):
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Handle vault unlock/login."""
        # Check if vault file exists, redirect to setup if not
        if not check_vault_file_exists():
            return redirect(url_for('vault_setup'))

        # Auto-login on GET if config password exists
        if request.method == 'GET':
            # Skip auto-login if user just logged out explicitly
            if session.get('logged_out'):
                return render_template('login.html')

            from satorineuron import config
            config_password = config.get().get('vault password')

            if config_password:
                # Attempt auto-login with config password
                wallet_manager = get_or_create_session_vault()
                if wallet_manager:
                    try:
                        vault = wallet_manager.openVault(password=config_password, create=True)

                        if vault and vault.isDecrypted:
                            # Successfully auto-logged in
                            session['vault_open'] = True
                            session.pop('logged_out', None)  # Clear logout flag

                            # Encrypt and store password in session for future use
                            encrypted_pw, session_key = encrypt_vault_password(config_password)
                            session['encrypted_vault_password'] = encrypted_pw
                            session['session_key'] = session_key

                            # Register peer with API server (non-blocking)
                            try:
                                peer_info = ensure_peer_registered(
                                    current_app, wallet_manager, max_retries=1)
                                if peer_info:
                                    session['peer_id'] = peer_info.get('peer_id')
                                    logger.info(f"Auto-login successful, peer registered: {peer_info}")
                            except Exception as e:
                                logger.warning(f"Auto-login peer registration warning: {e}")

                            # Auto-login successful - redirect to dashboard
                            logger.info("Auto-login successful with config password")
                            return redirect(url_for('dashboard'))
                    except Exception as e:
                        logger.warning(f"Auto-login failed: {e}")
                        # Fall through to show login form

            # No config password or auto-login failed - show login form
            return render_template('login.html')

        if request.method == 'POST':
            password = request.form.get('password', '')

            # Get session-specific wallet manager
            wallet_manager = get_or_create_session_vault()
            if wallet_manager:
                try:
                    # WalletManager.openVault(password) unlocks the vault
                    vault = wallet_manager.openVault(password=password, create=True)

                    # Verify the vault was actually decrypted
                    # If wrong password, decryption fails silently and vault remains encrypted
                    if vault is None:
                        flash('Error: Could not open vault', 'error')
                    elif not vault.isDecrypted:
                        # Wrong password - vault is still encrypted
                        flash('Error: Invalid password', 'error')
                    else:
                        # Successfully decrypted - vault stays unlocked for this session
                        session['vault_open'] = True
                        session.pop('logged_out', None)  # Clear logout flag

                        # Encrypt and store password in session for future use
                        encrypted_pw, session_key = encrypt_vault_password(password)
                        session['encrypted_vault_password'] = encrypted_pw
                        session['session_key'] = session_key

                        # Register peer with API server (non-blocking)
                        try:
                            peer_info = ensure_peer_registered(
                                current_app, wallet_manager, max_retries=1)
                            if peer_info:
                                session['peer_id'] = peer_info.get('peer_id')
                                logger.info(f"Peer registered: {peer_info}")
                        except Exception as e:
                            # Don't block login if registration fails
                            logger.warning(f"Peer registration warning: {e}")

                        return redirect(url_for('dashboard'))
                except Exception as e:
                    flash(f'Error: Invalid password or vault error', 'error')
            else:
                # For testing without vault
                if app.config.get('TESTING'):
                    session['vault_open'] = True
                    return redirect(url_for('dashboard'))
                flash('Error: Vault not initialized', 'error')

        return render_template('login.html')

    @app.route('/vault-setup', methods=['GET', 'POST'])
    def vault_setup():
        """Handle initial vault password creation."""
        # If vault file already exists, redirect to login with message
        if check_vault_file_exists():
            flash('Vault already exists. Please log in with your existing password.', 'info')
            logger.info("Attempted to access vault setup but password already exists")
            return redirect(url_for('login'))

        if request.method == 'POST':
            password = request.form.get('password', '')
            password_confirm = request.form.get('password_confirm', '')

            # Validate password length
            if len(password) < 4:
                flash('Password must be at least 4 characters long', 'error')
                return render_template('vault_setup.html')

            # Validate passwords match
            if password != password_confirm:
                flash('Passwords do not match. Please try again.', 'error')
                return render_template('vault_setup.html')

            # Create vault with password (DO NOT save password to config for security)
            try:
                # Get session-specific wallet manager to create the vault
                wallet_manager = get_or_create_session_vault()

                if wallet_manager:
                    # Create the vault with the password
                    vault = wallet_manager.openVault(password=password, create=True)

                    if vault and vault.isDecrypted:
                        flash('Vault created successfully! Please log in with your password.', 'success')
                        logger.info("Vault created via web UI (password NOT saved to config)")
                        return redirect(url_for('login'))
                    else:
                        flash('Error creating vault. Please try again.', 'error')
                        return render_template('vault_setup.html')
                else:
                    flash('Error initializing wallet manager', 'error')
                    return render_template('vault_setup.html')
            except Exception as e:
                logger.error(f"Failed to create vault: {e}")
                flash(f'Error creating vault: {str(e)}', 'error')
                return render_template('vault_setup.html')

        return render_template('vault_setup.html')

    @app.route('/logout')
    def logout():
        """Handle logout/vault lock."""
        # Cleanup session-specific vault
        session_id = session.get('session_id')
        if session_id:
            cleanup_session_vault(session_id)

        # Set flag before clearing to prevent auto-login
        session.clear()
        session['logged_out'] = True
        return redirect(url_for('login'))

    @app.route('/dashboard')
    @login_required
    def dashboard():
        """Main dashboard page."""
        from satorineuron import VERSION

        # Derive eth_wallet_address locally from wallet pubkey
        eth_wallet_address = None
        is_signer = False
        try:
            wallet_manager = get_or_create_session_vault()
            if wallet_manager and wallet_manager.wallet and hasattr(wallet_manager.wallet, 'pubkey'):
                wallet_pubkey = wallet_manager.wallet.pubkey
                if wallet_pubkey:
                    from satorineuron.common.eth_address import derive_eth_wallet_address_from_pubkey
                    eth_wallet_address = derive_eth_wallet_address_from_pubkey(wallet_pubkey)
            # Check if user is an authorized signer
            if wallet_manager and wallet_manager.wallet:
                try:
                    from satorip2p.protocol.signer import is_authorized_signer
                    evrmore_address = wallet_manager.wallet.address
                    if evrmore_address:
                        is_signer = is_authorized_signer(evrmore_address)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not derive eth_wallet_address: {e}")

        return render_template('dashboard.html', version=VERSION, eth_wallet_address=eth_wallet_address, is_signer=is_signer)

    @app.route('/stake')
    @login_required
    def stake_management():
        """Stake management page for pool staking and pool management."""
        from satorineuron import VERSION
        return render_template('stake.html', version=VERSION)

    @app.route('/guide')
    @login_required
    def guide():
        """Comprehensive rewards guide page."""
        return render_template('guide.html')

    @app.route('/badges')
    @login_required
    def badges_page():
        """Full badge catalog and achievements page."""
        return render_template('badges.html')

    @app.route('/health')
    def health():
        """Health check endpoint - checks API server connectivity."""
        api_url = current_app.config.get('SATORI_API_URL', get_api_url())
        try:
            resp = requests.get(f"{api_url}/health", timeout=5)
            if resp.status_code == 200:
                return jsonify({'status': 'ok', 'api': 'connected'})
        except requests.RequestException:
            pass
        return jsonify({'status': 'ok', 'api': 'disconnected'})

    @app.route('/api/engine/status')
    def get_engine_status():
        """Check AI engine initialization status (no auth required for testing)."""
        startup = get_startup()
        if startup is None:
            return jsonify({
                'initialized': False,
                'error': 'Startup instance not set'
            })

        has_aiengine = hasattr(startup, 'aiengine')
        aiengine_not_none = startup.aiengine is not None if has_aiengine else False
        has_streams = False
        stream_count = 0

        if aiengine_not_none and hasattr(startup.aiengine, 'streamModels'):
            has_streams = bool(startup.aiengine.streamModels)
            stream_count = len(startup.aiengine.streamModels)

        return jsonify({
            'initialized': aiengine_not_none,
            'has_aiengine_attr': has_aiengine,
            'aiengine_not_none': aiengine_not_none,
            'has_streams': has_streams,
            'stream_count': stream_count,
            'subscriptions_count': len(startup.subscriptions) if hasattr(startup, 'subscriptions') and startup.subscriptions else 0,
            'publications_count': len(startup.publications) if hasattr(startup, 'publications') and startup.publications else 0
        })

    # API Proxy Routes
    def get_auth_headers():
        """Get authentication headers for API requests using JWT.

        This implements JWT authentication:
        1. Check if we have a valid access token in session
        2. If not or expired, login with wallet signature to get JWT tokens
        3. If expiring soon, refresh the access token
        4. Return Authorization header with Bearer token
        """
        from datetime import datetime, timedelta
        import time

        wallet_manager = get_or_create_session_vault()
        if not wallet_manager or not wallet_manager.wallet:
            return None

        wallet = wallet_manager.wallet
        if not hasattr(wallet, 'pubkey') or not hasattr(wallet, 'sign'):
            logger.warning("Wallet missing pubkey or sign method")
            return None

        api_url = current_app.config.get('SATORI_API_URL', get_api_url())

        # Use lock to prevent concurrent login attempts
        with _auth_lock:
            try:
                # Check if we have a valid JWT token in session
                access_token = session.get('access_token')
                refresh_token = session.get('refresh_token')
                token_expiry = session.get('token_expiry')

                # Convert token_expiry back to datetime if it exists
                if token_expiry:
                    token_expiry = datetime.fromisoformat(token_expiry)

                now = datetime.now()

                # If no token or expired, perform JWT login
                if not access_token or not token_expiry or now >= token_expiry:
                    logger.info("JWT token missing or expired, performing login")

                    # Generate challenge (timestamp)
                    challenge = str(time.time())

                    # Sign with wallet
                    signature = wallet.sign(message=challenge)
                    if isinstance(signature, bytes):
                        signature = signature.decode('utf-8')

                    # Call JWT login endpoint
                    resp = requests.post(
                        f"{api_url}/api/v1/auth/login",
                        headers={
                            'wallet-pubkey': wallet.pubkey,
                            'message': challenge,
                            'signature': signature
                        },
                        timeout=10
                    )

                    if resp.status_code != 200:
                        logger.warning(f"JWT login failed: {resp.text}")
                        return None

                    data = resp.json()
                    access_token = data['access_token']
                    refresh_token = data['refresh_token']
                    expires_in = data['expires_in']

                    # Store tokens in session
                    session['access_token'] = access_token
                    session['refresh_token'] = refresh_token
                    session['token_expiry'] = (now + timedelta(seconds=expires_in)).isoformat()

                    logger.info("JWT login successful")

                # Token expiring soon (within 5 minutes)? Refresh it
                elif now >= (token_expiry - timedelta(minutes=5)) and refresh_token:
                    logger.info("JWT token expiring soon, refreshing")

                    try:
                        resp = requests.post(
                            f"{api_url}/api/v1/auth/refresh",
                            headers={'Authorization': f'Bearer {refresh_token}'},
                            timeout=10
                        )

                        if resp.status_code == 200:
                            data = resp.json()
                            access_token = data['access_token']
                            expires_in = data['expires_in']

                            # Update session with new token
                            session['access_token'] = access_token
                            session['token_expiry'] = (now + timedelta(seconds=expires_in)).isoformat()

                            logger.info("JWT token refreshed successfully")
                        else:
                            logger.warning(f"JWT refresh failed: {resp.text}")
                    except Exception as e:
                        logger.warning(f"JWT refresh failed: {e}")

                # Return JWT Bearer token header
                return {
                    'Authorization': f'Bearer {access_token}'
                }

            except Exception as e:
                logger.warning(f"JWT auth header generation failed: {e}")
                return None

    def proxy_api(endpoint, method='GET', data=None, authenticated=True):
        """Proxy requests to the Satori API server."""
        api_url = current_app.config.get('SATORI_API_URL', get_api_url())
        url = f"{api_url}/api/v1{endpoint}"

        # Get auth headers for authenticated requests
        headers = {}
        if authenticated:
            auth_headers = get_auth_headers()
            if auth_headers:
                headers.update(auth_headers)
            else:
                logger.warning(f"No auth headers available for {endpoint}")

        try:
            if method == 'GET':
                # Forward query parameters from the incoming request
                resp = requests.get(url, params=request.args, headers=headers, timeout=10)
            elif method == 'POST':
                resp = requests.post(url, json=data, headers=headers, timeout=10)
            elif method == 'DELETE':
                resp = requests.delete(url, json=data, headers=headers, timeout=10)
            else:
                return jsonify({'error': 'Invalid method'}), 400

            return jsonify(resp.json()), resp.status_code
        except requests.RequestException as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/balance/get')
    @login_required
    def api_balance():
        """Proxy balance request."""
        return proxy_api('/balance/get')

    @app.route('/api/peer/reward-address', methods=['GET', 'POST'])
    @login_required
    def api_reward_address():
        """
        Get or set reward address via IPC (synced with server by neuron process).

        GET: Reads from neuron process via IPC
        POST: Updates via neuron process which syncs with server
        """
        from web.wsgi import _ipc_get, _ipc_post

        if request.method == 'POST':
            # Get the new reward address from request
            data = request.get_json()
            if not data or 'reward_address' not in data:
                return jsonify({'error': 'Missing reward_address'}), 400

            # Update via IPC
            try:
                result = _ipc_post('/reward-address', data)
                if result and result.get('success'):
                    return jsonify({'success': True, 'reward_address': result.get('reward_address', '')})
                else:
                    error = result.get('error', 'Failed to set reward address') if result else 'IPC failed'
                    return jsonify({'error': error}), 500
            except Exception as e:
                logger.error(f"Error setting reward address via IPC: {e}")
                return jsonify({'error': str(e)}), 500

        # GET request - read via IPC
        try:
            result = _ipc_get('/reward-address')
            if result:
                return jsonify({'reward_address': result.get('reward_address', '')})
            else:
                return jsonify({'reward_address': ''})
        except Exception as e:
            logger.warning(f"Error getting reward address via IPC: {e}")
            return jsonify({'reward_address': ''})

    # =========================================================================
    # STAKE MANAGEMENT - P2P ENABLED ROUTES
    # =========================================================================

    @app.route('/api/lender/status')
    @login_required
    def api_lender_status():
        """Get lender status - P2P first, central fallback."""
        mode = _get_current_networking_mode()

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager

                    async def get_status():
                        if hasattr(manager, 'get_my_lending_status'):
                            return await manager.get_my_lending_status()
                        return None

                    try:
                        loop = asyncio.new_event_loop()
                        status = loop.run_until_complete(get_status())
                        if status is not None:
                            return jsonify(status)
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P lender status failed, trying central: {e}")

        # Fallback to central - proxy without authentication (public endpoint)
        try:
            from satorineuron.init import start
            startup = start.getStart() if hasattr(start, 'getStart') else None
            wallet_address = None
            if startup and hasattr(startup, 'wallet') and startup.wallet:
                wallet_address = startup.wallet.address

            if wallet_address:
                result = proxy_api(f'/lender/status?wallet_address={wallet_address}', authenticated=False)
                # Check if proxy returned an error
                if hasattr(result, 'status_code') and result.status_code >= 400:
                    raise Exception(f"Central API returned {result.status_code}")
                return result
            else:
                # Try without wallet address if not available
                return proxy_api('/lender/status', authenticated=False)
        except Exception as e:
            logger.warning(f"Central lender status also failed: {e}")
            # Return empty/default status
            return jsonify({
                'is_lending': False,
                'lending_to': None,
                'lent_amount': 0,
                'available_pools': [],
                'error': None
            })

    @app.route('/api/lender/lend', methods=['POST', 'DELETE'])
    @login_required
    def api_lender_lend():
        """Lend to pool or remove lending - P2P first, central fallback."""
        mode = _get_current_networking_mode()
        data = request.get_json(silent=True) if request.method == 'POST' else None

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager

                    if request.method == 'POST':
                        # Lend to pool
                        pool_address = data.get('pool_address') if data else None
                        if pool_address:
                            async def lend_to():
                                if hasattr(manager, 'lend_to_vault'):
                                    return await manager.lend_to_vault(pool_address)
                                return False

                            try:
                                loop = asyncio.new_event_loop()
                                success = loop.run_until_complete(lend_to())
                                if success:
                                    return jsonify({'success': True})
                            finally:
                                loop.close()
                    else:
                        # Remove lending (DELETE)
                        async def remove_lend():
                            if hasattr(manager, 'remove_lending'):
                                return await manager.remove_lending()
                            return False

                        try:
                            loop = asyncio.new_event_loop()
                            success = loop.run_until_complete(remove_lend())
                            if success:
                                return jsonify({'success': True})
                        finally:
                            loop.close()
            except Exception as e:
                logger.warning(f"P2P lend operation failed, trying central: {e}")

        # Fallback to central
        return proxy_api('/lender/lend', request.method, data)

    @app.route('/api/pool/worker', methods=['POST'])
    @login_required
    def api_pool_worker_add():
        """Add worker to pool - P2P first, central fallback."""
        mode = _get_current_networking_mode()
        data = request.json

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager
                    worker_address = data.get('worker_address') if data else None

                    if worker_address:
                        async def add_worker():
                            if hasattr(manager, 'add_pool_worker'):
                                return await manager.add_pool_worker(worker_address)
                            return False

                        try:
                            loop = asyncio.new_event_loop()
                            success = loop.run_until_complete(add_worker())
                            if success:
                                return jsonify({'success': True})
                        finally:
                            loop.close()
            except Exception as e:
                logger.warning(f"P2P add worker failed, trying central: {e}")

        # Fallback to central
        return proxy_api('/pool/worker', 'POST', data)

    @app.route('/api/pool/worker/<worker_address>', methods=['DELETE'])
    @login_required
    def api_pool_worker_delete(worker_address):
        """Remove worker from pool - P2P first, central fallback."""
        mode = _get_current_networking_mode()

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager

                    async def remove_worker():
                        if hasattr(manager, 'remove_pool_worker'):
                            return await manager.remove_pool_worker(worker_address)
                        return False

                    try:
                        loop = asyncio.new_event_loop()
                        success = loop.run_until_complete(remove_worker())
                        if success:
                            return jsonify({'success': True})
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P remove worker failed, trying central: {e}")

        # Fallback to central
        return proxy_api(f'/pool/worker/{worker_address}', 'DELETE')

    @app.route('/api/pool/toggle-open', methods=['POST'])
    @login_required
    def api_pool_toggle():
        """Toggle pool open/closed - P2P first, central fallback."""
        mode = _get_current_networking_mode()
        data = request.json

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager
                    is_open = data.get('open', False) if data else False

                    async def toggle_pool():
                        if is_open:
                            if hasattr(manager, 'register_pool'):
                                return await manager.register_pool()
                        else:
                            if hasattr(manager, 'unregister_pool'):
                                return await manager.unregister_pool()
                        return False

                    try:
                        loop = asyncio.new_event_loop()
                        success = loop.run_until_complete(toggle_pool())
                        if success:
                            return jsonify({'success': True, 'open': is_open})
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P toggle pool failed, trying central: {e}")

        # Fallback to central
        return proxy_api('/pool/toggle-open', 'POST', data)

    @app.route('/api/pool/open', methods=['GET'])
    @login_required
    def api_pool_open():
        """Get list of open pools - P2P first, central fallback."""
        mode = _get_current_networking_mode()

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager

                    async def get_pools():
                        if hasattr(manager, 'get_available_pools'):
                            pools = await manager.get_available_pools()
                            return [
                                {
                                    'address': p.vault_address,
                                    'pool_size_limit': p.pool_size_limit,
                                    'worker_reward_pct': getattr(p, 'worker_reward_pct', 0),
                                    'accepting': p.accepting,
                                }
                                for p in pools if p.accepting
                            ]
                        return None

                    try:
                        loop = asyncio.new_event_loop()
                        pools = loop.run_until_complete(get_pools())
                        if pools is not None:
                            return jsonify({'pools': pools})
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P get open pools failed, trying central: {e}")

        # Fallback to central
        return proxy_api('/pool/open', 'GET')

    @app.route('/api/pool/commission', methods=['GET'])
    @login_required
    def api_pool_commission():
        """Get pool commission status - P2P first, central fallback."""
        mode = _get_current_networking_mode()

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager

                    if hasattr(manager, '_my_pool_config') and manager._my_pool_config:
                        config = manager._my_pool_config
                        return jsonify({
                            'commission': getattr(config, 'worker_reward_pct', 0),
                            'open': config.accepting,
                            'pool_size_limit': config.pool_size_limit,
                        })
            except Exception as e:
                logger.warning(f"P2P get commission failed, trying central: {e}")

        # Fallback to central
        return proxy_api('/pool/commission', 'GET')

    @app.route('/api/pool/workers', methods=['GET'])
    @login_required
    def api_pool_workers():
        """Get list of workers for pool - P2P first, central fallback."""
        mode = _get_current_networking_mode()

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager

                    async def get_workers():
                        if hasattr(manager, 'get_pool_workers'):
                            return await manager.get_pool_workers()
                        return None

                    try:
                        loop = asyncio.new_event_loop()
                        workers = loop.run_until_complete(get_workers())
                        if workers is not None:
                            return jsonify({
                                'workers': [
                                    {
                                        'address': w.worker_address if hasattr(w, 'worker_address') else str(w),
                                        'added_at': getattr(w, 'timestamp', None),
                                    }
                                    for w in workers
                                ]
                            })
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P get workers failed, trying central: {e}")

        # Fallback to central
        return proxy_api('/pool/workers', 'GET')

    @app.route('/api/pool/lenders', methods=['GET'])
    @login_required
    def api_pool_lenders():
        """Get list of lenders for pool - P2P first, central fallback."""
        mode = _get_current_networking_mode()

        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                    manager = startup._lending_manager

                    async def get_lenders():
                        if hasattr(manager, 'get_pool_participants'):
                            return await manager.get_pool_participants()
                        return None

                    try:
                        loop = asyncio.new_event_loop()
                        lenders = loop.run_until_complete(get_lenders())
                        if lenders is not None:
                            return jsonify({
                                'lenders': [
                                    {
                                        'address': l.lender_address if hasattr(l, 'lender_address') else str(l),
                                        'amount': getattr(l, 'lent_out', 0),
                                        'timestamp': getattr(l, 'timestamp', None),
                                    }
                                    for l in lenders
                                ]
                            })
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P get lenders failed, trying central: {e}")

        # Fallback to central
        return proxy_api('/pool/lenders', 'GET')

    @app.route('/api/wallet/address')
    @login_required
    def api_wallet_address():
        """Get wallet and vault addresses with public keys."""
        wallet_manager = get_or_create_session_vault()
        if wallet_manager:
            result = {}
            # Get wallet (identity) address and public key
            if wallet_manager.wallet:
                if hasattr(wallet_manager.wallet, 'address'):
                    result['wallet_address'] = wallet_manager.wallet.address
                if hasattr(wallet_manager.wallet, 'publicKey'):
                    result['wallet_public_key'] = wallet_manager.wallet.publicKey
                elif hasattr(wallet_manager.wallet, 'pubkey'):
                    result['wallet_public_key'] = wallet_manager.wallet.pubkey
            # Get vault address and public key
            if wallet_manager.vault:
                if hasattr(wallet_manager.vault, 'address'):
                    result['vault_address'] = wallet_manager.vault.address
                if hasattr(wallet_manager.vault, 'publicKey'):
                    result['vault_public_key'] = wallet_manager.vault.publicKey
                elif hasattr(wallet_manager.vault, 'pubkey'):
                    result['vault_public_key'] = wallet_manager.vault.pubkey
            return jsonify(result)
        return jsonify({'error': 'Wallet not initialized'}), 500

    @app.route('/api/wallet/private-key')
    @login_required
    def api_wallet_private_key():
        """Get vault's private key (sensitive - requires login)."""
        wallet_manager = get_or_create_session_vault()
        if wallet_manager and wallet_manager.vault:
            try:
                privkey = wallet_manager.vault.privkey
                return jsonify({'private_key': privkey})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'Vault not initialized'}), 500

    @app.route('/api/wallet/identity-private-key')
    @login_required
    def api_wallet_identity_private_key():
        """Get identity wallet's private key (for backup - requires login)."""
        wallet_manager = get_or_create_session_vault()
        if wallet_manager and wallet_manager.wallet:
            try:
                privkey = wallet_manager.wallet.privkey
                return jsonify({'private_key': privkey})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'Identity wallet not initialized'}), 500

    @app.route('/api/wallet/send-from-wallet', methods=['POST'])
    @login_required
    def api_wallet_send_from_wallet():
        """Send SATORI tokens from wallet (identity wallet)."""
        wallet_manager = get_or_create_session_vault()
        if not wallet_manager or not wallet_manager.wallet:
            return jsonify({'error': 'Wallet not initialized'}), 500

        data = request.json or {}
        address = data.get('address', '').strip()
        amount = data.get('amount')
        sweep = data.get('sweep', False)

        # Validate address
        if not address:
            return jsonify({'error': 'Address is required'}), 400
        if not address.startswith('E') or len(address) != 34:
            return jsonify({'error': 'Invalid address format'}), 400

        # Validate amount (unless sweep)
        if not sweep:
            try:
                amount = float(amount)
                if amount <= 0:
                    return jsonify({'error': 'Amount must be greater than 0'}), 400
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid amount'}), 400

        try:
            wallet = wallet_manager.wallet
            # Get ready to send
            wallet.get()
            wallet.getReadyToSend()

            if sweep:
                # Send all tokens - returns string (txid)
                txid = wallet.sendAllTransaction(address)
                if txid and len(txid) == 64:
                    return jsonify({'success': True, 'txid': txid})
                else:
                    return jsonify({'error': 'Transaction failed', 'details': str(txid)}), 500
            else:
                # Send specific amount - use satoriTransaction for wallet (like CLI does)
                txid = wallet.satoriTransaction(amount=amount, address=address)

                if txid and len(txid) == 64:
                    return jsonify({'success': True, 'txid': txid})
                else:
                    return jsonify({'error': 'Transaction failed', 'details': str(txid)}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/wallet/send', methods=['POST'])
    @login_required
    def api_wallet_send():
        """Send SATORI tokens from vault."""
        wallet_manager = get_or_create_session_vault()
        if not wallet_manager or not wallet_manager.vault:
            return jsonify({'error': 'Vault not initialized'}), 500

        data = request.json or {}
        address = data.get('address', '').strip()
        amount = data.get('amount')
        sweep = data.get('sweep', False)

        # Validate address
        if not address:
            return jsonify({'error': 'Address is required'}), 400
        if not address.startswith('E') or len(address) != 34:
            return jsonify({'error': 'Invalid address format'}), 400

        # Validate amount (unless sweep)
        if not sweep:
            try:
                amount = float(amount)
                if amount <= 0:
                    return jsonify({'error': 'Amount must be greater than 0'}), 400
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid amount'}), 400

        try:
            vault = wallet_manager.vault
            # Get ready to send
            vault.get()
            vault.getReadyToSend()

            if sweep:
                # Send all tokens - returns string (txid)
                txid = vault.sendAllTransaction(address)
                if txid and len(txid) == 64:
                    return jsonify({'success': True, 'txid': txid})
                else:
                    return jsonify({'error': 'Transaction failed', 'details': str(txid)}), 500
            else:
                # Send specific amount - returns TransactionResult object
                result = vault.typicalNeuronTransaction(
                    amount=amount,
                    address=address)

                # Check if result is a TransactionResult object
                if hasattr(result, 'success') and hasattr(result, 'msg'):
                    if result.success and result.msg and len(result.msg) == 64:
                        return jsonify({'success': True, 'txid': result.msg})
                    else:
                        error_msg = result.msg or 'Transaction failed'
                        return jsonify({'error': error_msg}), 500
                # Fallback for unexpected return type
                else:
                    return jsonify({'error': 'Unexpected transaction result', 'details': str(result)}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/lender/pay', methods=['POST'])
    @login_required
    def api_lender_pay():
        """Pay lenders from audit - simple single-transaction approach."""
        try:
            # Get wallet manager
            wallet_manager = get_or_create_session_vault()
            if not wallet_manager:
                return jsonify({'error': 'Wallet not initialized'}), 500

            vault = wallet_manager.vault

            # Check vault is unlocked
            if not vault.isDecrypted:
                return jsonify({'error': 'Vault is locked'}), 401

            # Get payments from request
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            payments = data.get('payments', {})

            if not payments:
                return jsonify({'error': 'No payments provided'}), 400

            # Check limit (blockchain max is 1000)
            if len(payments) > 1000:
                return jsonify({
                    'error': f'Too many lenders ({len(payments)}). Maximum is 1000.'
                }), 400

            # Calculate total
            total = sum(payments.values())

            # Check balance
            vault.get()
            if vault.balance.amount < total:
                return jsonify({
                    'error': f'Insufficient balance: have {vault.balance.amount}, need {total}'
                }), 400

            # Prepare and send
            vault.getReadyToSend()
            txhash = vault.satoriDistribution(
                amountByAddress=payments,
                memo='Lender rewards',
                broadcast=True
            )

            # Validate txhash
            if txhash and isinstance(txhash, str) and len(txhash) == 64:
                return jsonify({
                    'success': True,
                    'txhash': txhash,
                    'lenders_paid': len(payments),
                    'total_sent': total
                })
            else:
                return jsonify({'error': f'Invalid transaction hash: {txhash}'}), 500

        except Exception as e:
            app.logger.error(f"Error in lender payment: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/wallet/qr/<address>')
    @login_required
    def api_wallet_qr(address: str):
        """Generate QR code for an address."""
        import io
        import base64
        try:
            import qrcode
            qr = qrcode.QRCode(version=1, box_size=4, border=2)
            qr.add_data(address)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return jsonify({'qr_code': f'data:image/png;base64,{img_base64}'})
        except ImportError:
            return jsonify({'error': 'QR code library not available'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/wallet/balance/direct')
    @login_required
    def api_wallet_balance_direct():
        """Get combined wallet and vault balance directly from electrumx.

        This bypasses the Satori API server and queries the blockchain directly
        via the electromax server using the wallet objects.
        """
        wallet_manager = get_or_create_session_vault()
        if not wallet_manager:
            return jsonify({'error': 'Wallet manager not initialized'}), 500

        try:
            # Ensure electrumx connection with retry
            logger.info("Attempting to connect to ElectrumX...")
            connected = False
            if hasattr(wallet_manager, 'connect'):
                for attempt in range(5):
                    connected = wallet_manager.connect()
                    logger.info(f"ElectrumX connection attempt {attempt + 1}: {connected}")
                    if connected:
                        break
                    time.sleep(2)  # Give more time for connection to establish
                if not connected:
                    logger.warning("Failed to connect to ElectrumX after retries")
                    return jsonify({'error': 'Could not connect to electrumx'}), 500
            else:
                logger.warning("WalletManager has no connect method")

            total_satori = 0.0
            wallet_balance = 0.0
            vault_balance = 0.0
            total_evr = 0.0
            wallet_evr = 0.0
            vault_evr = 0.0

            # Get wallet (identity) balance
            if wallet_manager.wallet:
                wallet = wallet_manager.wallet
                if hasattr(wallet, 'getBalances'):
                    logger.info("Getting wallet balances...")
                    # Retry if connection not ready yet
                    for _ in range(3):
                        if wallet.electrumx and wallet.electrumx.connected():
                            wallet.getBalances()
                            break
                        time.sleep(1)
                    else:
                        wallet.getBalances()  # Final attempt
                    if hasattr(wallet, 'balance') and wallet.balance:
                        wallet_balance = wallet.balance.amount if hasattr(wallet.balance, 'amount') else 0.0
                    if hasattr(wallet, 'currency') and wallet.currency:
                        wallet_evr = wallet.currency.amount if hasattr(wallet.currency, 'amount') else 0.0
                    logger.info(f"Wallet balance: SATORI={wallet_balance}, EVR={wallet_evr}")

            # Get vault balance
            if wallet_manager.vault:
                vault = wallet_manager.vault
                if hasattr(vault, 'getBalances'):
                    logger.info("Getting vault balances...")
                    # Retry if connection not ready yet
                    for _ in range(3):
                        if vault.electrumx and vault.electrumx.connected():
                            vault.getBalances()
                            break
                        time.sleep(1)
                    else:
                        vault.getBalances()  # Final attempt
                    if hasattr(vault, 'balance') and vault.balance:
                        vault_balance = vault.balance.amount if hasattr(vault.balance, 'amount') else 0.0
                    if hasattr(vault, 'currency') and vault.currency:
                        vault_evr = vault.currency.amount if hasattr(vault.currency, 'amount') else 0.0
                    logger.info(f"Vault balance: SATORI={vault_balance}, EVR={vault_evr}")

            total_satori = wallet_balance + vault_balance
            total_evr = wallet_evr + vault_evr

            return jsonify({
                'total': total_satori,
                'wallet_balance': wallet_balance,
                'vault_balance': vault_balance,
                'total_evr': total_evr,
                'wallet_evr': wallet_evr,
                'vault_evr': vault_evr
            })
        except Exception as e:
            import traceback
            logger.error(f"Failed to get direct balance: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # P2P NETWORK HEALTH ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/health')
    @login_required
    def api_p2p_health():
        """Get P2P network health status."""
        try:
            from satorineuron.init import start
            from web.wsgi import get_web_p2p_peers, get_web_uptime

            result = {
                'mode': 'unknown',
                'peer_count': 0,
                'uptime_pct': 0.0,
                'last_heartbeat_ago': None,
                'relay_eligible': False,
                'connected': False,
            }

            # Get networking mode
            if hasattr(start, '_get_networking_mode'):
                result['mode'] = start._get_networking_mode()

            # Get P2P peers info using helper (via IPC)
            peers = get_web_p2p_peers()
            if peers:
                result['connected'] = True
                if hasattr(peers, 'get_peer_count'):
                    result['peer_count'] = peers.get_peer_count()

            # Get uptime tracker info via IPC API
            uptime_data = get_web_uptime()
            if uptime_data and uptime_data.get('success'):
                result['uptime_pct'] = uptime_data.get('uptime_percentage', 0.0)
                # relay_eligible threshold is 95% (0.95)
                result['relay_eligible'] = result['uptime_pct'] >= 0.95
                # Calculate last_heartbeat_ago from heartbeats_sent count
                # (approximation - actual timestamp would need IPC extension)
                if uptime_data.get('heartbeats_sent', 0) > 0:
                    result['last_heartbeat_ago'] = 60  # Heartbeats sent every 60s

            return jsonify(result)
        except Exception as e:
            logger.warning(f"P2P health check failed: {e}")
            return jsonify({
                'mode': 'central',
                'peer_count': 0,
                'connected': False,
                'error': str(e)
            })

    @app.route('/api/p2p/heartbeats')
    @login_required
    def api_p2p_heartbeats():
        """Get recent heartbeats from the network."""
        try:
            from web.wsgi import get_web_heartbeats

            limit = request.args.get('limit', 20, type=int)
            heartbeats = get_web_heartbeats(limit=limit)
            return jsonify({'heartbeats': heartbeats})
        except Exception as e:
            logger.warning(f"Failed to get heartbeats: {e}")
            return jsonify({'heartbeats': [], 'error': str(e)})

    @app.route('/api/p2p/heartbeat/stats')
    @login_required
    def api_p2p_heartbeat_stats():
        """Get heartbeat statistics for our own node."""
        try:
            from web.wsgi import get_web_uptime
            from datetime import datetime, timezone

            result = {
                'active': False,
                'current_round': None,
                'last_heartbeat': None,
                'last_heartbeat_ago': None,
                'heartbeats_expected': 0,
                'heartbeats_sent': 0,
                'heartbeats_received': 0,
                'uptime_percentage': 0.0,
                'last_status_message': None,
                'is_heartbeating': False,
            }

            # Get uptime data via IPC API
            uptime_data = get_web_uptime()
            if uptime_data and uptime_data.get('success'):
                result['heartbeats_sent'] = uptime_data.get('heartbeats_sent', 0)
                result['heartbeats_received'] = uptime_data.get('heartbeats_received', 0)
                result['uptime_percentage'] = uptime_data.get('uptime_percentage', 0.0)
                result['current_round'] = uptime_data.get('current_round')
                result['heartbeat_round'] = uptime_data.get('current_round')
                result['streak_days'] = uptime_data.get('streak_days', 0)
                result['last_status_message'] = uptime_data.get('last_status_message')

                # If heartbeats are being sent, we're active
                if result['heartbeats_sent'] > 0:
                    result['active'] = True
                    result['is_heartbeating'] = True
                    # Approximate last heartbeat as ~60 seconds ago (heartbeat interval)
                    result['last_heartbeat_ago'] = 60

                # Expected heartbeats based on time since round start
                # (approximation - rounds are daily)
                now = datetime.now(timezone.utc)
                seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second
                result['heartbeats_expected'] = max(1, seconds_since_midnight // 60)

            # Network Epoch and Round (for rewards/badges)
            now = datetime.now(timezone.utc)
            iso_cal = now.isocalendar()
            result['network_epoch'] = f"{iso_cal.year}-W{iso_cal.week:02d}"
            result['network_round'] = iso_cal.weekday  # 1=Monday to 7=Sunday

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get heartbeat stats: {e}")
            return jsonify({'error': str(e), 'active': False})

    @app.route('/api/p2p/uptime/streak')
    @login_required
    def api_p2p_uptime_streak():
        """Get our own uptime streak info."""
        try:
            from web.wsgi import get_web_uptime

            result = {
                'streak_days': 0,
                'streak_tier': None,
                'streak_bonus': 0.0,
                'longest_streak': 0,
                'streak_start_date': None,
            }

            # Get uptime data via IPC API
            uptime_data = get_web_uptime()
            if uptime_data and uptime_data.get('success'):
                streak_days = uptime_data.get('streak_days', 0)
                result['streak_days'] = streak_days
                result['longest_streak'] = streak_days  # TODO: Track separately in IPC

                # Get tier and bonus from rewards module
                try:
                    from satorip2p.protocol.rewards import (
                        get_uptime_streak_tier,
                        calculate_uptime_streak_bonus
                    )
                    result['streak_tier'] = get_uptime_streak_tier(streak_days)
                    result['streak_bonus'] = calculate_uptime_streak_bonus(streak_days)
                except ImportError:
                    # Calculate manually if rewards module not available
                    if streak_days >= 90:
                        result['streak_tier'] = 'Legend'
                        result['streak_bonus'] = 0.10
                    elif streak_days >= 30:
                        result['streak_tier'] = 'Veteran'
                        result['streak_bonus'] = 0.05
                    elif streak_days >= 7:
                        result['streak_tier'] = 'Regular'
                        result['streak_bonus'] = 0.02

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get uptime streak: {e}")
            return jsonify({'error': str(e), 'streak_days': 0})

    @app.route('/api/p2p/uptime/leaderboard')
    @login_required
    def api_p2p_uptime_leaderboard():
        """Get top uptime streak leaderboard."""
        try:
            from satorineuron.init import start

            limit = request.args.get('limit', 10, type=int)
            streaks = []

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_uptime_tracker') and startup._uptime_tracker:
                tracker = startup._uptime_tracker

                if hasattr(tracker, 'get_top_uptime_streaks'):
                    top_streaks = tracker.get_top_uptime_streaks(limit=limit)
                    for record in top_streaks:
                        if record.streak_days > 0:
                            from satorip2p.protocol.rewards import (
                                get_uptime_streak_tier,
                                calculate_uptime_streak_bonus
                            )
                            streaks.append({
                                'node_id': record.node_id,
                                'streak_days': record.streak_days,
                                'streak_tier': get_uptime_streak_tier(record.streak_days),
                                'streak_bonus': calculate_uptime_streak_bonus(record.streak_days),
                                'longest_streak': record.longest_streak,
                                'streak_start_date': record.streak_start_date,
                            })

            return jsonify({'streaks': streaks})
        except Exception as e:
            logger.warning(f"Failed to get uptime leaderboard: {e}")
            return jsonify({'streaks': [], 'error': str(e)})

    @app.route('/api/p2p/titles')
    @login_required
    def api_p2p_titles():
        """Get earned titles for our node."""
        try:
            from satorineuron.init import start

            result = {
                'titles': [],
                'all_titles': ['historic', 'friendly', 'charity', 'whale', 'legend'],
                'title_descriptions': {
                    'historic': '90+ day uptime streak',
                    'friendly': 'Diamond referral tier (2000+ referrals)',
                    'charity': 'Diamond donor tier (500k+ EVR donated)',
                    'whale': 'Maxed stake bonus (55+ SATORI)',
                    'legend': 'Maxed ALL multipliers (2.10x)',
                },
                'multiplier_breakdown': {},
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None

            # Gather data needed to calculate titles
            uptime_streak_days = 0
            stake_amount = 50.0  # Default minimum
            referral_count = 0
            total_donated_evr = 0.0
            is_signer = False

            # Get uptime streak
            if startup and hasattr(startup, '_uptime_tracker') and startup._uptime_tracker:
                tracker = startup._uptime_tracker
                if hasattr(tracker, 'get_uptime_streak'):
                    uptime_streak_days = tracker.get_uptime_streak(tracker.node_id)

            # Get referral count
            if startup and hasattr(startup, '_referral_manager') and startup._referral_manager:
                manager = startup._referral_manager
                if hasattr(manager, 'get_my_referral_count'):
                    import asyncio
                    try:
                        loop = asyncio.new_event_loop()
                        referral_count = loop.run_until_complete(manager.get_my_referral_count())
                    except:
                        pass
                    finally:
                        loop.close()

            # Get donation total
            if startup and hasattr(startup, '_donation_manager') and startup._donation_manager:
                manager = startup._donation_manager
                if hasattr(manager, 'get_my_total_donated'):
                    import asyncio
                    try:
                        loop = asyncio.new_event_loop()
                        total_donated_evr = loop.run_until_complete(manager.get_my_total_donated())
                    except:
                        pass
                    finally:
                        loop.close()

            # Calculate titles using rewards module
            try:
                from satorip2p.protocol.rewards import (
                    NodeRoles,
                    calculate_stake_bonus,
                    calculate_uptime_streak_bonus,
                    calculate_referral_bonus,
                    calculate_donation_bonus,
                    get_uptime_streak_tier,
                    get_referral_tier,
                    get_donation_tier,
                    STAKE_BONUS_CAP,
                    UPTIME_STREAK_TIER_5,
                    REFERRAL_TIER_DIAMOND,
                    DONATION_TIER_DIAMOND,
                )

                # Build NodeRoles to get titles
                node_roles = NodeRoles(
                    node_id='self',
                    uptime_streak_days=uptime_streak_days,
                    stake_amount=stake_amount,
                    referral_count=referral_count,
                    total_donated_evr=total_donated_evr,
                    signer_qualified=is_signer,
                )
                result['titles'] = node_roles.get_earned_titles()

                # Add multiplier breakdown
                result['multiplier_breakdown'] = {
                    'stake': {
                        'bonus': calculate_stake_bonus(stake_amount),
                        'amount': stake_amount,
                        'maxed': calculate_stake_bonus(stake_amount) >= STAKE_BONUS_CAP,
                    },
                    'uptime_streak': {
                        'bonus': calculate_uptime_streak_bonus(uptime_streak_days),
                        'days': uptime_streak_days,
                        'tier': get_uptime_streak_tier(uptime_streak_days),
                        'maxed': uptime_streak_days >= UPTIME_STREAK_TIER_5,
                    },
                    'referral': {
                        'bonus': calculate_referral_bonus(referral_count),
                        'count': referral_count,
                        'tier': get_referral_tier(referral_count),
                        'maxed': referral_count >= REFERRAL_TIER_DIAMOND,
                    },
                    'donation': {
                        'bonus': calculate_donation_bonus(total_donated_evr),
                        'total_evr': total_donated_evr,
                        'tier': get_donation_tier(total_donated_evr),
                        'maxed': total_donated_evr >= DONATION_TIER_DIAMOND,
                    },
                }
            except ImportError as e:
                logger.warning(f"Could not import rewards module: {e}")

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get titles: {e}")
            return jsonify({'titles': [], 'error': str(e)})

    # ========================================================================
    # CURATOR PROTOCOL ENDPOINTS (status endpoint moved to line ~2048)
    # ========================================================================

    @app.route('/api/p2p/curator/streams')
    @login_required
    def api_p2p_curator_streams():
        """Get all curated streams."""
        try:
            from satorineuron.init import start

            result = {'streams': []}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_curator') and startup._curator:
                curator = startup._curator
                if hasattr(curator, 'get_all_curated_streams'):
                    streams = curator.get_all_curated_streams()
                    result['streams'] = [s.to_dict() for s in streams.values()]

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get curated streams: {e}")
            return jsonify({'streams': [], 'error': str(e)})

    @app.route('/api/p2p/curator/stream/<stream_id>')
    @login_required
    def api_p2p_curator_stream(stream_id):
        """Get curation status for a specific stream."""
        try:
            from satorineuron.init import start

            result = {'stream': None}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_curator') and startup._curator:
                curator = startup._curator
                if hasattr(curator, 'get_stream_status'):
                    curation = curator.get_stream_status(stream_id)
                    if curation:
                        result['stream'] = curation.to_dict()
                        result['votes'] = curator.get_vote_summary(stream_id)

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get stream curation: {e}")
            return jsonify({'stream': None, 'error': str(e)})

    @app.route('/api/p2p/curator/flags')
    @login_required
    def api_p2p_curator_flags():
        """Get all flagged oracles."""
        try:
            from satorineuron.init import start

            result = {'flags': []}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_curator') and startup._curator:
                curator = startup._curator
                if hasattr(curator, 'get_flagged_oracles'):
                    flags = curator.get_flagged_oracles(unresolved_only=True)
                    result['flags'] = [f.to_dict() for f in flags]

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get flags: {e}")
            return jsonify({'flags': [], 'error': str(e)})

    @app.route('/api/p2p/curator/oracle/<oracle_address>')
    @login_required
    def api_p2p_curator_oracle(oracle_address):
        """Get reputation for a specific oracle."""
        try:
            from satorineuron.init import start

            result = {'reputation': None}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_curator') and startup._curator:
                curator = startup._curator
                if hasattr(curator, 'get_oracle_reputation'):
                    rep = curator.get_oracle_reputation(oracle_address)
                    if rep:
                        result['reputation'] = rep.to_dict()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get oracle reputation: {e}")
            return jsonify({'reputation': None, 'error': str(e)})

    @app.route('/api/p2p/curator/vote', methods=['POST'])
    @login_required
    def api_p2p_curator_vote():
        """Vote on a stream's quality (curators only)."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            stream_id = data.get('stream_id', '')
            vote_value = data.get('vote', 'abstain')
            comment = data.get('comment', '')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id required'})

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_curator') or not startup._curator:
                return jsonify({'success': False, 'error': 'Curator protocol not available'})

            curator = startup._curator
            if not curator.is_curator:
                return jsonify({'success': False, 'error': 'Not authorized as curator'})

            from satorip2p.protocol.curator import QualityVote
            vote = QualityVote(vote_value)

            loop = asyncio.new_event_loop()
            try:
                success = loop.run_until_complete(
                    curator.vote_stream_quality(stream_id, vote, comment)
                )
            finally:
                loop.close()

            return jsonify({'success': success})
        except Exception as e:
            logger.warning(f"Failed to vote: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/curator/flag', methods=['POST'])
    @login_required
    def api_p2p_curator_flag():
        """Flag an oracle (curators only)."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            oracle_address = data.get('oracle_address', '')
            stream_id = data.get('stream_id', '')
            reason = data.get('reason', 'other')
            evidence = data.get('evidence', '')

            if not oracle_address or not stream_id:
                return jsonify({'success': False, 'error': 'oracle_address and stream_id required'})

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_curator') or not startup._curator:
                return jsonify({'success': False, 'error': 'Curator protocol not available'})

            curator = startup._curator
            if not curator.is_curator:
                return jsonify({'success': False, 'error': 'Not authorized as curator'})

            from satorip2p.protocol.curator import FlagReason
            flag_reason = FlagReason(reason)

            loop = asyncio.new_event_loop()
            try:
                success = loop.run_until_complete(
                    curator.flag_oracle(oracle_address, stream_id, flag_reason, evidence)
                )
            finally:
                loop.close()

            return jsonify({'success': success})
        except Exception as e:
            logger.warning(f"Failed to flag oracle: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/curator/confirm-flag', methods=['POST'])
    @login_required
    def api_p2p_curator_confirm_flag():
        """Confirm/resolve a flag on an oracle (curators only).

        This endpoint allows curators to vote to confirm or dismiss a flag.
        When enough curators (3-of-5) confirm a flag, the oracle is rejected.
        """
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            oracle_address = data.get('oracle_address', '')
            stream_id = data.get('stream_id', '')
            confirmed = data.get('confirmed', True)  # True = confirm flag, False = dismiss
            resolution = data.get('resolution', '')

            if not oracle_address or not stream_id:
                return jsonify({'success': False, 'error': 'oracle_address and stream_id required'})

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_curator') or not startup._curator:
                return jsonify({'success': False, 'error': 'Curator protocol not available'})

            curator = startup._curator
            if not curator.is_curator:
                return jsonify({'success': False, 'error': 'Not authorized as curator'})

            async def do_confirm():
                # If confirmed, reject the oracle; if dismissed, approve
                if confirmed:
                    success = await curator.reject_oracle(oracle_address, stream_id)
                else:
                    success = await curator.approve_oracle(oracle_address, stream_id)

                # Mark the flag as resolved if we have access to stream curation
                if success:
                    curation = curator._stream_curations.get(stream_id)
                    if curation:
                        for flag in curation.flags:
                            if flag.oracle_address == oracle_address and not flag.resolved:
                                flag.resolved = True
                                flag.resolution = resolution or ('confirmed' if confirmed else 'dismissed')
                                break

                return success

            loop = asyncio.new_event_loop()
            try:
                success = loop.run_until_complete(do_confirm())
            finally:
                loop.close()

            return jsonify({
                'success': success,
                'action': 'rejected' if confirmed else 'approved',
                'oracle_address': oracle_address,
                'stream_id': stream_id
            })
        except Exception as e:
            logger.warning(f"Failed to confirm flag: {e}")
            return jsonify({'success': False, 'error': str(e)})

    # ========================================================================
    # CURATOR PROTOCOL ENDPOINTS
    # ========================================================================

    @app.route('/api/p2p/curator/status')
    @login_required
    def api_p2p_curator_status():
        """Get curator protocol status and statistics.

        Note: Curators are the multi-sig signers - they perform curation as one of their jobs.
        The number of active curators equals the number of signers online.
        """
        try:
            from satorineuron.init import start

            # Get signer count for active_curators (signers do curation)
            signer_count = 0
            _, peers = get_p2p_state()
            if peers and hasattr(peers, 'get_connected_signers'):
                signer_count = len(peers.get_connected_signers())

            result = {
                'is_curator': False,
                'started': False,
                'curated_streams': 0,
                'tracked_oracles': 0,
                'unresolved_flags': 0,
                'avg_quality_score': None,
                'active_curators': signer_count,  # Curators = signers
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_curator') and startup._curator:
                curator = startup._curator
                if hasattr(curator, 'get_stats'):
                    stats = curator.get_stats()
                    result.update(stats)
                    # Ensure we don't override signer-based counts with zeros
                    if result['active_curators'] == 0:
                        result['active_curators'] = signer_count

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get curator status: {e}")
            return jsonify({'error': str(e)})

    # ========================================================================
    # ARCHIVER PROTOCOL ENDPOINTS
    # ========================================================================

    @app.route('/api/p2p/archiver/status')
    @login_required
    def api_p2p_archiver_status():
        """Get archiver protocol status and statistics.

        Note: Archivers are the multi-sig signers - they perform archival as one of their jobs.
        The number of archivers online equals the number of signers online.

        Returns different integrity data based on node type:
        - Signers/Archivers: See their local_integrity (their own archive health)
        - Non-signers: See network_integrity (average across all archivers)

        The 'integrity' field is set to the appropriate value based on node type,
        and 'integrity_label' indicates what the value represents.
        """
        try:
            from satorineuron.init import start

            # Get signer count for network_archivers (signers do archival)
            signer_count = 0
            _, peers = get_p2p_state()
            if peers and hasattr(peers, 'get_connected_signers'):
                signer_count = len(peers.get_connected_signers())

            result = {
                'is_archiver': False,
                'started': False,
                'local_archives': 0,
                'total_size_mb': 0,
                'network_archivers': signer_count,  # Archivers = signers
                'redundancy': signer_count,  # Redundancy level = signer count
                # Integrity fields
                'integrity': None,  # Primary display value (local or network depending on node type)
                'integrity_label': 'Network Integrity',  # Label for the integrity value
                'local_integrity': None,  # This node's archive integrity (signers only)
                'network_integrity': None,  # Average integrity across all archivers
                'valid_archives': 0,
                'invalid_archives': 0,
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_archiver') and startup._archiver:
                archiver = startup._archiver
                if hasattr(archiver, 'get_stats'):
                    stats = archiver.get_stats()
                    result.update(stats)
                    # Ensure we don't override signer-based counts with zeros
                    if result['network_archivers'] == 0:
                        result['network_archivers'] = signer_count
                    if result.get('redundancy', 0) == 0:
                        result['redundancy'] = signer_count

                # Get network integrity from announcements
                if hasattr(archiver, 'get_network_integrity'):
                    network_data = archiver.get_network_integrity()
                    result['network_integrity'] = network_data.get('network_integrity')

                # Calculate local integrity for archivers
                if result['is_archiver']:
                    if hasattr(archiver, 'get_available_rounds') and hasattr(archiver, 'verify_archive_integrity'):
                        rounds = archiver.get_available_rounds()
                        if rounds:
                            valid_count = 0
                            for round_id in rounds:
                                try:
                                    if archiver.verify_archive_integrity(round_id):
                                        valid_count += 1
                                except Exception:
                                    pass  # Count as invalid
                            result['valid_archives'] = valid_count
                            result['invalid_archives'] = len(rounds) - valid_count
                            result['local_integrity'] = round((valid_count / len(rounds)) * 100, 1) if rounds else 100.0
                        else:
                            result['local_integrity'] = 100.0  # No archives = 100% integrity

                    # For archivers, show local integrity
                    result['integrity'] = result['local_integrity']
                    result['integrity_label'] = 'Local Integrity'
                else:
                    # For non-archivers, show network integrity
                    result['integrity'] = result['network_integrity']
                    result['integrity_label'] = 'Network Integrity'

            # Fallback if no archiver initialized
            if result['integrity'] is None:
                result['integrity'] = result.get('network_integrity') or 100.0

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get archiver status: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/archiver/rounds')
    @login_required
    def api_p2p_archiver_rounds():
        """Get list of available archived rounds."""
        try:
            from satorineuron.init import start

            result = {'rounds': [], 'metadata': {}}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_archiver') and startup._archiver:
                archiver = startup._archiver
                if hasattr(archiver, 'get_available_rounds'):
                    rounds = archiver.get_available_rounds()
                    result['rounds'] = rounds

                    # Get metadata for each round
                    if hasattr(archiver, 'get_archive_metadata'):
                        for round_id in rounds[:50]:  # Limit to 50 for performance
                            meta = archiver.get_archive_metadata(round_id)
                            if meta:
                                result['metadata'][round_id] = meta.to_dict()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get archived rounds: {e}")
            return jsonify({'rounds': [], 'error': str(e)})

    @app.route('/api/p2p/archiver/round/<round_id>')
    @login_required
    def api_p2p_archiver_round(round_id):
        """Get archived data for a specific round."""
        try:
            from satorineuron.init import start

            result = {'round': None, 'metadata': None}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_archiver') and startup._archiver:
                archiver = startup._archiver
                if hasattr(archiver, 'get_local_archive'):
                    archived = archiver.get_local_archive(round_id)
                    if archived:
                        result['round'] = archived.to_dict()
                        # Don't include full data in list view, just summary
                        result['round']['predictions'] = len(archived.predictions)
                        result['round']['observations'] = len(archived.observations)
                        result['round']['rewards'] = len(archived.rewards)

                if hasattr(archiver, 'get_archive_metadata'):
                    meta = archiver.get_archive_metadata(round_id)
                    if meta:
                        result['metadata'] = meta.to_dict()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get round archive: {e}")
            return jsonify({'round': None, 'error': str(e)})

    @app.route('/api/p2p/archiver/verify/<round_id>')
    @login_required
    def api_p2p_archiver_verify(round_id):
        """Verify integrity of an archived round."""
        try:
            from satorineuron.init import start

            result = {'valid': False, 'round_id': round_id}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_archiver') and startup._archiver:
                archiver = startup._archiver
                if hasattr(archiver, 'verify_archive_integrity'):
                    result['valid'] = archiver.verify_archive_integrity(round_id)

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to verify archive: {e}")
            return jsonify({'valid': False, 'error': str(e)})

    @app.route('/api/p2p/archiver/network/<round_id>')
    @login_required
    def api_p2p_archiver_network(round_id):
        """Get list of archivers that have a round."""
        try:
            from satorineuron.init import start

            result = {'archivers': [], 'round_id': round_id}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_archiver') and startup._archiver:
                archiver = startup._archiver
                if hasattr(archiver, 'get_network_availability'):
                    result['archivers'] = archiver.get_network_availability(round_id)

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get network availability: {e}")
            return jsonify({'archivers': [], 'error': str(e)})

    @app.route('/api/p2p/archiver/archive-round', methods=['POST'])
    @login_required
    def api_p2p_archiver_archive_round():
        """Archive the current round's data (signers only)."""
        try:
            from satorineuron.init import start
            import asyncio

            # Check if user is authorized signer
            from satorip2p.protocol.signer import is_authorized_signer
            startup = start.getStart() if hasattr(start, 'getStart') else None

            wallet_address = None
            if startup and hasattr(startup, 'wallet') and startup.wallet:
                wallet_address = startup.wallet.address

            if not wallet_address or not is_authorized_signer(wallet_address):
                return jsonify({'success': False, 'error': 'Only authorized signers can archive'}), 403

            if not startup or not hasattr(startup, '_archiver') or not startup._archiver:
                return jsonify({'success': False, 'error': 'Archiver not initialized'}), 500

            archiver = startup._archiver
            data = request.get_json(silent=True) or {}
            round_id = data.get('round_id')

            if not round_id:
                # Use current round
                from satorip2p.protocol.uptime import get_current_round
                round_id, _ = get_current_round()

            # Get data to archive (from engine/neuron)
            predictions = data.get('predictions', [])
            observations = data.get('observations', [])
            rewards = data.get('rewards', [])

            async def do_archive():
                return await archiver.archive_round(
                    round_id=round_id,
                    predictions=predictions,
                    observations=observations,
                    rewards=rewards
                )

            loop = asyncio.new_event_loop()
            try:
                success = loop.run_until_complete(do_archive())
                return jsonify({'success': success, 'round_id': round_id})
            finally:
                loop.close()

        except Exception as e:
            logger.warning(f"Failed to archive round: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/archiver/verify-all', methods=['POST'])
    @login_required
    def api_p2p_archiver_verify_all():
        """Verify integrity of all local archives."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_archiver') or not startup._archiver:
                return jsonify({'success': False, 'error': 'Archiver not initialized'}), 500

            archiver = startup._archiver
            results = {
                'total': 0,
                'valid': 0,
                'invalid': 0,
                'invalid_rounds': [],
                'integrity_percentage': 100.0
            }

            if hasattr(archiver, 'get_available_rounds') and hasattr(archiver, 'verify_archive_integrity'):
                rounds = archiver.get_available_rounds()
                results['total'] = len(rounds)

                for round_id in rounds:
                    if archiver.verify_archive_integrity(round_id):
                        results['valid'] += 1
                    else:
                        results['invalid'] += 1
                        results['invalid_rounds'].append(round_id)

                if results['total'] > 0:
                    results['integrity_percentage'] = (results['valid'] / results['total']) * 100

            return jsonify({'success': True, **results})

        except Exception as e:
            logger.warning(f"Failed to verify archives: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/archiver/announce', methods=['POST'])
    @login_required
    def api_p2p_archiver_announce():
        """Announce our archives to the network (signers only)."""
        try:
            from satorineuron.init import start
            import asyncio

            # Check if user is authorized signer
            from satorip2p.protocol.signer import is_authorized_signer
            startup = start.getStart() if hasattr(start, 'getStart') else None

            wallet_address = None
            if startup and hasattr(startup, 'wallet') and startup.wallet:
                wallet_address = startup.wallet.address

            if not wallet_address or not is_authorized_signer(wallet_address):
                return jsonify({'success': False, 'error': 'Only authorized signers can announce'}), 403

            if not startup or not hasattr(startup, '_archiver') or not startup._archiver:
                return jsonify({'success': False, 'error': 'Archiver not initialized'}), 500

            archiver = startup._archiver

            async def do_announce():
                await archiver.announce_archives()
                return True

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(do_announce())
                return jsonify({'success': True, 'message': 'Archives announced to network'})
            finally:
                loop.close()

        except Exception as e:
            logger.warning(f"Failed to announce archives: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/archiver/sync', methods=['POST'])
    @login_required
    def api_p2p_archiver_sync():
        """Sync missing archives from other archivers (signers only)."""
        try:
            from satorineuron.init import start
            import asyncio

            # Check if user is authorized signer
            from satorip2p.protocol.signer import is_authorized_signer
            startup = start.getStart() if hasattr(start, 'getStart') else None

            wallet_address = None
            if startup and hasattr(startup, 'wallet') and startup.wallet:
                wallet_address = startup.wallet.address

            if not wallet_address or not is_authorized_signer(wallet_address):
                return jsonify({'success': False, 'error': 'Only authorized signers can sync'}), 403

            if not startup or not hasattr(startup, '_archiver') or not startup._archiver:
                return jsonify({'success': False, 'error': 'Archiver not initialized'}), 500

            archiver = startup._archiver

            # Get list of rounds we're missing that others have
            missing_rounds = []
            synced_rounds = []

            if hasattr(archiver, '_network_archives') and hasattr(archiver, '_local_archives'):
                local_rounds = set(archiver._local_archives.keys())
                network_rounds = set(archiver._network_archives.keys())
                missing_rounds = list(network_rounds - local_rounds)

                # TODO: Actually request and download missing archives
                # For now, just report what's missing

            return jsonify({
                'success': True,
                'missing_rounds': missing_rounds,
                'synced_rounds': synced_rounds,
                'message': f'Found {len(missing_rounds)} missing rounds'
            })

        except Exception as e:
            logger.warning(f"Failed to sync archives: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    # ========================================================================
    # GOVERNANCE PROTOCOL ENDPOINTS
    # ========================================================================

    @app.route('/api/p2p/governance/status')
    @login_required
    def api_p2p_governance_status():
        """Get governance protocol status and statistics via IPC."""
        try:
            from web.wsgi import _ipc_get
            result = _ipc_get('/p2p/governance/status')
            if result:
                return jsonify(result)
            return jsonify({
                'started': False,
                'total_proposals': 0,
                'active_proposals': 0,
                'passed_proposals': 0,
                'rejected_proposals': 0,
                'my_stake': 0,
                'my_voting_power': 0,
                'can_propose': False,
                'can_vote': False,
            })
        except Exception as e:
            logger.warning(f"Failed to get governance status: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/governance/proposals')
    @login_required
    def api_p2p_governance_proposals():
        """Get all governance proposals via IPC."""
        try:
            from web.wsgi import _ipc_get
            result = _ipc_get('/p2p/governance/proposals')
            if result:
                return jsonify(result)
            return jsonify({'proposals': [], 'active': []})
        except Exception as e:
            logger.warning(f"Failed to get proposals: {e}")
            return jsonify({'proposals': [], 'error': str(e)})

    @app.route('/api/p2p/governance/proposal/<proposal_id>')
    @login_required
    def api_p2p_governance_proposal(proposal_id):
        """Get a specific proposal with tally via IPC."""
        try:
            from web.wsgi import _ipc_get
            result = _ipc_get(f'/p2p/governance/proposal/{proposal_id}')
            if result:
                return jsonify(result)
            return jsonify({'proposal': None, 'tally': None, 'my_vote': None})
        except Exception as e:
            logger.warning(f"Failed to get proposal: {e}")
            return jsonify({'proposal': None, 'error': str(e)})

    @app.route('/api/p2p/governance/propose', methods=['POST'])
    @login_required
    def api_p2p_governance_propose():
        """Create a new governance proposal via IPC."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            result = _ipc_post('/p2p/governance/propose', data)
            if result:
                return jsonify(result)
            return jsonify({'success': False, 'error': 'IPC request failed'})
        except Exception as e:
            logger.warning(f"Failed to create proposal: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/governance/vote', methods=['POST'])
    @login_required
    def api_p2p_governance_vote():
        """Vote on a proposal via IPC."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            result = _ipc_post('/p2p/governance/vote', data)
            if result:
                return jsonify(result)
            return jsonify({'success': False, 'error': 'IPC request failed'})
        except Exception as e:
            logger.warning(f"Failed to vote: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/governance/comment', methods=['POST'])
    @login_required
    def api_p2p_governance_comment():
        """Add a comment to a proposal via IPC."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            result = _ipc_post('/p2p/governance/comment', data)
            if result:
                return jsonify(result)
            return jsonify({'success': False, 'error': 'IPC request failed'})
        except Exception as e:
            logger.warning(f"Failed to add comment: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/governance/comments/<proposal_id>')
    @login_required
    def api_p2p_governance_comments(proposal_id):
        """Get comments for a proposal via IPC."""
        try:
            from web.wsgi import _ipc_get
            result = _ipc_get(f'/p2p/governance/comments/{proposal_id}')
            if result:
                return jsonify(result)
            return jsonify({'comments': []})
        except Exception as e:
            logger.warning(f"Failed to get comments: {e}")
            return jsonify({'comments': [], 'error': str(e)})

    @app.route('/api/p2p/governance/pin', methods=['POST'])
    @login_required
    def api_p2p_governance_pin():
        """Pin or unpin a proposal (signers only) via IPC."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            result = _ipc_post('/p2p/governance/pin', data)
            if result:
                return jsonify(result)
            return jsonify({'success': False, 'error': 'IPC request failed'})
        except Exception as e:
            logger.warning(f"Failed to pin proposal: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/governance/execute', methods=['POST'])
    @login_required
    def api_p2p_governance_execute():
        """Mark a proposal as executed (signers only) via IPC."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            result = _ipc_post('/p2p/governance/execute', data)
            if result:
                return jsonify(result)
            return jsonify({'success': False, 'error': 'IPC request failed'})
        except Exception as e:
            logger.warning(f"Failed to execute proposal: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/governance/emergency-cancel', methods=['POST'])
    @login_required
    def api_p2p_governance_emergency_cancel():
        """Vote to emergency cancel a proposal (signers only, 3-of-5) via IPC."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            result = _ipc_post('/p2p/governance/emergency-cancel', data)
            if result:
                return jsonify(result)
            return jsonify({'success': False, 'error': 'IPC request failed'})
        except Exception as e:
            logger.warning(f"Failed to emergency cancel: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/governance/pinned')
    @login_required
    def api_p2p_governance_pinned():
        """Get all pinned proposals via IPC."""
        try:
            from web.wsgi import _ipc_get
            result = _ipc_get('/p2p/governance/pinned')
            if result:
                # The IPC returns 'pinned' key, adapt to 'proposals' for consistency
                return jsonify({'proposals': result.get('pinned', [])})
            return jsonify({'proposals': []})
        except Exception as e:
            logger.warning(f"Failed to get pinned: {e}")
            return jsonify({'proposals': [], 'error': str(e)})

    @app.route('/api/p2p/governance/voting-power')
    @login_required
    def api_p2p_governance_voting_power():
        """Get detailed voting power breakdown for the current user via IPC."""
        try:
            from web.wsgi import _ipc_get
            result = _ipc_get('/p2p/governance/voting-power')
            if result:
                return jsonify(result)
            return jsonify({
                'success': True,
                'base_stake': 0.0,
                'stake_weight': 1.0,
                'uptime_days': 0,
                'uptime_bonus_pct': 0,
                'is_signer': False,
                'signer_bonus_pct': 0,
                'total_voting_power': 0.0,
                'network_total_power': 0.0,
                'active_voters': 0,
            })
        except Exception as e:
            logger.warning(f"Failed to get voting power: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/uptime/details')
    @login_required
    def api_p2p_uptime_details():
        """Get detailed uptime information for the current user.

        Uses IPC API to get uptime data from main process since gunicorn
        worker can't access main process memory directly.
        """
        try:
            from web.wsgi import get_web_uptime

            # Get uptime data via IPC API
            result = get_web_uptime()

            # Ensure success flag is set
            if 'success' not in result:
                result['success'] = True

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get uptime details: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/infrastructure')
    @login_required
    def api_p2p_infrastructure():
        """Get network infrastructure information (rendezvous, bootstrap peers, etc.)."""
        try:
            from web.wsgi import get_web_p2p_peers, _ipc_get

            result = {
                'success': True,
                'rendezvous_peer': '--',
                'rendezvous_latency_ms': None,
                'bootstrap_peers': [],
                'dht_nodes': 0,
                'mesh_peers': 0,
                'relay_enabled': None,
            }

            # Get P2P peers proxy via IPC
            peers = get_web_p2p_peers()
            if not peers:
                return jsonify(result)

            # Rendezvous status via IPC
            if hasattr(peers, 'get_rendezvous_status'):
                rendezvous_status = peers.get_rendezvous_status()
                if rendezvous_status:
                    result['rendezvous_peer'] = rendezvous_status.get('current_peer', '--')
                    result['rendezvous_latency_ms'] = rendezvous_status.get('latency_ms')
                    # Bootstrap peers from rendezvous status
                    if 'bootstrap_peers' in rendezvous_status:
                        result['bootstrap_peers'] = rendezvous_status['bootstrap_peers']

            # Mesh peers from status
            if hasattr(peers, 'get_status'):
                status = peers.get_status()
                if status:
                    result['mesh_peers'] = status.get('mesh_peer_count', 0)
                    result['dht_nodes'] = status.get('connected_count', 0)

            # Relay status - check if relay is ENABLED (can use relays), not if we ARE a relay
            # Use full-status IPC endpoint to get enable_relay flag
            full_status = _ipc_get('/p2p/full-status')
            if full_status:
                result['relay_enabled'] = full_status.get('enable_relay', False)

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get infrastructure: {e}")
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # GOVERNANCE PAGE
    # =========================================================================

    @app.route('/governance')
    @login_required
    def governance():
        """Render the Governance page."""
        return render_template('governance.html')

    # =========================================================================
    # PREDICTIONS PAGE
    # =========================================================================

    @app.route('/predictions')
    @login_required
    def predictions_page():
        """Render the Predictions page."""
        return render_template('predictions.html')

    # =========================================================================
    # ORACLE PAGE (Oracles + Streams)
    # =========================================================================

    @app.route('/oracle')
    @login_required
    def oracle_page():
        """Render the Oracle/Streams page."""
        return render_template('oracle.html')

    # =========================================================================
    # SIGNER PAGE
    # =========================================================================

    @app.route('/signer')
    @login_required
    def signer_dashboard():
        """Render the Signer Dashboard page (signers only)."""
        return render_template('signer.html')

    @app.route('/api/p2p/signer/status')
    @login_required
    def api_p2p_signer_status():
        """Get signer status - whether user is a signer."""
        try:
            from satorip2p.protocol.signer import is_authorized_signer
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            evrmore_address = ""
            if startup and hasattr(startup, '_identity_bridge') and startup._identity_bridge:
                evrmore_address = startup._identity_bridge.evrmore_address or ""

            is_signer = is_authorized_signer(evrmore_address)

            return jsonify({
                'is_signer': is_signer,
                'evrmore_address': evrmore_address,
            })
        except Exception as e:
            logger.warning(f"Failed to get signer status: {e}")
            return jsonify({'is_signer': False, 'error': str(e)})

    @app.route('/api/p2p/delegations')
    @login_required
    def api_p2p_delegations():
        """Get delegation info for current user."""
        try:
            from satorineuron.init import start
            import asyncio

            result = {'my_delegate': None, 'my_children': []}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_delegation_manager') and startup._delegation_manager:
                manager = startup._delegation_manager

                # Run async methods
                async def get_delegation_data():
                    data = {'my_delegate': None, 'my_children': []}
                    if hasattr(manager, 'get_my_delegate'):
                        delegate = await manager.get_my_delegate()
                        if delegate:
                            data['my_delegate'] = {
                                'parent_address': delegate.parent_address,
                                'amount': delegate.amount,
                                'charity': delegate.charity,
                            }
                    if hasattr(manager, 'get_proxy_children'):
                        children = await manager.get_proxy_children()
                        for child in children:
                            data['my_children'].append({
                                'child_address': child.child_address,
                                'amount': child.amount,
                                'charity': child.charity,
                            })
                    return data

                try:
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(get_delegation_data())
                finally:
                    loop.close()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get delegations: {e}")
            return jsonify({'my_delegate': None, 'my_children': [], 'error': str(e)})

    @app.route('/api/p2p/pools')
    @login_required
    def api_p2p_pools():
        """Get P2P pool information."""
        try:
            from satorineuron.init import start
            import asyncio

            result = {'my_pool': None, 'lending_to': None, 'available_pools': []}

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                manager = startup._lending_manager

                # Get my pool config
                if hasattr(manager, '_my_pool_config') and manager._my_pool_config:
                    config = manager._my_pool_config
                    result['my_pool'] = {
                        'vault_address': config.vault_address,
                        'pool_size_limit': config.pool_size_limit,
                        'accepting': config.accepting,
                    }

                # Get lending target
                async def get_lend_addr():
                    if hasattr(manager, 'get_current_lend_address'):
                        return await manager.get_current_lend_address()
                    return None

                try:
                    loop = asyncio.new_event_loop()
                    result['lending_to'] = loop.run_until_complete(get_lend_addr())
                finally:
                    loop.close()

                # Get available pools
                if hasattr(manager, '_pool_configs'):
                    for addr, cfg in list(manager._pool_configs.items())[:10]:
                        if cfg.accepting:
                            result['available_pools'].append({
                                'address': addr,
                                'pool_size_limit': cfg.pool_size_limit,
                            })

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get P2P pools: {e}")
            return jsonify({'my_pool': None, 'lending_to': None, 'available_pools': [], 'error': str(e)})

    # =========================================================================
    # NETWORK DASHBOARD PAGE & API ENDPOINTS
    # =========================================================================

    @app.route('/network')
    @login_required
    def network():
        """Render the P2P Network Dashboard page."""
        return render_template('network.html')

    @app.route('/api/p2p-status')
    @login_required
    def api_p2p_status():
        """Get comprehensive P2P status for network dashboard via IPC."""
        try:
            from web.wsgi import _ipc_get
            from satorineuron.init import start

            result = {
                'peer_count': 0,
                'peer_id': None,
                'nat_type': 'Unknown',
                'networking_mode': 'central',
                'uptime_pct': 0.0,
                'consensus_phase': 'Idle',
                'peers': [],
            }

            # Get networking mode
            if hasattr(start, '_get_networking_mode'):
                result['networking_mode'] = start._get_networking_mode()

            # Get all data via IPC (web worker can't access main process memory)
            full_status = _ipc_get('/p2p/full-status') or {}
            peers_result = _ipc_get('/p2p/peers') or {}
            latencies_result = _ipc_get('/p2p/latencies') or {}
            identify_result = _ipc_get('/p2p/identify/known') or {}
            oracle_result = _ipc_get('/p2p/oracle/known') or {}
            identity_result = _ipc_get('/p2p/identity') or {}
            multiaddrs_result = _ipc_get('/p2p/multiaddrs') or {}

            # Set peer_id from full status
            if full_status.get('peer_id'):
                result['peer_id'] = full_status['peer_id']

            # Set peer count
            result['peer_count'] = full_status.get('connected_count', 0)

            # Set evrmore address from identity
            if identity_result.get('address'):
                result['evrmore_address'] = identity_result['address']

            # Build oracle peer IDs set for role supplementation
            oracle_peer_ids = set()
            if oracle_result.get('oracles'):
                for oracle in oracle_result['oracles']:
                    if oracle.get('peer_id'):
                        oracle_peer_ids.add(oracle['peer_id'])

            # Build known peers map from identify protocol
            known_peers = {}
            if identify_result.get('peers'):
                for peer in identify_result['peers']:
                    if peer.get('peer_id'):
                        known_peers[peer['peer_id']] = peer

            # Get latencies map
            latencies = latencies_result.get('latencies', {})

            # Build peers list
            if peers_result.get('peers'):
                for peer_info in peers_result['peers'][:20]:  # Limit to 20
                    peer_id = peer_info.get('peer_id')
                    if not peer_id:
                        continue

                    # Get roles from identify protocol
                    roles = ['node']  # Default base role
                    if peer_id in known_peers:
                        known = known_peers[peer_id]
                        if known.get('roles'):
                            roles = list(known['roles'])

                    # Supplement with oracle role if peer is a registered oracle
                    if peer_id in oracle_peer_ids and 'oracle' not in roles:
                        roles.append('oracle')

                    # Format role for display
                    role = ', '.join(r.capitalize() for r in roles) if roles else 'Node'

                    result['peers'].append({
                        'id': peer_id,
                        'latency': latencies.get(peer_id),
                        'location': None,
                        'streams': 0,
                        'role': role,
                    })

            # Add multiaddress for sharing with friends
            addrs = multiaddrs_result.get('multiaddrs', [])
            if addrs and result.get('peer_id'):
                full_addrs = []
                for addr in addrs:
                    if '/p2p/' not in addr:
                        full_addrs.append(f"{addr}/p2p/{result['peer_id']}")
                    else:
                        full_addrs.append(addr)
                result['multiaddrs'] = full_addrs
                if full_addrs:
                    result['multiaddr'] = full_addrs[0]

            # Get NAT type from full status
            nat = full_status.get('nat_type')
            if nat == "PUBLIC":
                result['nat_type'] = "Public"
            elif nat == "PRIVATE":
                result['nat_type'] = "Private (NAT)"
            else:
                # NAT is UNKNOWN or not set - check if we're in Docker
                try:
                    from satorip2p.nat.docker import detect_docker_environment
                    docker_info = detect_docker_environment()
                    if docker_info.in_container:
                        if docker_info.is_bridge_mode:
                            result['nat_type'] = "Private (Docker)"
                        elif docker_info.is_host_mode:
                            result['nat_type'] = "Host Mode"
                        else:
                            result['nat_type'] = "Docker"
                except Exception:
                    pass

            # Get uptime tracker info from StartupDag
            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup:
                if hasattr(startup, '_uptime_tracker') and startup._uptime_tracker:
                    tracker = startup._uptime_tracker
                    if hasattr(tracker, 'get_uptime_percentage'):
                        result['uptime_pct'] = tracker.get_uptime_percentage()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"P2P status failed: {e}")
            return jsonify({
                'peer_count': 0,
                'networking_mode': 'central',
                'error': str(e)
            })

    @app.route('/api/p2p/unified-status')
    @login_required
    def api_p2p_unified_status():
        """
        Get comprehensive P2P status with atomic snapshot of all peer data.

        This is the single source of truth for all P2P stats on the network page.
        Returns connected peers, known peers, and all counts in one atomic call
        to prevent inconsistencies from multiple API calls.
        """
        try:
            from satorineuron.init import start
            from web.wsgi import get_web_uptime
            identity, peers = get_p2p_state()

            # Get connected peer IDs first (atomic snapshot)
            connected_peer_ids = set()
            if peers and hasattr(peers, 'get_connected_peers'):
                connected_peer_ids = set(str(p) for p in peers.get_connected_peers())

            # Get known peer identities
            known_peers = []
            role_counts = {
                'predictor': 0, 'oracle': 0, 'signer': 0,
                'relay': 0, 'pool-operator': 0, 'delegate': 0, 'charity': 0
            }

            # Get peer latencies if available
            peer_latencies = {}
            if peers and hasattr(peers, 'get_all_peer_latencies'):
                peer_latencies = peers.get_all_peer_latencies()

            # Track which peer IDs we've added to known_peers
            known_peer_ids = set()

            if peers and hasattr(peers, 'get_known_peer_identities'):
                identities = peers.get_known_peer_identities()
                for peer_id, ident in identities.items():
                    peer_id_str = str(peer_id)
                    known_peer_ids.add(peer_id_str)
                    is_connected = peer_id_str in connected_peer_ids

                    roles = getattr(ident, 'roles', [])
                    role = roles[0] if roles else 'predictor'

                    # Count by role
                    role_key = role.lower().replace(' ', '-')
                    if role_key in role_counts:
                        role_counts[role_key] += 1
                    else:
                        role_counts['predictor'] += 1

                    # Get latency for this peer
                    latency_ms = peer_latencies.get(peer_id_str)

                    known_peers.append({
                        'peer_id': peer_id_str,
                        'evrmore_address': getattr(ident, 'evrmore_address', ''),
                        'roles': roles,
                        'role': role,
                        'protocols': getattr(ident, 'protocols', []),
                        'agent_version': getattr(ident, 'agent_version', ''),
                        'timestamp': getattr(ident, 'timestamp', 0),
                        'connected': is_connected,
                        'listen_addresses': getattr(ident, 'listen_addresses', []),
                        'latency': latency_ms,  # RTT in milliseconds
                    })

            # Add connected peers that don't have identities yet
            # Try to fill in from cache if available
            cached_data = _get_cached_peers()
            cached_peers_by_id = {p['peer_id']: p for p in cached_data.get('known_peers', [])}

            for peer_id_str in connected_peer_ids:
                if peer_id_str not in known_peer_ids:
                    latency_ms = peer_latencies.get(peer_id_str)

                    # Check if we have cached identify data for this peer
                    cached_peer = cached_peers_by_id.get(peer_id_str)
                    if cached_peer and cached_peer.get('evrmore_address'):
                        # Use cached data but mark as connected and update latency
                        role = cached_peer.get('role', 'predictor')
                        role_counts[role] = role_counts.get(role, 0) + 1
                        known_peers.append({
                            'peer_id': peer_id_str,
                            'evrmore_address': cached_peer.get('evrmore_address', ''),
                            'roles': cached_peer.get('roles', ['predictor']),
                            'role': role,
                            'protocols': cached_peer.get('protocols', []),
                            'agent_version': cached_peer.get('agent_version', ''),
                            'timestamp': cached_peer.get('timestamp', 0),
                            'connected': True,
                            'listen_addresses': cached_peer.get('listen_addresses', []),
                            'latency': latency_ms,
                            'from_cache': True,  # Flag to indicate data is from cache
                        })
                    else:
                        # No cached data, use skeleton with pending flag
                        role_counts['predictor'] += 1
                        known_peers.append({
                            'peer_id': peer_id_str,
                            'evrmore_address': '',
                            'roles': ['predictor'],
                            'role': 'predictor',
                            'protocols': [],
                            'agent_version': '',
                            'timestamp': 0,
                            'connected': True,
                            'listen_addresses': [],
                            'latency': latency_ms,
                            'pending_identify': True,  # Flag for UI to show loading
                        })

            # Build result with atomic counts
            connected_count = len(connected_peer_ids)
            known_count = len(known_peers)

            result = {
                'success': True,
                'timestamp': int(time.time() * 1000),  # For cache busting

                # Counts (single source of truth)
                'connected_count': connected_count,
                'known_count': known_count,
                'role_counts': role_counts,

                # Connected peer IDs (for Peer Tools section)
                'connected_peers': list(connected_peer_ids),

                # Full known peer list with connected status
                'known_peers': known_peers,

                # My identity info
                'my_peer_id': str(peers.peer_id) if peers and hasattr(peers, 'peer_id') else None,
                'my_evrmore_address': identity.address if identity and hasattr(identity, 'address') else None,
            }

            # Add public key with multiple fallbacks
            public_key = None
            # Try identity.pubkey first
            if identity and hasattr(identity, 'pubkey'):
                public_key = identity.pubkey
            # Try peers.public_key second
            if not public_key and peers and hasattr(peers, 'public_key'):
                public_key = peers.public_key
            # Try vault.publicKey third
            if not public_key:
                try:
                    if start.StartupDag.vault and hasattr(start.StartupDag.vault, 'publicKey'):
                        public_key = start.StartupDag.vault.publicKey
                except Exception:
                    pass
            if public_key:
                result['my_public_key'] = public_key

            # Add multiaddress if available
            if peers and hasattr(peers, 'public_addresses') and result['my_peer_id']:
                addrs = peers.public_addresses
                if addrs:
                    full_addrs = []
                    for addr in addrs:
                        if '/p2p/' not in addr:
                            full_addrs.append(f"{addr}/p2p/{result['my_peer_id']}")
                        else:
                            full_addrs.append(addr)
                    result['multiaddrs'] = full_addrs
                    if full_addrs:
                        result['multiaddr'] = full_addrs[0]

            # Add uptime info via IPC (gunicorn can't access main process memory)
            uptime_data = get_web_uptime()
            if uptime_data and uptime_data.get('success'):
                result['uptime_pct'] = uptime_data.get('uptime_percentage', 0)
                # last_heartbeat_message would need separate IPC endpoint if needed

            # Add network average latency
            if peers and hasattr(peers, 'get_network_avg_latency'):
                avg_latency = peers.get_network_avg_latency()
                if avg_latency is not None:
                    result['avg_latency'] = round(avg_latency, 1)

            # Add mesh peer count from GossipSub
            if peers and hasattr(peers, 'get_pubsub_debug'):
                try:
                    debug_info = peers.get_pubsub_debug()
                    mesh_info = debug_info.get('mesh', {})
                    # Count unique peers across all topic meshes
                    all_mesh_peers = set()
                    for topic, peer_list in mesh_info.items():
                        all_mesh_peers.update(peer_list)
                    result['mesh_peer_count'] = len(all_mesh_peers)
                    logger.info(f"Mesh peer count from pubsub: {len(all_mesh_peers)}")
                except Exception as e:
                    logger.warning(f"Failed to get mesh peer count: {e}")
                    result['mesh_peer_count'] = 0
            else:
                logger.warning(f"Cannot get mesh peers: peers={peers is not None}, has_method={hasattr(peers, 'get_pubsub_debug') if peers else False}")

            # Save peers to cache for Option D (instant display on restart)
            if known_peers:
                _save_peer_cache(known_peers)

            return jsonify(result)

        except Exception as e:
            logger.warning(f"Unified P2P status failed: {e}")
            # Try to use cached peers on failure (Option D)
            cached = _get_cached_peers()
            cached_peers = cached.get('known_peers', [])
            # Mark all cached peers as disconnected since we can't verify
            for peer in cached_peers:
                peer['connected'] = False
                peer['cached'] = True  # Flag to indicate stale data
            return jsonify({
                'success': False,
                'error': str(e),
                'connected_count': 0,
                'known_count': len(cached_peers),
                'connected_peers': [],
                'known_peers': cached_peers,
                'from_cache': True,
                'cache_timestamp': cached.get('timestamp', 0),
            })

    @app.route('/api/network/stats')
    @login_required
    def api_network_stats():
        """Get network statistics for dashboard charts."""
        try:
            from satorineuron.init import start

            result = {
                'active_streams': 0,
                'message_rate': 0,
                'avg_latency': 0,
                'consensus_rate': 0,
                'predictions_count': 0,
                'observations_count': 0,
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup:
                # Get stream registry info
                if hasattr(startup, '_stream_registry') and startup._stream_registry:
                    registry = startup._stream_registry
                    if hasattr(registry, 'get_active_count'):
                        result['active_streams'] = registry.get_active_count()

                # Get prediction protocol info
                if hasattr(startup, '_prediction_protocol') and startup._prediction_protocol:
                    proto = startup._prediction_protocol
                    if hasattr(proto, 'get_recent_prediction_count'):
                        result['predictions_count'] = proto.get_recent_prediction_count()

                # Get oracle network info
                if hasattr(startup, '_oracle_network') and startup._oracle_network:
                    oracle = startup._oracle_network
                    if hasattr(oracle, 'get_recent_observation_count'):
                        result['observations_count'] = oracle.get_recent_observation_count()

                # Estimate message rate from heartbeats
                if hasattr(startup, '_uptime_tracker') and startup._uptime_tracker:
                    tracker = startup._uptime_tracker
                    if hasattr(tracker, 'get_message_rate'):
                        result['message_rate'] = tracker.get_message_rate()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Network stats failed: {e}")
            return jsonify(result)

    @app.route('/api/network/activity')
    @login_required
    def api_network_activity():
        """Get hourly network activity data for charts.

        Returns real-time activity counts from the P2P WebSocket bridge via IPC.
        """
        try:
            from web.wsgi import get_web_activity_stats

            result = get_web_activity_stats()
            if result and result.get('success'):
                return jsonify({
                    'success': True,
                    'activity': result.get('hourly', {}),
                    'counts': result.get('counts', {}),
                })
            else:
                return jsonify({
                    'success': False,
                    'activity': {},
                    'counts': {
                        'predictions': 0,
                        'observations': 0,
                        'heartbeats': 0,
                        'consensus_votes': 0,
                        'governance': 0,
                    },
                    'error': result.get('error', 'IPC unavailable')
                })
        except Exception as e:
            logger.warning(f"Network activity failed: {e}")
            # Return empty data structure on error
            return jsonify({
                'success': False,
                'activity': {
                    'labels': [],
                    'predictions': [],
                    'observations': [],
                    'heartbeats': [],
                    'consensus': [],
                    'governance': [],
                },
                'counts': {
                    'predictions': 0,
                    'observations': 0,
                    'heartbeats': 0,
                    'consensus_votes': 0,
                    'governance': 0,
                },
                'error': str(e)
            })

    @app.route('/api/network/recent-events')
    @login_required
    def api_network_recent_events():
        """Get recent network events for polling-based live updates.

        This is a fallback for when WebSocket is not available.
        The frontend can poll this endpoint to get real-time events.
        """
        try:
            from web.wsgi import get_web_recent_events

            limit = request.args.get('limit', 50, type=int)
            since = request.args.get('since', None, type=float)
            events = get_web_recent_events(limit=limit, since=since)
            return jsonify({
                'success': True,
                'events': events,
            })
        except Exception as e:
            logger.warning(f"Recent events failed: {e}")
            return jsonify({
                'success': False,
                'events': [],
                'error': str(e)
            })

    @app.route('/api/peers/list')
    @login_required
    def api_peers_list():
        """Get list of connected peers via IPC API."""
        try:
            from web.wsgi import _ipc_get

            # Get connected peers from IPC
            peers_result = _ipc_get('/p2p/peers')
            if not peers_result or 'peers' not in peers_result:
                return jsonify({'peers': []})

            # Get identify info for roles
            identify_result = _ipc_get('/p2p/identify/known')
            identities = identify_result.get('identities', {}) if identify_result else {}

            # Get latencies
            latencies_result = _ipc_get('/p2p/latencies')
            latencies = latencies_result.get('latencies', {}) if latencies_result else {}

            # Get oracle registrations to supplement roles
            oracle_result = _ipc_get('/p2p/oracle/known')
            oracle_peer_ids = set()
            if oracle_result and 'oracles' in oracle_result:
                for oracle in oracle_result['oracles']:
                    if oracle.get('peer_id'):
                        oracle_peer_ids.add(oracle['peer_id'])

            peers = []
            for p in peers_result['peers']:
                peer_id = p.get('peer_id', '')
                # Get roles from identify protocol
                roles = ['node']  # Default base role
                if peer_id in identities:
                    peer_info = identities[peer_id]
                    if peer_info.get('roles'):
                        roles = list(peer_info['roles'])
                # Supplement with oracle info if peer is a registered oracle
                if peer_id in oracle_peer_ids and 'oracle' not in roles:
                    roles.append('oracle')
                # Format roles for display
                role = ', '.join(r.capitalize() for r in roles) if roles else 'Node'
                peers.append({
                    'id': peer_id,
                    'latency': latencies.get(peer_id),
                    'location': None,
                    'streams': 0,
                    'role': role,
                    'roles': roles,  # Also include raw roles array for frontend
                })

            return jsonify({'peers': peers})
        except Exception as e:
            logger.warning(f"Peers list failed: {e}")
            return jsonify({'peers': [], 'error': str(e)})

    @app.route('/api/rewards/pending')
    @login_required
    def api_rewards_pending():
        """Get pending rewards for claiming."""
        try:
            from satorineuron.init import start
            import asyncio

            rewards = []
            startup = start.getStart() if hasattr(start, 'getStart') else None

            # Try P2P rewards first
            if startup and hasattr(startup, '_prediction_protocol') and startup._prediction_protocol:
                proto = startup._prediction_protocol
                if hasattr(proto, 'get_pending_rewards'):
                    async def get_rewards():
                        return await proto.get_pending_rewards()
                    try:
                        loop = asyncio.new_event_loop()
                        pending = loop.run_until_complete(get_rewards())
                        for r in pending:
                            rewards.append({
                                'round_id': r.round_id,
                                'stream_id': getattr(r, 'stream_id', None),
                                'amount': r.amount,
                                'score': getattr(r, 'score', None),
                                'multiplier': getattr(r, 'multiplier', 1.0),
                            })
                    finally:
                        loop.close()

            return jsonify({'rewards': rewards})
        except Exception as e:
            logger.warning(f"Pending rewards failed: {e}")
            return jsonify({'rewards': [], 'error': str(e)})

    @app.route('/api/rewards/history')
    @login_required
    def api_rewards_history():
        """Get reward history."""
        try:
            from satorineuron.init import start
            import asyncio

            limit = request.args.get('limit', 20, type=int)
            history = []
            startup = start.getStart() if hasattr(start, 'getStart') else None

            if startup and hasattr(startup, '_prediction_protocol') and startup._prediction_protocol:
                proto = startup._prediction_protocol
                if hasattr(proto, 'get_reward_history'):
                    async def get_history():
                        return await proto.get_reward_history(limit=limit)
                    try:
                        loop = asyncio.new_event_loop()
                        hist = loop.run_until_complete(get_history())
                        for h in hist:
                            history.append({
                                'date': h.date if hasattr(h, 'date') else '',
                                'round_id': h.round_id,
                                'amount': h.amount,
                                'txid': getattr(h, 'txid', None),
                            })
                    finally:
                        loop.close()

            return jsonify({'history': history})
        except Exception as e:
            logger.warning(f"Reward history failed: {e}")
            return jsonify({'history': [], 'error': str(e)})

    @app.route('/api/rewards/claim', methods=['POST'])
    @login_required
    def api_rewards_claim():
        """Claim specific rewards."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            round_ids = data.get('round_ids', [])

            if not round_ids:
                return jsonify({'success': False, 'error': 'No round_ids provided'})

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_prediction_protocol') and startup._prediction_protocol:
                proto = startup._prediction_protocol
                if hasattr(proto, 'claim_rewards'):
                    async def do_claim():
                        return await proto.claim_rewards(round_ids)
                    try:
                        loop = asyncio.new_event_loop()
                        result = loop.run_until_complete(do_claim())
                        return jsonify({
                            'success': True,
                            'txid': getattr(result, 'txid', None),
                        })
                    finally:
                        loop.close()

            return jsonify({'success': False, 'error': 'P2P prediction protocol not available'})
        except Exception as e:
            logger.error(f"Claim rewards failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/rewards/claim-all', methods=['POST'])
    @login_required
    def api_rewards_claim_all():
        """Claim all pending rewards."""
        try:
            from satorineuron.init import start
            import asyncio

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_prediction_protocol') and startup._prediction_protocol:
                proto = startup._prediction_protocol
                if hasattr(proto, 'claim_all_rewards'):
                    async def do_claim_all():
                        return await proto.claim_all_rewards()
                    try:
                        loop = asyncio.new_event_loop()
                        result = loop.run_until_complete(do_claim_all())
                        return jsonify({
                            'success': True,
                            'txid': getattr(result, 'txid', None),
                        })
                    finally:
                        loop.close()

            return jsonify({'success': False, 'error': 'P2P prediction protocol not available'})
        except Exception as e:
            logger.error(f"Claim all rewards failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/networking-mode', methods=['GET', 'POST'])
    @login_required
    def api_networking_mode():
        """Get or set the networking mode."""
        try:
            from satorineuron.init import start

            if request.method == 'GET':
                mode = 'central'
                if hasattr(start, '_get_networking_mode'):
                    mode = start._get_networking_mode()
                return jsonify({'mode': mode})

            # POST - set mode
            data = request.get_json() or {}
            new_mode = data.get('mode')

            if new_mode not in ('central', 'hybrid', 'p2p'):
                return jsonify({'success': False, 'error': 'Invalid mode. Must be central, hybrid, or p2p'})

            # Use the proper setter function
            if hasattr(start, '_set_networking_mode'):
                success = start._set_networking_mode(new_mode)
                if success:
                    return jsonify({
                        'success': True,
                        'mode': new_mode,
                        'message': 'Mode updated. Restart required for full effect.',
                        'restart_required': True
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to set mode'})
            else:
                # Fallback: save to YAML config file
                import os
                import yaml
                from satorineuron import config as neuron_config
                config_path = neuron_config.root('config', 'config.yaml')
                cfg = {}
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        cfg = yaml.safe_load(f) or {}
                cfg['networking mode'] = new_mode
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                return jsonify({
                    'success': True,
                    'mode': new_mode,
                    'message': 'Mode saved. Restart required for full effect.',
                    'restart_required': True
                })

        except Exception as e:
            logger.error(f"Networking mode failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # RESTART ENDPOINT
    # =========================================================================

    @app.route('/restart')
    @login_required
    def restart_neuron():
        """Restart the neuron to apply configuration changes.

        Shows a restart page then triggers system exit so Docker can restart.
        """
        import threading
        import os
        import signal

        def do_restart():
            import time
            time.sleep(1)  # Give time for response to be sent
            os.kill(os.getpid(), signal.SIGTERM)

        # Start restart in background thread
        threading.Thread(target=do_restart, daemon=True).start()

        return '''<!DOCTYPE html>
<html>
<head>
    <title>Restarting...</title>
    <meta http-equiv="refresh" content="5;url=/">
    <style>
        body {
            background: #0d0d0d;
            color: #fff;
            font-family: Inter, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container { text-align: center; }
        .spinner {
            border: 4px solid #333;
            border-top: 4px solid #5a5aff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="spinner"></div>
        <h2>Restarting Satori Neuron...</h2>
        <p>This page will refresh automatically.</p>
    </div>
</body>
</html>'''

    # =========================================================================
    # AI ENGINE TRAINING DELAY CONTROL (from upstream)
    # =========================================================================

    @app.route('/api/engine/training-delay', methods=['GET'])
    @login_required
    def get_training_delay():
        """Get current AI engine training delay setting.

        Returns:
            JSON with delay_seconds (int)
        """
        try:
            from satorineuron import config

            # Default to 10 minutes if not set
            delay = config.get().get('training_delay', 600)

            return jsonify({
                'delay_seconds': delay
            })
        except Exception as e:
            logger.error(f"Error getting training delay: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/engine/training-delay', methods=['POST'])
    @login_required
    def set_training_delay():
        """Set AI engine training delay.

        Request Body:
            {
                "delay_seconds": int  // 0 to 86400
            }

        Returns:
            JSON with updated delay_seconds
        """
        try:
            from satorineuron import config
            from start import getStart

            data = request.get_json()
            delay_seconds = data.get('delay_seconds')

            # Validate
            if delay_seconds is None:
                return jsonify({'error': 'delay_seconds required'}), 400

            delay_seconds = int(delay_seconds)

            if delay_seconds < 0 or delay_seconds > 86400:
                return jsonify({'error': 'delay_seconds must be between 0 and 86400'}), 400

            # Save to config
            config.add(data={'training_delay': delay_seconds})

            # Update running engine instances
            try:
                startup = getStart()
                if hasattr(startup, 'aiengine') and startup.aiengine is not None:
                    for streamUuid, streamModel in startup.aiengine.streamModels.items():
                        streamModel.trainingDelay = delay_seconds
                        logger.info(f"Updated training delay for stream {streamUuid}: {delay_seconds}s")
            except Exception as e:
                # Log but don't fail if engine isn't running
                logger.warning(f"Could not update running engine instances: {e}")

            return jsonify({
                'delay_seconds': delay_seconds,
                'message': 'Training delay updated successfully'
            })

        except Exception as e:
            logger.error(f"Error setting training delay: {e}")
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # TREASURY ALERTS ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/treasury/status')
    @login_required
    def api_treasury_status():
        """Get current treasury status and active alerts."""
        try:
            from satorineuron.init import start

            result = {
                'severity': 'info',
                'satori_level': 'Normal',
                'evr_level': 'Normal',
                'active_alert': None,
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_alert_manager') and startup._alert_manager:
                manager = startup._alert_manager
                if hasattr(manager, 'get_current_status'):
                    status = manager.get_current_status()
                    if status:
                        result['severity'] = status.severity
                        result['satori_level'] = status.satori_level
                        result['evr_level'] = status.evr_level
                        result['satori_balance'] = getattr(status, 'satori_balance', None)
                        result['evr_balance'] = getattr(status, 'evr_balance', None)

                if hasattr(manager, 'get_active_alert'):
                    alert = manager.get_active_alert()
                    if alert:
                        result['active_alert'] = {
                            'type': alert.alert_type,
                            'severity': alert.severity,
                            'message': alert.message,
                            'timestamp': alert.timestamp,
                        }

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Treasury status failed: {e}")
            return jsonify({
                'severity': 'info',
                'satori_level': 'Unknown',
                'evr_level': 'Unknown',
                'error': str(e)
            })

    @app.route('/api/p2p/treasury/deferred')
    @login_required
    def api_treasury_deferred():
        """Get deferred rewards for current user."""
        try:
            from satorineuron.init import start

            result = {
                'total_pending': 0.0,
                'deferred_count': 0,
                'deferred_rewards': [],
            }

            # Get wallet address
            wallet_manager = get_or_create_session_vault()
            if not wallet_manager or not wallet_manager.vault:
                return jsonify(result)

            vault_address = wallet_manager.vault.address if hasattr(wallet_manager.vault, 'address') else None
            if not vault_address:
                return jsonify(result)

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_deferred_rewards_manager') and startup._deferred_rewards_manager:
                manager = startup._deferred_rewards_manager
                if hasattr(manager, 'get_deferred_for_address'):
                    summary = manager.get_deferred_for_address(vault_address)
                    if summary:
                        result['total_pending'] = summary.total_pending
                        result['deferred_count'] = summary.deferred_count
                        result['oldest_deferred_at'] = summary.oldest_deferred_at
                        result['newest_deferred_at'] = summary.newest_deferred_at
                        for r in summary.deferred_rewards[:10]:  # Limit to 10 items
                            result['deferred_rewards'].append({
                                'round_id': r.round_id,
                                'amount': r.amount,
                                'reason': r.reason,
                                'created_at': r.created_at,
                            })

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Deferred rewards failed: {e}")
            return jsonify({
                'total_pending': 0.0,
                'deferred_count': 0,
                'error': str(e)
            })

    @app.route('/api/p2p/treasury/alerts/history')
    @login_required
    def api_treasury_alert_history():
        """Get recent alert history."""
        try:
            from satorineuron.init import start

            limit = request.args.get('limit', 20, type=int)
            history = []

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_alert_manager') and startup._alert_manager:
                manager = startup._alert_manager
                if hasattr(manager, 'get_alert_history'):
                    raw = manager.get_alert_history(limit=limit)
                    for alert in raw:
                        history.append({
                            'type': alert.alert_type,
                            'severity': alert.severity,
                            'message': alert.message,
                            'timestamp': alert.timestamp,
                            'resolved': getattr(alert, 'resolved', False),
                            'resolved_at': getattr(alert, 'resolved_at', None),
                        })

            return jsonify({'history': history})
        except Exception as e:
            logger.warning(f"Alert history failed: {e}")
            return jsonify({'history': [], 'error': str(e)})

    @app.route('/api/p2p/treasury/total-deferred')
    @login_required
    def api_treasury_total_deferred():
        """Get total deferred rewards across all users (for signer dashboard)."""
        try:
            from satorineuron.init import start

            result = {
                'total_deferred': 0.0,
                'deferred_count': 0,
                'unique_addresses': 0,
                'deferred_rounds': 0,
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_deferred_rewards_manager') and startup._deferred_rewards_manager:
                manager = startup._deferred_rewards_manager
                if hasattr(manager, 'get_stats'):
                    stats = manager.get_stats()
                    result['total_deferred'] = stats.get('total_deferred', 0.0)
                    result['deferred_count'] = stats.get('deferred_count', 0)
                    result['unique_addresses'] = stats.get('unique_addresses', 0)
                    result['deferred_rounds'] = stats.get('deferred_rounds', 0)
                    result['oldest_deferral'] = stats.get('oldest_deferral')
                    result['newest_deferral'] = stats.get('newest_deferral')

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Total deferred failed: {e}")
            return jsonify(result)

    # =========================================================================
    # PROTOCOL VERSION ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/version')
    @login_required
    def api_p2p_version():
        """Get protocol version info and peer version distribution."""
        try:
            from web.wsgi import get_web_version

            # Use IPC to get version info (gunicorn can't access main process)
            version_data = get_web_version()

            result = {
                'current_version': version_data.get('current_version', '1.0.0'),
                'min_supported': '1.0.0',
                'peer_stats': {
                    'total_peers': 0,
                    'compatible_peers': 0,
                    'version_distribution': {},
                },
                'features': version_data.get('features', []),
                'upgrade_progress': 0.0,
            }

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Version info failed: {e}")
            return jsonify({'error': str(e), 'current_version': '1.0.0'})

    @app.route('/api/p2p/protocols/descriptions')
    @login_required
    def api_p2p_protocol_descriptions():
        """Get protocol descriptions for UI tooltips."""
        try:
            from satorip2p.protocol import get_all_protocol_descriptions
            descriptions = get_all_protocol_descriptions()
            return jsonify({
                'descriptions': descriptions,
                'count': len(descriptions)
            })
        except ImportError:
            # Fallback if satorip2p not available
            return jsonify({
                'descriptions': {},
                'count': 0,
                'error': 'Protocol registry not available'
            })
        except Exception as e:
            logger.warning(f"Protocol descriptions failed: {e}")
            return jsonify({'descriptions': {}, 'error': str(e)})

    @app.route('/api/p2p/protocols/info')
    @login_required
    def api_p2p_protocol_info():
        """Get full protocol info including categories and required flags."""
        try:
            from satorip2p.protocol import (
                ALL_PROTOCOLS,
                get_required_protocols,
                get_satori_protocols
            )

            protocols = {}
            for proto_id, proto in ALL_PROTOCOLS.items():
                protocols[proto_id] = {
                    'id': proto.id,
                    'name': proto.name,
                    'version': proto.version,
                    'description': proto.description,
                    'category': proto.category,
                    'required': proto.required,
                }

            return jsonify({
                'protocols': protocols,
                'required': get_required_protocols(),
                'satori_protocols': get_satori_protocols(),
                'count': len(protocols)
            })
        except ImportError:
            return jsonify({
                'protocols': {},
                'required': [],
                'satori_protocols': [],
                'count': 0,
                'error': 'Protocol registry not available'
            })
        except Exception as e:
            logger.warning(f"Protocol info failed: {e}")
            return jsonify({'protocols': {}, 'error': str(e)})

    # =========================================================================
    # PING & IDENTIFY PROTOCOL ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/ping/<peer_id>', methods=['POST'])
    @login_required
    def api_p2p_ping(peer_id):
        """Ping a peer to test connectivity and measure latency."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            count = request.json.get('count', 3) if request.json else 3

            # P2PProxy.ping_peer is synchronous (handles IPC internally)
            # It returns latencies in seconds, or None on failure
            latencies = peers.ping_peer(peer_id, count=count)

            if latencies:
                return jsonify({
                    'success': True,
                    'peer_id': peer_id,
                    'latencies_ms': [l * 1000 for l in latencies],
                    'avg_ms': sum(latencies) / len(latencies) * 1000,
                    'min_ms': min(latencies) * 1000,
                    'max_ms': max(latencies) * 1000,
                    'pings_sent': count,
                    'pings_received': len(latencies),
                })
            else:
                return jsonify({
                    'success': False,
                    'peer_id': peer_id,
                    'error': 'No response from peer',
                })

        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    @app.route('/api/p2p/ping/stats')
    @login_required
    def api_p2p_ping_stats():
        """Get ping protocol statistics."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503
            if hasattr(peers, '_ping_service') and peers._ping_service:
                stats = peers._ping_service.get_stats()
                return jsonify({
                    'success': True,
                    'started': stats.get('started', False),
                    'pending_pings': stats.get('pending_pings', 0),
                    'version': stats.get('version', '1.0.0'),
                })
            else:
                return jsonify({'success': False, 'error': 'Ping protocol not initialized'})

        except Exception as e:
            logger.warning(f"Ping stats failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/p2p/connect', methods=['POST'])
    @login_required
    def api_p2p_connect():
        """Connect to a peer by multiaddress."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            data = request.json or {}
            multiaddr = data.get('multiaddr')

            if not multiaddr:
                return jsonify({'error': 'multiaddr is required', 'success': False}), 400

            # P2PProxy.connect is synchronous (handles IPC internally)
            success = peers.connect(multiaddr)

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Connected to peer',
                    'multiaddr': multiaddr,
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to connect to peer',
                    'multiaddr': multiaddr,
                })

        except Exception as e:
            logger.warning(f"Connect failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    @app.route('/api/p2p/disconnect', methods=['POST'])
    @login_required
    def api_p2p_disconnect():
        """Disconnect from a peer."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            data = request.json or {}
            peer_id = data.get('peer_id')

            if not peer_id:
                return jsonify({'error': 'peer_id is required', 'success': False}), 400

            # P2PProxy.disconnect is synchronous (handles IPC internally)
            success = peers.disconnect(peer_id)

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Disconnected from peer',
                    'peer_id': peer_id,
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to disconnect from peer',
                    'peer_id': peer_id,
                })

        except Exception as e:
            logger.warning(f"Disconnect failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    @app.route('/api/p2p/debug')
    @login_required
    def api_p2p_debug():
        """Get detailed GossipSub/Pubsub debug information."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503

            if hasattr(peers, 'get_pubsub_debug'):
                debug_info = peers.get_pubsub_debug()
                return jsonify(debug_info)
            else:
                return jsonify({'error': 'Debug method not available'}), 501

        except Exception as e:
            logger.warning(f"P2P debug failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/p2p/identity')
    @login_required
    def api_p2p_identity():
        """Get our own identity info (peer_id, evrmore_address)."""
        try:
            identity, peers = get_p2p_state()

            result = {
                'peer_id': None,
                'evrmore_address': None,
            }

            # Get peer ID
            if peers:
                if hasattr(peers, 'peer_id') and peers.peer_id:
                    result['peer_id'] = str(peers.peer_id)
                elif hasattr(peers, 'node_id') and peers.node_id:
                    result['peer_id'] = str(peers.node_id)

            # Get evrmore address
            if identity and hasattr(identity, 'address'):
                result['evrmore_address'] = identity.address

            return jsonify(result)

        except Exception as e:
            logger.warning(f"P2P identity failed: {e}")
            return jsonify({'peer_id': None, 'evrmore_address': None})

    @app.route('/api/p2p/identify/stats')
    @login_required
    def api_p2p_identify_stats():
        """Get our own identity info including peer_id, evrmore_address, protocols, and prediction stats."""
        try:
            from satorineuron.init import start
            identity, peers = get_p2p_state()

            result = {
                'peer_id': None,
                'evrmore_address': None,
                'public_key': None,
                'multiaddr': None,
                'protocols': [],
                'active_predictions': 0,
                'consensus_participation': 0,
            }

            # Get peer ID
            if peers:
                if hasattr(peers, 'peer_id') and peers.peer_id:
                    result['peer_id'] = str(peers.peer_id)
                elif hasattr(peers, 'node_id') and peers.node_id:
                    result['peer_id'] = str(peers.node_id)

                # Get protocols from identify handler
                if hasattr(peers, '_identify_handler') and peers._identify_handler:
                    if hasattr(peers._identify_handler, '_get_supported_protocols'):
                        result['protocols'] = peers._identify_handler._get_supported_protocols()
                # Fallback: construct from peers capabilities
                if not result['protocols']:
                    protocols = ["/satori/stream/1.0.0", "/satori/ping/1.0.0", "/satori/identify/1.0.0"]
                    if hasattr(peers, 'enable_dht') and peers.enable_dht:
                        protocols.append("/ipfs/kad/1.0.0")
                    if hasattr(peers, 'enable_pubsub') and peers.enable_pubsub:
                        # Include meshsub v1.2 (latest), v1.1 (compat), and floodsub
                        protocols.extend(["/meshsub/1.2.0", "/meshsub/1.1.0", "/floodsub/1.0.0"])
                    if hasattr(peers, 'enable_relay') and peers.enable_relay:
                        protocols.append("/libp2p/circuit/relay/0.2.0/hop")
                    result['protocols'] = protocols

            # Get evrmore address
            if identity and hasattr(identity, 'address'):
                result['evrmore_address'] = identity.address

            # Get public key
            if identity and hasattr(identity, 'pubkey'):
                result['public_key'] = identity.pubkey
            elif peers and hasattr(peers, 'public_key'):
                result['public_key'] = peers.public_key

            # Get multiaddress
            if peers and hasattr(peers, 'public_addresses') and result['peer_id']:
                addrs = peers.public_addresses
                if addrs:
                    addr = addrs[0]
                    if '/p2p/' not in addr:
                        result['multiaddr'] = f"{addr}/p2p/{result['peer_id']}"
                    else:
                        result['multiaddr'] = addr

            # Get prediction stats from startup
            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup:
                # Active predictions - get from prediction manager if available
                if hasattr(startup, '_prediction_protocol') and startup._prediction_protocol:
                    pred_proto = startup._prediction_protocol
                    if hasattr(pred_proto, 'get_active_count'):
                        result['active_predictions'] = pred_proto.get_active_count()

                # Consensus participation
                if hasattr(startup, '_consensus_manager') and startup._consensus_manager:
                    consensus = startup._consensus_manager
                    if hasattr(consensus, 'participation_count'):
                        result['consensus_participation'] = consensus.participation_count

            # Determine role based on registrations
            roles = ['node']  # Base role
            try:
                from web.wsgi import _ipc_get
                # Check if we're a registered oracle
                oracle_result = _ipc_get('/p2p/oracle/my_registrations')
                if oracle_result and oracle_result.get('registrations') and len(oracle_result['registrations']) > 0:
                    roles.append('oracle')
                # Could also check for other roles (signer, relay, etc.) here
            except Exception:
                pass  # Role detection is supplementary
            result['roles'] = roles

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Get identify stats failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/p2p/identify/peers')
    @login_required
    def api_p2p_identify_peers():
        """Get known peer identities from the Identify protocol."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503
            identities = peers.get_known_peer_identities()

            # Get list of currently connected peers
            connected_peers = set()
            if hasattr(peers, 'get_connected_peers'):
                connected_peers = set(str(p) for p in peers.get_connected_peers())

            # Get oracle registrations to supplement role info
            from web.wsgi import _ipc_get
            oracle_peer_ids = set()
            try:
                oracle_result = _ipc_get('/p2p/oracle/known')
                if oracle_result and 'oracles' in oracle_result:
                    for oracle in oracle_result['oracles']:
                        if oracle.get('peer_id'):
                            oracle_peer_ids.add(oracle['peer_id'])
            except Exception:
                pass  # Oracle info is supplementary, don't fail if unavailable

            # Track which peer IDs we've added
            known_peer_ids = set()

            peer_list = []
            for peer_id, identity in identities.items():
                peer_id_str = str(peer_id)
                known_peer_ids.add(peer_id_str)
                roles = list(getattr(identity, 'roles', []))
                # Supplement with oracle role if peer is a registered oracle
                if peer_id_str in oracle_peer_ids and 'oracle' not in roles:
                    roles.append('oracle')
                peer_info = {
                    'peer_id': peer_id_str,
                    'evrmore_address': getattr(identity, 'evrmore_address', ''),
                    'roles': roles if roles else ['node'],
                    'protocols': getattr(identity, 'protocols', []),
                    'agent_version': getattr(identity, 'agent_version', ''),
                    'timestamp': getattr(identity, 'timestamp', 0),
                    'connected': peer_id_str in connected_peers,
                    'listen_addresses': getattr(identity, 'listen_addresses', []),
                }
                peer_list.append(peer_info)

            # Add connected peers that don't have identities yet (with default node role)
            for peer_id_str in connected_peers:
                if peer_id_str not in known_peer_ids:
                    roles = ['node']
                    # Check if this peer is a registered oracle
                    if peer_id_str in oracle_peer_ids:
                        roles.append('oracle')
                    peer_list.append({
                        'peer_id': peer_id_str,
                        'evrmore_address': '',
                        'roles': roles,
                        'protocols': [],
                        'agent_version': '',
                        'timestamp': 0,
                        'connected': True,
                        'listen_addresses': [],
                    })

            return jsonify({
                'success': True,
                'peers': peer_list,
                'count': len(peer_list),
                'connected_count': len(connected_peers),
            })

        except Exception as e:
            logger.warning(f"Get peer identities failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/p2p/identify/by-role/<role>')
    @login_required
    def api_p2p_identify_by_role(role):
        """Get peers with a specific role (predictor, relay, oracle, signer)."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized'}), 503
            role_peers = peers.get_peers_by_role(role)

            peer_list = []
            for identity in role_peers:
                peer_info = {
                    'peer_id': getattr(identity, 'peer_id', ''),
                    'evrmore_address': getattr(identity, 'evrmore_address', ''),
                    'roles': getattr(identity, 'roles', []),
                    'agent_version': getattr(identity, 'agent_version', ''),
                }
                peer_list.append(peer_info)

            return jsonify({
                'success': True,
                'role': role,
                'peers': peer_list,
                'count': len(peer_list),
            })

        except Exception as e:
            logger.warning(f"Get peers by role failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/p2p/identify/announce', methods=['POST'])
    @login_required
    def api_p2p_identify_announce():
        """Announce our identity to the network."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            # P2PProxy.announce_identity_sync is synchronous (handles IPC internally)
            if hasattr(peers, 'announce_identity_sync'):
                success = peers.announce_identity_sync()
            else:
                # Fallback for direct Peers object (async)
                import trio
                async def do_announce():
                    await peers.announce_identity()
                trio.run(do_announce)
                success = True

            return jsonify({
                'success': success,
                'message': 'Identity announced to network' if success else 'Announce failed',
            })

        except Exception as e:
            logger.warning(f"Identity announce failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    # ============================================
    # IDENTITY BROADCAST ENDPOINTS (Send OUR identity)
    # ============================================

    # Already have: /api/p2p/identify/announce - broadcasts to whole network

    @app.route('/api/p2p/identify/broadcast/known', methods=['POST'])
    @login_required
    def api_p2p_identify_broadcast_known():
        """Broadcast our identity to all known peers only."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            result = {'success': False, 'count': 0}

            async def do_broadcast():
                if hasattr(peers, '_identify_handler') and peers._identify_handler:
                    count = await peers._identify_handler.announce_to_known_peers()
                    result['count'] = count
                    result['success'] = True
                else:
                    logger.warning("No identify handler available")

            try:
                from web.wsgi import run_in_p2p_context
                run_in_p2p_context(do_broadcast)
            except RuntimeError:
                import trio
                trio.run(do_broadcast)

            return jsonify({
                'success': result['success'],
                'message': f'Identity broadcast to {result["count"]} known peers',
                'count': result['count'],
            })

        except Exception as e:
            logger.warning(f"Identity broadcast to known peers failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    @app.route('/api/p2p/identify/broadcast/<peer_id>', methods=['POST'])
    @login_required
    def api_p2p_identify_broadcast_peer(peer_id):
        """Broadcast our identity to a specific peer."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            if not peer_id:
                return jsonify({'error': 'peer_id is required', 'success': False}), 400

            result = {'success': False}

            async def do_broadcast():
                if hasattr(peers, '_identify_handler') and peers._identify_handler:
                    result['success'] = await peers._identify_handler.announce_to_peer(peer_id)
                else:
                    logger.warning("No identify handler available")

            try:
                from web.wsgi import run_in_p2p_context
                run_in_p2p_context(do_broadcast)
            except RuntimeError:
                import trio
                trio.run(do_broadcast)

            return jsonify({
                'success': result['success'],
                'message': f'Identity {"sent to" if result["success"] else "failed to send to"} peer {peer_id}',
                'peer_id': peer_id,
            })

        except Exception as e:
            logger.warning(f"Identity broadcast to peer {peer_id} failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    # ============================================
    # IDENTITY REQUEST ENDPOINTS (Request THEIR identity)
    # ============================================

    @app.route('/api/p2p/identify/request/network', methods=['POST'])
    @login_required
    def api_p2p_identify_request_network():
        """Request identity from whole network via GossipSub broadcast."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            async def do_request():
                if hasattr(peers, '_identify_handler') and peers._identify_handler:
                    await peers._identify_handler.request_identity(None)  # None = broadcast
                else:
                    logger.warning("No identify handler available")

            try:
                from web.wsgi import run_in_p2p_context
                run_in_p2p_context(do_request)
            except RuntimeError:
                import trio
                trio.run(do_request)

            return jsonify({
                'success': True,
                'message': 'Identity request broadcast to network',
            })

        except Exception as e:
            logger.warning(f"Identity request to network failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    @app.route('/api/p2p/identify/request/known', methods=['POST'])
    @login_required
    def api_p2p_identify_request_known():
        """Request identity from all known peers only."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            result = {'success': False, 'count': 0}

            async def do_request():
                if hasattr(peers, '_identify_handler') and peers._identify_handler:
                    count = await peers._identify_handler.request_identity_from_known_peers()
                    result['count'] = count
                    result['success'] = True
                else:
                    logger.warning("No identify handler available")

            try:
                from web.wsgi import run_in_p2p_context
                run_in_p2p_context(do_request)
            except RuntimeError:
                import trio
                trio.run(do_request)

            return jsonify({
                'success': result['success'],
                'message': f'Identity requested from {result["count"]} known peers',
                'count': result['count'],
            })

        except Exception as e:
            logger.warning(f"Identity request from known peers failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    @app.route('/api/p2p/identify/request/<peer_id>', methods=['POST'])
    @login_required
    def api_p2p_identify_request_peer(peer_id):
        """Request identity from a specific peer."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            if not peer_id:
                return jsonify({'error': 'peer_id is required', 'success': False}), 400

            result = {'success': False}

            async def do_request():
                if hasattr(peers, '_identify_handler') and peers._identify_handler:
                    result['success'] = await peers._identify_handler.request_identity_from_peer(peer_id)
                else:
                    logger.warning("No identify handler available")

            try:
                from web.wsgi import run_in_p2p_context
                run_in_p2p_context(do_request)
            except RuntimeError:
                import trio
                trio.run(do_request)

            return jsonify({
                'success': result['success'],
                'message': f'Identity request {"sent to" if result["success"] else "failed for"} peer {peer_id}',
                'peer_id': peer_id,
            })

        except Exception as e:
            logger.warning(f"Identity request from peer {peer_id} failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    @app.route('/api/p2p/identify/forget/<peer_id>', methods=['DELETE'])
    @login_required
    def api_p2p_forget_peer(peer_id):
        """Remove a peer from the known peers list."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            if not hasattr(peers, 'forget_peer'):
                return jsonify({'error': 'forget_peer not available', 'success': False}), 501

            success = peers.forget_peer(peer_id)
            if success:
                return jsonify({'success': True, 'message': f'Forgot peer {peer_id}'})
            else:
                return jsonify({'success': False, 'error': 'Peer not found'}), 404

        except Exception as e:
            logger.warning(f"Forget peer failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    @app.route('/api/p2p/identify/reconnect/<peer_id>', methods=['POST'])
    @login_required
    def api_p2p_reconnect_peer(peer_id):
        """Attempt to reconnect to a known peer."""
        try:
            _, peers = get_p2p_state()
            if not peers:
                return jsonify({'error': 'P2P not initialized', 'success': False}), 503

            # Get the peer's identity to find their listen addresses
            identities = peers.get_known_peer_identities()
            if peer_id not in identities:
                return jsonify({'error': 'Peer not found in known peers', 'success': False}), 404

            identity = identities[peer_id]
            listen_addrs = getattr(identity, 'listen_addresses', [])
            if not listen_addrs:
                return jsonify({'error': 'No listen addresses for peer', 'success': False}), 400

            # Try to connect using the first available address
            async def do_connect():
                for addr in listen_addrs:
                    try:
                        success = await peers.connect_peer(addr, timeout=10.0)
                        if success:
                            return True, addr
                    except Exception as e:
                        logger.debug(f"Failed to connect to {addr}: {e}")
                        continue
                return False, None

            try:
                from web.wsgi import run_in_p2p_context
                success, used_addr = run_in_p2p_context(do_connect)
            except RuntimeError:
                import trio
                success, used_addr = trio.run(do_connect)

            if success:
                return jsonify({
                    'success': True,
                    'message': f'Connected to peer {peer_id}',
                    'address': used_addr
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to connect to any address'
                }), 502

        except Exception as e:
            logger.warning(f"Reconnect peer failed: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    # =========================================================================
    # STORAGE REDUNDANCY ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/storage')
    @login_required
    def api_p2p_storage():
        """Get storage redundancy status and disk usage."""
        try:
            from web.wsgi import get_web_storage

            # Use IPC to get storage status (gunicorn can't access main process)
            result = get_web_storage()
            return jsonify(result)
        except Exception as e:
            logger.warning(f"Storage status failed: {e}")
            return jsonify({'status': 'error', 'error': str(e)})

    @app.route('/api/p2p/storage/toggle', methods=['POST'])
    @login_required
    def api_p2p_storage_toggle():
        """Enable or disable a storage backend."""
        try:
            from satorineuron.init import start

            data = request.get_json() or {}
            backend = data.get('backend')  # Only 'dht' is toggleable
            enabled = data.get('enabled', True)

            # Memory and File storage cannot be toggled - required for operation
            if backend == 'memory':
                return jsonify({
                    'success': False,
                    'error': 'Memory storage cannot be disabled - required for read/write operations'
                })
            if backend == 'file':
                return jsonify({
                    'success': False,
                    'error': 'File storage cannot be disabled - it is the core persistence layer'
                })

            if backend != 'dht':
                return jsonify({
                    'success': False,
                    'error': f'Invalid backend: {backend}. Only DHT can be toggled.'
                })

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup:
                return jsonify({
                    'success': False,
                    'error': 'P2P not initialized'
                })

            # Get or create storage manager
            if not hasattr(startup, '_storage_manager') or not startup._storage_manager:
                # Initialize storage manager if not exists
                try:
                    from satorip2p.protocol import StorageManager
                    startup._storage_manager = StorageManager()
                    logger.info("Initialized StorageManager")
                except ImportError:
                    return jsonify({
                        'success': False,
                        'error': 'StorageManager not available'
                    })

            manager = startup._storage_manager

            # Toggle the backend
            if hasattr(manager, 'toggle_backend'):
                success = manager.toggle_backend(backend, enabled)
                if success:
                    logger.info(f"Storage backend '{backend}' {'enabled' if enabled else 'disabled'}")
                    return jsonify({
                        'success': True,
                        'backend': backend,
                        'enabled': enabled
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Failed to toggle {backend}'
                    })
            else:
                # Fallback: try to enable/disable directly
                if backend == 'memory':
                    if enabled and hasattr(manager, 'enable_memory'):
                        manager.enable_memory()
                    elif hasattr(manager, 'disable_memory'):
                        manager.disable_memory()
                elif backend == 'file':
                    if enabled and hasattr(manager, 'enable_file'):
                        manager.enable_file()
                    elif hasattr(manager, 'disable_file'):
                        manager.disable_file()
                elif backend == 'dht':
                    if enabled and hasattr(manager, 'enable_dht'):
                        manager.enable_dht()
                    elif hasattr(manager, 'disable_dht'):
                        manager.disable_dht()

                return jsonify({
                    'success': True,
                    'backend': backend,
                    'enabled': enabled
                })

        except Exception as e:
            logger.warning(f"Storage toggle failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # BANDWIDTH & QOS ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/bandwidth')
    @login_required
    def api_p2p_bandwidth():
        """Get bandwidth usage statistics and QoS status."""
        try:
            from web.wsgi import get_web_bandwidth

            # Use IPC to get bandwidth stats (gunicorn can't access main process)
            result = get_web_bandwidth()
            return jsonify(result)
        except Exception as e:
            logger.warning(f"Bandwidth stats failed: {e}")
            return jsonify({'status': 'error', 'error': str(e)})

    @app.route('/api/p2p/bandwidth/history')
    @login_required
    def api_p2p_bandwidth_history():
        """Get bandwidth usage history for charting."""
        try:
            from web.wsgi import get_web_bandwidth_history

            # Use IPC to get bandwidth history (gunicorn can't access main process)
            result = get_web_bandwidth_history()
            return jsonify(result)
        except Exception as e:
            logger.warning(f"Bandwidth history failed: {e}")
            return jsonify({'history': [], 'error': str(e)})

    # =========================================================================
    # PRICING ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/pricing/satori')
    @login_required
    def api_p2p_pricing_satori():
        """Get SATORI price info from SafeTrade."""
        try:
            from satorineuron.init import start

            result = {
                'symbol': 'SATORI',
                'price_usd': 0.0,
                'price_btc': 0.0,
                'volume_24h_usd': 0.0,
                'change_24h_pct': 0.0,
                'source': 'safetrade',
                'updated_at': 0,
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_price_provider') and startup._price_provider:
                provider = startup._price_provider
                if hasattr(provider, 'get_satori_price'):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    try:
                        quote = loop.run_until_complete(provider.get_satori_price())
                        if quote:
                            result['price_usd'] = quote.price_usd
                            result['price_btc'] = quote.price_btc if hasattr(quote, 'price_btc') else 0.0
                            result['volume_24h_usd'] = quote.volume_24h if hasattr(quote, 'volume_24h') else 0.0
                            result['updated_at'] = quote.timestamp if hasattr(quote, 'timestamp') else 0
                    finally:
                        loop.close()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"SATORI pricing failed: {e}")
            return jsonify({'symbol': 'SATORI', 'price_usd': 0.0, 'error': str(e)})

    @app.route('/api/p2p/pricing/evr')
    @login_required
    def api_p2p_pricing_evr():
        """Get EVR price info from SafeTrade."""
        try:
            from satorineuron.init import start

            result = {
                'symbol': 'EVR',
                'price_usd': 0.0,
                'price_btc': 0.0,
                'volume_24h_usd': 0.0,
                'change_24h_pct': 0.0,
                'source': 'safetrade',
                'updated_at': 0,
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_price_provider') and startup._price_provider:
                provider = startup._price_provider
                if hasattr(provider, 'get_evr_price'):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    try:
                        quote = loop.run_until_complete(provider.get_evr_price())
                        if quote:
                            result['price_usd'] = quote.price_usd
                            result['price_btc'] = quote.price_btc if hasattr(quote, 'price_btc') else 0.0
                            result['volume_24h_usd'] = quote.volume_24h if hasattr(quote, 'volume_24h') else 0.0
                            result['updated_at'] = quote.timestamp if hasattr(quote, 'timestamp') else 0
                    finally:
                        loop.close()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"EVR pricing failed: {e}")
            return jsonify({'symbol': 'EVR', 'price_usd': 0.0, 'error': str(e)})

    @app.route('/api/p2p/pricing/exchange-rate')
    @login_required
    def api_p2p_pricing_exchange_rate():
        """Get EVR to SATORI exchange rate for donations."""
        try:
            from satorineuron.init import start

            result = {
                'evr_to_satori': 0.0,
                'satori_price_usd': 0.0,
                'evr_price_usd': 0.0,
                'discount_applied': 0.8,
                'source': 'safetrade',
                'updated_at': 0,
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_price_provider') and startup._price_provider:
                provider = startup._price_provider
                if hasattr(provider, 'get_exchange_rate'):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    try:
                        rate_info = loop.run_until_complete(provider.get_exchange_rate())
                        if rate_info:
                            result.update(rate_info)
                    finally:
                        loop.close()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Exchange rate failed: {e}")
            return jsonify({'evr_to_satori': 0.0, 'error': str(e)})

    # =========================================================================
    # DONATION ENDPOINTS
    # =========================================================================

    @app.route('/donate')
    @login_required
    def donate():
        """Treasury donation page."""
        from satorineuron import VERSION
        return render_template('donate.html', version=VERSION)

    @app.route('/donate/treasury-address')
    @login_required
    def donate_treasury_address():
        """Get the treasury multi-sig address."""
        try:
            from satorineuron.init import start
            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, 'server') and startup.server:
                if hasattr(startup.server, 'getTreasuryAddress'):
                    address = startup.server.getTreasuryAddress()
                    return address or 'Not available'
            return 'Not available'
        except Exception as e:
            logger.error(f"Get treasury address failed: {e}")
            return 'Error'

    @app.route('/donate/send', methods=['POST'])
    @login_required
    def donate_send():
        """Submit a donation to the treasury."""
        try:
            from satorineuron.init import start
            data = request.get_json() or {}
            amount = float(data.get('amount', 0))

            if amount <= 0:
                return jsonify({'success': False, 'error': 'Invalid amount'})

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, 'server') and startup.server:
                if hasattr(startup.server, 'donateToTreasury'):
                    result = startup.server.donateToTreasury(amount)
                    if result:
                        return jsonify({'success': True, 'message': 'Donation submitted!'})

            return jsonify({'success': False, 'error': 'Unable to process donation'})
        except Exception as e:
            logger.error(f"Donate send failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/donate/stats')
    @login_required
    def donate_stats():
        """Get donation stats for current user."""
        try:
            return jsonify({
                'total_donated': 0.0,
                'donation_count': 0,
                'tier': 'none',
                'badges_earned': [],
            })
        except Exception as e:
            logger.error(f"Get donation stats failed: {e}")
            return jsonify({'total_donated': 0.0, 'donation_count': 0, 'tier': 'none'})

    @app.route('/donate/history')
    @login_required
    def donate_history():
        """Get donation history for current user."""
        try:
            return jsonify([])
        except Exception as e:
            logger.error(f"Get donation history failed: {e}")
            return jsonify([])

    @app.route('/donate/top-donors')
    @login_required
    def donate_top_donors():
        """Get top donors list."""
        try:
            return jsonify([])
        except Exception as e:
            logger.error(f"Get top donors failed: {e}")
            return jsonify([])

    @app.route('/api/donate/treasury-address')
    @login_required
    def api_donate_treasury_address():
        """Get the treasury address for donations."""
        try:
            from satorineuron.init import start

            # Try to get from signer module
            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_signer_node') and startup._signer_node:
                signer = startup._signer_node
                if hasattr(signer, 'treasury_address') and signer.treasury_address:
                    return jsonify({'treasury_address': signer.treasury_address})

            # Fallback to P2P treasury config
            try:
                from satorip2p.protocol.signer import get_treasury_address
                treasury = get_treasury_address()
                if treasury:
                    return jsonify({'treasury_address': treasury})
            except ImportError:
                pass

            # Final fallback - return a placeholder
            return jsonify({'treasury_address': 'Not configured'})
        except Exception as e:
            logger.warning(f"Treasury address failed: {e}")
            return jsonify({'treasury_address': 'Error', 'error': str(e)})

    @app.route('/api/donate/stats')
    @login_required
    def api_donate_stats():
        """Get donation stats for current user."""
        try:
            from satorineuron.init import start
            import asyncio

            result = {
                'total_donated': 0.0,
                'tier': 'None',
                'rewards_received': 0.0,
                'progress_to_next': 0.0,
                'next_tier': None,
                'next_tier_threshold': 0.0,
            }

            # Get wallet address
            wallet_manager = get_or_create_session_vault()
            if not wallet_manager or not wallet_manager.vault:
                return jsonify(result)

            vault_address = wallet_manager.vault.address if hasattr(wallet_manager.vault, 'address') else None
            if not vault_address:
                return jsonify(result)

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_donation_manager') and startup._donation_manager:
                manager = startup._donation_manager

                async def get_donor_stats():
                    if hasattr(manager, 'get_donor_stats'):
                        return await manager.get_donor_stats(vault_address)
                    return None

                try:
                    loop = asyncio.new_event_loop()
                    stats = loop.run_until_complete(get_donor_stats())
                    if stats:
                        result['total_donated'] = stats.total_donated
                        result['tier'] = stats.current_tier
                        result['rewards_received'] = stats.total_rewards_received

                        # Get tier progress
                        if hasattr(manager, 'get_progress_to_next_tier'):
                            progress_info = loop.run_until_complete(
                                manager.get_progress_to_next_tier(vault_address))
                            if progress_info:
                                result['progress_to_next'] = progress_info.get('progress', 0.0)
                                result['next_tier'] = progress_info.get('next_tier')
                                result['next_tier_threshold'] = progress_info.get('threshold', 0.0)
                finally:
                    loop.close()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Donation stats failed: {e}")
            return jsonify(result)

    @app.route('/api/donate/send', methods=['POST'])
    @login_required
    def api_donate_send():
        """Send a donation to the treasury."""
        try:
            data = request.get_json() or {}
            amount = data.get('amount')

            if not amount or float(amount) <= 0:
                return jsonify({'success': False, 'error': 'Invalid amount'})

            # Get wallet manager
            wallet_manager = get_or_create_session_vault()
            if not wallet_manager or not wallet_manager.vault:
                return jsonify({'success': False, 'error': 'Vault not available'})

            # Get treasury address
            from satorineuron.init import start
            treasury_address = None

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_signer_node') and startup._signer_node:
                signer = startup._signer_node
                if hasattr(signer, 'treasury_address'):
                    treasury_address = signer.treasury_address

            if not treasury_address:
                try:
                    from satorip2p.protocol.signer import get_treasury_address
                    treasury_address = get_treasury_address()
                except ImportError:
                    pass

            if not treasury_address:
                return jsonify({'success': False, 'error': 'Treasury address not configured'})

            # Send EVR to treasury
            vault = wallet_manager.vault
            vault.get()
            vault.getReadyToSend()

            # Send EVR transaction
            txid = vault.evrTransaction(amount=float(amount), address=treasury_address)

            if txid and len(txid) == 64:
                return jsonify({'success': True, 'tx_hash': txid})
            else:
                return jsonify({'success': False, 'error': 'Transaction failed', 'details': str(txid)})

        except Exception as e:
            logger.error(f"Donation send failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/donate/top-donors')
    @login_required
    def api_donate_top_donors():
        """Get top donors leaderboard."""
        try:
            from satorineuron.init import start
            import asyncio

            limit = request.args.get('limit', 10, type=int)
            donors = []

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_donation_manager') and startup._donation_manager:
                manager = startup._donation_manager

                async def get_top():
                    if hasattr(manager, 'get_top_donors'):
                        return await manager.get_top_donors(limit=limit)
                    return []

                try:
                    loop = asyncio.new_event_loop()
                    top = loop.run_until_complete(get_top())
                    for d in top:
                        donors.append({
                            'address': d.address,
                            'total_donated': d.total_donated,
                            'tier': d.current_tier,
                        })
                finally:
                    loop.close()

            return jsonify({'donors': donors})
        except Exception as e:
            logger.warning(f"Top donors failed: {e}")
            return jsonify({'donors': [], 'error': str(e)})

    # =========================================================================
    # REFERRAL SYSTEM ENDPOINTS
    # =========================================================================

    @app.route('/api/referral/stats')
    @login_required
    def api_referral_stats():
        """Get referral stats for current user."""
        try:
            from satorineuron.init import start
            import asyncio

            result = {
                'referral_count': 0,
                'tier': 'None',
                'bonus_percent': 0,
            }

            # Get wallet address
            wallet_manager = get_or_create_session_vault()
            if not wallet_manager or not wallet_manager.wallet:
                return jsonify(result)

            wallet_address = wallet_manager.wallet.address if hasattr(wallet_manager.wallet, 'address') else None
            if not wallet_address:
                return jsonify(result)

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_referral_manager') and startup._referral_manager:
                manager = startup._referral_manager

                async def get_stats():
                    if hasattr(manager, 'get_referrer_stats'):
                        return await manager.get_referrer_stats(wallet_address)
                    return None

                try:
                    loop = asyncio.new_event_loop()
                    stats = loop.run_until_complete(get_stats())
                    if stats:
                        result['referral_count'] = stats.referral_count
                        result['tier'] = stats.tier
                        result['bonus_percent'] = stats.bonus_percent
                finally:
                    loop.close()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Referral stats failed: {e}")
            return jsonify(result)

    @app.route('/api/referral/top')
    @login_required
    def api_referral_top():
        """Get top referrers leaderboard."""
        try:
            from satorineuron.init import start
            import asyncio

            limit = request.args.get('limit', 10, type=int)
            referrers = []

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_referral_manager') and startup._referral_manager:
                manager = startup._referral_manager

                async def get_top():
                    if hasattr(manager, 'get_top_referrers'):
                        return await manager.get_top_referrers(limit=limit)
                    return []

                try:
                    loop = asyncio.new_event_loop()
                    top = loop.run_until_complete(get_top())
                    for r in top:
                        referrers.append({
                            'address': r.address,
                            'referral_count': r.referral_count,
                            'tier': r.tier,
                            'bonus_percent': r.bonus_percent,
                        })
                finally:
                    loop.close()

            return jsonify({'referrers': referrers})
        except Exception as e:
            logger.warning(f"Top referrers failed: {e}")
            return jsonify({'referrers': [], 'error': str(e)})

    @app.route('/api/referral/set', methods=['POST'])
    @login_required
    def api_referral_set():
        """Set referrer for current user."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            referrer_address = data.get('referrer_address', '').strip()

            if not referrer_address or len(referrer_address) != 34:
                return jsonify({'success': False, 'error': 'Invalid referrer address'})

            # Get wallet address
            wallet_manager = get_or_create_session_vault()
            if not wallet_manager or not wallet_manager.wallet:
                return jsonify({'success': False, 'error': 'Wallet not available'})

            wallet_address = wallet_manager.wallet.address

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_referral_manager') and startup._referral_manager:
                manager = startup._referral_manager

                async def set_referrer():
                    if hasattr(manager, 'register_referral'):
                        return await manager.register_referral(wallet_address, referrer_address)
                    return False

                try:
                    loop = asyncio.new_event_loop()
                    success = loop.run_until_complete(set_referrer())
                    return jsonify({'success': success})
                finally:
                    loop.close()

            return jsonify({'success': False, 'error': 'Referral manager not available'})
        except Exception as e:
            logger.error(f"Set referrer failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # PROXY/DELEGATION MANAGEMENT ENDPOINTS
    # =========================================================================

    def _get_current_networking_mode():
        """Helper to get current networking mode."""
        try:
            from satorineuron.init import start
            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_get_networking_mode'):
                return startup._get_networking_mode()
        except:
            pass
        return 'central'

    @app.route('/api/proxy/child/charity/<address>/<int:child_id>', methods=['POST'])
    @login_required
    def api_proxy_charity(address, child_id):
        """Mark a proxy child as charity."""
        mode = _get_current_networking_mode()

        # Try P2P first if in P2P/hybrid mode
        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_delegation_manager') and startup._delegation_manager:
                    manager = startup._delegation_manager

                    async def set_charity():
                        if hasattr(manager, 'set_charity_status'):
                            return await manager.set_charity_status(address, child_id, True)
                        return False

                    try:
                        loop = asyncio.new_event_loop()
                        success = loop.run_until_complete(set_charity())
                        if success:
                            return jsonify({'success': True})
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P charity failed, trying central: {e}")

        # Fallback to central
        try:
            return proxy_api('/stake/proxy/charity', 'POST', {'child': address, 'childId': child_id})
        except Exception as e:
            logger.error(f"Set charity failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/proxy/child/no_charity/<address>/<int:child_id>', methods=['POST'])
    @login_required
    def api_proxy_no_charity(address, child_id):
        """Remove charity status from a proxy child."""
        mode = _get_current_networking_mode()

        # Try P2P first if in P2P/hybrid mode
        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_delegation_manager') and startup._delegation_manager:
                    manager = startup._delegation_manager

                    async def remove_charity():
                        if hasattr(manager, 'set_charity_status'):
                            return await manager.set_charity_status(address, child_id, False)
                        return False

                    try:
                        loop = asyncio.new_event_loop()
                        success = loop.run_until_complete(remove_charity())
                        if success:
                            return jsonify({'success': True})
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P remove charity failed, trying central: {e}")

        # Fallback to central
        try:
            return proxy_api('/stake/proxy/charity/not', 'POST', {'child': address, 'childId': child_id})
        except Exception as e:
            logger.error(f"Remove charity failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/proxy/child/remove/<address>/<int:child_id>', methods=['POST'])
    @login_required
    def api_proxy_remove_child(address, child_id):
        """Remove a proxy child."""
        mode = _get_current_networking_mode()

        # Try P2P first if in P2P/hybrid mode
        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_delegation_manager') and startup._delegation_manager:
                    manager = startup._delegation_manager

                    async def remove_child():
                        if hasattr(manager, 'remove_proxy_child'):
                            return await manager.remove_proxy_child(address, child_id)
                        return False

                    try:
                        loop = asyncio.new_event_loop()
                        success = loop.run_until_complete(remove_child())
                        if success:
                            return jsonify({'success': True})
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P remove child failed, trying central: {e}")

        # Fallback to central
        try:
            return proxy_api('/stake/proxy/remove', 'POST', {'child': address, 'childId': child_id})
        except Exception as e:
            logger.error(f"Remove proxy child failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/delegation/remove', methods=['POST'])
    @login_required
    def api_delegation_remove():
        """Remove current delegation."""
        mode = _get_current_networking_mode()

        # Try P2P first if in P2P/hybrid mode
        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                wallet_manager = get_or_create_session_vault()
                if wallet_manager and wallet_manager.wallet:
                    wallet_address = wallet_manager.wallet.address

                    startup = start.getStart() if hasattr(start, 'getStart') else None
                    if startup and hasattr(startup, '_delegation_manager') and startup._delegation_manager:
                        manager = startup._delegation_manager

                        async def remove_delegation():
                            if hasattr(manager, 'remove_delegation'):
                                return await manager.remove_delegation(wallet_address)
                            return False

                        try:
                            loop = asyncio.new_event_loop()
                            success = loop.run_until_complete(remove_delegation())
                            if success:
                                return jsonify({'success': True})
                        finally:
                            loop.close()
            except Exception as e:
                logger.warning(f"P2P remove delegation failed, trying central: {e}")

        # Fallback to central
        try:
            return proxy_api('/stake/proxy/delegate/remove', 'GET')
        except Exception as e:
            logger.error(f"Remove delegation failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # POOL SIZE MANAGEMENT
    # =========================================================================

    @app.route('/api/pool/size/set', methods=['POST'])
    @login_required
    def api_pool_size_set():
        """Set pool size limit."""
        mode = _get_current_networking_mode()
        data = request.get_json() or {}
        size_limit = float(data.get('size_limit', 0))

        if size_limit < 0:
            return jsonify({'success': False, 'error': 'Size limit must be >= 0'})

        # Try P2P first if in P2P/hybrid mode
        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                wallet_manager = get_or_create_session_vault()
                if wallet_manager and wallet_manager.wallet:
                    vault_address = wallet_manager.wallet.address

                    startup = start.getStart() if hasattr(start, 'getStart') else None
                    if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                        manager = startup._lending_manager

                        async def set_size():
                            if hasattr(manager, 'set_pool_size'):
                                return await manager.set_pool_size(vault_address, size_limit)
                            return False

                        try:
                            loop = asyncio.new_event_loop()
                            success = loop.run_until_complete(set_size())
                            if success:
                                return jsonify({'success': True})
                        finally:
                            loop.close()
            except Exception as e:
                logger.warning(f"P2P set pool size failed, trying central: {e}")

        # Fallback to central
        try:
            return proxy_api('/pool/size/set', 'POST', {'amount': size_limit})
        except Exception as e:
            logger.error(f"Set pool size failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # REWARD ADDRESS MANAGEMENT
    # =========================================================================

    @app.route('/api/peer/reward-address/remove', methods=['POST'])
    @login_required
    def api_reward_address_remove():
        """Remove custom reward address (revert to wallet address)."""
        mode = _get_current_networking_mode()

        # Try P2P first if in P2P/hybrid mode
        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                startup = start.getStart() if hasattr(start, 'getStart') else None
                if startup and hasattr(startup, '_reward_address_manager') and startup._reward_address_manager:
                    manager = startup._reward_address_manager

                    async def remove_addr():
                        if hasattr(manager, 'remove_reward_address'):
                            return await manager.remove_reward_address()
                        return False

                    try:
                        loop = asyncio.new_event_loop()
                        success = loop.run_until_complete(remove_addr())
                        if success:
                            return jsonify({'success': True})
                    finally:
                        loop.close()
            except Exception as e:
                logger.warning(f"P2P remove reward address failed, trying central: {e}")

        # Fallback to central - POST empty address to remove
        try:
            return proxy_api('/peer/reward-address', 'POST', {'reward_address': ''})
        except Exception as e:
            logger.error(f"Remove reward address failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # LENDING MANAGEMENT
    # =========================================================================

    @app.route('/api/lend/addresses')
    @login_required
    def api_lend_addresses():
        """Get all pools user is lending to."""
        mode = _get_current_networking_mode()

        # Try P2P first if in P2P/hybrid mode
        if mode in ('p2p', 'hybrid'):
            try:
                from satorineuron.init import start
                import asyncio

                wallet_manager = get_or_create_session_vault()
                if wallet_manager and wallet_manager.wallet:
                    wallet_address = wallet_manager.wallet.address

                    startup = start.getStart() if hasattr(start, 'getStart') else None
                    if startup and hasattr(startup, '_lending_manager') and startup._lending_manager:
                        manager = startup._lending_manager

                        async def get_lendings():
                            if hasattr(manager, 'get_my_lendings'):
                                return await manager.get_my_lendings(wallet_address)
                            return []

                        try:
                            loop = asyncio.new_event_loop()
                            my_lendings = loop.run_until_complete(get_lendings())
                            lendings = []
                            for l in my_lendings:
                                lendings.append({
                                    'vault_address': l.vault_address,
                                    'lent_out': l.lent_out,
                                    'timestamp': l.timestamp,
                                })
                            if lendings:
                                return jsonify({'lendings': lendings})
                        finally:
                            loop.close()
            except Exception as e:
                logger.warning(f"P2P get lendings failed, trying central: {e}")

        # Fallback to central
        try:
            return proxy_api('/stake/lend/addresses', 'GET')
        except Exception as e:
            logger.warning(f"Get lendings failed: {e}")
            return jsonify({'lendings': [], 'error': str(e)})

    # =========================================================================
    # SIGNER API ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/signer/pending')
    @login_required
    def api_p2p_signer_pending():
        """Get pending signing actions for this signer."""
        try:
            from satorineuron.init import start
            from satorip2p.protocol.signer import SigningPhase

            result = {
                'pending': [],
                'total': 0,
                'current_round': None,
                'phase': 'waiting',
                'signatures_collected': 0,
                'signatures_required': 3
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_signer') and startup._signer:
                signer = startup._signer

                # Get current signing status
                if hasattr(signer, 'check_signing_status'):
                    status = signer.check_signing_status()
                    result['current_round'] = status.round_id
                    result['phase'] = status.phase.value if status.phase else 'waiting'
                    result['signatures_collected'] = status.signatures_collected
                    result['signatures_required'] = status.signatures_required

                    # If there's an active request, add it to pending
                    if signer._current_request and status.phase == SigningPhase.COLLECTING:
                        req = signer._current_request
                        result['pending'].append({
                            'action_id': req.round_id,
                            'type': 'distribution',
                            'round_id': req.round_id,
                            'merkle_root': req.merkle_root[:16] + '...' if req.merkle_root else '',
                            'total_reward': req.total_reward,
                            'num_recipients': req.num_recipients,
                            'timestamp': req.timestamp,
                            'signatures': status.signatures_collected,
                            'required': status.signatures_required,
                            'signers': status.signers
                        })
                        result['total'] = 1

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Failed to get signer pending: {e}")
            return jsonify({'pending': [], 'error': str(e)})

    @app.route('/api/p2p/signer/sign', methods=['POST'])
    @login_required
    def api_p2p_signer_sign():
        """Sign a pending action (for authorized signers only)."""
        try:
            from satorineuron.init import start
            from satorip2p.protocol.signer import is_authorized_signer

            data = request.get_json() or {}
            action_id = data.get('action_id')
            if not action_id:
                return jsonify({'success': False, 'error': 'action_id required'})

            startup = start.getStart() if hasattr(start, 'getStart') else None

            # Check if user is authorized signer
            wallet_address = None
            if startup and hasattr(startup, 'wallet') and startup.wallet:
                wallet_address = startup.wallet.address

            if not wallet_address or not is_authorized_signer(wallet_address):
                return jsonify({'success': False, 'error': 'Only authorized signers can sign'}), 403

            # Get the signer node and process the current request
            if not startup or not hasattr(startup, '_signer') or not startup._signer:
                return jsonify({'success': False, 'error': 'Signer not initialized'}), 500

            signer = startup._signer

            # Verify action_id matches current request
            if not signer._current_request or signer._current_request.round_id != action_id:
                return jsonify({'success': False, 'error': 'Action not found or already processed'})

            # Process the signature request (signs and broadcasts)
            if hasattr(signer, 'process_signature_request'):
                success = signer.process_signature_request(signer._current_request)
                if success:
                    status = signer.check_signing_status()
                    return jsonify({
                        'success': True,
                        'action_id': action_id,
                        'signatures_collected': status.signatures_collected,
                        'phase': status.phase.value
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to sign'})

            return jsonify({'success': False, 'error': 'Signing not available'})
        except Exception as e:
            logger.warning(f"Failed to sign action: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/signer/consensus')
    @login_required
    def api_p2p_signer_consensus():
        """Get consensus status and other signers."""
        try:
            from satorip2p.protocol.signer import AUTHORIZED_SIGNERS, is_authorized_signer
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            my_address = ""
            if startup and hasattr(startup, '_identity_bridge') and startup._identity_bridge:
                my_address = startup._identity_bridge.evrmore_address or ""

            # Build signer list
            signers = []
            for i, addr in enumerate(AUTHORIZED_SIGNERS):
                signers.append({
                    'address': addr,
                    'position': i + 1,
                    'is_self': addr == my_address,
                    'online': True,  # TODO: Check actual online status
                    'last_seen': 'now'
                })

            return jsonify({
                'signers': signers,
                'online_count': len(signers),
                'threshold': 3,
                'pending_actions': []
            })
        except Exception as e:
            logger.warning(f"Failed to get consensus status: {e}")
            return jsonify({'signers': [], 'error': str(e)})

    # =========================================================================
    # PREDICTION PROTOCOL API ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/predictions/submit', methods=['POST'])
    @login_required
    def api_p2p_predictions_submit():
        """Submit a prediction for a stream."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            stream_id = data.get('stream_id')
            value = data.get('value')
            target_time = data.get('target_time')
            confidence = data.get('confidence', 0.0)
            metadata = data.get('metadata', {})

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400
            if value is None:
                return jsonify({'success': False, 'error': 'value is required'}), 400
            if not target_time:
                return jsonify({'success': False, 'error': 'target_time is required'}), 400

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_prediction_protocol') or not startup._prediction_protocol:
                return jsonify({'success': False, 'error': 'Prediction protocol not initialized'}), 503

            protocol = startup._prediction_protocol

            # Check if we're an oracle for this stream
            is_oracle = False
            oracle_network = getattr(startup, '_oracle_network', None)
            if oracle_network is not None:
                try:
                    oracle_role = oracle_network.get_oracle_role(stream_id)
                    is_oracle = oracle_role in ('primary', 'secondary')
                except Exception:
                    pass

            async def do_publish():
                return await protocol.publish_prediction(
                    stream_id=stream_id,
                    value=value,
                    target_time=int(target_time),
                    confidence=float(confidence),
                    metadata=metadata,
                    is_oracle=is_oracle
                )

            try:
                loop = asyncio.new_event_loop()
                prediction = loop.run_until_complete(do_publish())
            finally:
                loop.close()

            if prediction:
                return jsonify({
                    'success': True,
                    'prediction': {
                        'hash': prediction.hash,
                        'stream_id': prediction.stream_id,
                        'value': prediction.value,
                        'target_time': prediction.target_time,
                        'predictor': prediction.predictor,
                        'created_at': prediction.created_at,
                        'confidence': prediction.confidence,
                        'is_oracle': prediction.is_oracle,
                    }
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to publish prediction'})

        except Exception as e:
            logger.warning(f"Failed to submit prediction: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/predictions/list')
    @login_required
    def api_p2p_predictions_list():
        """List all cached predictions across all subscribed streams."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_prediction_protocol') or not startup._prediction_protocol:
                return jsonify({'predictions': [], 'error': 'Prediction protocol not initialized'})

            protocol = startup._prediction_protocol
            limit = request.args.get('limit', 100, type=int)

            # Get predictions from all cached streams
            all_predictions = []
            if hasattr(protocol, '_prediction_cache'):
                for stream_id, predictions in protocol._prediction_cache.items():
                    for pred in predictions[-limit:]:
                        all_predictions.append({
                            'hash': pred.hash,
                            'stream_id': pred.stream_id,
                            'value': pred.value,
                            'target_time': pred.target_time,
                            'predictor': pred.predictor,
                            'created_at': pred.created_at,
                            'confidence': pred.confidence,
                            'is_oracle': getattr(pred, 'is_oracle', False),
                        })

            # Sort by created_at descending
            all_predictions.sort(key=lambda x: x['created_at'], reverse=True)

            return jsonify({
                'predictions': all_predictions[:limit],
                'total': len(all_predictions),
            })

        except Exception as e:
            logger.warning(f"Failed to list predictions: {e}")
            return jsonify({'predictions': [], 'error': str(e)})

    @app.route('/api/p2p/predictions/stream/<stream_id>')
    @login_required
    def api_p2p_predictions_stream(stream_id):
        """Get predictions for a specific stream."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_prediction_protocol') or not startup._prediction_protocol:
                return jsonify({'predictions': [], 'error': 'Prediction protocol not initialized'})

            protocol = startup._prediction_protocol
            limit = request.args.get('limit', 100, type=int)

            predictions = protocol.get_cached_predictions(stream_id, limit)
            result = []
            for pred in predictions:
                result.append({
                    'hash': pred.hash,
                    'stream_id': pred.stream_id,
                    'value': pred.value,
                    'target_time': pred.target_time,
                    'predictor': pred.predictor,
                    'created_at': pred.created_at,
                    'confidence': pred.confidence,
                    'is_oracle': getattr(pred, 'is_oracle', False),
                })

            return jsonify({
                'stream_id': stream_id,
                'predictions': result,
                'total': len(result),
            })

        except Exception as e:
            logger.warning(f"Failed to get stream predictions: {e}")
            return jsonify({'predictions': [], 'error': str(e)})

    @app.route('/api/p2p/predictions/my')
    @login_required
    def api_p2p_predictions_my():
        """Get my predictions via IPC API."""
        try:
            import requests as req
            stream_id = request.args.get('stream_id')  # Optional filter
            limit = request.args.get('limit', 50, type=int)

            # Build IPC URL with query params
            ipc_url = 'http://127.0.0.1:24602/p2p/predictions/my'
            params = {}
            if stream_id:
                params['stream_id'] = stream_id
            if limit:
                params['limit'] = limit

            resp = req.get(ipc_url, params=params, timeout=5)
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({'predictions': [], 'error': f'IPC error: {resp.status_code}'})

        except Exception as e:
            logger.warning(f"Failed to get my predictions: {e}")
            return jsonify({'predictions': [], 'error': str(e)})

    @app.route('/api/p2p/predictions/scores')
    @login_required
    def api_p2p_predictions_scores():
        """Get prediction scores (leaderboard)."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_prediction_protocol') or not startup._prediction_protocol:
                return jsonify({'scores': [], 'error': 'Prediction protocol not initialized'})

            protocol = startup._prediction_protocol
            stream_id = request.args.get('stream_id')  # Optional filter
            limit = request.args.get('limit', 50, type=int)

            # Aggregate scores by predictor
            predictor_scores = {}
            if hasattr(protocol, '_score_cache'):
                for sid, scores in protocol._score_cache.items():
                    if stream_id and sid != stream_id:
                        continue
                    for score in scores:
                        predictor = score.predictor
                        if predictor not in predictor_scores:
                            predictor_scores[predictor] = {
                                'predictor': predictor,
                                'total_predictions': 0,
                                'total_score': 0.0,
                                'avg_score': 0.0,
                                'streams': set(),
                            }
                        predictor_scores[predictor]['total_predictions'] += 1
                        predictor_scores[predictor]['total_score'] += score.score
                        predictor_scores[predictor]['streams'].add(score.stream_id)

            # Calculate averages and format
            leaderboard = []
            for predictor, data in predictor_scores.items():
                if data['total_predictions'] > 0:
                    data['avg_score'] = data['total_score'] / data['total_predictions']
                data['streams'] = len(data['streams'])
                del data['total_score']
                leaderboard.append(data)

            # Sort by average score descending
            leaderboard.sort(key=lambda x: x['avg_score'], reverse=True)

            # Add rank
            for i, entry in enumerate(leaderboard[:limit]):
                entry['rank'] = i + 1

            return jsonify({
                'scores': leaderboard[:limit],
                'total_predictors': len(leaderboard),
            })

        except Exception as e:
            logger.warning(f"Failed to get prediction scores: {e}")
            return jsonify({'scores': [], 'error': str(e)})

    @app.route('/api/p2p/predictions/subscribe', methods=['POST'])
    @login_required
    def api_p2p_predictions_subscribe():
        """Subscribe to predictions for a stream."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_prediction_protocol') or not startup._prediction_protocol:
                return jsonify({'success': False, 'error': 'Prediction protocol not initialized'}), 503

            protocol = startup._prediction_protocol

            # Define a callback that will emit to websocket
            def on_prediction(prediction):
                try:
                    from web.p2p_bridge import get_bridge
                    bridge = get_bridge()
                    if bridge:
                        bridge._on_prediction(prediction)
                except Exception as e:
                    logger.debug(f"Failed to emit prediction: {e}")

            async def do_subscribe():
                return await protocol.subscribe_to_predictions(stream_id, on_prediction)

            try:
                loop = asyncio.new_event_loop()
                success = loop.run_until_complete(do_subscribe())
            finally:
                loop.close()

            if success:
                return jsonify({
                    'success': True,
                    'stream_id': stream_id,
                    'message': f'Subscribed to predictions for {stream_id}'
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to subscribe'})

        except Exception as e:
            logger.warning(f"Failed to subscribe to predictions: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/predictions/unsubscribe', methods=['POST'])
    @login_required
    def api_p2p_predictions_unsubscribe():
        """Unsubscribe from predictions for a stream."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_prediction_protocol') or not startup._prediction_protocol:
                return jsonify({'success': False, 'error': 'Prediction protocol not initialized'}), 503

            protocol = startup._prediction_protocol

            async def do_unsubscribe():
                return await protocol.unsubscribe_from_predictions(stream_id)

            try:
                loop = asyncio.new_event_loop()
                success = loop.run_until_complete(do_unsubscribe())
            finally:
                loop.close()

            if success:
                return jsonify({
                    'success': True,
                    'stream_id': stream_id,
                    'message': f'Unsubscribed from predictions for {stream_id}'
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to unsubscribe or not subscribed'})

        except Exception as e:
            logger.warning(f"Failed to unsubscribe from predictions: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/predictions/stats')
    @login_required
    def api_p2p_predictions_stats():
        """Get prediction protocol statistics via IPC API."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/predictions/stats')
            if result:
                return jsonify(result)

            # Fallback if IPC not available
            return jsonify({
                'started': False,
                'subscribed_streams': 0,
                'my_predictions': 0,
                'cached_predictions': 0,
                'cached_scores': 0,
                'my_average_score': 0.0,
            })

        except Exception as e:
            logger.warning(f"Failed to get prediction stats: {e}")
            return jsonify({'error': str(e)})

    # =========================================================================
    # ORACLE/OBSERVATION PROTOCOL API ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/oracle/publish', methods=['POST'])
    @login_required
    def api_p2p_oracle_publish():
        """Publish an observation (must be registered oracle)."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            stream_id = data.get('stream_id')
            value = data.get('value')
            timestamp = data.get('timestamp')
            metadata = data.get('metadata', {})

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400
            if value is None:
                return jsonify({'success': False, 'error': 'value is required'}), 400

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_oracle_network') or not startup._oracle_network:
                return jsonify({'success': False, 'error': 'Oracle network not initialized'}), 503

            oracle = startup._oracle_network

            async def do_publish():
                return await oracle.publish_observation(
                    stream_id=stream_id,
                    value=value,
                    timestamp=int(timestamp) if timestamp else None,
                    metadata=metadata
                )

            try:
                loop = asyncio.new_event_loop()
                observation = loop.run_until_complete(do_publish())
            finally:
                loop.close()

            if observation:
                return jsonify({
                    'success': True,
                    'observation': {
                        'hash': observation.hash,
                        'stream_id': observation.stream_id,
                        'value': observation.value,
                        'timestamp': observation.timestamp,
                        'oracle': observation.oracle,
                    }
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to publish observation'})

        except Exception as e:
            logger.warning(f"Failed to publish observation: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/oracle/observations/stream/<stream_id>')
    @login_required
    def api_p2p_oracle_observations_stream(stream_id):
        """Get cached observations for a stream."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_oracle_network') or not startup._oracle_network:
                return jsonify({'observations': [], 'error': 'Oracle network not initialized'})

            oracle = startup._oracle_network
            limit = request.args.get('limit', 100, type=int)

            observations = oracle.get_cached_observations(stream_id, limit)
            result = []
            for obs in observations:
                result.append({
                    'hash': obs.hash,
                    'stream_id': obs.stream_id,
                    'value': obs.value,
                    'timestamp': obs.timestamp,
                    'oracle': obs.oracle,
                })

            return jsonify({
                'stream_id': stream_id,
                'observations': result,
                'total': len(result),
            })

        except Exception as e:
            logger.warning(f"Failed to get stream observations: {e}")
            return jsonify({'observations': [], 'error': str(e)})

    @app.route('/api/p2p/oracle/observations')
    @login_required
    def api_p2p_oracle_observations():
        """Get recent observations via IPC API."""
        try:
            from web.wsgi import _ipc_get

            limit = request.args.get('limit', 20, type=int)
            result = _ipc_get(f'/p2p/oracle/observations?limit={limit}')
            if result:
                return jsonify(result)

            return jsonify({'observations': [], 'count': 0})

        except Exception as e:
            logger.warning(f"Failed to get observations: {e}")
            return jsonify({'observations': [], 'error': str(e)})

    @app.route('/api/p2p/oracle/observations/latest')
    @login_required
    def api_p2p_oracle_observations_latest():
        """Get latest observations via IPC API."""
        try:
            from web.wsgi import _ipc_get

            include_own = request.args.get('include_own', 'true').lower()
            result = _ipc_get(f'/p2p/oracle/observations?limit=100&include_own={include_own}')
            if result and result.get('observations'):
                # Get latest per stream
                latest_by_stream = {}
                for obs in result['observations']:
                    stream_id = obs.get('stream_id', '')
                    if stream_id not in latest_by_stream:
                        latest_by_stream[stream_id] = obs

                latest = list(latest_by_stream.values())
                latest.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

                return jsonify({
                    'observations': latest,
                    'total': len(latest),
                })

            return jsonify({'observations': [], 'total': 0})

        except Exception as e:
            logger.warning(f"Failed to get latest observations: {e}")
            return jsonify({'observations': [], 'error': str(e)})

    @app.route('/api/p2p/oracle/observations/my')
    @login_required
    def api_p2p_oracle_observations_my():
        """Get our own published observations via IPC API with pagination."""
        try:
            from web.wsgi import _ipc_get

            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            result = _ipc_get(f'/p2p/oracle/observations/my?page={page}&per_page={per_page}')
            if result:
                return jsonify(result)

            return jsonify({'observations': [], 'total': 0, 'page': page, 'per_page': per_page})

        except Exception as e:
            logger.warning(f"Failed to get my published observations: {e}")
            return jsonify({'observations': [], 'error': str(e)})

    @app.route('/api/p2p/oracle/oracles')
    @login_required
    def api_p2p_oracle_oracles():
        """List all registered oracles via IPC API (includes our own registrations)."""
        try:
            from web.wsgi import _ipc_get

            # Get known oracles from network and our own registrations
            known_result = _ipc_get('/p2p/oracle/known')
            summary_result = _ipc_get('/p2p/oracle/summary')
            identity_result = _ipc_get('/p2p/identity')

            oracles = known_result.get('oracles', []) if known_result else []

            # Add our own primary/secondary registrations if not already in list
            if summary_result and identity_result:
                my_address = identity_result.get('evrmore_address') or identity_result.get('address', '')
                my_peer_id = identity_result.get('peer_id', '')

                # Existing oracle addresses for deduplication
                existing = {(o.get('stream_id'), o.get('oracle_address')) for o in oracles}

                # Add our primary registrations
                for stream_id in summary_result.get('primary_streams', []):
                    if (stream_id, my_address) not in existing:
                        oracles.append({
                            'stream_id': stream_id,
                            'oracle_address': my_address,
                            'peer_id': my_peer_id,
                            'reputation': 1.0,
                            'is_primary': True,
                            'timestamp': 0,
                        })

                # Add our secondary registrations
                for stream_id in summary_result.get('secondary_streams', []):
                    if (stream_id, my_address) not in existing:
                        oracles.append({
                            'stream_id': stream_id,
                            'oracle_address': my_address,
                            'peer_id': my_peer_id,
                            'reputation': 1.0,
                            'is_primary': False,
                            'timestamp': 0,
                        })

            # Filter by stream_id if specified
            stream_id_filter = request.args.get('stream_id')
            if stream_id_filter:
                oracles = [o for o in oracles if o.get('stream_id') == stream_id_filter]

            return jsonify({
                'oracles': oracles,
                'total': len(oracles),
            })

        except Exception as e:
            logger.warning(f"Failed to list oracles: {e}")
            return jsonify({'oracles': [], 'error': str(e)})

    @app.route('/api/p2p/oracle/register', methods=['POST'])
    @login_required
    def api_p2p_oracle_register():
        """Register as an oracle for a stream via IPC API."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            stream_id = data.get('stream_id')
            is_primary = data.get('is_primary', False)
            data_source = data.get('data_source')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            # Use appropriate IPC endpoint based on role
            if is_primary:
                endpoint = '/p2p/oracle/register-primary'
                payload = {'stream_id': stream_id}
                if data_source:
                    payload['data_source'] = data_source
            else:
                endpoint = '/p2p/oracle/register-secondary'
                payload = {'stream_id': stream_id}

            result = _ipc_post(endpoint, payload)

            if result and result.get('success'):
                return jsonify(result)
            else:
                error = result.get('error', 'Failed to register') if result else 'IPC request failed'
                return jsonify({'success': False, 'error': error}), 500

        except Exception as e:
            logger.warning(f"Failed to register as oracle: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/oracle/subscribe', methods=['POST'])
    @login_required
    def api_p2p_oracle_subscribe():
        """Subscribe to observations for a stream via IPC API."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            result = _ipc_post('/p2p/oracle/subscribe', {'stream_id': stream_id})
            if result:
                return jsonify(result)

            # Fallback if IPC not available
            return jsonify({'success': False, 'error': 'IPC API not available'}), 503

        except Exception as e:
            logger.warning(f"Failed to subscribe to observations: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/oracle/unsubscribe', methods=['POST'])
    @login_required
    def api_p2p_oracle_unsubscribe():
        """Unsubscribe from observations for a stream via IPC API."""
        try:
            from web.wsgi import _ipc_post

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            result = _ipc_post('/p2p/oracle/unsubscribe', {'stream_id': stream_id})
            if result:
                return jsonify(result)

            # Fallback if IPC not available
            return jsonify({'success': False, 'error': 'IPC API not available'}), 503

        except Exception as e:
            logger.warning(f"Failed to unsubscribe from observations: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/oracle/stats')
    @login_required
    def api_p2p_oracle_stats():
        """Get oracle network statistics via IPC API."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/oracle/stats')
            if result:
                return jsonify(result)

            # Fallback if IPC not available
            return jsonify({
                'started': False,
                'subscribed_streams': 0,
                'my_oracle_registrations': 0,
                'known_oracles': 0,
                'cached_observations': 0,
            })

        except Exception as e:
            logger.warning(f"Failed to get oracle stats: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/oracle/known')
    @login_required
    def api_p2p_oracle_known():
        """Get list of known oracles via IPC API."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/oracle/known')
            if result:
                return jsonify(result)

            return jsonify({'oracles': [], 'count': 0})

        except Exception as e:
            logger.warning(f"Failed to get known oracles: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/oracle/my_registrations')
    @login_required
    def api_p2p_oracle_my_registrations():
        """Get list of our oracle registrations via IPC API."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/oracle/my_registrations')
            if result:
                return jsonify(result)

            return jsonify({'registrations': [], 'count': 0})

        except Exception as e:
            logger.warning(f"Failed to get my oracle registrations: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/oracle/subscriptions')
    @login_required
    def api_p2p_oracle_subscriptions():
        """Get list of streams we're subscribed to via IPC API."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/oracle/subscriptions')
            if result:
                return jsonify(result)

            return jsonify({'subscriptions': [], 'count': 0})

        except Exception as e:
            logger.warning(f"Failed to get oracle subscriptions: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/oracle/summary')
    @login_required
    def api_p2p_oracle_summary():
        """Get summary of our oracle registrations (primary/secondary)."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/oracle/summary')
            if result:
                return jsonify(result)

            return jsonify({
                'primary_count': 0,
                'secondary_count': 0,
                'total_count': 0,
                'primary_streams': [],
                'secondary_streams': [],
            })

        except Exception as e:
            logger.warning(f"Failed to get oracle summary: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/oracle/role/<stream_id>')
    @login_required
    def api_p2p_oracle_role(stream_id: str):
        """Get our oracle role for a specific stream."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get(f'/p2p/oracle/role/{stream_id}')
            if result:
                return jsonify(result)

            return jsonify({'stream_id': stream_id, 'role': 'none'})

        except Exception as e:
            logger.warning(f"Failed to get oracle role: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/oracle/register-secondary', methods=['POST'])
    @login_required
    def api_p2p_oracle_register_secondary():
        """Register as a secondary oracle for a stream."""
        try:
            import requests as req

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            resp = req.post(
                'http://127.0.0.1:24602/p2p/oracle/register-secondary',
                json={'stream_id': stream_id},
                timeout=10
            )

            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {'error': resp.text}
                return jsonify(error_data), resp.status_code

        except Exception as e:
            logger.warning(f"Failed to register as secondary oracle: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/oracle/stream-info/<stream_id>')
    @login_required
    def api_p2p_oracle_stream_info(stream_id: str):
        """Get oracle info for a specific stream (primary, secondaries)."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get(f'/p2p/oracle/stream-info/{stream_id}')
            if result:
                return jsonify(result)

            return jsonify({
                'stream_id': stream_id,
                'primary': None,
                'secondaries': [],
                'my_role': 'none',
            })

        except Exception as e:
            logger.warning(f"Failed to get oracle stream info: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/oracle/templates')
    @login_required
    def api_p2p_oracle_templates():
        """Get available oracle data source templates."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/oracle/templates')
            if result:
                return jsonify(result)

            return jsonify({'success': False, 'templates': []})

        except Exception as e:
            logger.warning(f"Failed to get oracle templates: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/p2p/oracle/register-primary', methods=['POST'])
    @login_required
    def api_p2p_oracle_register_primary():
        """Register as a primary oracle with data source configuration."""
        try:
            import requests as req

            data = request.get_json() or {}
            stream_id = data.get('stream_id')
            data_source = data.get('data_source')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            # Call IPC API
            resp = req.post(
                'http://127.0.0.1:24602/p2p/oracle/register-primary',
                json={'stream_id': stream_id, 'data_source': data_source},
                timeout=30
            )

            if resp.ok:
                return jsonify(resp.json())
            else:
                error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {'error': resp.text}
                return jsonify(error_data), resp.status_code

        except Exception as e:
            logger.warning(f"Failed to register as primary oracle: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/oracle/unregister', methods=['POST'])
    @login_required
    def api_p2p_oracle_unregister():
        """Unregister as an oracle for a stream."""
        try:
            import requests as req

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            # Call IPC API
            resp = req.post(
                'http://127.0.0.1:24602/p2p/oracle/unregister',
                json={'stream_id': stream_id},
                timeout=10
            )

            if resp.ok:
                return jsonify(resp.json())
            else:
                error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {'error': resp.text}
                return jsonify(error_data), resp.status_code

        except Exception as e:
            logger.warning(f"Failed to unregister as oracle: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    # =========================================================================
    # STREAM REGISTRY API ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/streams/discover')
    @login_required
    def api_p2p_streams_discover():
        """Discover available streams via IPC API."""
        try:
            from web.wsgi import _ipc_get

            # Build query string from request args
            params = []
            if request.args.get('source'):
                params.append(f"source={request.args.get('source')}")
            if request.args.get('datatype'):
                params.append(f"datatype={request.args.get('datatype')}")
            for tag in request.args.getlist('tag'):
                params.append(f"tag={tag}")
            limit = request.args.get('limit', 100, type=int)
            params.append(f"limit={limit}")

            query_string = '&'.join(params)
            result = _ipc_get(f'/p2p/streams/discover?{query_string}')

            if result:
                return jsonify(result)
            return jsonify({'streams': [], 'total': 0})

        except Exception as e:
            logger.warning(f"Failed to discover streams: {e}")
            return jsonify({'streams': [], 'error': str(e)})

    @app.route('/api/p2p/streams/claim', methods=['POST'])
    @login_required
    def api_p2p_streams_claim():
        """Claim a predictor slot on a stream via IPC API."""
        try:
            import requests as req

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            # Call IPC API
            resp = req.post(
                'http://127.0.0.1:24602/p2p/streams/claim',
                json=data,
                timeout=10
            )

            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {'error': resp.text}
                return jsonify(error_data), resp.status_code

        except Exception as e:
            logger.warning(f"Failed to claim stream: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/streams/release', methods=['POST'])
    @login_required
    def api_p2p_streams_release():
        """Release claim on a stream via IPC API."""
        try:
            import requests as req

            data = request.get_json() or {}
            stream_id = data.get('stream_id')

            if not stream_id:
                return jsonify({'success': False, 'error': 'stream_id is required'}), 400

            # Call IPC API
            resp = req.post(
                'http://127.0.0.1:24602/p2p/streams/release',
                json=data,
                timeout=10
            )

            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {'error': resp.text}
                return jsonify(error_data), resp.status_code

        except Exception as e:
            logger.warning(f"Failed to release claim: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/streams/my')
    @login_required
    def api_p2p_streams_my():
        """Get my claimed streams via IPC API."""
        try:
            import requests as req

            resp = req.get('http://127.0.0.1:24602/p2p/streams/my-claims', timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                claims = data.get('claims', [])
                # Return both 'streams' and 'claims' for compatibility
                return jsonify({
                    'streams': claims,
                    'claims': claims,
                    'total': data.get('count', 0),
                    'count': data.get('count', 0),
                })
            else:
                return jsonify({'streams': [], 'claims': [], 'error': f'IPC error: {resp.status_code}'})

        except Exception as e:
            logger.warning(f"Failed to get my streams: {e}")
            return jsonify({'streams': [], 'claims': [], 'error': str(e)})

    @app.route('/api/p2p/streams/my-claims')
    @login_required
    def api_p2p_streams_my_claims():
        """Get my claimed streams with full claim details via IPC API."""
        try:
            import requests as req

            resp = req.get('http://127.0.0.1:24602/p2p/streams/my-claims', timeout=5)
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({'claims': [], 'count': 0, 'error': f'IPC error: {resp.status_code}'})

        except Exception as e:
            logger.warning(f"Failed to get my claims: {e}")
            return jsonify({'claims': [], 'count': 0, 'error': str(e)})

    @app.route('/api/p2p/streams/definition/<stream_id>')
    @login_required
    def api_p2p_streams_definition(stream_id):
        """Get stream definition by ID."""
        try:
            from satorineuron.init import start
            import asyncio

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_stream_registry') or not startup._stream_registry:
                return jsonify({'stream': None, 'error': 'Stream registry not initialized'})

            registry = startup._stream_registry

            async def do_get():
                return await registry.get_stream(stream_id)

            try:
                loop = asyncio.new_event_loop()
                stream = loop.run_until_complete(do_get())
            finally:
                loop.close()

            if stream:
                return jsonify({
                    'stream': {
                        'stream_id': stream.stream_id,
                        'source': stream.source,
                        'stream': stream.stream,
                        'target': stream.target,
                        'datatype': stream.datatype,
                        'cadence': stream.cadence,
                        'predictor_slots': stream.predictor_slots,
                        'creator': stream.creator,
                        'timestamp': stream.timestamp,
                        'description': stream.description,
                        'tags': stream.tags,
                    }
                })
            else:
                return jsonify({'stream': None, 'error': 'Stream not found'})

        except Exception as e:
            logger.warning(f"Failed to get stream: {e}")
            return jsonify({'stream': None, 'error': str(e)})

    @app.route('/api/p2p/streams/predictors/<stream_id>')
    @login_required
    def api_p2p_streams_predictors(stream_id):
        """Get predictors for a stream."""
        try:
            from satorineuron.init import start
            import asyncio

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_stream_registry') or not startup._stream_registry:
                return jsonify({'predictors': [], 'error': 'Stream registry not initialized'})

            registry = startup._stream_registry

            async def do_get():
                claims = await registry.get_stream_claims(stream_id)
                return claims

            try:
                loop = asyncio.new_event_loop()
                claims = loop.run_until_complete(do_get())
            finally:
                loop.close()

            predictors = []
            for c in claims:
                predictors.append({
                    'predictor': c.predictor,
                    'slot_index': c.slot_index,
                    'peer_id': c.peer_id,
                    'timestamp': c.timestamp,
                    'expires': c.expires,
                    'stake': c.stake,
                })

            return jsonify({
                'stream_id': stream_id,
                'predictors': predictors,
                'total': len(predictors),
            })

        except Exception as e:
            logger.warning(f"Failed to get stream predictors: {e}")
            return jsonify({'predictors': [], 'error': str(e)})

    @app.route('/api/p2p/streams/register', methods=['POST'])
    @login_required
    def api_p2p_streams_register():
        """Register a new stream definition."""
        try:
            from satorineuron.init import start
            import asyncio

            data = request.get_json() or {}
            source = data.get('source')
            stream = data.get('stream')
            target = data.get('target')
            datatype = data.get('datatype', 'numeric')
            cadence = data.get('cadence', 60)
            predictor_slots = data.get('predictor_slots', 10)
            description = data.get('description', '')
            tags = data.get('tags', [])

            if not source or not stream or not target:
                return jsonify({'success': False, 'error': 'source, stream, and target are required'}), 400

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if not startup or not hasattr(startup, '_stream_registry') or not startup._stream_registry:
                return jsonify({'success': False, 'error': 'Stream registry not initialized'}), 503

            registry = startup._stream_registry

            async def do_register():
                return await registry.register_stream(
                    source=source,
                    stream=stream,
                    target=target,
                    datatype=datatype,
                    cadence=int(cadence),
                    predictor_slots=int(predictor_slots),
                    description=description,
                    tags=tags
                )

            try:
                loop = asyncio.new_event_loop()
                definition = loop.run_until_complete(do_register())
            finally:
                loop.close()

            if definition:
                return jsonify({
                    'success': True,
                    'stream': {
                        'stream_id': definition.stream_id,
                        'source': definition.source,
                        'stream': definition.stream,
                        'target': definition.target,
                    }
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to register stream'})

        except Exception as e:
            logger.warning(f"Failed to register stream: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/p2p/streams/stats')
    @login_required
    def api_p2p_streams_stats():
        """Get stream registry statistics via IPC API."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/streams/stats')
            if result:
                return jsonify(result)

            # Fallback if IPC not available
            return jsonify({
                'started': False,
                'known_streams': 0,
                'my_claims': 0,
                'total_claims': 0,
            })

        except Exception as e:
            logger.warning(f"Failed to get stream stats: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/streams/engine-status')
    @login_required
    def api_p2p_streams_engine_status():
        """Get Engine stream model status via IPC API."""
        try:
            from web.wsgi import _ipc_get

            result = _ipc_get('/p2p/streams/engine-status')
            if result:
                return jsonify(result)

            # Fallback if IPC not available
            return jsonify({
                'active_models': 0,
                'stream_uuids': [],
                'engine_ready': False,
            })

        except Exception as e:
            logger.warning(f"Failed to get engine status: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/streams/renew-claims', methods=['POST'])
    @login_required
    def api_p2p_streams_renew_claims():
        """Manually trigger renewal of stream claims."""
        try:
            import requests as req

            resp = req.post('http://127.0.0.1:24602/p2p/streams/renew-claims', timeout=30)

            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {'error': resp.text}
                return jsonify(error_data), resp.status_code

        except Exception as e:
            logger.warning(f"Failed to renew claims: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    # =========================================================================
    # CONSENSUS ROUND ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/consensus/status')
    @login_required
    def api_p2p_consensus_status():
        """Get current consensus round status."""
        try:
            from web.wsgi import get_web_consensus_status

            status = get_web_consensus_status()

            # Return placeholder data when consensus not running
            if not status or not status.get('success'):
                return jsonify({
                    'current_round': 0,
                    'phase': 'inactive',
                    'progress_percent': 0,
                    'start_time': None,
                    'end_time': None,
                    'validators': 0,
                    'total_votes': 0,
                    'streams_in_round': 0,
                    'quorum_percent': 0,
                    'my_participation': None,
                })

            return jsonify({
                'current_round': status.get('current_round', 0),
                'phase': status.get('phase', 'unknown'),
                'progress_percent': 0,  # TODO: Calculate from timestamps
                'start_time': status.get('round_start_time'),
                'end_time': status.get('phase_deadline'),
                'validators': 0,  # TODO: Add to IPC
                'total_votes': status.get('vote_count', 0),
                'streams_in_round': 0,  # TODO: Add to IPC
                'quorum_percent': 0,  # TODO: Calculate
                'my_participation': 'voted' if status.get('my_vote_submitted') else None,
            })

        except Exception as e:
            logger.warning(f"Failed to get consensus status: {e}")
            return jsonify({
                'error': str(e),
                'current_round': 0,
                'phase': 'error',
            })

    @app.route('/api/p2p/consensus/history')
    @login_required
    def api_p2p_consensus_history():
        """Get recent consensus round history."""
        try:
            from web.wsgi import get_web_consensus_history

            limit = request.args.get('limit', 10, type=int)
            limit = min(limit, 50)  # Cap at 50

            rounds = get_web_consensus_history(limit=limit)

            return jsonify({
                'rounds': rounds,
                'total_completed': len(rounds),
            })

        except Exception as e:
            logger.warning(f"Failed to get consensus history: {e}")
            return jsonify({
                'error': str(e),
                'rounds': [],
                'total_completed': 0,
            })

    @app.route('/api/p2p/consensus/vote', methods=['POST'])
    @login_required
    def api_p2p_consensus_vote():
        """Submit a consensus vote for prediction scores."""
        try:
            from satorineuron.init import start

            data = request.get_json() or {}
            stream_id = data.get('stream_id')
            scores = data.get('scores')  # Dict of predictor_address -> score

            if not stream_id or not scores:
                return jsonify({
                    'success': False,
                    'error': 'stream_id and scores are required',
                })

            startup = start.getStart() if hasattr(start, 'getStart') else None
            consensus = None
            if startup:
                consensus = getattr(startup, '_consensus_manager', None)

            if not consensus:
                return jsonify({
                    'success': False,
                    'error': 'Consensus manager not available',
                })

            result = consensus.submit_vote(stream_id=stream_id, scores=scores)

            return jsonify({
                'success': result.get('success', False),
                'vote_id': result.get('vote_id'),
                'error': result.get('error'),
            })

        except Exception as e:
            logger.warning(f"Failed to submit consensus vote: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            })

    # =========================================================================
    # DISTRIBUTION STATUS ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/distribution/status')
    @login_required
    def api_p2p_distribution_status():
        """Get current distribution status (read-only for all users)."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None

            # Try to get distribution manager
            distribution = None
            if startup:
                distribution = getattr(startup, '_distribution_manager', None)

            if not distribution:
                # Return placeholder data when distribution not running
                return jsonify({
                    'last_distribution': None,
                    'pending_distribution': None,
                    'my_pending_reward': 0,
                })

            status = distribution.get_status()

            return jsonify({
                'last_distribution': status.get('last_distribution'),
                'pending_distribution': status.get('pending_distribution'),
                'my_pending_reward': status.get('my_pending_reward', 0),
            })

        except Exception as e:
            logger.warning(f"Failed to get distribution status: {e}")
            return jsonify({
                'error': str(e),
                'last_distribution': None,
                'pending_distribution': None,
                'my_pending_reward': 0,
            })

    @app.route('/api/p2p/distribution/history')
    @login_required
    def api_p2p_distribution_history():
        """Get distribution history for the current user."""
        try:
            from satorineuron.init import start

            limit = request.args.get('limit', 20, type=int)
            limit = min(limit, 100)

            startup = start.getStart() if hasattr(start, 'getStart') else None
            distribution = None
            if startup:
                distribution = getattr(startup, '_distribution_manager', None)

            if not distribution:
                return jsonify({
                    'distributions': [],
                    'total_earned': 0,
                })

            history = distribution.get_my_history(limit=limit)

            return jsonify({
                'distributions': history.get('distributions', []),
                'total_earned': history.get('total_earned', 0),
            })

        except Exception as e:
            logger.warning(f"Failed to get distribution history: {e}")
            return jsonify({
                'error': str(e),
                'distributions': [],
                'total_earned': 0,
            })

    # =========================================================================
    # SIGNER-ONLY DISTRIBUTION ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/signer/distribution/status')
    @login_required
    def api_p2p_signer_distribution_status():
        """Get distribution status for signers (includes control info)."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None

            # Check if user is a signer
            # TODO: Add proper signer check
            is_signer = True  # Placeholder

            distribution = None
            if startup:
                distribution = getattr(startup, '_distribution_manager', None)

            if not distribution:
                return jsonify({
                    'status': 'not_ready',
                    'pool_amount': 0,
                    'eligible_count': 0,
                    'consensus_round': None,
                    'pending_distribution': None,
                })

            status = distribution.get_signer_status()

            return jsonify({
                'status': status.get('status', 'not_ready'),
                'pool_amount': status.get('pool_amount', 0),
                'eligible_count': status.get('eligible_count', 0),
                'consensus_round': status.get('consensus_round'),
                'pending_distribution': status.get('pending_distribution'),
            })

        except Exception as e:
            logger.warning(f"Failed to get signer distribution status: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error',
            })

    @app.route('/api/p2p/signer/distribution/history')
    @login_required
    def api_p2p_signer_distribution_history():
        """Get distribution history for signers."""
        try:
            from satorineuron.init import start

            limit = request.args.get('limit', 10, type=int)
            limit = min(limit, 50)

            startup = start.getStart() if hasattr(start, 'getStart') else None
            distribution = None
            if startup:
                distribution = getattr(startup, '_distribution_manager', None)

            if not distribution:
                return jsonify({'distributions': []})

            history = distribution.get_all_history(limit=limit)

            return jsonify({
                'distributions': history.get('distributions', []),
            })

        except Exception as e:
            logger.warning(f"Failed to get signer distribution history: {e}")
            return jsonify({'error': str(e), 'distributions': []})

    @app.route('/api/p2p/signer/distribution/trigger', methods=['POST'])
    @login_required
    def api_p2p_signer_distribution_trigger():
        """Trigger a new distribution (signer only)."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None

            # TODO: Add proper signer verification
            distribution = None
            if startup:
                distribution = getattr(startup, '_distribution_manager', None)

            if not distribution:
                return jsonify({
                    'success': False,
                    'error': 'Distribution manager not available',
                })

            result = distribution.trigger_distribution()

            return jsonify({
                'success': result.get('success', False),
                'distribution_id': result.get('distribution_id'),
                'required_signatures': result.get('required_signatures', 3),
                'error': result.get('error'),
            })

        except Exception as e:
            logger.warning(f"Failed to trigger distribution: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            })

    @app.route('/api/p2p/signer/distribution/preview')
    @login_required
    def api_p2p_signer_distribution_preview():
        """Preview the pending distribution (signer only)."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            distribution = None
            if startup:
                distribution = getattr(startup, '_distribution_manager', None)

            if not distribution:
                return jsonify({'error': 'Distribution manager not available'})

            preview = distribution.get_preview()

            return jsonify({
                'round_id': preview.get('round_id'),
                'pool_amount': preview.get('pool_amount', 0),
                'recipient_count': preview.get('recipient_count', 0),
                'top_recipients': preview.get('top_recipients', []),
            })

        except Exception as e:
            logger.warning(f"Failed to get distribution preview: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/p2p/signer/distribution/sign', methods=['POST'])
    @login_required
    def api_p2p_signer_distribution_sign():
        """Sign the pending distribution (signer only)."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            distribution = None
            if startup:
                distribution = getattr(startup, '_distribution_manager', None)

            if not distribution:
                return jsonify({
                    'success': False,
                    'error': 'Distribution manager not available',
                })

            result = distribution.sign_distribution()

            return jsonify({
                'success': result.get('success', False),
                'signatures': result.get('signatures', 0),
                'required': result.get('required', 3),
                'error': result.get('error'),
            })

        except Exception as e:
            logger.warning(f"Failed to sign distribution: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            })

    @app.route('/api/p2p/signer/distribution/reject', methods=['POST'])
    @login_required
    def api_p2p_signer_distribution_reject():
        """Reject the pending distribution (signer only)."""
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            distribution = None
            if startup:
                distribution = getattr(startup, '_distribution_manager', None)

            if not distribution:
                return jsonify({
                    'success': False,
                    'error': 'Distribution manager not available',
                })

            result = distribution.reject_distribution()

            return jsonify({
                'success': result.get('success', False),
                'error': result.get('error'),
            })

        except Exception as e:
            logger.warning(f"Failed to reject distribution: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            })

    # ========================================================================
    # BADGE SYSTEM ENDPOINTS
    # ========================================================================

    @app.route('/api/badges/my')
    @login_required
    def api_badges_my():
        """Get badges earned by the current user.

        Returns all badges earned by this node's wallet address, grouped by
        category with metadata for each badge.
        """
        try:
            from satorineuron.init import start

            # Get wallet address
            startup = start.getStart() if hasattr(start, 'getStart') else None
            wallet_address = None
            if startup and hasattr(startup, 'wallet') and startup.wallet:
                wallet_address = startup.wallet.address

            if not wallet_address:
                return jsonify({
                    'badges': [],
                    'total': 0,
                    'error': 'Wallet not available'
                })

            # Try to get badge manager from startup
            badge_manager = None
            if startup:
                badge_manager = getattr(startup, '_badge_manager', None)

            # If no badge manager in startup, create a temporary one for querying
            if not badge_manager:
                try:
                    from satorip2p.protocol.badges import BadgeManager
                    badge_manager = BadgeManager()
                except ImportError:
                    return jsonify({
                        'badges': [],
                        'total': 0,
                        'by_category': {},
                        'error': 'Badge system not available'
                    })

            # Get earned badges
            earned_badges = badge_manager.get_badges_for_address(wallet_address)

            # Group by category
            by_category = {}
            for badge in earned_badges:
                # Get badge definition for full metadata
                badge_def = badge_manager.get_badge_definition(badge.badge_id)
                badge_data = badge.to_dict()

                # Add definition metadata if available
                if badge_def:
                    badge_data['name'] = badge_def.name
                    badge_data['description'] = badge_def.description
                    badge_data['category'] = badge_def.category
                    badge_data['rarity'] = badge_def.rarity

                category = badge_data.get('category', 'unknown')
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(badge_data)

            return jsonify({
                'badges': [b.to_dict() for b in earned_badges],
                'total': len(earned_badges),
                'by_category': by_category,
                'address': wallet_address
            })

        except Exception as e:
            logger.warning(f"Failed to get user badges: {e}")
            return jsonify({
                'badges': [],
                'total': 0,
                'by_category': {},
                'error': str(e)
            })

    @app.route('/api/badges/all')
    @login_required
    def api_badges_all():
        """Get all available badge definitions.

        Returns the complete catalog of badges that can be earned,
        grouped by category.
        """
        try:
            from satorip2p.protocol.badges import (
                BadgeManager,
                ACHIEVEMENTS,
                STREAK_BADGES,
                COMMUNITY_BADGES,
                SPECIAL_BADGES,
                DONATION_TIERS,
                ROLE_BADGES,
                BadgeCategory,
            )

            badge_manager = BadgeManager()
            all_badges = badge_manager.list_all_badges()

            # Group by category
            by_category = {}
            for badge in all_badges:
                badge_dict = badge.to_dict()
                category = badge_dict.get('category', 'unknown')
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(badge_dict)

            # Get counts
            counts = badge_manager.get_badge_count_by_category()

            return jsonify({
                'badges': [b.to_dict() for b in all_badges],
                'total': len(all_badges),
                'by_category': by_category,
                'counts': counts
            })

        except Exception as e:
            logger.warning(f"Failed to get badge catalog: {e}")
            return jsonify({
                'badges': [],
                'total': 0,
                'by_category': {},
                'counts': {},
                'error': str(e)
            })

    @app.route('/api/badges/stats')
    @login_required
    def api_badges_stats():
        """Get badge system statistics.

        Returns overall stats about badge issuance across the network.
        """
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            badge_manager = None
            if startup:
                badge_manager = getattr(startup, '_badge_manager', None)

            if not badge_manager:
                try:
                    from satorip2p.protocol.badges import BadgeManager
                    badge_manager = BadgeManager()
                except ImportError:
                    return jsonify({'error': 'Badge system not available'})

            stats = badge_manager.get_stats()
            counts = badge_manager.get_badge_count_by_category()

            return jsonify({
                **stats,
                'badge_types': counts,
                'total_badge_types': sum(counts.values())
            })

        except Exception as e:
            logger.warning(f"Failed to get badge stats: {e}")
            return jsonify({'error': str(e)})

    @app.route('/api/badges/leaderboard')
    @login_required
    def api_badges_leaderboard():
        """Get badge leaderboard - top badge holders.

        Returns the top addresses by badge count.
        """
        try:
            from satorineuron.init import start

            limit = request.args.get('limit', 10, type=int)

            startup = start.getStart() if hasattr(start, 'getStart') else None
            badge_manager = None
            if startup:
                badge_manager = getattr(startup, '_badge_manager', None)

            if not badge_manager:
                return jsonify({
                    'leaderboard': [],
                    'error': 'Badge manager not initialized'
                })

            # Build leaderboard from earned badges
            address_counts = {}
            for address, badges in badge_manager._earned_badges.items():
                address_counts[address] = len(badges)

            # Sort by count descending
            sorted_addresses = sorted(
                address_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

            leaderboard = [
                {'address': addr, 'badge_count': count, 'rank': i + 1}
                for i, (addr, count) in enumerate(sorted_addresses)
            ]

            return jsonify({
                'leaderboard': leaderboard,
                'total_holders': len(address_counts)
            })

        except Exception as e:
            logger.warning(f"Failed to get badge leaderboard: {e}")
            return jsonify({
                'leaderboard': [],
                'error': str(e)
            })

    @app.route('/api/badges/progress')
    @login_required
    def api_badges_progress():
        """Get badge progress for current user.

        Returns progress towards various badges (streaks, achievements, etc.)
        """
        try:
            from satorineuron.init import start

            startup = start.getStart() if hasattr(start, 'getStart') else None
            wallet_address = None
            if startup and hasattr(startup, 'wallet') and startup.wallet:
                wallet_address = startup.wallet.address

            if not wallet_address:
                return jsonify({'error': 'Wallet not available'})

            progress = {
                'streak': {
                    'current_days': 0,
                    'next_badge': 'STREAK_EPOCH',
                    'days_needed': 7,
                    'progress_percent': 0
                },
                'voting': {
                    'total_votes': 0,
                    'next_badge': 'VOTER100',
                    'votes_needed': 100,
                    'progress_percent': 0
                },
                'mentoring': {
                    'nodes_mentored': 0,
                    'next_badge': 'MENTOR',
                    'needed': 5,
                    'progress_percent': 0
                },
                'data_service': {
                    'requests_served': 0,
                    'next_badge': 'DATA_PROVIDER',
                    'needed': 1000,
                    'progress_percent': 0
                }
            }

            # Get uptime tracker for streak info
            if startup:
                uptime = getattr(startup, '_uptime', None)
                if uptime and hasattr(uptime, 'get_streak'):
                    streak_data = uptime.get_streak(wallet_address)
                    if streak_data:
                        days = streak_data.get('streak_days', 0)
                        progress['streak']['current_days'] = days

                        # Calculate next badge
                        if days < 7:
                            progress['streak']['next_badge'] = 'STREAK_EPOCH'
                            progress['streak']['days_needed'] = 7
                        elif days < 28:
                            progress['streak']['next_badge'] = 'STREAK_EPOCH_4'
                            progress['streak']['days_needed'] = 28
                        elif days < 91:
                            progress['streak']['next_badge'] = 'STREAK_EPOCH_13'
                            progress['streak']['days_needed'] = 91
                        else:
                            progress['streak']['next_badge'] = 'STREAK_EPOCH_52'
                            progress['streak']['days_needed'] = 364

                        target = progress['streak']['days_needed']
                        progress['streak']['progress_percent'] = min(100, int((days / target) * 100))

                # Get governance for vote count
                governance = getattr(startup, '_governance', None)
                if governance and hasattr(governance, 'get_vote_count'):
                    votes = governance.get_vote_count(wallet_address)
                    progress['voting']['total_votes'] = votes
                    if votes < 100:
                        progress['voting']['next_badge'] = 'VOTER100'
                        progress['voting']['votes_needed'] = 100
                    else:
                        progress['voting']['next_badge'] = 'VOTER500'
                        progress['voting']['votes_needed'] = 500
                    target = progress['voting']['votes_needed']
                    progress['voting']['progress_percent'] = min(100, int((votes / target) * 100))

            return jsonify(progress)

        except Exception as e:
            logger.warning(f"Failed to get badge progress: {e}")
            return jsonify({'error': str(e)})

    # =========================================================================
    # ENGINE PERFORMANCE ENDPOINTS (from upstream)
    # =========================================================================

    @app.route('/api/engine/performance', methods=['GET'])
    @login_required
    def get_engine_performance():
        """Get engine performance metrics via IPC.

        Returns last 100 observations and predictions with accuracy calculations.

        Returns:
            JSON with:
            - observations: [{ts, value}, ...]
            - predictions: [{ts, value}, ...]
            - accuracy: [{ts, error, abs_error}, ...]
            - stats: {avg_error, avg_abs_error, accuracy_pct}
            - available: bool indicating if engine is available
        """
        from web.wsgi import _ipc_get

        # Return empty data structure when engine isn't available
        empty_response = {
            'observations': [],
            'predictions': [],
            'accuracy': [],
            'stats': {},
            'available': False
        }

        try:
            # Get engine performance via IPC (neuron process has access to AI engine)
            result = _ipc_get('/engine/performance')
            if result:
                return jsonify(result)
            else:
                logger.debug("Engine performance IPC returned no data")
                return jsonify(empty_response)

        except Exception as e:
            logger.error(f"Error getting engine performance via IPC: {e}")
            return jsonify(empty_response)

    @app.route('/api/wallet/import', methods=['POST'])
    @login_required
    def api_wallet_import():
        """Import wallet from uploaded files with automatic rollback on failure.

        Security: Requires vault to be unlocked (login_required decorator)
        Process:
        1. Validate uploaded files
        2. Create timestamped backup of existing wallet directory
        3. Save uploaded files to temporary directory
        4. Validate YAML structure
        5. Move files to wallet directory
        6. Trigger container restart (critical for safety)

        Rollback: If ANY step fails, automatically restores from backup

        Expected files in upload:
        - wallet.yaml (required)
        - vault.yaml (required)
        - wallet.yaml.bak (optional)
        - vault.yaml.bak (optional)
        """
        import tempfile
        import shutil
        import datetime
        from pathlib import Path
        import yaml

        try:
            # 1. Validate request has files
            if 'files' not in request.files:
                return jsonify({'error': 'No files uploaded'}), 400

            files = request.files.getlist('files')
            if not files:
                return jsonify({'error': 'No files selected'}), 400

            # 2. Validate required files are present
            file_names = [f.filename for f in files if f.filename]
            required_files = {'wallet.yaml', 'vault.yaml'}
            uploaded_files = set(file_names)

            if not required_files.issubset(uploaded_files):
                missing = required_files - uploaded_files
                return jsonify({
                    'error': f'Missing required files: {", ".join(missing)}'
                }), 400

            # 3. Validate file names (security: prevent directory traversal)
            allowed_files = {'wallet.yaml', 'vault.yaml', 'wallet.yaml.bak', 'vault.yaml.bak'}
            for name in file_names:
                if name not in allowed_files:
                    return jsonify({
                        'error': f'Invalid file: {name}. Only wallet files allowed.'
                    }), 400

            # 4. Get wallet directory path
            from satorineuron import config
            wallet_dir = Path(config.walletPath())

            # 5. Create timestamped backup of wallet files inside wallet directory
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = wallet_dir / f'backup_{timestamp}'

            try:
                if wallet_dir.exists():
                    # Create backup directory inside wallet folder
                    backup_dir.mkdir(parents=True, exist_ok=True)

                    # Copy all wallet files to backup (excluding other backup folders)
                    for item in wallet_dir.iterdir():
                        if item.is_file():
                            shutil.copy2(item, backup_dir / item.name)

                    logger.info(f"Created backup at {backup_dir}")
                else:
                    logger.warning(f"Wallet directory doesn't exist yet: {wallet_dir}")
                    wallet_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Backup failed: {e}")
                return jsonify({
                    'error': f'Failed to create backup: {str(e)}'
                }), 500

            # 6. Save uploaded files to temporary location first
            temp_dir = Path(tempfile.mkdtemp())
            try:
                # Save files to temp
                for file in files:
                    if file.filename:
                        file_path = temp_dir / file.filename
                        file.save(str(file_path))
                        logger.info(f"Saved {file.filename} to temp: {file_path}")

                # 7. Validate file contents (basic YAML check)
                for file_name in ['wallet.yaml', 'vault.yaml']:
                    file_path = temp_dir / file_name
                    try:
                        with open(file_path, 'r') as f:
                            yaml.safe_load(f)
                        logger.info(f"Validated {file_name}")
                    except Exception as e:
                        # Validation failed - cleanup and rollback
                        shutil.rmtree(temp_dir)
                        logger.error(f"Invalid {file_name}: {e}")

                        # Rollback: This shouldn't be needed as we haven't modified wallet_dir yet
                        # but include for safety
                        if backup_dir.exists():
                            logger.info("Validation failed before any changes - no rollback needed")

                        return jsonify({
                            'error': f'Invalid {file_name}: {str(e)}',
                            'rolled_back': False
                        }), 400

                # 8. Files validated - move to wallet directory
                for file_name in file_names:
                    src = temp_dir / file_name
                    dst = wallet_dir / file_name
                    shutil.move(str(src), str(dst))
                    logger.info(f"Moved {file_name} to wallet directory")

                logger.info(f"Wallet import: saved {len(file_names)} files")

                # 8.5. Clear reward address from config since wallet changed
                try:
                    # Read config, remove reward address, and write back
                    config_data = config.get()
                    logger.info(f"Current config before clearing reward address: {list(config_data.keys())}")

                    if 'reward address' in config_data:
                        old_address = config_data['reward address']
                        del config_data['reward address']
                        # Use put to write the entire config back (this will remove the key)
                        config.put(data=config_data)
                        logger.info(f"Cleared old reward address '{old_address}' from config")

                        # Verify it was removed
                        verify_config = config.get()
                        if 'reward address' in verify_config:
                            logger.error("Failed to remove reward address from config!")
                        else:
                            logger.info("Verified: reward address successfully removed from config")
                    else:
                        logger.info("No reward address found in config to clear")
                except Exception as config_error:
                    logger.error(f"Error clearing reward address from config: {config_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Don't fail the import for this - it's not critical

            except Exception as e:
                # Error during file operations - ROLLBACK
                logger.error(f"Import failed during file operations: {e}. Rolling back...")

                # Delete any partial changes in wallet_dir
                for file_name in file_names:
                    try:
                        (wallet_dir / file_name).unlink(missing_ok=True)
                    except Exception as cleanup_error:
                        logger.warning(f"Cleanup error for {file_name}: {cleanup_error}")

                # Restore from backup
                if backup_dir.exists():
                    try:
                        for item in backup_dir.iterdir():
                            if item.is_file():
                                shutil.copy2(item, wallet_dir / item.name)
                        logger.info("Rollback completed. Old wallet restored.")
                    except Exception as rollback_error:
                        logger.error(f"Rollback failed: {rollback_error}")
                        return jsonify({
                            'error': f'Import failed AND rollback failed: {str(e)}. Backup at: {backup_dir}',
                            'rolled_back': False
                        }), 500

                return jsonify({
                    'error': str(e),
                    'rolled_back': True
                }), 500

            finally:
                # Cleanup temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

            # 9. Schedule container restart (happens after response sent)
            def delayed_restart():
                import time
                time.sleep(2)  # Give response time to be sent
                startup = get_startup()
                if startup:
                    logger.info("Triggering restart after wallet import")
                    startup.triggerRestart()
                else:
                    logger.error("Cannot restart - startup instance not available")

            import threading
            threading.Thread(target=delayed_restart, daemon=True).start()

            return jsonify({
                'success': True,
                'message': 'Wallet imported successfully. Container restarting...',
                'backup_location': str(backup_dir)
            })

        except Exception as e:
            logger.error(f"Wallet import error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
