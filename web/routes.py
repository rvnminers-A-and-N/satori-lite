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

# Session-specific vaults - each browser session gets its own WalletManager
_session_vaults = {}
_vault_lock = Lock()

# JWT authentication lock to prevent concurrent login attempts
_auth_lock = Lock()


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
        session.permanent = False  # Don't persist sessions indefinitely
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
        # Ensure session ID exists for tracking
        if not session.get('session_id'):
            session['session_id'] = str(uuid.uuid4())
            session.permanent = False  # Don't persist sessions indefinitely

        # Check if user is logged in via session flag
        if not session.get('vault_open'):
            # Not logged in - check if vault file exists
            if not check_vault_file_exists():
                # No vault file - redirect to create password
                return redirect(url_for('vault_setup'))
            # Vault exists but not logged in - redirect to login
            return redirect(url_for('login'))

        # Validate that vault actually exists and is open (handles container restart)
        # Check _session_vaults directly - don't call get_or_create (which creates new vault)
        session_id = session.get('session_id')
        if session_id and session_id not in _session_vaults:
            # Vault doesn't exist (container restarted) - require re-login
            logger.info(f"Session vault missing for {session_id} - forcing re-login")
            session.pop('vault_open', None)
            session['logged_out'] = True  # Prevent auto-login
            return redirect(url_for('login'))

        # Also validate vault is actually decrypted
        if session_id and session_id in _session_vaults:
            wallet_manager = _session_vaults[session_id]
            if not wallet_manager or not wallet_manager.vault or not wallet_manager.vault.isDecrypted:
                logger.info(f"Session vault not decrypted for {session_id} - forcing re-login")
                session.pop('vault_open', None)
                session['logged_out'] = True  # Prevent auto-login
                return redirect(url_for('login'))

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
        return render_template('dashboard.html', version=VERSION)

    @app.route('/stake')
    @login_required
    def stake_management():
        """Stake management page for pool staking and pool management."""
        from satorineuron import VERSION
        return render_template('stake.html', version=VERSION)

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
                resp = requests.get(url, headers=headers, timeout=10)
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
        """Proxy reward address request."""
        if request.method == 'POST':
            return proxy_api('/peer/reward-address', 'POST', request.json)
        return proxy_api('/peer/reward-address')

    @app.route('/api/lender/status')
    @login_required
    def api_lender_status():
        """Proxy lender status request."""
        return proxy_api('/lender/status')

    @app.route('/api/lender/lend', methods=['POST', 'DELETE'])
    @login_required
    def api_lender_lend():
        """Proxy lend request."""
        data = request.get_json(silent=True) if request.method == 'POST' else None
        return proxy_api('/lender/lend', request.method, data)

    @app.route('/api/pool/worker', methods=['POST'])
    @login_required
    def api_pool_worker_add():
        """Proxy pool worker add request."""
        return proxy_api('/pool/worker', 'POST', request.json)

    @app.route('/api/pool/worker/<worker_address>', methods=['DELETE'])
    @login_required
    def api_pool_worker_delete(worker_address):
        """Proxy pool worker delete request."""
        return proxy_api(f'/pool/worker/{worker_address}', 'DELETE')

    @app.route('/api/pool/toggle-open', methods=['POST'])
    @login_required
    def api_pool_toggle():
        """Proxy pool toggle request."""
        return proxy_api('/pool/toggle-open', 'POST', request.json)

    @app.route('/api/pool/open', methods=['GET'])
    @login_required
    def api_pool_open():
        """Get list of open pools."""
        return proxy_api('/pool/open', 'GET')

    @app.route('/api/pool/commission', methods=['GET'])
    @login_required
    def api_pool_commission():
        """Get pool commission status."""
        return proxy_api('/pool/commission', 'GET')

    @app.route('/api/pool/workers', methods=['GET'])
    @login_required
    def api_pool_workers():
        """Get list of workers for authenticated user's pool."""
        return proxy_api('/pool/workers', 'GET')

    @app.route('/api/pool/lenders', methods=['GET'])
    @login_required
    def api_pool_lenders():
        """Get list of lenders for authenticated user's pool."""
        return proxy_api('/pool/lenders', 'GET')

    @app.route('/api/wallet/address')
    @login_required
    def api_wallet_address():
        """Get wallet and vault addresses."""
        wallet_manager = get_or_create_session_vault()
        if wallet_manager:
            result = {}
            # Get wallet address
            if wallet_manager.wallet and hasattr(wallet_manager.wallet, 'address'):
                result['wallet_address'] = wallet_manager.wallet.address
            # Get vault address
            if wallet_manager.vault and hasattr(wallet_manager.vault, 'address'):
                result['vault_address'] = wallet_manager.vault.address
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

            # Get P2P peers info
            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_p2p_peers') and startup._p2p_peers:
                result['connected'] = True
                peers = startup._p2p_peers
                if hasattr(peers, 'get_peer_count'):
                    result['peer_count'] = peers.get_peer_count()

            # Get uptime tracker info
            if startup and hasattr(startup, '_uptime_tracker') and startup._uptime_tracker:
                tracker = startup._uptime_tracker
                if hasattr(tracker, 'get_uptime_percentage'):
                    result['uptime_pct'] = tracker.get_uptime_percentage()
                if hasattr(tracker, '_last_heartbeat') and tracker._last_heartbeat:
                    result['last_heartbeat_ago'] = int(time.time() - tracker._last_heartbeat)
                result['relay_eligible'] = result['uptime_pct'] >= 95.0

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
            from satorineuron.init import start

            heartbeats = []
            limit = request.args.get('limit', 20, type=int)

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_uptime_tracker') and startup._uptime_tracker:
                tracker = startup._uptime_tracker
                if hasattr(tracker, 'get_recent_heartbeats'):
                    raw = tracker.get_recent_heartbeats(limit=limit)
                    for hb in raw:
                        heartbeats.append({
                            'node_id': hb.node_id[:12] + '...' if len(hb.node_id) > 12 else hb.node_id,
                            'address': getattr(hb, 'evrmore_address', '')[:12] + '...',
                            'timestamp': hb.timestamp,
                            'roles': getattr(hb, 'roles', []),
                        })

            return jsonify({'heartbeats': heartbeats})
        except Exception as e:
            logger.warning(f"Failed to get heartbeats: {e}")
            return jsonify({'heartbeats': [], 'error': str(e)})

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
        """Get comprehensive P2P status for network dashboard."""
        try:
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

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup:
                # Get P2P peers info
                if hasattr(startup, '_p2p_peers') and startup._p2p_peers:
                    peers = startup._p2p_peers
                    if hasattr(peers, 'get_peer_count'):
                        result['peer_count'] = peers.get_peer_count()
                    if hasattr(peers, 'node_id'):
                        result['peer_id'] = peers.node_id
                    if hasattr(peers, 'get_connected_peers'):
                        connected = peers.get_connected_peers()
                        for p in connected[:20]:  # Limit to 20
                            result['peers'].append({
                                'id': getattr(p, 'peer_id', str(p))[:16] + '...',
                                'latency': getattr(p, 'latency', None),
                                'location': getattr(p, 'location', None),
                                'streams': getattr(p, 'stream_count', 0),
                                'role': getattr(p, 'role', 'Predictor'),
                            })

                # Get uptime tracker info
                if hasattr(startup, '_uptime_tracker') and startup._uptime_tracker:
                    tracker = startup._uptime_tracker
                    if hasattr(tracker, 'get_uptime_percentage'):
                        result['uptime_pct'] = tracker.get_uptime_percentage()

                # Get NAT info
                if hasattr(startup, '_nat_type'):
                    result['nat_type'] = startup._nat_type or 'Unknown'

            return jsonify(result)
        except Exception as e:
            logger.warning(f"P2P status failed: {e}")
            return jsonify({
                'peer_count': 0,
                'networking_mode': 'central',
                'error': str(e)
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

    @app.route('/api/peers/list')
    @login_required
    def api_peers_list():
        """Get list of connected peers."""
        try:
            from satorineuron.init import start

            peers = []
            startup = start.getStart() if hasattr(start, 'getStart') else None

            if startup and hasattr(startup, '_p2p_peers') and startup._p2p_peers:
                p2p = startup._p2p_peers
                if hasattr(p2p, 'get_connected_peers'):
                    connected = p2p.get_connected_peers()
                    for p in connected:
                        peers.append({
                            'id': getattr(p, 'peer_id', str(p)),
                            'latency': getattr(p, 'latency', None),
                            'location': getattr(p, 'location', None),
                            'streams': getattr(p, 'stream_count', 0),
                            'role': getattr(p, 'role', 'Predictor'),
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

            # Save to config
            if hasattr(start, '_set_networking_mode'):
                start._set_networking_mode(new_mode)
                return jsonify({'success': True, 'mode': new_mode})
            else:
                # Fallback: try to save to config file
                import os
                import json
                config_path = os.path.expanduser('~/.satori/config.json')
                config = {}
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                config['networking_mode'] = new_mode
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                return jsonify({'success': True, 'mode': new_mode})

        except Exception as e:
            logger.error(f"Networking mode failed: {e}")
            return jsonify({'success': False, 'error': str(e)})

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
            from satorineuron.init import start

            result = {
                'current_version': '1.0.0',
                'min_supported': '1.0.0',
                'peer_stats': {
                    'total_peers': 0,
                    'compatible_peers': 0,
                    'version_distribution': {},
                },
                'features': [],
                'upgrade_progress': 0.0,
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup:
                # Get protocol version
                if hasattr(startup, '_protocol_version'):
                    result['current_version'] = startup._protocol_version

                # Get version tracker stats
                if hasattr(startup, '_version_tracker') and startup._version_tracker:
                    tracker = startup._version_tracker
                    if hasattr(tracker, 'get_network_stats'):
                        stats = tracker.get_network_stats()
                        result['peer_stats'] = {
                            'total_peers': stats.get('total_peers', 0),
                            'compatible_peers': stats.get('compatible_peers', 0),
                            'version_distribution': stats.get('version_distribution', {}),
                        }
                    if hasattr(tracker, 'get_upgrade_progress'):
                        result['upgrade_progress'] = tracker.get_upgrade_progress()

                # Get current features
                try:
                    from satorip2p.protocol.versioning import get_current_features
                    result['features'] = get_current_features()
                except ImportError:
                    pass

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Version info failed: {e}")
            return jsonify({'error': str(e), 'current_version': '1.0.0'})

    # =========================================================================
    # STORAGE REDUNDANCY ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/storage')
    @login_required
    def api_p2p_storage():
        """Get storage redundancy status and disk usage."""
        try:
            from satorineuron.init import start
            import os

            result = {
                'status': 'unavailable',
                'disk_usage': {
                    'used_bytes': 0,
                    'used_mb': 0.0,
                    'storage_dir': '~/.satori/storage',
                },
                'backends': {
                    'memory': {'enabled': False, 'items': 0},
                    'file': {'enabled': False, 'items': 0},
                    'dht': {'enabled': False, 'items': 0},
                },
                'deferred_rewards': {
                    'stored_count': 0,
                    'pending_sync': 0,
                },
                'alerts': {
                    'stored_count': 0,
                    'pending_sync': 0,
                },
            }

            # Calculate actual disk usage
            storage_dir = os.path.expanduser('~/.satori/storage')
            if os.path.exists(storage_dir):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(storage_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                result['disk_usage']['used_bytes'] = total_size
                result['disk_usage']['used_mb'] = round(total_size / (1024 * 1024), 2)
                result['disk_usage']['storage_dir'] = storage_dir

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup:
                # Get storage manager status
                if hasattr(startup, '_storage_manager') and startup._storage_manager:
                    manager = startup._storage_manager
                    result['status'] = 'active'
                    if hasattr(manager, 'get_status'):
                        status = manager.get_status()
                        result['backends'] = status.get('backends', result['backends'])

                # Get deferred rewards storage status
                if hasattr(startup, '_deferred_rewards_storage') and startup._deferred_rewards_storage:
                    storage = startup._deferred_rewards_storage
                    result['backends']['file']['enabled'] = True
                    if hasattr(storage, 'count'):
                        result['deferred_rewards']['stored_count'] = storage.count()
                    if hasattr(storage, 'pending_sync_count'):
                        result['deferred_rewards']['pending_sync'] = storage.pending_sync_count()

                # Get alert storage status
                if hasattr(startup, '_alert_storage') and startup._alert_storage:
                    storage = startup._alert_storage
                    if hasattr(storage, 'count'):
                        result['alerts']['stored_count'] = storage.count()
                    if hasattr(storage, 'pending_sync_count'):
                        result['alerts']['pending_sync'] = storage.pending_sync_count()

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Storage status failed: {e}")
            return jsonify({'status': 'error', 'error': str(e)})

    # =========================================================================
    # BANDWIDTH & QOS ENDPOINTS
    # =========================================================================

    @app.route('/api/p2p/bandwidth')
    @login_required
    def api_p2p_bandwidth():
        """Get bandwidth usage statistics and QoS status."""
        try:
            from satorineuron.init import start

            result = {
                'status': 'unavailable',
                'global': {
                    'bytes_sent': 0,
                    'bytes_received': 0,
                    'messages_sent': 0,
                    'messages_received': 0,
                    'bytes_per_second': 0.0,
                    'messages_per_second': 0.0,
                },
                'topics': {},
                'qos': {
                    'enabled': False,
                    'drops_low_priority': 0,
                    'drops_rate_limited': 0,
                    'policy': 'none',
                },
                'peers': {},
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup:
                # Get bandwidth tracker stats
                if hasattr(startup, '_bandwidth_tracker') and startup._bandwidth_tracker:
                    tracker = startup._bandwidth_tracker
                    result['status'] = 'active'

                    # Global metrics
                    if hasattr(tracker, 'get_global_metrics'):
                        global_metrics = tracker.get_global_metrics()
                        if hasattr(global_metrics, 'to_dict'):
                            result['global'] = global_metrics.to_dict()
                        elif isinstance(global_metrics, dict):
                            result['global'] = global_metrics

                    # Per-topic metrics
                    if hasattr(tracker, 'get_topic_metrics'):
                        topic_metrics = tracker.get_topic_metrics()
                        for topic, metrics in topic_metrics.items():
                            if hasattr(metrics, 'to_dict'):
                                result['topics'][topic] = metrics.to_dict()
                            elif isinstance(metrics, dict):
                                result['topics'][topic] = metrics

                    # Per-peer metrics (summarized)
                    if hasattr(tracker, 'get_peer_metrics'):
                        peer_metrics = tracker.get_peer_metrics()
                        result['peers'] = {
                            'count': len(peer_metrics),
                            'top_senders': [],
                            'top_receivers': [],
                        }

                # Get QoS manager stats
                if hasattr(startup, '_qos_manager') and startup._qos_manager:
                    qos = startup._qos_manager
                    result['qos']['enabled'] = True
                    if hasattr(qos, 'get_stats'):
                        qos_stats = qos.get_stats()
                        result['qos']['drops_low_priority'] = qos_stats.get('drops_low_priority', 0)
                        result['qos']['drops_rate_limited'] = qos_stats.get('drops_rate_limited', 0)
                        result['qos']['policy'] = qos_stats.get('policy', 'default')

            return jsonify(result)
        except Exception as e:
            logger.warning(f"Bandwidth stats failed: {e}")
            return jsonify({'status': 'error', 'error': str(e)})

    @app.route('/api/p2p/bandwidth/history')
    @login_required
    def api_p2p_bandwidth_history():
        """Get bandwidth usage history for charting."""
        try:
            from satorineuron.init import start

            result = {
                'history': [],
                'interval_seconds': 60,
                'points': 60,  # Last 60 minutes
            }

            startup = start.getStart() if hasattr(start, 'getStart') else None
            if startup and hasattr(startup, '_bandwidth_tracker') and startup._bandwidth_tracker:
                tracker = startup._bandwidth_tracker
                if hasattr(tracker, 'get_history'):
                    result['history'] = tracker.get_history(points=60)

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
        return render_template('donate.html', **get_base_context())

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
