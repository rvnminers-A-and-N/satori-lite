#!/usr/bin/env python3
"""
Satori Neuron CLI - Interactive command-line interface.
Starts the neuron in the background and provides commands to interact with it.
"""

import sys
import os
import time
import threading
import io
import logging as stdlib_logging
import tty
import termios

# Save original file descriptors BEFORE any redirection
_original_stdout_fd = os.dup(1)
_original_stderr_fd = os.dup(2)
_original_stdin_fd = os.dup(0)

# Create file objects from the duplicated fds
_console_out = os.fdopen(os.dup(_original_stdout_fd), 'w', buffering=1)
_console_in = os.fdopen(os.dup(_original_stdin_fd), 'r', buffering=1)

# Global log buffer
_log_buffer: list[str] = []
_max_logs = 1000

# Command history
_command_history: list[str] = []
_history_index: int = 0


class LogInterceptHandler(stdlib_logging.Handler):
    """Handler that captures all logs to buffer."""
    def emit(self, record):
        msg = self.format(record)
        timestamp = time.strftime("%H:%M:%S")
        _log_buffer.append(f"[{timestamp}] {msg}")
        if len(_log_buffer) > _max_logs:
            _log_buffer.pop(0)


class OutputCapture:
    """Captures stdout/stderr to buffer."""
    def __init__(self, original):
        self.original = original
        # Need fileno for subprocess compatibility
        self._devnull = open(os.devnull, 'w')

    def write(self, msg):
        msg = msg.strip()
        if msg:
            timestamp = time.strftime("%H:%M:%S")
            _log_buffer.append(f"[{timestamp}] {msg}")
            if len(_log_buffer) > _max_logs:
                _log_buffer.pop(0)

    def flush(self):
        pass

    def fileno(self):
        # Return devnull's fileno so subprocesses write there
        return self._devnull.fileno()


def setup_log_capture():
    """Set up log capture before importing neuron modules."""
    # Install our handler on root logger
    root = stdlib_logging.getLogger()
    root.handlers.clear()
    handler = LogInterceptHandler()
    handler.setFormatter(stdlib_logging.Formatter('%(levelname)s - %(message)s'))
    root.addHandler(handler)
    root.setLevel(stdlib_logging.DEBUG)

    # Capture stdout/stderr
    sys.stdout = OutputCapture(sys.__stdout__)
    sys.stderr = OutputCapture(sys.__stderr__)

    # Also redirect the actual file descriptors for subprocess output
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)  # stdout
    os.dup2(devnull_fd, 2)  # stderr
    os.close(devnull_fd)


def console_print(msg: str = ""):
    """Print to actual console."""
    _console_out.write(msg + "\n")
    _console_out.flush()


def console_write(msg: str):
    """Write to actual console without newline."""
    _console_out.write(msg)
    _console_out.flush()


def console_readline() -> str:
    """Read a line from console."""
    return _console_in.readline()


def console_input(prompt: str = "") -> str:
    """Read input with arrow key support for command history."""
    global _history_index

    console_write(prompt)

    # Use the original stdin fd for reading
    fd = _original_stdin_fd
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        line = ""
        cursor_pos = 0
        _history_index = len(_command_history)
        saved_line = ""  # Save current line when browsing history

        while True:
            # Read one character
            ch = os.read(fd, 1).decode('utf-8', errors='ignore')

            if ch == '\r' or ch == '\n':  # Enter
                console_write('\r\n')
                break
            elif ch == '\x03':  # Ctrl+C
                console_write('\r\n')
                raise KeyboardInterrupt
            elif ch == '\x04':  # Ctrl+D
                console_write('\r\n')
                raise EOFError
            elif ch == '\x7f' or ch == '\x08':  # Backspace
                if cursor_pos > 0:
                    line = line[:cursor_pos-1] + line[cursor_pos:]
                    cursor_pos -= 1
                    # Redraw line
                    console_write(f'\r{prompt}{line} \r{prompt}{line[:cursor_pos]}')
            elif ch == '\x1b':  # Escape sequence (arrow keys)
                seq1 = os.read(fd, 1).decode('utf-8', errors='ignore')
                if seq1 == '[':
                    seq2 = os.read(fd, 1).decode('utf-8', errors='ignore')
                    if seq2 == 'A':  # Up arrow - previous command
                        if _command_history and _history_index > 0:
                            if _history_index == len(_command_history):
                                saved_line = line
                            _history_index -= 1
                            line = _command_history[_history_index]
                            cursor_pos = len(line)
                            # Clear and redraw
                            console_write(f'\r{prompt}{" " * 50}\r{prompt}{line}')
                    elif seq2 == 'B':  # Down arrow - next command
                        if _history_index < len(_command_history):
                            _history_index += 1
                            if _history_index == len(_command_history):
                                line = saved_line
                            else:
                                line = _command_history[_history_index]
                            cursor_pos = len(line)
                            # Clear and redraw
                            console_write(f'\r{prompt}{" " * 50}\r{prompt}{line}')
                    elif seq2 == 'C':  # Right arrow
                        if cursor_pos < len(line):
                            cursor_pos += 1
                            console_write('\x1b[C')
                    elif seq2 == 'D':  # Left arrow
                        if cursor_pos > 0:
                            cursor_pos -= 1
                            console_write('\x1b[D')
            elif ch >= ' ':  # Printable character
                line = line[:cursor_pos] + ch + line[cursor_pos:]
                cursor_pos += 1
                # Redraw from cursor
                console_write(f'{line[cursor_pos-1:]}\r{prompt}{line[:cursor_pos]}')

        # Add to history if non-empty
        if line.strip() and (not _command_history or _command_history[-1] != line):
            _command_history.append(line)

        return line

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class NeuronCLI:
    """Interactive CLI for Satori Neuron."""

    def __init__(self, env: str = 'prod', runMode: str = 'worker'):
        self.env = env
        self.runMode = runMode
        self.startup = None
        self.neuron_started = False
        self._vault_password = None  # Store password in memory for session

    def add_log(self, message: str):
        """Add a log message to buffer."""
        timestamp = time.strftime("%H:%M:%S")
        _log_buffer.append(f"[{timestamp}] {message}")

    def check_vault_file_exists(self) -> bool:
        """Check if vault.yaml file exists (like web UI does)."""
        try:
            from satorineuron import config
            vault_path = config.walletPath('vault.yaml')
            return os.path.exists(vault_path)
        except Exception:
            return False

    def check_vault_password_exists(self) -> bool:
        """Check if vault password exists in config."""
        from satorineuron import config
        vault_password = config.get().get('vault password')
        return vault_password is not None and len(str(vault_password)) > 0

    def prompt_mandatory_vault_password(self) -> bool:
        """Prompt user to create a vault password. Returns True if successful."""
        # Check if vault password already exists (could be set via web UI or another CLI instance)
        if self.check_vault_password_exists():
            console_print()
            console_print("=" * 60)
            console_print("  VAULT PASSWORD ALREADY EXISTS")
            console_print("=" * 60)
            console_print()
            console_print("A vault password has already been created.")
            console_print("You can use it to unlock your vault.")
            console_print()
            console_print("If you set it in another interface (web UI), use that password.")
            console_print("=" * 60)
            console_print()
            return True

        console_print()
        console_print("=" * 60)
        console_print("  VAULT PASSWORD SETUP REQUIRED")
        console_print("=" * 60)
        console_print()
        console_print("A vault password is required to secure your wallet.")
        console_print("This password encrypts your private keys and funds.")
        console_print()
        console_print("IMPORTANT: Save this password in a secure location!")
        console_print("If you lose this password, you will lose access to your vault.")
        console_print("There is no way to recover a lost vault password.")
        console_print()
        console_print("=" * 60)
        console_print()

        while True:
            console_write("Enter new vault password (min 4 characters): ")
            password1 = console_readline().strip()

            if len(password1) < 4:
                console_print("Password must be at least 4 characters. Please try again.")
                console_print()
                continue

            console_write("Confirm password: ")
            password2 = console_readline().strip()

            if password1 != password2:
                console_print("Passwords do not match. Please try again.")
                console_print()
                continue

            # Double-check if password was created elsewhere (race condition protection)
            if self.check_vault_password_exists():
                console_print()
                console_print("Vault password was already created by another process.")
                console_print("Please use the existing password to unlock your vault.")
                console_print()
                return True

            # Save password to config
            from satorineuron import config
            config.add(data={'vault password': password1})
            self._vault_password_set = True

            console_print()
            console_print("Vault password saved successfully!")
            console_print()
            console_print("REMINDER: Please save your password securely!")
            console_print("         You will need it to access your vault.")
            console_print()
            return True

    def prompt_vault_setup_or_unlock(self) -> bool:
        """
        Prompt user to either create new vault or unlock existing vault.
        Returns True if successful, False otherwise.
        This matches the web UI behavior - passwords are NOT saved to config.
        """
        from satorineuron import config

        # Check if config password exists (backward compatibility)
        config_password = config.get().get('vault password')

        if not self.check_vault_file_exists():
            # No vault file - need to create new vault
            console_print()
            console_print("=" * 60)
            console_print("  VAULT SETUP")
            console_print("=" * 60)
            console_print()
            console_print("No vault found. Let's create a new one.")
            console_print()
            console_print("Create a secure password to protect your vault.")
            console_print("This password encrypts your private keys and funds.")
            console_print()
            console_print("IMPORTANT: Save this password in a secure location!")
            console_print("If you lose this password, you will lose access to your vault.")
            console_print("There is no way to recover a lost vault password.")
            console_print()
            console_print("=" * 60)
            console_print()

            return self.setup_new_vault()
        else:
            # Vault file exists - need to unlock it
            console_print()
            console_print("=" * 60)
            console_print("  VAULT UNLOCK")
            console_print("=" * 60)
            console_print()

            # Try config password first (backward compatibility, like web UI)
            if config_password:
                console_print("Attempting to unlock vault with config password...")
                if self.unlock_existing_vault(config_password):
                    console_print("Vault unlocked successfully!")
                    console_print()
                    return True
                else:
                    console_print("Config password failed. Please enter your password manually.")
                    console_print()

            console_print("Please enter your vault password to unlock.")
            console_print("(Press Ctrl+C to exit)")
            console_print()
            console_print("=" * 60)
            console_print()

            # Prompt for password (unlimited attempts, like web UI)
            while True:
                console_write("Enter vault password: ")
                password = console_readline().strip()

                if self.unlock_existing_vault(password):
                    console_print()
                    console_print("Vault unlocked successfully!")
                    console_print()
                    return True
                else:
                    console_print("Invalid password. Please try again.")
                    console_print()

    def setup_new_vault(self) -> bool:
        """
        Create a new vault with password (DO NOT save password to config).
        Matches web UI behavior.
        """
        while True:
            console_write("Enter new vault password (min 4 characters): ")
            password1 = console_readline().strip()

            if len(password1) < 4:
                console_print("Password must be at least 4 characters. Please try again.")
                console_print()
                continue

            console_write("Confirm password: ")
            password2 = console_readline().strip()

            if password1 != password2:
                console_print("Passwords do not match. Please try again.")
                console_print()
                continue

            # Try to create the vault using WalletManager
            try:
                from satorineuron.init.wallet import WalletManager

                # Create wallet manager without using config password
                wallet_manager = WalletManager.create(useConfigPassword=False)
                vault = wallet_manager.openVault(password=password1, create=True)

                if vault and vault.isDecrypted:
                    self._vault_password = password1  # Store in memory for session
                    console_print()
                    console_print("Vault created successfully!")
                    console_print()
                    console_print("REMINDER: Please save your password securely!")
                    console_print("         You will need it to access your vault.")
                    console_print()
                    return True
                else:
                    console_print("Error creating vault. Please try again.")
                    console_print()
                    return False
            except Exception as e:
                console_print(f"Error creating vault: {e}")
                console_print()
                return False

    def unlock_existing_vault(self, password: str) -> bool:
        """
        Unlock existing vault with password.
        Returns True if successful, False otherwise.
        """
        try:
            from satorineuron.init.wallet import WalletManager

            # Create wallet manager without using config password
            wallet_manager = WalletManager.create(useConfigPassword=False)
            vault = wallet_manager.openVault(password=password, create=False)

            if vault and vault.isDecrypted:
                self._vault_password = password  # Store in memory for session
                return True
            else:
                return False
        except Exception:
            return False

    def start_neuron_background(self):
        """Start the neuron in a background thread."""
        def run():
            try:
                # Import here after log capture is set up
                from start import StartupDag
                self.add_log("Creating StartupDag...")
                # Create instance without calling startFunction yet
                startupDag = StartupDag(
                    env=self.env,
                    runMode=self.runMode)
                # Assign to self.startup BEFORE the blocking call
                self.startup = startupDag

                # Unlock the vault with the password we got during CLI startup
                if self._vault_password:
                    if startupDag.walletManager:
                        try:
                            vault = startupDag.walletManager.openVault(password=self._vault_password)
                            if vault and hasattr(vault, 'isDecrypted') and vault.isDecrypted:
                                self.add_log("Vault unlocked successfully in StartupDag")
                            else:
                                self.add_log("Warning: Vault unlock failed - vault not decrypted")
                        except Exception as e:
                            self.add_log(f"Warning: Could not unlock vault in StartupDag: {e}")
                    else:
                        self.add_log("Warning: walletManager not available in StartupDag")
                else:
                    self.add_log("Warning: No vault password stored from CLI startup")

                self.add_log("Neuron started successfully")
                # Now call the blocking startFunction
                startupDag.startFunction()
            except Exception as e:
                self.add_log(f"Neuron startup error: {e}")
                import traceback
                self.add_log(traceback.format_exc())

        self.add_log("Starting neuron in background...")
        threading.Thread(target=run, daemon=True).start()

    def interactive_menu(self, title: str, options: list[dict]) -> int | None:
        """
        Display an interactive menu with arrow key navigation.

        Args:
            title: Menu title to display
            options: List of dicts with 'label' and 'action' keys
                     e.g., [{'label': 'Option 1', 'action': callable}, ...]

        Returns:
            Index of selected option, or None if cancelled
        """
        selected = 0

        # Use the original stdin fd for reading
        fd = _original_stdin_fd
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)

            # Initial render
            console_write('\r\n')
            console_write(title + '\r\n')
            console_write("-" * len(title) + '\r\n')
            for i, option in enumerate(options):
                if i == selected:
                    console_write(f"  > \x1b[7m{option['label']}\x1b[0m\r\n")  # Highlighted
                else:
                    console_write(f"    {option['label']}\r\n")
            console_write('\r\n')
            console_write("(Use ↑/↓ arrows to navigate, Enter to select, Esc to cancel)\r\n")

            # Number of lines printed (title + separator + options + blank + help text)
            lines_printed = 2 + len(options) + 1 + 1

            while True:
                # Read one character
                ch = os.read(fd, 1).decode('utf-8', errors='ignore')

                if ch == '\r' or ch == '\n':  # Enter - select
                    # Clear menu
                    console_write(f'\x1b[{lines_printed}A')  # Move cursor up
                    for _ in range(lines_printed):
                        console_write('\x1b[2K\n')  # Clear line and move down
                    console_write(f'\x1b[{lines_printed}A')  # Move back up
                    return selected

                elif ch == '\x03':  # Ctrl+C - cancel
                    # Clear menu
                    console_write(f'\x1b[{lines_printed}A')
                    for _ in range(lines_printed):
                        console_write('\x1b[2K\n')
                    console_write(f'\x1b[{lines_printed}A')
                    raise KeyboardInterrupt

                elif ch == '\x1b':  # Escape sequence
                    seq1 = os.read(fd, 1).decode('utf-8', errors='ignore')
                    if seq1 == '[':
                        seq2 = os.read(fd, 1).decode('utf-8', errors='ignore')

                        old_selected = selected

                        if seq2 == 'A':  # Up arrow
                            selected = max(0, selected - 1)
                        elif seq2 == 'B':  # Down arrow
                            selected = min(len(options) - 1, selected + 1)

                        # If selection changed, redraw
                        if old_selected != selected:
                            # Move cursor back to start of menu
                            console_write(f'\x1b[{lines_printed}A')

                            # Redraw menu
                            console_write('\x1b[2K')  # Clear title line
                            console_write(title + '\r\n')
                            console_write('\x1b[2K')
                            console_write("-" * len(title) + '\r\n')
                            for i, option in enumerate(options):
                                console_write('\x1b[2K')  # Clear line
                                if i == selected:
                                    console_write(f"  > \x1b[7m{option['label']}\x1b[0m\r\n")
                                else:
                                    console_write(f"    {option['label']}\r\n")
                            console_write('\x1b[2K\r\n')  # Clear blank line
                            console_write('\x1b[2K')
                            console_write("(Use ↑/↓ arrows to navigate, Enter to select, Esc to cancel)\r\n")

                    elif seq1 == '\x1b':  # Double escape (Esc key) - cancel
                        # Clear menu
                        console_write(f'\x1b[{lines_printed}A')
                        for _ in range(lines_printed):
                            console_write('\x1b[2K\n')
                        console_write(f'\x1b[{lines_printed}A')
                        return None

        except KeyboardInterrupt:
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_wallet_balance_electrumx(self) -> str:
        """Fetch wallet balance from ElectrumX servers."""
        if not self.startup or not hasattr(self.startup, 'walletManager'):
            return "Neuron is starting... Use /logs neuron to see progress."

        console_print("Fetching wallet balance...")
        console_print()

        try:
            # Get wallet manager
            wallet_manager = self.startup.walletManager
            if not wallet_manager:
                return "Wallet manager not initialized."

            # Ensure ElectrumX connection
            if hasattr(wallet_manager, 'connect'):
                wallet_manager.connect()

            # Get wallet
            wallet = wallet_manager.wallet
            if not wallet:
                return "Wallet not available."

            # Fetch balances from ElectrumX
            if hasattr(wallet, 'getBalances'):
                wallet.getBalances()

            # Format and return balance information
            lines = ["Wallet Balance:"]
            lines.append("-" * 40)

            # SATORI balance
            if hasattr(wallet, 'balance') and wallet.balance:
                satori_amount = wallet.balance.amount
                lines.append(f"  SATORI: {satori_amount:.8f}")
            else:
                lines.append("  SATORI: 0.00000000")

            # EVR balance
            if hasattr(wallet, 'currency') and wallet.currency:
                evr_amount = wallet.currency.amount
                lines.append(f"  EVR:    {evr_amount:.8f}")
            else:
                lines.append("  EVR:    0.00000000")

            return "\n".join(lines)

        except Exception as e:
            return f"Error fetching wallet balance: {e}"

    def get_vault_balance_electrumx(self) -> str:
        """Fetch vault balance from ElectrumX servers."""
        if not self.startup or not hasattr(self.startup, 'walletManager'):
            return "Neuron is starting... Use /logs neuron to see progress."

        console_print("Fetching vault balance...")
        console_print()

        try:
            # Get wallet manager
            wallet_manager = self.startup.walletManager
            if not wallet_manager:
                return "Wallet manager not initialized."

            # Ensure ElectrumX connection
            if hasattr(wallet_manager, 'connect'):
                wallet_manager.connect()

            # Re-open vault with stored password to ensure it's decrypted
            if self._vault_password:
                vault = wallet_manager.openVault(password=self._vault_password)
            else:
                vault = wallet_manager.vault

            if not vault:
                return "Vault not available.\nPlease create or unlock your vault first."

            # Check if vault is decrypted
            if hasattr(vault, 'isDecrypted') and not vault.isDecrypted:
                return "Vault is locked.\nPlease unlock your vault first with /vault-open."

            # Fetch balances from ElectrumX
            if hasattr(vault, 'getBalances'):
                vault.getBalances()

            # Format and return balance information
            lines = ["Vault Balance:"]
            lines.append("-" * 40)

            # SATORI balance
            if hasattr(vault, 'balance') and vault.balance:
                satori_amount = vault.balance.amount
                lines.append(f"  SATORI: {satori_amount:.8f}")
            else:
                lines.append("  SATORI: 0.00000000")

            # EVR balance
            if hasattr(vault, 'currency') and vault.currency:
                evr_amount = vault.currency.amount
                lines.append(f"  EVR:    {evr_amount:.8f}")
            else:
                lines.append("  EVR:    0.00000000")

            return "\n".join(lines)

        except Exception as e:
            return f"Error fetching vault balance: {e}"

    def handle_command(self, user_input: str) -> str | None:
        """Process user commands and return response."""
        user_input = user_input.strip()

        if user_input == "/help":
            return """Satori Neuron CLI - Available Commands

Monitoring:
  /logs          - View neuron or engine logs (interactive menu)

Wallet & Rewards:
  /balance       - Show wallet or vault balance (interactive menu)
  /stake         - Check stake status
  /pool          - Show pool status

Streams & Engine:
  /streams       - Show stream assignments
  /pause         - Pause the engine
  /unpause       - Unpause the engine

System:
  /restart       - Restart the neuron
  /clear         - Clear the screen
  /help          - Show this help message
  /exit          - Exit CLI (neuron keeps running)"""

        elif user_input == "/logs":
            # Interactive menu for logs
            options = [
                {'label': 'Neuron Logs', 'action': 'neuron'},
                {'label': 'Engine Logs', 'action': 'engine'}
            ]

            selected = self.interactive_menu("Logs Menu", options)

            if selected is None:
                return "Cancelled."

            # Execute the selected option
            action = options[selected]['action']
            if action == 'neuron':
                logs = _log_buffer[-50:]
                if not logs:
                    return "No logs yet."
                return "\n".join(logs)
            elif action == 'engine':
                # Filter logs for engine-related messages
                engine_keywords = [
                    'engine', 'Engine', 'adapter', 'Adapter', 'prediction',
                    'Prediction', 'StreamModel', 'stream model', 'forecast',
                    'XgbAdapter', 'StarterAdapter', 'XgbChronosAdapter',
                    'model training', 'inference', 'Engine DB'
                ]
                engine_logs = [
                    log for log in _log_buffer
                    if any(keyword in log for keyword in engine_keywords)
                ]
                logs = engine_logs[-50:]
                if not logs:
                    return "No engine logs yet."
                return "\n".join(logs)

        elif user_input == "/logs neuron":
            logs = _log_buffer[-50:]
            if not logs:
                return "No logs yet."
            return "\n".join(logs)

        elif user_input == "/logs engine":
            # Filter logs for engine-related messages
            engine_keywords = [
                'engine', 'Engine', 'adapter', 'Adapter', 'prediction',
                'Prediction', 'StreamModel', 'stream model', 'forecast',
                'XgbAdapter', 'StarterAdapter', 'XgbChronosAdapter',
                'model training', 'inference', 'Engine DB'
            ]
            engine_logs = [
                log for log in _log_buffer
                if any(keyword in log for keyword in engine_keywords)
            ]
            logs = engine_logs[-50:]
            if not logs:
                return "No engine logs yet."
            return "\n".join(logs)

        elif user_input == "/status":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            status_lines = [
                f"Mode: {self.startup.runMode.name}",
                f"Paused: {self.startup.paused}",
                f"Mining Mode: {self.startup.miningMode}",
                f"Environment: {self.startup.env}",
                f"Version: {self.startup.version}",
            ]
            if self.startup.wallet:
                status_lines.append(f"Wallet: {self.startup.wallet.address}")
            return "\n".join(status_lines)

        elif user_input == "/balance":
            # Check if startup object exists and has wallet manager
            if not self.startup or not hasattr(self.startup, 'walletManager'):
                return "Neuron is starting... Please wait a moment and try again."

            # Interactive menu for balance
            # Both options always shown since vault is unlocked at CLI startup
            options = [
                {'label': 'Wallet Balance', 'action': 'wallet'},
                {'label': 'Vault Balance', 'action': 'vault'}
            ]

            selected = self.interactive_menu("Balance Menu", options)

            if selected is None:
                return "Cancelled."

            # Execute the selected option
            action = options[selected]['action']
            if action == 'wallet':
                return self.get_wallet_balance_electrumx()
            elif action == 'vault':
                return self.get_vault_balance_electrumx()

        elif user_input == "/streams":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            lines = [f"Subscriptions: {len(self.startup.subscriptions)}"]
            for s in self.startup.subscriptions[:5]:
                lines.append(f"  - {s.streamId.source}/{s.streamId.stream}")
            if len(self.startup.subscriptions) > 5:
                lines.append(f"  ... and {len(self.startup.subscriptions) - 5} more")
            lines.append(f"Publications: {len(self.startup.publications)}")
            for p in self.startup.publications[:5]:
                lines.append(f"  - {p.streamId.source}/{p.streamId.stream}")
            if len(self.startup.publications) > 5:
                lines.append(f"  ... and {len(self.startup.publications) - 5} more")
            return "\n".join(lines)

        elif user_input == "/pause":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            self.startup.pause()
            return "Engine paused"

        elif user_input == "/unpause":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            self.startup.unpause()
            return "Engine unpaused"

        elif user_input == "/restart":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            self.startup.triggerRestart()

        elif user_input == "/stake":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            status = self.startup.performStakeCheck()
            return f"Stake Status: {status}\nStake Required: {self.startup.stakeRequired}"

        elif user_input == "/pool":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            return f"Pool Accepting: {self.startup.poolIsAccepting}"

        elif user_input == "/vault-status":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            vault = self.startup.vault
            if vault is None:
                return "Vault: Not created\nUse /vault-create to create a new vault."
            status_lines = [
                f"Vault Address: {vault.address}",
                f"Decrypted: {vault.isDecrypted if hasattr(vault, 'isDecrypted') else 'Unknown'}",
            ]
            return "\n".join(status_lines)

        elif user_input == "/vault-create":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            if self.startup.vault is not None:
                return "Vault already exists. Use /vault-open to unlock it."
            return "VAULT_CREATE_PROMPT"

        elif user_input == "/vault-open":
            if not self.neuron_started:
                return "Neuron is starting... Use /logs neuron to see progress."
            vault = self.startup.vault
            if vault is None:
                return "No vault exists. Use /vault-create to create one first."
            if hasattr(vault, 'isDecrypted') and vault.isDecrypted:
                return "Vault is already open."
            return "VAULT_OPEN_PROMPT"

        elif user_input == "/clear":
            # ANSI escape code to clear screen and move cursor to top
            console_write("\033[2J\033[H")
            return "Satori Neuron CLI. Type /help for commands."

        elif user_input in ("/exit", "/quit"):
            return "EXIT_CLI"

        elif user_input.startswith("/"):
            return f"Unknown command: {user_input}. Type /help for available commands."

        else:
            return "Type /help for available commands."

    def prompt_vault_password(self, create: bool = False) -> str | None:
        """Prompt for vault password securely."""
        import getpass
        try:
            if create:
                console_print("Creating new vault...")
                console_write("Enter new vault password: ")
                password1 = console_readline().strip()
                console_write("Confirm password: ")
                password2 = console_readline().strip()
                if password1 != password2:
                    return None, "Passwords do not match."
                if len(password1) < 4:
                    return None, "Password must be at least 4 characters."
                return password1, None
            else:
                console_write("Enter vault password: ")
                password = console_readline().strip()
                return password, None
        except Exception as e:
            return None, str(e)

    def create_vault(self, password: str) -> str:
        """Create a new vault with the given password."""
        try:
            # Save password to config
            from satorineuron import config
            config.add(data={'vault password': password})

            # Create vault
            vault = self.startup.openVault(password=password, create=True)
            if vault:
                return f"Vault created successfully!\nVault Address: {vault.address}"
            return "Failed to create vault."
        except Exception as e:
            return f"Error creating vault: {e}"

    def open_vault(self, password: str) -> str:
        """Open an existing vault with the given password."""
        try:
            vault = self.startup.openVault(password=password)
            if vault and hasattr(vault, 'isDecrypted') and vault.isDecrypted:
                return f"Vault opened successfully!\nVault Address: {vault.address}"
            return "Failed to open vault. Wrong password?"
        except Exception as e:
            return f"Error opening vault: {e}"

    def run(self):
        """Run interactive CLI loop."""
        console_print("Satori Neuron CLI. Type /help for commands, /exit to quit.")
        console_print()

        # Prompt for vault setup or unlock (matches web UI behavior)
        try:
            if not self.prompt_vault_setup_or_unlock():
                console_print("Vault access required. Exiting.")
                return
        except (KeyboardInterrupt, EOFError):
            console_print()
            console_print("Exiting...")
            return

        # Start neuron in background
        self.start_neuron_background()

        while True:
            try:
                # Use console_input() for readline support (arrow keys, history)
                user_input = console_input("> ").strip()

                if not user_input:
                    continue

                response = self.handle_command(user_input)

                if response == "EXIT_CLI":
                    console_print("Goodbye!")
                    break

                if response == "VAULT_CREATE_PROMPT":
                    password, error = self.prompt_vault_password(create=True)
                    if error:
                        console_print(error)
                    elif password:
                        result = self.create_vault(password)
                        console_print(result)
                    console_print()
                    continue

                if response == "VAULT_OPEN_PROMPT":
                    password, error = self.prompt_vault_password(create=False)
                    if error:
                        console_print(error)
                    elif password:
                        result = self.open_vault(password)
                        console_print(result)
                    console_print()
                    continue

                console_print(response)
                console_print()

            except KeyboardInterrupt:
                console_print("\nGoodbye!")
                break
            except EOFError:
                console_print("\nGoodbye!")
                break


if __name__ == "__main__":
    # Set up log capture FIRST before any imports
    setup_log_capture()

    cli = NeuronCLI(env='prod', runMode='worker')
    cli.run()
