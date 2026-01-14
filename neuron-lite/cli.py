#!/usr/bin/env python3
"""
Satori Neuron CLI - Interactive command-line interface for wallet, vault, and P2P management.
Provides commands to check balances, manage vault, and interact with P2P network.

Features:
- Wallet & Vault Management
- P2P Network Operations (Ping, Identify, Peers)
- Stream Operations (Subscribe, Publish, Discover)
- Lending/Delegation Management
- Rewards & Referrals
- Oracle/Prediction Features
"""

import sys
import os
import time
import tty
import termios
import asyncio
from typing import Optional, List, Dict, Any, Callable

# Save original file descriptors for console I/O
_original_stdout_fd = os.dup(1)
_original_stdin_fd = os.dup(0)

# Create file objects from the duplicated fds
_console_out = os.fdopen(os.dup(_original_stdout_fd), 'w', buffering=1)
_console_in = os.fdopen(os.dup(_original_stdin_fd), 'r', buffering=1)

# Command history
_command_history: list[str] = []
_history_index: int = 0


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
            elif ch == '\x1b':  # Escape sequence (arrow keys or Esc)
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
                else:
                    # Esc key pressed (not followed by '[')
                    console_write('\r\n')
                    return None  # Return None to indicate cancellation
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


def console_password_input(prompt: str = "") -> str:
    """Read password input with asterisk masking."""
    console_write(prompt)

    # Use the original stdin fd for reading
    fd = _original_stdin_fd
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        password = ""

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
                if len(password) > 0:
                    password = password[:-1]
                    # Erase last asterisk
                    console_write('\b \b')
            elif ch >= ' ':  # Printable character
                password += ch
                console_write('*')

        return password

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_async(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(coro)


class NeuronCLI:
    """Interactive CLI for Satori Neuron."""

    def __init__(self, env: str = 'prod', runMode: str = 'worker'):
        self.env = env
        self.runMode = runMode
        self.wallet_manager = None
        self._vault_password = None  # Store password in memory for session
        self._startup_dag = None  # Reference to StartupDag for P2P access

    def _get_startup_dag(self):
        """Get or create StartupDag singleton for P2P access."""
        if self._startup_dag is None:
            try:
                from satorineuron.init.start import getStart
                self._startup_dag = getStart()
            except Exception as e:
                console_print(f"Warning: Could not access StartupDag: {e}")
        return self._startup_dag

    def _get_p2p_peers(self):
        """Get P2P peers instance from StartupDag."""
        dag = self._get_startup_dag()
        if dag and hasattr(dag, '_p2p_peers'):
            return dag._p2p_peers
        return None

    def _is_p2p_available(self) -> bool:
        """Check if P2P networking is available."""
        try:
            from satorineuron.init.start import is_p2p_available
            return is_p2p_available()
        except ImportError:
            return False

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
            password1 = console_password_input("Enter new vault password (min 4 characters): ")

            if len(password1) < 4:
                console_print("Password must be at least 4 characters. Please try again.")
                console_print()
                continue

            password2 = console_password_input("Confirm password: ")

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
                password = console_password_input("Enter vault password: ")

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
            password1 = console_password_input("Enter new vault password (min 4 characters): ")

            if len(password1) < 4:
                console_print("Password must be at least 4 characters. Please try again.")
                console_print()
                continue

            password2 = console_password_input("Confirm password: ")

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
                            # Move cursor back to start of menu and clear all lines atomically
                            console_write(f'\x1b[{lines_printed}A\r')

                            # Redraw menu (atomic writes to prevent duplication)
                            console_write(f'\x1b[2K{title}\r\n')
                            console_write(f'\x1b[2K{"-" * len(title)}\r\n')
                            for i, option in enumerate(options):
                                if i == selected:
                                    console_write(f"\x1b[2K  > \x1b[7m{option['label']}\x1b[0m\r\n")
                                else:
                                    console_write(f"\x1b[2K    {option['label']}\r\n")
                            console_write('\x1b[2K\r\n')  # Clear blank line
                            console_write("\x1b[2K(Use ↑/↓ arrows to navigate, Enter to select, Esc to cancel)\r\n")

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

    # ========== Wallet & Vault Operations ==========

    def get_wallet_balance_electrumx(self) -> str:
        """Fetch wallet balance from ElectrumX servers."""
        if not self.wallet_manager:
            return "Wallet manager not initialized."

        console_print("Fetching wallet balance...")
        console_print()

        try:
            # Ensure ElectrumX connection with retry
            if hasattr(self.wallet_manager, 'connect'):
                for _ in range(3):
                    if self.wallet_manager.connect():
                        break
                    time.sleep(1)

            # Get wallet
            wallet = self.wallet_manager.wallet
            if not wallet:
                return "Wallet not available."

            # Fetch balances from ElectrumX (retry if connection not ready)
            if hasattr(wallet, 'getBalances'):
                for _ in range(3):
                    if wallet.electrumx and wallet.electrumx.connected():
                        wallet.getBalances()
                        break
                    time.sleep(1)
                else:
                    # Final attempt even if connection check failed
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
        if not self.wallet_manager:
            return "Wallet manager not initialized."

        console_print("Fetching vault balance...")
        console_print()

        try:
            # Ensure ElectrumX connection with retry
            if hasattr(self.wallet_manager, 'connect'):
                for _ in range(3):
                    if self.wallet_manager.connect():
                        break
                    time.sleep(1)

            # Re-open vault with stored password to ensure it's decrypted
            if self._vault_password:
                vault = self.wallet_manager.openVault(password=self._vault_password)
            else:
                vault = self.wallet_manager.vault

            if not vault:
                return "Vault not available.\nPlease create or unlock your vault first."

            # Check if vault is decrypted
            if hasattr(vault, 'isDecrypted') and not vault.isDecrypted:
                return "Vault is locked.\nPlease unlock your vault first."

            # Fetch balances from ElectrumX (retry if connection not ready)
            if hasattr(vault, 'getBalances'):
                for _ in range(3):
                    if vault.electrumx and vault.electrumx.connected():
                        vault.getBalances()
                        break
                    time.sleep(1)
                else:
                    # Final attempt even if connection check failed
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

    def send_transaction_wallet(self) -> str:
        """Send SATORI transaction from wallet."""
        if not self.wallet_manager:
            return "Wallet manager not initialized."

        try:
            # Ensure ElectrumX connection
            if hasattr(self.wallet_manager, 'connect'):
                for _ in range(3):
                    if self.wallet_manager.connect():
                        break
                    time.sleep(1)

            # Get wallet
            wallet = self.wallet_manager.wallet
            if not wallet:
                return "Wallet not available."

            # Fetch current balance
            console_print("Fetching current balance...")
            if hasattr(wallet, 'getBalances'):
                wallet.getBalances()

            console_print()
            console_print("Send Transaction from Wallet                    [Esc to cancel]")
            console_print("=" * 60)
            console_print()
            console_print("Current Balance:")
            console_print(f"  SATORI: {wallet.balance.amount:.8f}")
            console_print(f"  EVR:    {wallet.currency.amount:.8f}")
            console_print()

            # Get recipient address
            address = console_input("Enter recipient address: ")
            if address is None:
                return "Transaction cancelled. Returning to main menu."
            address = address.strip()
            if not address:
                return "Transaction cancelled. Returning to main menu."

            # Validate address
            from satorilib.wallet.utils.validate import Validate
            if not Validate.address(address, wallet.symbol):
                return f"Invalid address: {address}\nReturning to main menu."

            # Clear screen and show amount prompt
            console_write("\033[2J\033[H")
            console_print()
            console_print("Send Transaction from Wallet                    [Esc to cancel]")
            console_print("=" * 60)
            console_print(f"Recipient: {address}")
            console_print()
            console_print("Current Balance:")
            console_print(f"  SATORI: {wallet.balance.amount:.8f}")
            console_print(f"  EVR:    {wallet.currency.amount:.8f}")
            console_print()

            # Get amount
            amount_str = console_input("Enter amount of SATORI to send: ")
            if amount_str is None:
                return "Transaction cancelled. Returning to main menu."
            amount_str = amount_str.strip()
            if not amount_str:
                return "Transaction cancelled. Returning to main menu."

            try:
                amount = float(amount_str)
            except ValueError:
                return f"Invalid amount: {amount_str}\nReturning to main menu."

            if amount <= 0:
                return "Amount must be greater than 0.\nReturning to main menu."

            if amount > wallet.balance.amount:
                return f"Insufficient balance. You have {wallet.balance.amount:.8f} SATORI\nReturning to main menu."

            # Show confirmation
            console_print()
            console_print("Transaction Confirmation                        [Esc to cancel]")
            console_print("=" * 60)
            console_print(f"  From:   Wallet")
            console_print(f"  To:     {address}")
            console_print(f"  Amount: {amount} SATORI")
            console_print("=" * 60)
            console_print()

            confirm = console_input("Type 'yes' to confirm: ")
            if confirm is None:
                return "Transaction cancelled. Returning to main menu."
            confirm = confirm.strip().lower()
            if confirm not in ['yes', 'y']:
                return "Transaction cancelled. Returning to main menu."

            # Prepare for transaction
            console_print()
            console_print("Preparing transaction...")
            wallet.getReadyToSend(balance=True, save=True)

            # Send transaction
            console_print("Sending transaction...")
            txid = wallet.satoriTransaction(amount=amount, address=address)

            return f"\nTransaction successful!\nTransaction ID: {txid}"

        except Exception as e:
            return f"Error sending transaction: {e}"

    def send_transaction_vault(self) -> str:
        """Send SATORI transaction from vault."""
        if not self.wallet_manager:
            return "Wallet manager not initialized."

        try:
            # Ensure ElectrumX connection
            if hasattr(self.wallet_manager, 'connect'):
                for _ in range(3):
                    if self.wallet_manager.connect():
                        break
                    time.sleep(1)

            # Get vault
            if self._vault_password:
                vault = self.wallet_manager.openVault(password=self._vault_password)
            else:
                vault = self.wallet_manager.vault

            if not vault:
                return "Vault not available.\nPlease create or unlock your vault first."

            if hasattr(vault, 'isDecrypted') and not vault.isDecrypted:
                return "Vault is locked.\nPlease unlock your vault first."

            # Fetch current balance
            console_print("Fetching current balance...")
            if hasattr(vault, 'getBalances'):
                vault.getBalances()

            console_print()
            console_print("Send Transaction from Vault                     [Esc to cancel]")
            console_print("=" * 60)
            console_print()
            console_print("Current Balance:")
            console_print(f"  SATORI: {vault.balance.amount:.8f}")
            console_print(f"  EVR:    {vault.currency.amount:.8f}")
            console_print()

            # Get recipient address
            address = console_input("Enter recipient address: ")
            if address is None:
                return "Transaction cancelled. Returning to main menu."
            address = address.strip()
            if not address:
                return "Transaction cancelled. Returning to main menu."

            # Validate address
            from satorilib.wallet.utils.validate import Validate
            if not Validate.address(address, vault.symbol):
                return f"Invalid address: {address}\nReturning to main menu."

            # Clear screen and show amount prompt
            console_write("\033[2J\033[H")
            console_print()
            console_print("Send Transaction from Vault                     [Esc to cancel]")
            console_print("=" * 60)
            console_print(f"Recipient: {address}")
            console_print()
            console_print("Current Balance:")
            console_print(f"  SATORI: {vault.balance.amount:.8f}")
            console_print(f"  EVR:    {vault.currency.amount:.8f}")
            console_print()

            # Get amount
            amount_str = console_input("Enter amount of SATORI to send: ")
            if amount_str is None:
                return "Transaction cancelled. Returning to main menu."
            amount_str = amount_str.strip()
            if not amount_str:
                return "Transaction cancelled. Returning to main menu."

            try:
                amount = float(amount_str)
            except ValueError:
                return f"Invalid amount: {amount_str}\nReturning to main menu."

            if amount <= 0:
                return "Amount must be greater than 0.\nReturning to main menu."

            if amount > vault.balance.amount:
                return f"Insufficient balance. You have {vault.balance.amount:.8f} SATORI\nReturning to main menu."

            # Show confirmation
            console_print()
            console_print("Transaction Confirmation                        [Esc to cancel]")
            console_print("=" * 60)
            console_print(f"  From:   Vault")
            console_print(f"  To:     {address}")
            console_print(f"  Amount: {amount} SATORI")
            console_print("=" * 60)
            console_print()

            confirm = console_input("Type 'yes' to confirm: ")
            if confirm is None:
                return "Transaction cancelled. Returning to main menu."
            confirm = confirm.strip().lower()
            if confirm not in ['yes', 'y']:
                return "Transaction cancelled. Returning to main menu."

            # Prepare for transaction
            console_print()
            console_print("Preparing transaction...")
            vault.getReadyToSend(balance=True, save=True)

            # Send transaction
            console_print("Sending transaction...")
            txid = vault.satoriTransaction(amount=amount, address=address)

            return f"\nTransaction successful!\nTransaction ID: {txid}"

        except Exception as e:
            return f"Error sending transaction: {e}"

    # ========== P2P Network Operations ==========

    def p2p_menu(self) -> str | None:
        """Show P2P network menu."""
        if not self._is_p2p_available():
            return "P2P networking not available. Install satorip2p or enable P2P mode."

        options = [
            {'label': 'Ping Peer', 'action': 'ping'},
            {'label': 'View Known Peers', 'action': 'peers'},
            {'label': 'View Peer Identities', 'action': 'identities'},
            {'label': 'Announce Identity', 'action': 'announce'},
            {'label': 'Network Status', 'action': 'status'},
            {'label': 'Discover Peers', 'action': 'discover'},
            {'label': 'Back to Main Menu', 'action': 'back'}
        ]

        selected = self.interactive_menu("P2P Network Menu", options)
        if selected is None:
            return None

        action = options[selected]['action']
        if action == 'back':
            return None
        elif action == 'ping':
            return self.ping_peer()
        elif action == 'peers':
            return self.view_known_peers()
        elif action == 'identities':
            return self.view_peer_identities()
        elif action == 'announce':
            return self.announce_identity()
        elif action == 'status':
            return self.network_status()
        elif action == 'discover':
            return self.discover_peers()

        return None

    def ping_peer(self) -> str:
        """Ping a specific peer to test connectivity."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized. Start the node first."

        console_print()
        console_print("Ping Peer                                       [Esc to cancel]")
        console_print("=" * 60)
        console_print()

        peer_id = console_input("Enter peer ID to ping: ")
        if peer_id is None or not peer_id.strip():
            return "Cancelled."

        peer_id = peer_id.strip()

        try:
            console_print(f"\nPinging {peer_id[:20]}...")

            async def do_ping():
                return await peers.ping_peer(peer_id, count=3, timeout=10.0)

            latencies = run_async(do_ping())

            if latencies:
                avg_rtt = sum(latencies) / len(latencies)
                lines = [
                    f"\nPing Results for {peer_id[:20]}...",
                    "-" * 40,
                    f"  Pings sent:     3",
                    f"  Pings received: {len(latencies)}",
                    f"  Min RTT:        {min(latencies)*1000:.2f} ms",
                    f"  Max RTT:        {max(latencies)*1000:.2f} ms",
                    f"  Avg RTT:        {avg_rtt*1000:.2f} ms",
                ]
                return "\n".join(lines)
            else:
                return f"Ping failed - no response from {peer_id[:20]}..."

        except Exception as e:
            return f"Ping error: {e}"

    def view_known_peers(self) -> str:
        """View list of known/connected peers."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized. Start the node first."

        try:
            connected = peers.connected_peers
            peer_id = peers.peer_id or "Not set"

            lines = [
                "Known Peers",
                "=" * 60,
                f"  Our Peer ID: {peer_id[:40]}...",
                f"  Connected Peers: {connected}",
                "-" * 60,
            ]

            # Try to get more peer info if available
            if hasattr(peers, 'get_connected_peers'):
                peer_list = peers.get_connected_peers()
                if peer_list:
                    lines.append("  Connected Peer IDs:")
                    for p in peer_list[:10]:  # Limit to 10
                        lines.append(f"    - {p[:40]}...")
                    if len(peer_list) > 10:
                        lines.append(f"    ... and {len(peer_list) - 10} more")
                else:
                    lines.append("  No peers currently connected.")
            else:
                lines.append("  Detailed peer list not available.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error getting peer info: {e}"

    def view_peer_identities(self) -> str:
        """View identities of known peers from Identify protocol."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized. Start the node first."

        try:
            identities = peers.get_known_peer_identities()

            lines = [
                "Peer Identities",
                "=" * 60,
            ]

            if not identities:
                lines.append("  No peer identities cached.")
                lines.append("  Peers announce their identity when they connect.")
            else:
                lines.append(f"  Known Peers: {len(identities)}")
                lines.append("-" * 60)

                for peer_id, identity in list(identities.items())[:10]:
                    lines.append(f"  Peer: {peer_id[:30]}...")
                    if hasattr(identity, 'evrmore_address'):
                        lines.append(f"    Evrmore: {identity.evrmore_address[:30]}...")
                    if hasattr(identity, 'roles'):
                        lines.append(f"    Roles:   {', '.join(identity.roles)}")
                    if hasattr(identity, 'agent_version'):
                        lines.append(f"    Agent:   {identity.agent_version}")
                    lines.append("")

                if len(identities) > 10:
                    lines.append(f"  ... and {len(identities) - 10} more peers")

            return "\n".join(lines)

        except Exception as e:
            return f"Error getting peer identities: {e}"

    def announce_identity(self) -> str:
        """Announce our identity to the network."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized. Start the node first."

        try:
            console_print("Announcing identity to network...")

            async def do_announce():
                await peers.announce_identity()

            run_async(do_announce())

            return "Identity announced successfully!\nPeers will receive our peer info, address, and capabilities."

        except Exception as e:
            return f"Error announcing identity: {e}"

    def network_status(self) -> str:
        """Show overall P2P network status."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized. Start the node first."

        try:
            lines = [
                "P2P Network Status",
                "=" * 60,
            ]

            # Basic info
            lines.append(f"  Peer ID:         {peers.peer_id[:40]}..." if peers.peer_id else "  Peer ID: Not set")
            lines.append(f"  Evrmore Address: {peers.evrmore_address[:40]}..." if peers.evrmore_address else "  Evrmore Address: Not set")
            lines.append(f"  Connected:       {peers.is_connected}")
            lines.append(f"  Connected Peers: {peers.connected_peers}")
            lines.append(f"  NAT Type:        {peers.nat_type}")
            lines.append(f"  Is Relay:        {peers.is_relay}")

            # Public addresses
            if hasattr(peers, 'public_addresses') and peers.public_addresses:
                lines.append(f"  Public Addresses:")
                for addr in peers.public_addresses[:3]:
                    lines.append(f"    - {addr}")

            # Protocol info
            lines.append("")
            lines.append("Protocol Status:")
            lines.append("-" * 40)

            # Ping protocol stats
            if hasattr(peers, '_ping_service') and peers._ping_service:
                ping_stats = peers._ping_service.get_stats()
                lines.append(f"  Ping Protocol:   v{ping_stats.get('version', '?')} (started: {ping_stats.get('started', False)})")
            else:
                lines.append("  Ping Protocol:   Not initialized")

            # Identify protocol stats
            if hasattr(peers, '_identify_handler') and peers._identify_handler:
                id_stats = peers._identify_handler.get_stats()
                lines.append(f"  Identify Protocol: v{id_stats.get('version', '?')} (known peers: {id_stats.get('known_peers', 0)})")
            else:
                lines.append("  Identify Protocol: Not initialized")

            return "\n".join(lines)

        except Exception as e:
            return f"Error getting network status: {e}"

    def discover_peers(self) -> str:
        """Discover peers on the network."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized. Start the node first."

        try:
            console_print("Discovering peers on the network...")

            async def do_discover():
                return await peers.discover_peers()

            discovered = run_async(do_discover())

            if discovered:
                lines = [
                    f"Discovered {len(discovered)} peers:",
                    "-" * 40,
                ]
                for p in discovered[:10]:
                    lines.append(f"  - {p[:40]}...")
                if len(discovered) > 10:
                    lines.append(f"  ... and {len(discovered) - 10} more")
                return "\n".join(lines)
            else:
                return "No new peers discovered."

        except Exception as e:
            return f"Error discovering peers: {e}"

    # ========== Stream Operations ==========

    def streams_menu(self) -> str | None:
        """Show stream operations menu."""
        if not self._is_p2p_available():
            return "P2P networking not available. Install satorip2p or enable P2P mode."

        options = [
            {'label': 'View My Subscriptions', 'action': 'subscriptions'},
            {'label': 'View My Publications', 'action': 'publications'},
            {'label': 'Subscribe to Stream', 'action': 'subscribe'},
            {'label': 'Publish to Stream', 'action': 'publish'},
            {'label': 'Discover Streams', 'action': 'discover'},
            {'label': 'Back to Main Menu', 'action': 'back'}
        ]

        selected = self.interactive_menu("Stream Operations", options)
        if selected is None:
            return None

        action = options[selected]['action']
        if action == 'back':
            return None
        elif action == 'subscriptions':
            return self.view_subscriptions()
        elif action == 'publications':
            return self.view_publications()
        elif action == 'subscribe':
            return self.subscribe_to_stream()
        elif action == 'publish':
            return self.publish_to_stream()
        elif action == 'discover':
            return self.discover_streams()

        return None

    def view_subscriptions(self) -> str:
        """View current stream subscriptions."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized."

        try:
            subs = peers.get_my_subscriptions()
            lines = ["Current Subscriptions", "=" * 40]

            if subs:
                for s in subs:
                    lines.append(f"  - {s}")
            else:
                lines.append("  No active subscriptions.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def view_publications(self) -> str:
        """View current stream publications."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized."

        try:
            pubs = peers.get_my_publications()
            lines = ["Current Publications", "=" * 40]

            if pubs:
                for p in pubs:
                    lines.append(f"  - {p}")
            else:
                lines.append("  No active publications.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def subscribe_to_stream(self) -> str:
        """Subscribe to a stream."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized."

        console_print()
        stream_id = console_input("Enter stream ID to subscribe: ")
        if stream_id is None or not stream_id.strip():
            return "Cancelled."

        try:
            async def do_sub():
                await peers.subscribe_async(stream_id.strip())

            run_async(do_sub())
            return f"Subscribed to stream: {stream_id.strip()}"

        except Exception as e:
            return f"Error subscribing: {e}"

    def publish_to_stream(self) -> str:
        """Publish data to a stream."""
        peers = self._get_p2p_peers()
        if not peers:
            return "P2P not initialized."

        console_print()
        stream_id = console_input("Enter stream ID: ")
        if stream_id is None or not stream_id.strip():
            return "Cancelled."

        data = console_input("Enter data to publish: ")
        if data is None or not data.strip():
            return "Cancelled."

        try:
            async def do_pub():
                await peers.publish(stream_id.strip(), data.strip().encode())

            run_async(do_pub())
            return f"Published to stream: {stream_id.strip()}"

        except Exception as e:
            return f"Error publishing: {e}"

    def discover_streams(self) -> str:
        """Discover available streams."""
        dag = self._get_startup_dag()
        if not dag or not hasattr(dag, '_stream_registry'):
            return "Stream registry not available."

        try:
            registry = dag._stream_registry
            if not registry:
                return "Stream registry not initialized."

            # Get known streams
            streams = registry.get_all_streams() if hasattr(registry, 'get_all_streams') else []

            lines = ["Available Streams", "=" * 40]
            if streams:
                for s in streams[:20]:
                    lines.append(f"  - {s}")
                if len(streams) > 20:
                    lines.append(f"  ... and {len(streams) - 20} more")
            else:
                lines.append("  No streams discovered yet.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    # ========== Rewards & Referrals ==========

    def rewards_menu(self) -> str | None:
        """Show rewards and referrals menu."""
        options = [
            {'label': 'View Reward Stats', 'action': 'stats'},
            {'label': 'View Referral Status', 'action': 'referrals'},
            {'label': 'View Stake Bonuses', 'action': 'stake'},
            {'label': 'View Role Multipliers', 'action': 'roles'},
            {'label': 'Custom Reward Address', 'action': 'address'},
            {'label': 'Back to Main Menu', 'action': 'back'}
        ]

        selected = self.interactive_menu("Rewards & Referrals", options)
        if selected is None:
            return None

        action = options[selected]['action']
        if action == 'back':
            return None
        elif action == 'stats':
            return self.view_reward_stats()
        elif action == 'referrals':
            return self.view_referral_status()
        elif action == 'stake':
            return self.view_stake_bonuses()
        elif action == 'roles':
            return self.view_role_multipliers()
        elif action == 'address':
            return self.manage_reward_address()

        return None

    def view_reward_stats(self) -> str:
        """View reward statistics."""
        try:
            from satorineuron.init.start import get_p2p_module
            RewardCalculator = get_p2p_module('RewardCalculator')

            lines = [
                "Reward Statistics",
                "=" * 40,
            ]

            if RewardCalculator:
                lines.append("  Reward calculation module available.")
                lines.append("  Use the Neuron web UI for detailed reward history.")
            else:
                lines.append("  Reward module not available.")
                lines.append("  Enable P2P mode for local reward calculation.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def view_referral_status(self) -> str:
        """View referral program status."""
        try:
            from satorineuron.init.start import get_p2p_module

            lines = [
                "Referral Program Status",
                "=" * 40,
            ]

            ReferralManager = get_p2p_module('ReferralManager')
            if ReferralManager:
                lines.append("  Referral system available.")
                lines.append("")
                lines.append("  Tier Thresholds:")
                lines.append("    Bronze:   1+ referrals   (+1% bonus)")
                lines.append("    Silver:   5+ referrals   (+2% bonus)")
                lines.append("    Gold:     10+ referrals  (+3% bonus)")
                lines.append("    Platinum: 25+ referrals  (+4% bonus)")
                lines.append("    Diamond:  50+ referrals  (+5% bonus)")
            else:
                lines.append("  Referral module not available.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def view_stake_bonuses(self) -> str:
        """View stake bonus information."""
        try:
            from satorineuron.init.start import get_p2p_module

            lines = [
                "Stake Bonuses",
                "=" * 40,
                "",
                "  Stake bonuses reward holding SATORI:",
                "",
                "  Minimum Stake:     1,000 SATORI",
                "  Bonus per 1,000:   +0.5%",
                "  Maximum Bonus:     +10%",
                "",
                "  Your vault balance determines your stake bonus.",
            ]

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def view_role_multipliers(self) -> str:
        """View role-based reward multipliers."""
        lines = [
            "Role Multipliers",
            "=" * 40,
            "",
            "  Bonus rewards for running network roles:",
            "",
            "  Predictor:  Base (all nodes)",
            "  Relay:      +5% bonus (relay nodes)",
            "  Oracle:     +3% bonus (oracle publishers)",
            "  Signer:     +2% bonus (multisig signers)",
            "",
            "  Multiplier Cap: +10% maximum combined",
        ]

        return "\n".join(lines)

    def manage_reward_address(self) -> str:
        """Manage custom reward address."""
        try:
            from satorineuron.init.start import get_p2p_module
            RewardAddressManager = get_p2p_module('RewardAddressManager')

            if not RewardAddressManager:
                return "Reward address management not available in current mode."

            lines = [
                "Custom Reward Address",
                "=" * 40,
                "",
                "  Set a custom address to receive rewards.",
                "  By default, rewards go to your vault address.",
                "",
                "  Use the web UI to configure a custom reward address.",
            ]

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    # ========== Treasury Donation ==========

    def donation_menu(self) -> str | None:
        """Show treasury donation menu."""
        options = [
            {'label': 'Make Donation', 'action': 'donate'},
            {'label': 'View My Donation Stats', 'action': 'stats'},
            {'label': 'View Donation History', 'action': 'history'},
            {'label': 'View Top Donors', 'action': 'top'},
            {'label': 'View Treasury Address', 'action': 'address'},
            {'label': 'Back to Main Menu', 'action': 'back'}
        ]

        selected = self.interactive_menu("Treasury Donation", options)
        if selected is None:
            return None

        action = options[selected]['action']
        if action == 'back':
            return None
        elif action == 'donate':
            return self.make_donation()
        elif action == 'stats':
            return self.view_donation_stats()
        elif action == 'history':
            return self.view_donation_history()
        elif action == 'top':
            return self.view_top_donors()
        elif action == 'address':
            return self.view_treasury_address()

        return None

    def view_treasury_address(self) -> str:
        """View the treasury donation address."""
        try:
            dag = self._get_startup_dag()
            if dag and hasattr(dag, '_donation_manager') and dag._donation_manager:
                dm = dag._donation_manager
                if hasattr(dm, 'get_treasury_address'):
                    addr = dm.get_treasury_address()
                    return f"Treasury Address:\n  {addr}\n\nSend EVR to this address to donate to the Satori Network."

            # Fallback: try to get from config
            from satorineuron import config
            treasury = config.get().get('treasury_address', 'EeEYJBZBjQA1g3G2RJGz11kqPuSTwtEpf9')
            return f"Treasury Address:\n  {treasury}\n\nSend EVR to this address to donate to the Satori Network."

        except Exception as e:
            return f"Error: {e}"

    def view_donation_stats(self) -> str:
        """View your donation statistics."""
        try:
            dag = self._get_startup_dag()

            lines = [
                "Your Donation Stats",
                "=" * 40,
            ]

            if dag and hasattr(dag, '_donation_manager') and dag._donation_manager:
                dm = dag._donation_manager
                if hasattr(dm, 'get_donor_stats'):
                    stats = dm.get_donor_stats()
                    lines.append(f"  Total Donated: {stats.get('total_donated', 0):.2f} EVR")
                    lines.append(f"  Donation Count: {stats.get('donation_count', 0)}")
                    lines.append(f"  Current Tier: {stats.get('tier', 'None').capitalize()}")
                    lines.append(f"  Badges: {', '.join(stats.get('badges_earned', [])) or 'None'}")
                else:
                    lines.append("  Donation stats not available.")
            else:
                lines.append("  Donation manager not initialized.")
                lines.append("")
                lines.append("  Tier Thresholds:")
                lines.append("    Bronze:   100 EVR")
                lines.append("    Silver:   1,000 EVR")
                lines.append("    Gold:     10,000 EVR")
                lines.append("    Platinum: 100,000 EVR")
                lines.append("    Diamond:  1,000,000 EVR")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def view_donation_history(self) -> str:
        """View your donation history."""
        try:
            dag = self._get_startup_dag()

            lines = [
                "Donation History",
                "=" * 60,
            ]

            if dag and hasattr(dag, '_donation_manager') and dag._donation_manager:
                dm = dag._donation_manager
                if hasattr(dm, 'get_donation_history'):
                    history = dm.get_donation_history()
                    if history:
                        lines.append(f"  {'Date':<12} {'Amount':>12} {'SATORI':>12} {'Status':>10}")
                        lines.append("  " + "-" * 48)
                        for d in history[:10]:
                            import datetime
                            date = datetime.datetime.fromtimestamp(d.get('timestamp', 0)).strftime('%Y-%m-%d')
                            amount = f"{d.get('amount', 0):.2f}"
                            satori = f"{d.get('satori_reward', 0):.2f}"
                            status = d.get('status', 'unknown')
                            lines.append(f"  {date:<12} {amount:>12} {satori:>12} {status:>10}")
                        if len(history) > 10:
                            lines.append(f"  ... and {len(history) - 10} more")
                    else:
                        lines.append("  No donations yet.")
                else:
                    lines.append("  Donation history not available.")
            else:
                lines.append("  Donation manager not initialized.")
                lines.append("  Use the web UI to view your donation history.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def view_top_donors(self) -> str:
        """View top donors leaderboard."""
        try:
            dag = self._get_startup_dag()

            lines = [
                "Top Donors",
                "=" * 60,
            ]

            if dag and hasattr(dag, '_donation_manager') and dag._donation_manager:
                dm = dag._donation_manager
                if hasattr(dm, 'get_top_donors'):
                    donors = dm.get_top_donors()
                    if donors:
                        lines.append(f"  {'Rank':<6} {'Address':<20} {'Total':>12} {'Tier':>10}")
                        lines.append("  " + "-" * 50)
                        for i, d in enumerate(donors[:10], 1):
                            addr = d.get('donor_address', '')[:16] + '...'
                            total = f"{d.get('total_donated', 0):.2f}"
                            tier = d.get('tier', 'none').capitalize()
                            lines.append(f"  #{i:<5} {addr:<20} {total:>12} {tier:>10}")
                    else:
                        lines.append("  No donors yet.")
                else:
                    lines.append("  Top donors not available.")
            else:
                lines.append("  Donation manager not initialized.")
                lines.append("  Use the web UI to view top donors.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def make_donation(self) -> str:
        """Make a treasury donation."""
        if not self.wallet_manager:
            return "Wallet manager not initialized."

        try:
            console_print()
            console_print("Treasury Donation                               [Esc to cancel]")
            console_print("=" * 60)
            console_print()
            console_print("Donate EVR to the Satori Network treasury.")
            console_print("You'll receive SATORI tokens in return!")
            console_print()

            # Get current vault balance
            if self._vault_password:
                vault = self.wallet_manager.openVault(password=self._vault_password)
            else:
                vault = self.wallet_manager.vault

            if vault and hasattr(vault, 'getBalances'):
                vault.getBalances()

            if vault and hasattr(vault, 'currency') and vault.currency:
                evr_balance = vault.currency.amount
                console_print(f"Your EVR Balance: {evr_balance:.8f} EVR")
            else:
                console_print("Could not fetch EVR balance.")
                evr_balance = 0

            console_print()

            # Get treasury address
            try:
                from satorineuron import config
                treasury = config.get().get('treasury_address', 'EeEYJBZBjQA1g3G2RJGz11kqPuSTwtEpf9')
            except:
                treasury = 'EeEYJBZBjQA1g3G2RJGz11kqPuSTwtEpf9'

            console_print(f"Treasury Address: {treasury}")
            console_print()

            # Get amount
            amount_str = console_input("Enter EVR amount to donate: ")
            if amount_str is None or not amount_str.strip():
                return "Cancelled."

            try:
                amount = float(amount_str.strip())
            except ValueError:
                return f"Invalid amount: {amount_str}"

            if amount <= 0:
                return "Amount must be greater than 0."

            if evr_balance > 0 and amount > evr_balance:
                return f"Insufficient balance. You have {evr_balance:.8f} EVR"

            # Estimate SATORI reward (approximate exchange rate)
            try:
                dag = self._get_startup_dag()
                if dag and hasattr(dag, '_price_provider') and dag._price_provider:
                    rate = dag._price_provider.get_exchange_rate()
                else:
                    rate = 0.018  # Default estimate
            except:
                rate = 0.018

            satori_estimate = amount * rate * 0.9  # 90% after treasury fee

            console_print()
            console_print(f"Estimated SATORI reward: {satori_estimate:.2f} SATORI")
            console_print()

            # Confirmation
            console_print("Donation Confirmation")
            console_print("-" * 40)
            console_print(f"  Amount: {amount:.2f} EVR")
            console_print(f"  To:     {treasury[:30]}...")
            console_print(f"  Reward: ~{satori_estimate:.2f} SATORI")
            console_print()

            confirm = console_input("Type 'yes' to confirm donation: ")
            if confirm is None or confirm.strip().lower() not in ['yes', 'y']:
                return "Donation cancelled."

            # Execute donation
            console_print()
            console_print("Processing donation...")

            if vault:
                vault.getReadyToSend(balance=True, save=True)
                txid = vault.evrTransaction(amount=amount, address=treasury)
                return f"\nDonation successful!\nTransaction ID: {txid}\n\nThank you for supporting the Satori Network!"
            else:
                return "Error: Vault not available for transaction."

        except Exception as e:
            return f"Error making donation: {e}"

    # ========== Lending & Delegation ==========

    def lending_menu(self) -> str | None:
        """Show lending and delegation menu."""
        if not self._is_p2p_available():
            return "P2P networking required for lending/delegation."

        options = [
            {'label': 'View Pool Status', 'action': 'pool-status'},
            {'label': 'Join Lending Pool', 'action': 'join-pool'},
            {'label': 'Leave Pool', 'action': 'leave-pool'},
            {'label': 'View Delegations', 'action': 'delegations'},
            {'label': 'Set Delegation', 'action': 'set-delegate'},
            {'label': 'Back to Main Menu', 'action': 'back'}
        ]

        selected = self.interactive_menu("Lending & Delegation", options)
        if selected is None:
            return None

        action = options[selected]['action']
        if action == 'back':
            return None
        elif action == 'pool-status':
            return self.view_pool_status()
        elif action == 'join-pool':
            return self.join_lending_pool()
        elif action == 'leave-pool':
            return self.leave_lending_pool()
        elif action == 'delegations':
            return self.view_delegations()
        elif action == 'set-delegate':
            return self.set_delegation()

        return None

    def view_pool_status(self) -> str:
        """View lending pool status."""
        dag = self._get_startup_dag()
        if not dag:
            return "Startup not initialized."

        try:
            lines = [
                "Lending Pool Status",
                "=" * 40,
            ]

            if hasattr(dag, '_lending_manager') and dag._lending_manager:
                lm = dag._lending_manager
                lines.append(f"  Pool Active: Yes")
                # Add more status info
            else:
                lines.append("  Lending manager not initialized.")
                lines.append("  Enable P2P mode and restart.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def join_lending_pool(self) -> str:
        """Join a lending pool."""
        console_print()
        vault_addr = console_input("Enter pool vault address: ")
        if vault_addr is None or not vault_addr.strip():
            return "Cancelled."

        return "Pool joining requires the web UI for security confirmation."

    def leave_lending_pool(self) -> str:
        """Leave current lending pool."""
        return "Pool leaving requires the web UI for security confirmation."

    def view_delegations(self) -> str:
        """View delegation status."""
        dag = self._get_startup_dag()
        if not dag:
            return "Startup not initialized."

        try:
            lines = [
                "Delegation Status",
                "=" * 40,
            ]

            if hasattr(dag, '_delegation_manager') and dag._delegation_manager:
                dm = dag._delegation_manager
                lines.append("  Delegation manager active.")
            else:
                lines.append("  Delegation manager not initialized.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def set_delegation(self) -> str:
        """Set stake delegation."""
        return "Delegation setup requires the web UI for security confirmation."

    # ========== Oracle & Predictions ==========

    def oracle_menu(self) -> str | None:
        """Show oracle and predictions menu."""
        if not self._is_p2p_available():
            return "P2P networking required for oracle/predictions."

        options = [
            {'label': 'View Oracle Status', 'action': 'oracle-status'},
            {'label': 'View Predictions', 'action': 'predictions'},
            {'label': 'Submit Prediction', 'action': 'submit'},
            {'label': 'View Observations', 'action': 'observations'},
            {'label': 'Back to Main Menu', 'action': 'back'}
        ]

        selected = self.interactive_menu("Oracle & Predictions", options)
        if selected is None:
            return None

        action = options[selected]['action']
        if action == 'back':
            return None
        elif action == 'oracle-status':
            return self.view_oracle_status()
        elif action == 'predictions':
            return self.view_predictions()
        elif action == 'submit':
            return self.submit_prediction()
        elif action == 'observations':
            return self.view_observations()

        return None

    def view_oracle_status(self) -> str:
        """View oracle network status."""
        dag = self._get_startup_dag()
        if not dag:
            return "Startup not initialized."

        try:
            lines = [
                "Oracle Network Status",
                "=" * 40,
            ]

            if hasattr(dag, '_oracle_network') and dag._oracle_network:
                oracle = dag._oracle_network
                lines.append("  Oracle network active.")
                if hasattr(oracle, 'get_stats'):
                    stats = oracle.get_stats()
                    lines.append(f"  Registered Oracles: {stats.get('registered_oracles', 0)}")
            else:
                lines.append("  Oracle network not initialized.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def view_predictions(self) -> str:
        """View recent predictions."""
        dag = self._get_startup_dag()
        if not dag:
            return "Startup not initialized."

        try:
            lines = [
                "Recent Predictions",
                "=" * 40,
            ]

            if hasattr(dag, '_prediction_protocol') and dag._prediction_protocol:
                pp = dag._prediction_protocol
                lines.append("  Prediction protocol active.")
            else:
                lines.append("  Prediction protocol not initialized.")

            lines.append("")
            lines.append("  Use the web UI for detailed prediction history.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    def submit_prediction(self) -> str:
        """Submit a prediction."""
        return "Prediction submission requires the engine to be running.\nUse the automated prediction system."

    def view_observations(self) -> str:
        """View recent observations."""
        dag = self._get_startup_dag()
        if not dag:
            return "Startup not initialized."

        try:
            lines = [
                "Recent Observations",
                "=" * 40,
            ]

            if hasattr(dag, '_oracle_network') and dag._oracle_network:
                oracle = dag._oracle_network
                lines.append("  Oracle network active.")
                # Would need to implement get_recent_observations
            else:
                lines.append("  Oracle network not initialized.")

            lines.append("")
            lines.append("  Use the web UI for observation history.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    # ========== Command Handling ==========

    def handle_command(self, user_input: str) -> str | None:
        """Process user commands and return response."""
        user_input = user_input.strip()

        if user_input == "/help":
            return """Satori Neuron CLI - Available Commands

Wallet & Vault:
  /balance       - Show wallet or vault balance
  /vault-status  - Show vault status
  /send          - Send SATORI transaction

P2P Network:
  /p2p           - P2P network menu (ping, peers, identity)
  /ping          - Quick ping a peer
  /peers         - View connected peers
  /network       - Network status

Streams:
  /streams       - Stream operations menu
  /subscribe     - Subscribe to a stream
  /publish       - Publish to a stream

Rewards:
  /rewards       - Rewards & referrals menu
  /referrals     - View referral status

Donations:
  /donate        - Treasury donation menu
  /treasury      - View treasury address

Lending:
  /lending       - Lending & delegation menu
  /pool          - View pool status

Oracle:
  /oracle        - Oracle & predictions menu
  /predictions   - View predictions

System:
  /clear         - Clear the screen
  /help          - Show this help message
  /exit          - Exit CLI"""

        elif user_input == "/balance":
            # Check if wallet manager exists
            if not self.wallet_manager:
                return "Wallet manager not initialized."

            # Interactive menu for balance
            options = [
                {'label': 'Wallet Balance', 'action': 'wallet'},
                {'label': 'Vault Balance', 'action': 'vault'},
                {'label': 'Back to Main Menu', 'action': 'back'}
            ]

            selected = self.interactive_menu("Balance Menu", options)

            if selected is None:
                return "Cancelled."

            # Execute the selected option
            action = options[selected]['action']
            if action == 'back':
                return None
            elif action == 'wallet':
                return self.get_wallet_balance_electrumx()
            elif action == 'vault':
                return self.get_vault_balance_electrumx()

        elif user_input == "/vault-status":
            if not self.wallet_manager:
                return "Wallet manager not initialized."

            vault = self.wallet_manager.vault
            if vault is None:
                return "Vault: Not created"

            status_lines = [
                f"Vault Address: {vault.address}",
                f"Decrypted: {vault.isDecrypted if hasattr(vault, 'isDecrypted') else 'Unknown'}",
            ]
            return "\n".join(status_lines)

        # P2P Commands
        elif user_input == "/p2p":
            return self.p2p_menu()
        elif user_input == "/ping":
            return self.ping_peer()
        elif user_input == "/peers":
            return self.view_known_peers()
        elif user_input == "/network":
            return self.network_status()

        # Stream Commands
        elif user_input == "/streams":
            return self.streams_menu()
        elif user_input == "/subscribe":
            return self.subscribe_to_stream()
        elif user_input == "/publish":
            return self.publish_to_stream()

        # Rewards Commands
        elif user_input == "/rewards":
            return self.rewards_menu()
        elif user_input == "/referrals":
            return self.view_referral_status()

        # Donation Commands
        elif user_input == "/donate":
            return self.donation_menu()
        elif user_input == "/treasury":
            return self.view_treasury_address()

        # Lending Commands
        elif user_input == "/lending":
            return self.lending_menu()
        elif user_input == "/pool":
            return self.view_pool_status()

        # Oracle Commands
        elif user_input == "/oracle":
            return self.oracle_menu()
        elif user_input == "/predictions":
            return self.view_predictions()

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


    def show_main_menu(self) -> str | None:
        """Show interactive main menu and return selected action."""
        from satorineuron import VERSION
        # Clear screen and show logo
        console_write("\033[2J\033[H")
        console_print()
        console_print("           @@@@")
        console_print("      @@@@@@@@@@@@@@@@@")
        console_print("    @@@@@@@@@@@@@@@@@@@@@@@")
        console_print("  @@@@@@@@      @@@@@  @@@@@@")
        console_print(" @@@@@@@           @@@ @@@@@@")
        console_print("@@@@@@               @@@@@@@@@")
        console_print("@@@@@                  @@@@@@@")
        console_print("@@@@                    @@@@@@")
        console_print("@@@@                     @@@@@")
        console_print("@@@@           @         @@@@@")
        console_print("@@@@         @@@@@       @@@@@")
        console_print("@@@@@        @@@@@      @@@@@@")
        console_print(" @@@@@    @@@@@@@@@    @@@@@")
        console_print("  @@@@@@  @@@@@@@@@   @@@@@")
        console_print("   @@@@@@ @@@@@@@@@ @@@@@@")
        console_print("     @@@@@@@@@@@@@@@@@@@@")
        console_print("        @@@@@@@@@@@@@@")
        console_print()
        console_print(f"    Satori Neuron CLI - {VERSION}")
        console_print()

        # Check P2P status
        p2p_status = "Available" if self._is_p2p_available() else "Not Available"
        console_print(f"    P2P Status: {p2p_status}")
        console_print()

        options = [
            {'label': 'Check Balance', 'action': 'balance'},
            {'label': 'Send Transaction', 'action': 'send'},
            {'label': 'P2P Network', 'action': 'p2p'},
            {'label': 'Streams', 'action': 'streams'},
            {'label': 'Rewards & Referrals', 'action': 'rewards'},
            {'label': 'Treasury Donation', 'action': 'donate'},
            {'label': 'Lending & Delegation', 'action': 'lending'},
            {'label': 'Oracle & Predictions', 'action': 'oracle'},
            {'label': 'Vault Status', 'action': 'vault-status'},
            {'label': 'Clear Screen', 'action': 'clear'},
            {'label': 'Exit', 'action': 'exit'}
        ]

        selected = self.interactive_menu("Main Menu", options)

        if selected is None:
            return None

        return options[selected]['action']

    def run(self):
        """Run interactive CLI loop."""
        console_print()
        console_print("           @@@@")
        console_print("      @@@@@@@@@@@@@@@@@")
        console_print("    @@@@@@@@@@@@@@@@@@@@@@@")
        console_print("  @@@@@@@@      @@@@@  @@@@@@")
        console_print(" @@@@@@@           @@@ @@@@@@")
        console_print("@@@@@@               @@@@@@@@@")
        console_print("@@@@@                  @@@@@@@")
        console_print("@@@@                    @@@@@@")
        console_print("@@@@                     @@@@@")
        console_print("@@@@           @         @@@@@")
        console_print("@@@@         @@@@@       @@@@@")
        console_print("@@@@@        @@@@@      @@@@@@")
        console_print(" @@@@@    @@@@@@@@@    @@@@@")
        console_print("  @@@@@@  @@@@@@@@@   @@@@@")
        console_print("   @@@@@@ @@@@@@@@@ @@@@@@")
        console_print("     @@@@@@@@@@@@@@@@@@@@")
        console_print("        @@@@@@@@@@@@@@")
        from satorineuron import VERSION
        console_print()
        console_print("    Satori Neuron CLI")
        console_print(f"    Full P2P Node Manager - {VERSION}")
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

        # Initialize wallet manager after vault is unlocked
        try:
            from satorineuron.init.wallet import WalletManager
            self.wallet_manager = WalletManager.create(useConfigPassword=False)

            # Unlock vault with stored password
            if self._vault_password:
                vault = self.wallet_manager.openVault(password=self._vault_password)
                if vault and hasattr(vault, 'isDecrypted') and vault.isDecrypted:
                    console_print()  # Just add blank line after vault unlock
                else:
                    console_print("Warning: Could not unlock vault in wallet manager.")
                    console_print()
        except Exception as e:
            console_print(f"Error initializing wallet manager: {e}")
            console_print("Exiting.")
            return

        while True:
            try:
                # Show interactive main menu
                action = self.show_main_menu()

                if action is None:
                    continue

                if action == 'exit':
                    console_print("Goodbye!")
                    break

                # Execute the selected action
                if action == 'balance':
                    response = self.handle_command("/balance")
                elif action == 'send':
                    # Show submenu for wallet vs vault
                    send_options = [
                        {'label': 'From Wallet', 'action': 'wallet'},
                        {'label': 'From Vault', 'action': 'vault'},
                        {'label': 'Back to Main Menu', 'action': 'back'}
                    ]
                    send_selected = self.interactive_menu("Send Transaction", send_options)

                    if send_selected is None:
                        continue

                    send_action = send_options[send_selected]['action']
                    if send_action == 'back':
                        continue
                    elif send_action == 'wallet':
                        response = self.send_transaction_wallet()
                    elif send_action == 'vault':
                        response = self.send_transaction_vault()
                    else:
                        response = "Unknown send action"
                elif action == 'p2p':
                    response = self.p2p_menu()
                elif action == 'streams':
                    response = self.streams_menu()
                elif action == 'rewards':
                    response = self.rewards_menu()
                elif action == 'donate':
                    response = self.donation_menu()
                elif action == 'lending':
                    response = self.lending_menu()
                elif action == 'oracle':
                    response = self.oracle_menu()
                elif action == 'vault-status':
                    response = self.handle_command("/vault-status")
                elif action == 'clear':
                    response = self.handle_command("/clear")
                    continue  # Don't print response for clear
                else:
                    response = "Unknown action"

                if response:
                    console_print(response)
                    console_print()
                    console_print("Press Enter to continue...")
                    console_readline()

            except KeyboardInterrupt:
                console_print("\nGoodbye!")
                break
            except EOFError:
                console_print("\nGoodbye!")
                break


if __name__ == "__main__":
    cli = NeuronCLI(env=os.environ.get('SATORI_ENV', 'prod'), runMode='worker')
    cli.run()
