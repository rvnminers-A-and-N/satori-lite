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

# Save original file descriptors BEFORE any redirection
_original_stdout_fd = os.dup(1)
_original_stderr_fd = os.dup(2)
_original_stdin_fd = os.dup(0)

# Create file objects from the duplicated fds
_console_out = os.fdopen(_original_stderr_fd, 'w', buffering=1)
_console_in = os.fdopen(_original_stdin_fd, 'r', buffering=1)

# Global log buffer
_log_buffer: list[str] = []
_max_logs = 1000


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


class NeuronCLI:
    """Interactive CLI for Satori Neuron."""

    def __init__(self, env: str = 'prod', runMode: str = 'worker'):
        self.env = env
        self.runMode = runMode
        self.startup = None
        self.neuron_started = False

    def add_log(self, message: str):
        """Add a log message to buffer."""
        timestamp = time.strftime("%H:%M:%S")
        _log_buffer.append(f"[{timestamp}] {message}")

    def start_neuron_background(self):
        """Start the neuron in a background thread."""
        def run():
            try:
                # Import here after log capture is set up
                from start import StartupDag
                self.add_log("Creating StartupDag...")
                self.startup = StartupDag.create(
                    env=self.env,
                    runMode=self.runMode)
                self.neuron_started = True
                self.add_log("Neuron started successfully")
            except Exception as e:
                self.add_log(f"Neuron startup error: {e}")
                import traceback
                self.add_log(traceback.format_exc())

        self.add_log("Starting neuron in background...")
        threading.Thread(target=run, daemon=True).start()

    def handle_command(self, user_input: str) -> str | None:
        """Process user commands and return response."""
        user_input = user_input.strip()

        if user_input == "/help":
            return """Available commands:
  /neuron-logs - Show neuron logs
  /status     - Show current status
  /balance    - Show wallet balance
  /streams    - Show stream assignments
  /pause      - Pause the engine
  /unpause    - Unpause the engine
  /restart    - Restart the neuron
  /stake      - Check stake status
  /pool       - Show pool status
  /clear      - Clear the screen
  /help       - Show this help message
  /exit       - Exit CLI (neuron keeps running)"""

        elif user_input == "/neuron-logs":
            logs = _log_buffer[-50:]
            if not logs:
                return "No logs yet."
            return "\n".join(logs)

        elif user_input == "/status":
            if not self.neuron_started:
                return "Neuron is starting... Use /neuron-logs to see progress."
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
            if not self.neuron_started:
                return "Neuron is starting... Use /neuron-logs to see progress."
            balance_lines = [
                f"Holding Balance: {self.startup.holdingBalance}",
                f"Server Balance: {self.startup.getBalance('currency')}",
            ]
            if self.startup.balances:
                for k, v in self.startup.balances.items():
                    balance_lines.append(f"  {k}: {v}")
            return "\n".join(balance_lines)

        elif user_input == "/streams":
            if not self.neuron_started:
                return "Neuron is starting... Use /neuron-logs to see progress."
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
                return "Neuron is starting... Use /neuron-logs to see progress."
            self.startup.pause()
            return "Engine paused"

        elif user_input == "/unpause":
            if not self.neuron_started:
                return "Neuron is starting... Use /neuron-logs to see progress."
            self.startup.unpause()
            return "Engine unpaused"

        elif user_input == "/restart":
            if not self.neuron_started:
                return "Neuron is starting... Use /neuron-logs to see progress."
            self.startup.triggerRestart()

        elif user_input == "/stake":
            if not self.neuron_started:
                return "Neuron is starting... Use /neuron-logs to see progress."
            status = self.startup.performStakeCheck()
            return f"Stake Status: {status}\nStake Required: {self.startup.stakeRequired}"

        elif user_input == "/pool":
            if not self.neuron_started:
                return "Neuron is starting... Use /neuron-logs to see progress."
            return f"Pool Accepting: {self.startup.poolIsAccepting}"

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

    def run(self):
        """Run interactive CLI loop."""
        console_print("Satori Neuron CLI. Type /help for commands, /exit to quit.")
        console_print()

        # Start neuron in background
        self.start_neuron_background()

        while True:
            try:
                console_write("> ")
                user_input = console_readline().strip()

                if not user_input:
                    continue

                response = self.handle_command(user_input)

                if response == "EXIT_CLI":
                    console_print("Goodbye!")
                    break

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
