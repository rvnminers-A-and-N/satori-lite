# =============================================================================
# Satori Lite - Lightweight Self-Contained Neuron
# =============================================================================
#
# This container runs a complete Satori neuron including:
# - Lib (core library)
# - Engine (AI/ML predictions)
# - Neuron (coordinator + web UI)
# - Streams (oracle data sources)
# - P2P networking (satorip2p)
#
# NETWORKING MODES:
#   Set SATORI_NETWORKING_MODE environment variable:
#   - central: Legacy mode, connects to central server (default)
#   - hybrid:  P2P with central fallback (recommended)
#   - p2p:     Pure P2P, fully decentralized
#
# PORTS:
#   - 24601: Web UI
#   - 24600: P2P networking
#
# USAGE:
#   # Build
#   docker build -t satori-lite .
#
#   # Run in hybrid mode (recommended)
#   docker run -p 24601:24601 -p 24600:24600 \
#       -e SATORI_NETWORKING_MODE=hybrid \
#       -v ~/.satori:/root/.satori \
#       satori-lite
#
# =============================================================================

FROM python:3.12-slim

LABEL maintainer="Satori Network"
LABEL description="Satori Lite - Self-contained lightweight neuron"
LABEL version="1.0.0"

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libleveldb-dev \
        libgmp-dev \
        git \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# Directory Structure
# =============================================================================

RUN mkdir -p /Satori/Lib \
             /Satori/Engine \
             /Satori/Neuron \
             /Satori/Streams \
             /Satori/web \
             /root/.satori/wallet \
             /root/.satori/data \
             /root/.satori/models

# =============================================================================
# Copy Application Code
# =============================================================================

# Core components
COPY lib-lite /Satori/Lib
COPY engine-lite /Satori/Engine
COPY neuron-lite /Satori/Neuron
COPY web /Satori/web

# Streams (oracle data sources)
COPY streams-lite /Satori/Streams

# Tests
COPY tests /Satori/tests

# =============================================================================
# Python Dependencies
# =============================================================================

# Install requirements first
WORKDIR /Satori
COPY requirements.txt /Satori/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    grep -v "satorip2p" /Satori/requirements.txt > /tmp/requirements-filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements-filtered.txt && \
    pip install pytest

# Install satorip2p from GitHub
# TODO: Update to SatoriNetwork/satorip2p once merged
RUN pip install git+https://github.com/rvnminers-A-and-N/satorip2p.git@main && \
    pip uninstall py-multihash -y 2>/dev/null || true && \
    pip install pymultihash==0.8.2 --force-reinstall

# =============================================================================
# Environment Configuration
# =============================================================================

# Python path - include all components
ENV PYTHONPATH="/Satori/Lib:/Satori/Neuron:/Satori/Engine:/Satori/Streams:/Satori"

# Networking mode is read from config file (/Satori/Neuron/config/config.yaml)
# Can be overridden at runtime with -e SATORI_NETWORKING_MODE=hybrid
# Options: central, hybrid, p2p
# ENV SATORI_NETWORKING_MODE is intentionally NOT set here to allow config file control

# Wallet path
ENV SATORI_WALLET_PATH="/root/.satori/wallet"

# Data paths
ENV SATORI_DATA_PATH="/root/.satori/data"
ENV SATORI_MODELS_PATH="/root/.satori/models"

# =============================================================================
# Default Configuration
# =============================================================================

# Copy config template as default config (hybrid mode enabled by default)
RUN cp /Satori/Neuron/config/config.yaml.template /Satori/Neuron/config/config.yaml

# =============================================================================
# Symbolic Links for Compatibility
# =============================================================================

RUN rm -rf /Satori/Neuron/data /Satori/Neuron/models && \
    ln -s /Satori/Engine/db /Satori/Neuron/data && \
    ln -s /root/.satori/models /Satori/Neuron/models

# =============================================================================
# Make Scripts Executable
# =============================================================================

RUN chmod +x /Satori/Neuron/satorineuron/web/start.sh 2>/dev/null || true

# =============================================================================
# Working Directory & Ports
# =============================================================================

WORKDIR /Satori

# Web UI
EXPOSE 24601

# P2P networking
EXPOSE 24600

# =============================================================================
# Health Check
# =============================================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:24601/health || exit 1

# =============================================================================
# Default Command
# =============================================================================

# Start the neuron (includes web UI, P2P, and streams)
CMD ["python", "/Satori/Neuron/start.py"]
