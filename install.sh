#!/bin/bash
set -e

echo "Installing Satori Lite..."

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running. Please start Docker."
    exit 1
fi

# Pull the image
echo "Pulling Satori Lite image..."
docker pull satorinet/satori-lite:latest

# Stop and remove existing container if exists
if docker ps -a --format '{{.Names}}' | grep -q '^satori$'; then
    echo "Removing existing satori container..."
    docker stop satori 2>/dev/null || true
    docker rm satori 2>/dev/null || true
fi

# Start the container
echo "Starting Satori neuron..."
docker run -d --name satori \
    --restart unless-stopped \
    -v satori-data:/Satori/Neuron/data \
    satorinet/satori-lite:latest

# Install CLI command
echo "Installing satori command..."
INSTALL_DIR="/usr/local/bin"
if [ -w "$INSTALL_DIR" ]; then
    curl -fsSL https://raw.githubusercontent.com/SatoriNetwork/satori-lite/main/satori -o "$INSTALL_DIR/satori"
    chmod +x "$INSTALL_DIR/satori"
else
    sudo curl -fsSL https://raw.githubusercontent.com/SatoriNetwork/satori-lite/main/satori -o "$INSTALL_DIR/satori"
    sudo chmod +x "$INSTALL_DIR/satori"
fi

echo ""
echo "Satori Lite installed successfully!"
echo ""
echo "Usage:"
echo "  satori          - Enter the CLI"
echo "  satori --help   - Show help"
echo ""
echo "Run 'satori' to get started."
