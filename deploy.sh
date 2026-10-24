#!/usr/bin/env bash
# ==============================================================================
# UPI Spending Anomaly & Subscription Trap Detector - Automated Deploy Script
# For Ubuntu 22.04 / 24.04 on Azure Linux VM
# ==============================================================================

set -euo pipefail

echo "=================================================="
echo "🚀 Starting Automated Deployment on Azure VM"
echo "=================================================="

# 1. Setup 2GB Swap to prevent out-of-memory during builds on B1s/B2s VMs
if [ ! -f /swapfile ]; then
    echo "⚙️ Creating 2GB swap space for smooth build..."
    sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "✅ Swap enabled."
fi

# 2. Update package list and install Docker + Git
echo "⚙️ Updating system packages and installing Docker..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release git

if ! command -v docker &> /dev/null; then
    echo "Installing Docker & Docker Compose..."
    sudo apt-get install -y docker.io docker-compose-v2
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER" || true
fi

# 3. Navigate to application directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/work" ]; then
    APP_DIR="$SCRIPT_DIR/work"
else
    APP_DIR="$SCRIPT_DIR"
fi

cd "$APP_DIR"
echo "📂 Working directory: $(pwd)"

# 4. Build and start all services via Docker Compose
echo "⚙️ Building and launching Docker containers..."
sudo docker compose down --remove-orphans || true
sudo docker compose up -d --build

echo "=================================================="
echo "🎉 SUCCESS: All containers deployed and running!"
echo "=================================================="
echo ""
echo "🌐 Frontend Dashboard: http://$(curl -s ifconfig.me)/"
echo "🔌 Backend API:        http://$(curl -s ifconfig.me)/api/"
echo "📚 Swagger Docs:       http://$(curl -s ifconfig.me):8000/docs"
echo "📊 MLflow UI:          http://$(curl -s ifconfig.me):5000/"
echo ""
echo "To check logs anytime: sudo docker compose -f $APP_DIR/docker-compose.yml logs -f"
echo "=================================================="
