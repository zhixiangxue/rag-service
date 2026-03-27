#!/bin/bash
set -e

# ── Argument parsing ──────────────────────────────────────────────────────────
ACTION=""
TARGET="all"
AUTO_YES=false

for arg in "$@"; do
    case $arg in
        --install|--upgrade) ACTION="$arg" ;;
        server|worker|all)   TARGET="$arg" ;;
        --yes|-y)            AUTO_YES=true ;;
        *) echo "Unknown argument: $arg"; usage; exit 1 ;;
    esac
done

usage() {
    echo "Usage: ./deploy.sh <action> [target] [-y]"
    echo ""
    echo "Actions:"
    echo "  --install [server|worker|all]   Fresh install from scratch (default: all)"
    echo "  --upgrade [server|worker|all]   Pull latest code + sync deps (default: all)"
    echo ""
    echo "Targets:"
    echo "  server   API server only (no GPU / MinerU)"
    echo "  worker   Worker only     (GPU + MinerU required)"
    echo "  all      Both (default)"
    echo ""
    echo "Flags:"
    echo "  -y, --yes   Skip confirmation prompts (for non-interactive use)"
}

if [[ "$ACTION" != "--install" && "$ACTION" != "--upgrade" ]]; then
    usage
    exit 1
fi

DEPLOY_SERVER=false
DEPLOY_WORKER=false
[[ "$TARGET" == "server" || "$TARGET" == "all" ]] && DEPLOY_SERVER=true
[[ "$TARGET" == "worker" || "$TARGET" == "all" ]] && DEPLOY_WORKER=true

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "==========================================="
echo "RAG Service Deployment  [$ACTION $TARGET]"
echo "==========================================="
echo ""

# Root check
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}[ERROR] Please do not run this script as root${NC}"
    exit 1
fi


# ════════════════════════════════════════════════════════════════════════════
# UPGRADE mode
# ════════════════════════════════════════════════════════════════════════════
if [[ "$ACTION" == "--upgrade" ]]; then

    # Navigate to rag-service if running from parent directory
    if [ -d "rag-service" ]; then
        cd rag-service
    fi

    if [ ! -f "app/main.py" ]; then
        echo -e "${RED}[ERROR] Cannot locate rag-service. Run from rag-service/ or its parent directory.${NC}"
        exit 1
    fi

    # Step 1: Pull latest code
    echo -e "${GREEN}[1/3] Pulling latest code...${NC}"
    git pull
    if [ -d "../zag-ai" ]; then
        echo "Pulling zag-ai..."
        git -C ../zag-ai pull
    else
        echo -e "${YELLOW}[WARN] ../zag-ai not found, skipping${NC}"
    fi
    echo ""

    # Step 2: Activate venv
    echo -e "${GREEN}[2/3] Activating virtual environment...${NC}"
    if [ ! -f ".venv/bin/activate" ]; then
        echo -e "${RED}[ERROR] .venv not found. Run --install first.${NC}"
        exit 1
    fi
    source .venv/bin/activate
    echo "Using: $(python --version)"
    echo ""

    # Step 3: Sync dependencies
    echo -e "${GREEN}[3/3] Syncing dependencies...${NC}"
    pip install -r requirements.txt
    if [ -d "../zag-ai" ]; then
        pip install -e "../zag-ai[all]"
    fi
    if $DEPLOY_WORKER; then
        echo ""
        echo "Upgrading MinerU..."
        pip install "mineru[all]>=2.7.6" --upgrade
        echo ""
        echo -e "${GREEN}[3.5/3] Verifying document readers...${NC}"
        if python scripts/verify_readers.py; then
            echo -e "${GREEN}[SUCCESS] Readers verified${NC}"
        else
            echo -e "${YELLOW}[WARN] Reader verification had issues. Check output above.${NC}"
        fi
    fi
    echo ""

    # Summary
    echo "==========================================="
    echo -e "${GREEN}Upgrade Complete! [$TARGET]${NC}"
    echo "==========================================="
    echo ""
    echo "Restart services to apply changes:"
    echo ""
    if $DEPLOY_SERVER; then
        echo "  Server:"
        echo "    pkill -f 'gunicorn app.main:app' || true"
        echo "    ./scripts/start_server.sh"
        echo ""
    fi
    if $DEPLOY_WORKER; then
        echo "  Worker:"
        echo "    tmux kill-session -t worker 2>/dev/null || true"
        echo "    tmux new -s worker"
        echo "    ./scripts/start_worker.sh"
        echo ""
    fi
    echo "==========================================="
    exit 0
fi


# ════════════════════════════════════════════════════════════════════════════
# INSTALL mode
# ════════════════════════════════════════════════════════════════════════════

# Wipe existing dirs if present
if [ -d "rag-service" ] || [ -d "zag-ai" ]; then
    echo -e "${YELLOW}[WARN] Existing installation found. --install will wipe and reinstall:${NC}"
    [ -d "rag-service" ] && echo -e "${YELLOW}  ./rag-service${NC}"
    [ -d "zag-ai" ]      && echo -e "${YELLOW}  ./zag-ai${NC}"
    echo ""
    if $AUTO_YES; then
        echo "[-y] Auto-confirming wipe."
    else
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
    [ -d "rag-service" ] && rm -rf rag-service
    [ -d "zag-ai" ]      && rm -rf zag-ai
    echo ""
fi

# Step 1: Clone rag-service
echo -e "${GREEN}[1/9] Cloning rag-service repository...${NC}"
git clone https://github.com/zhixiangxue/rag-service.git
cd rag-service
chmod +x scripts/deploy.sh scripts/start_server.sh scripts/start_worker.sh
echo ""

# Step 1.5: Install and configure tmux
echo -e "${GREEN}[1.5/9] Installing and configuring tmux...${NC}"
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    fi
    case $OS in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y tmux
            ;;
        amzn|rhel|centos|fedora)
            sudo yum install -y tmux
            ;;
        *)
            echo -e "${YELLOW}[WARN] Could not install tmux automatically. Please install manually.${NC}"
            ;;
    esac
else
    echo "tmux already installed: $(tmux -V)"
fi

if [ ! -f ~/.tmux.conf ]; then
    echo "Configuring tmux..."
    cat > ~/.tmux.conf << 'EOF'
# Enable mouse support
set -g mouse on

# Increase history limit
set -g history-limit 10000

# Start window numbering at 1
set -g base-index 1
EOF
    echo "tmux configuration saved to ~/.tmux.conf"
else
    echo "~/.tmux.conf already exists, skipping configuration"
fi
echo ""

# Step 2: Install Python 3.12
echo -e "${GREEN}[2/9] Checking Python 3.12 installation...${NC}"
if command -v python3.12 &> /dev/null; then
    echo "Python 3.12 already installed: $(python3.12 --version)"

    echo "Testing venv functionality..."
    TEST_VENV_DIR="/tmp/test_venv_$$"
    VENV_WORKS=false

    if python3.12 -m venv "$TEST_VENV_DIR" &> /dev/null; then
        VENV_WORKS=true
        rm -rf "$TEST_VENV_DIR"
        echo "python3.12-venv is functional"
    else
        rm -rf "$TEST_VENV_DIR"
        echo -e "${YELLOW}[WARN] python3.12-venv not functional. Installing...${NC}"

        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        fi

        case $OS in
            ubuntu|debian)
                echo "Attempt 1: Installing python3.12-venv via apt..."
                sudo apt-get update
                if sudo apt-get install -y python3.12-venv python3.12-dev; then
                    echo -e "${GREEN}[SUCCESS] python3.12-venv installed${NC}"
                    VENV_WORKS=true
                else
                    echo -e "${YELLOW}[WARN] Failed. Attempt 2: Adding deadsnakes PPA...${NC}"
                    sudo apt-get install -y software-properties-common
                    sudo add-apt-repository -y ppa:deadsnakes/ppa
                    sudo apt-get update
                    if sudo apt-get install -y python3.12-venv python3.12-dev; then
                        echo -e "${GREEN}[SUCCESS] python3.12-venv installed${NC}"
                        VENV_WORKS=true
                    else
                        echo -e "${YELLOW}[WARN] Failed. Attempt 3: Using ensurepip...${NC}"
                        if python3.12 -m ensurepip --upgrade; then
                            echo -e "${GREEN}[SUCCESS] pip initialized via ensurepip${NC}"
                            VENV_WORKS=true
                        fi
                    fi
                fi
                ;;
            *)
                echo -e "${YELLOW}[WARN] Non-Debian system detected. Trying ensurepip...${NC}"
                if python3.12 -m ensurepip --upgrade; then
                    echo -e "${GREEN}[SUCCESS] pip initialized via ensurepip${NC}"
                    VENV_WORKS=true
                fi
                ;;
        esac
    fi

    if [ "$VENV_WORKS" = false ]; then
        echo -e "${RED}[ERROR] Failed to setup venv functionality${NC}"
        echo "Please manually run: sudo apt-get install -y python3.12-venv python3.12-dev"
        exit 1
    fi
else
    echo "Installing Python 3.12..."

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        echo -e "${RED}[ERROR] Cannot detect OS. Please install Python 3.12 manually.${NC}"
        exit 1
    fi

    case $OS in
        ubuntu|debian)
            echo "Detected Ubuntu/Debian"
            sudo apt-get update
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
            ;;
        amzn|rhel|centos|fedora)
            echo "Detected Amazon Linux/RHEL/CentOS/Fedora"
            sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel
            cd /tmp
            wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
            tar xzf Python-3.12.0.tgz
            cd Python-3.12.0
            ./configure --enable-optimizations
            sudo make altinstall
            cd -
            rm -rf /tmp/Python-3.12.0*
            ;;
        *)
            echo -e "${RED}[ERROR] Unsupported OS: $OS${NC}"
            echo "Please install Python 3.12 manually and re-run this script."
            exit 1
            ;;
    esac
fi
echo ""

# Step 3: Create virtual environment
echo -e "${GREEN}[3/9] Creating virtual environment...${NC}"
python3.12 -m venv .venv
source .venv/bin/activate
echo "Using: $(python --version)"
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Step 3.5: Install rag-service dependencies
echo -e "${GREEN}[3.5/9] Installing rag-service dependencies...${NC}"
pip install -r requirements.txt
echo ""

# Step 4: Clone zag-ai and install in editable mode
echo -e "${GREEN}[4/9] Cloning zag-ai and installing in editable mode...${NC}"
git clone https://github.com/zhixiangxue/zag-ai.git ../zag-ai
pip install -e "../zag-ai[all]"
echo ""

# Step 5: Install chak-ai from PyPI
echo -e "${GREEN}[5/9] Installing chak-ai from PyPI...${NC}"
pip install "chakpy[all]"
echo ""

# Step 5.5: Install MinerU (worker only)
if $DEPLOY_WORKER; then
    echo -e "${GREEN}[5.5/9] Installing MinerU...${NC}"
    pip install "mineru[all]>=2.7.6"
    echo ""
fi

# Step 6: Verify installation
echo -e "${GREEN}[6/9] Verifying installed packages...${NC}"
echo "----------------------------------------"
pip list | grep -E "(zagpy|chakpy|fastapi|uvicorn)" || true
echo "----------------------------------------"
echo ""

# Step 7: Check GPU availability (worker only)
if $DEPLOY_WORKER; then
    echo -e "${GREEN}[7/9] Checking GPU availability...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        echo ""
    else
        echo -e "${YELLOW}[WARN] NVIDIA GPU not detected or drivers not installed${NC}"
        echo ""
        echo -e "${YELLOW}Installing NVIDIA drivers for GPU support...${NC}"
        echo -e "${YELLOW}Note: This requires system reboot after installation.${NC}"
        echo ""
        sudo apt-get update
        sudo apt-get install -y nvidia-driver-550
        echo ""
        echo -e "${GREEN}[SUCCESS] NVIDIA driver installed${NC}"
        echo -e "${RED}[ACTION REQUIRED] System reboot is required for GPU to work${NC}"
        echo ""
        echo "After reboot, verify GPU with: nvidia-smi"
        echo ""
        echo "The deployment will continue with CPU-only mode for now..."
        echo ""
    fi
    echo ""
fi

# Step 8: Configuration reminder
echo -e "${GREEN}[8/9] Configuration reminder...${NC}"
echo ""
echo -e "${YELLOW}[ACTION REQUIRED] Please create and configure .env file:${NC}"
echo -e "${YELLOW}  1. Copy from template: cp .env.example .env${NC}"
echo -e "${YELLOW}  2. Edit with your configuration: nano .env${NC}"
echo ""
echo -e "${YELLOW}[IMPORTANT] For distributed deployment (API + Worker on separate machines):${NC}"
echo -e "${YELLOW}  - Set API_HOST=0.0.0.0 (listen on all interfaces)${NC}"
echo -e "${YELLOW}  - Set API_PUBLIC_HOST=<your_server_public_ip> (e.g., 13.56.109.233)${NC}"
echo -e "${YELLOW}  - This allows Worker to download files via HTTP${NC}"
echo ""

# Step 9: Verify readers (worker only)
if $DEPLOY_WORKER; then
    echo -e "${GREEN}[9/9] Verifying document readers...${NC}"
    echo ""
    echo -e "${YELLOW}Running reader verification script...${NC}"
    echo -e "${YELLOW}This will test MinerU and Docling to ensure they work correctly.${NC}"
    echo ""
    if python scripts/verify_readers.py; then
        echo -e "${GREEN}[SUCCESS] All readers verified successfully!${NC}"
    else
        echo -e "${RED}[WARNING] Some readers failed verification${NC}"
        echo -e "${YELLOW}You may still continue, but document processing may not work properly.${NC}"
        echo -e "${YELLOW}Please check the error messages above and fix the issues.${NC}"
        echo ""
        if $AUTO_YES; then
            echo "[-y] Auto-confirming, continuing despite reader issues."
        else
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Deployment aborted."
                exit 1
            fi
        fi
    fi
    echo ""
fi

# Summary
echo "==========================================="
echo -e "${GREEN}Install Complete! [$TARGET]${NC}"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Configure .env file:"
echo "     nano .env"
echo ""
if $DEPLOY_SERVER; then
    echo "  3. Start API server:"
    echo "     ./scripts/start_server.sh"
    echo ""
fi
if $DEPLOY_WORKER; then
    echo "  4. Start Worker (in tmux session):"
    echo "     tmux new -s worker"
    echo "     ./scripts/start_worker.sh"
    echo ""
fi
echo "==========================================="
