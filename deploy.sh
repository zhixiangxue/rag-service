#!/bin/bash

set -e  # Exit on any error

echo "=========================================="
echo "RAG Service Deployment Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 0: Check if script is run as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}[ERROR] Please do not run this script as root${NC}"
    exit 1
fi

# Step 1: Clone repository
echo -e "${GREEN}[1/8] Cloning rag-service repository...${NC}"
if [ -d "rag-service" ]; then
    echo -e "${YELLOW}[WARN] Directory 'rag-service' already exists. Skipping clone.${NC}"
    cd rag-service
else
    git clone https://github.com/zhixiangxue/rag-service.git
    cd rag-service
fi
echo ""

# Step 1.5: Install and configure tmux
echo -e "${GREEN}[1.5/8] Installing and configuring tmux...${NC}"
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    # Detect OS
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

# Configure tmux for better UX
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
echo -e "${GREEN}[2/8] Checking Python 3.12 installation...${NC}"
if command -v python3.12 &> /dev/null; then
    echo "Python 3.12 already installed: $(python3.12 --version)"
    
    # Check if venv module is available by testing actual venv creation
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
        
        # Detect OS
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        fi
        
        case $OS in
            ubuntu|debian)
                # Try 1: Direct installation
                echo "Attempt 1: Installing python3.12-venv via apt..."
                sudo apt-get update
                if sudo apt-get install -y python3.12-venv python3.12-dev; then
                    echo -e "${GREEN}[SUCCESS] python3.12-venv installed${NC}"
                    VENV_WORKS=true
                else
                    # Try 2: Add deadsnakes PPA and retry
                    echo -e "${YELLOW}[WARN] Failed. Attempt 2: Adding deadsnakes PPA...${NC}"
                    sudo apt-get install -y software-properties-common
                    sudo add-apt-repository -y ppa:deadsnakes/ppa
                    sudo apt-get update
                    
                    if sudo apt-get install -y python3.12-venv python3.12-dev; then
                        echo -e "${GREEN}[SUCCESS] python3.12-venv installed${NC}"
                        VENV_WORKS=true
                    else
                        # Try 3: Use ensurepip fallback
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
    
    # Final verification
    if [ "$VENV_WORKS" = false ]; then
        echo -e "${RED}[ERROR] Failed to setup venv functionality${NC}"
        echo "Please manually run: sudo apt-get install -y python3.12-venv python3.12-dev"
        exit 1
    fi
else
    echo "Installing Python 3.12..."
    
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        echo -e "${RED}[ERROR] Cannot detect OS. Please install Python 3.12 manually.${NC}"
        exit 1
    fi
    
    # Install based on OS
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
            
            # Build Python 3.12 from source
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
echo -e "${GREEN}[3/8] Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}[WARN] Virtual environment already exists. Removing old one...${NC}"
    rm -rf .venv
fi

python3.12 -m venv .venv
source .venv/bin/activate

# Verify Python version in venv
PYTHON_VERSION=$(python --version)
echo "Using: $PYTHON_VERSION"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Step 4: Install zag-ai from GitHub
echo -e "${GREEN}[4/8] Installing zag-ai from GitHub...${NC}"
pip install "zagpy[all] @ git+https://github.com/zhixiangxue/zag-ai.git"
echo ""

# Step 5: Install chak-ai from GitHub
echo -e "${GREEN}[5/8] Installing chak-ai from GitHub...${NC}"
pip install "chakpy @ git+https://github.com/zhixiangxue/chak-ai.git"
echo ""

# Step 6: Install rag-service dependencies
echo -e "${GREEN}[6/8] Installing rag-service dependencies...${NC}"

# Install from pyproject.toml (includes core dependencies)
if [ -f "pyproject.toml" ]; then
    echo "Installing dependencies from pyproject.toml..."
    pip install -e .
fi

# Install additional dependencies from requirements.txt (if any extras)
if [ -f "requirements.txt" ]; then
    echo "Installing additional dependencies from requirements.txt..."
    # Filter out zag-ai line (already installed)
    grep -v "git+https://github.com/zhixiangxue/zag-ai.git" requirements.txt > /tmp/requirements_filtered.txt || true
    if [ -s /tmp/requirements_filtered.txt ]; then
        pip install -r /tmp/requirements_filtered.txt
    fi
    rm -f /tmp/requirements_filtered.txt
fi

echo ""

# Step 7: Verify installation
echo -e "${GREEN}[Verification] Checking installed packages...${NC}"
echo "----------------------------------------"
pip list | grep -E "(zagpy|chak-ai|fastapi|uvicorn)" || true
echo "----------------------------------------"
echo ""

# Step 7: Check GPU availability
echo -e "${GREEN}[7/8] Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo -e "${YELLOW}[WARN] NVIDIA GPU not detected or drivers not installed${NC}"
    echo ""
    echo -e "${YELLOW}For GPU support, you need to install NVIDIA drivers and CUDA Toolkit.${NC}"
    echo -e "${YELLOW}This requires system reboot after installation.${NC}"
    echo ""
    echo -e "${YELLOW}Installation commands (Ubuntu/Debian):${NC}"
    echo ""
    echo "  # 1. Update package list"
    echo "  sudo apt update"
    echo ""
    echo "  # 2. Install NVIDIA driver (replace 550 with your preferred version)"
    echo "  sudo apt install -y nvidia-driver-550"
    echo ""
    echo "  # 3. Reboot system (REQUIRED)"
    echo "  sudo reboot"
    echo ""
    echo "  # 4. After reboot, verify GPU is working"
    echo "  nvidia-smi"
    echo ""
    echo -e "${YELLOW}Alternative: Use AWS Deep Learning AMI (recommended, drivers pre-installed):${NC}"
    echo "  AMI: Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)"
    echo ""
    echo "Continuing with CPU-only mode for now..."
    echo ""
fi
echo ""

# Step 8: Configuration reminder
echo -e "${GREEN}[8/8] Configuration reminder...${NC}"
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

# Step 9: Display next steps
echo "=========================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Configure .env file:"
echo "     nano .env"
echo ""
echo "  3. Initialize database (if needed):"
echo "     python -c 'from app.database import init_db; init_db()'"
echo ""
echo "  4. Start API server:"
echo "     python -m app.main"
echo ""
echo "  5. Start Worker (in tmux session, recommended):"
echo "     tmux new -s worker"
echo "     ./scripts/start_worker.sh"
echo ""
echo "=========================================="
