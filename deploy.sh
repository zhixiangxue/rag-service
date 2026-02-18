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
echo -e "${GREEN}[1/6] Cloning rag-service repository...${NC}"
if [ -d "rag-service" ]; then
    echo -e "${YELLOW}[WARN] Directory 'rag-service' already exists. Skipping clone.${NC}"
    cd rag-service
else
    git clone https://github.com/zhixiangxue/rag-service.git
    cd rag-service
fi
echo ""

# Step 2: Install Python 3.12
echo -e "${GREEN}[2/6] Checking Python 3.12 installation...${NC}"
if command -v python3.12 &> /dev/null; then
    echo "Python 3.12 already installed: $(python3.12 --version)"
    
    # Check if venv module is available
    if ! python3.12 -m venv --help &> /dev/null; then
        echo -e "${YELLOW}[WARN] python3.12-venv not installed. Installing...${NC}"
        
        # Detect OS
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        fi
        
        case $OS in
            ubuntu|debian)
                sudo apt-get update
                sudo apt-get install -y python3.12-venv
                ;;
            *)
                echo -e "${RED}[ERROR] Cannot install venv module automatically.${NC}"
                echo "Please run: sudo apt-get install -y python3.12-venv"
                exit 1
                ;;
        esac
    else
        echo "python3.12-venv already available"
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
echo -e "${GREEN}[3/6] Creating virtual environment...${NC}"
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
echo -e "${GREEN}[4/6] Installing zag-ai from GitHub...${NC}"
pip install "zagpy[all] @ git+https://github.com/zhixiangxue/zag-ai.git"
echo ""

# Step 5: Install chak-ai from GitHub
echo -e "${GREEN}[5/6] Installing chak-ai from GitHub...${NC}"
pip install "chak-ai @ git+https://github.com/zhixiangxue/chak-ai.git"
echo ""

# Step 6: Install rag-service dependencies
echo -e "${GREEN}[6/6] Installing rag-service dependencies...${NC}"

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

# Step 8: Setup .env file
echo -e "${GREEN}[Setup] Creating .env file...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}[INFO] Created .env from .env.example${NC}"
        echo -e "${YELLOW}[ACTION REQUIRED] Please edit .env file with your configuration!${NC}"
    else
        echo -e "${YELLOW}[WARN] No .env.example found. Please create .env manually.${NC}"
    fi
else
    echo ".env file already exists. Skipping."
fi
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
echo "  5. Start Worker (in another terminal):"
echo "     cd worker"
echo "     python daemon.py"
echo ""
echo "=========================================="
