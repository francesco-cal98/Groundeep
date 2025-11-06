#!/bin/bash
# GROUNDEEP Installation Script

set -e  # Exit on error

echo "========================================================================"
echo "GROUNDEEP Installation Script"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if version is >= 3.8
python3 -c "import sys; assert sys.version_info >= (3, 8), 'Python >= 3.8 required'" || {
    echo -e "${RED}Error: Python 3.8 or higher is required${NC}"
    exit 1
}
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check if virtual environment exists
if [ -d "groundeep" ]; then
    echo -e "${YELLOW}Virtual environment 'groundeep' already exists.${NC}"
    read -p "Do you want to recreate it? (y/N): " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        echo "Removing existing environment..."
        rm -rf groundeep
    else
        echo "Using existing environment..."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "groundeep" ]; then
    echo "Creating virtual environment..."
    python3 -m venv groundeep
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source groundeep/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

echo ""
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Install package in development mode
echo ""
echo "Installing GROUNDEEP in development mode..."
pip install -e .

echo ""
echo -e "${GREEN}✓ GROUNDEEP installed${NC}"

# Run tests
echo ""
echo "========================================================================"
echo "Running tests..."
echo "========================================================================"
python3 test_pipeline.py

echo ""
echo "========================================================================"
echo "Installation complete!"
echo "========================================================================"
echo ""
echo "To use GROUNDEEP:"
echo "  1. Activate environment:  source groundeep/bin/activate"
echo "  2. Configure analysis:    edit src/configs/analysis.yaml"
echo "  3. Run analysis:          python src/main_scripts/analyze_modular.py"
echo ""
echo "For more information, see README.md"
echo ""
