#!/bin/bash
# Quick verification script for GROUNDEEP refactored pipeline

echo "========================================================================"
echo "GROUNDEEP Refactored Pipeline - Installation Check"
echo "========================================================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
python --version 2>&1 | grep -q "Python 3\.[8-9]\|Python 3\.1[0-9]"
if [ $? -eq 0 ]; then
    echo "   ✅ Python version OK: $(python --version)"
else
    echo "   ⚠️  Python 3.8+ recommended, found: $(python --version)"
fi
echo ""

# Check directory structure
echo "2. Checking directory structure..."
DIRS=(
    "groundeep"
    "groundeep/core"
    "groundeep/analysis"
    "groundeep/utils"
    "groundeep/config"
    "examples"
)

for dir in "${DIRS[@]}"; do
    if [ -d "/home/student/Desktop/Groundeep/$dir" ]; then
        echo "   ✅ $dir/"
    else
        echo "   ❌ $dir/ NOT FOUND"
    fi
done
echo ""

# Check key files
echo "3. Checking key Python files..."
FILES=(
    "groundeep/__init__.py"
    "groundeep/pipeline.py"
    "groundeep/core/interfaces.py"
    "groundeep/analysis/probes.py"
    "groundeep/analysis/geometry.py"
    "groundeep/analysis/scaling.py"
    "groundeep/analysis/cka.py"
    "examples/run_pipeline_simple.py"
)

for file in "${FILES[@]}"; do
    if [ -f "/home/student/Desktop/Groundeep/$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file NOT FOUND"
    fi
done
echo ""

# Check documentation
echo "4. Checking documentation..."
DOCS=(
    "GROUNDEEP_REFACTORED_README.md"
    "REFACTORING_SUMMARY.md"
    "QUICKSTART.md"
    "STRUCTURE.txt"
    "REFACTORING_COMPLETE.txt"
)

for doc in "${DOCS[@]}"; do
    if [ -f "/home/student/Desktop/Groundeep/$doc" ]; then
        echo "   ✅ $doc"
    else
        echo "   ❌ $doc NOT FOUND"
    fi
done
echo ""

# Test import
echo "5. Testing Python imports..."
cd /home/student/Desktop/Groundeep
python -c "from groundeep.pipeline import AnalysisPipeline" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Can import AnalysisPipeline"
else
    echo "   ❌ Import failed - check Python path or dependencies"
fi

python -c "from groundeep.core.interfaces import BaseAnalysis, ModelContext, AnalysisResult" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Can import core interfaces"
else
    echo "   ❌ Import failed"
fi

python -c "from groundeep.analysis.geometry import GeometricAnalysis" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Can import analysis modules"
else
    echo "   ❌ Import failed"
fi
echo ""

# Check dependencies
echo "6. Checking Python dependencies..."
DEPS=("numpy" "torch" "sklearn" "scipy" "matplotlib" "yaml")

for dep in "${DEPS[@]}"; do
    python -c "import ${dep}" 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ $dep"
    else
        echo "   ❌ $dep NOT INSTALLED"
    fi
done
echo ""

# Optional dependencies
echo "7. Checking optional dependencies..."
python -c "import wandb" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ wandb (optional)"
else
    echo "   ⚠️  wandb not installed (optional, for W&B logging)"
fi

python -c "import umap" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ umap (optional)"
else
    echo "   ⚠️  umap not installed (optional, for UMAP dimensionality reduction)"
fi
echo ""

# Summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""
echo "✅ = OK  |  ⚠️  = Warning (non-critical)  |  ❌ = Error (needs fixing)"
echo ""
echo "Next steps:"
echo "  1. Fix any ❌ errors above"
echo "  2. Read QUICKSTART.md to get started"
echo "  3. Run: python examples/run_pipeline_simple.py"
echo ""
echo "========================================================================"
