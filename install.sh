#!/usr/bin/env bash
# =============================================================================
# MPMC Profiling Suite — Dependency Installer
# =============================================================================
# Detects the active PyTorch version and CUDA version at runtime, then installs
# the correct ABI-coupled wheels for torch-cluster, torch-scatter, torch-sparse,
# and torch-geometric from the official PyG wheel index.
#
# Usage (run once on the target VM before executing stress_test.py):
#   chmod +x install.sh && ./install.sh
#
# Requirements:
#   - Python with PyTorch already installed (CPU or CUDA build).
#   - pip accessible in the current environment.
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# 1. Detect Python and PyTorch/CUDA version pair
# --------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"

echo ""
echo "============================================================"
echo " MPMC Phase 1 — Dependency Installer"
echo "============================================================"
echo ""

echo "[1/5] Detecting Python executable..."
PYTHON_VERSION=$("$PYTHON" --version 2>&1)
echo "      Found: $PYTHON_VERSION"

echo ""
echo "[2/5] Detecting PyTorch and CUDA versions..."
TORCH_VERSION=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || true)
CUDA_VERSION=$("$PYTHON"  -c "import torch; print(torch.version.cuda if torch.version.cuda else 'cpu')" 2>/dev/null || true)

if [ -z "$TORCH_VERSION" ]; then
    echo ""
    echo "  ERROR: PyTorch is not importable in the current Python environment."
    echo "  Please install PyTorch first: https://pytorch.org/get-started/locally/"
    exit 1
fi

echo "      PyTorch version : $TORCH_VERSION"
echo "      CUDA version    : $CUDA_VERSION"

# --------------------------------------------------------------------------
# 2. Build the PyG wheel index URL
#    Format: https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
#    Examples:
#      torch-2.1.0+cu121 → https://data.pyg.org/whl/torch-2.1.0+cu121.html
#      torch-1.9.0+cpu   → https://data.pyg.org/whl/torch-1.9.0+cpu.html
# --------------------------------------------------------------------------
echo ""
echo "[3/5] Building PyG wheel index URL..."

# Strip any local suffix like "+cu121" that may already be embedded in
# torch.__version__ (e.g. "2.1.0+cu121") — we reconstruct it cleanly.
TORCH_BASE=$("$PYTHON" -c "
import torch, re
v = torch.__version__
# Remove any existing +suffix so we can reconstruct
v = re.split(r'[+]', v)[0]
print(v)
")

if [ "$CUDA_VERSION" = "cpu" ] || [ -z "$CUDA_VERSION" ]; then
    CUDA_TAG="cpu"
else
    # Convert "12.1" → "cu121", "11.8" → "cu118", etc.
    CUDA_TAG=$("$PYTHON" -c "
v = '${CUDA_VERSION}'.replace('.','')
# Guard against already-formatted tags like '121'
if len(v) <= 3:
    print('cu' + v)
else:
    print('cu' + v[:3])
")
fi

WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_BASE}+${CUDA_TAG}.html"
echo "      Wheel index URL: $WHEEL_URL"

# --------------------------------------------------------------------------
# 3. Install core C++ extension packages from PyG index
# --------------------------------------------------------------------------
echo ""
echo "[4/5] Installing torch-scatter, torch-sparse, torch-cluster..."
echo "      (This may take several minutes if building from source.)"
echo ""

"$PYTHON" -m pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    --find-links "$WHEEL_URL" \
    --no-cache-dir \
    -v

# --------------------------------------------------------------------------
# 4. Install torch-geometric (pure Python; no ABI coupling)
# --------------------------------------------------------------------------
echo ""
echo "[5/5] Installing torch-geometric and profiling dependencies..."
"$PYTHON" -m pip install \
    torch-geometric \
    matplotlib \
    numpy \
    --quiet

# --------------------------------------------------------------------------
# 5. Verification
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Verification"
echo "============================================================"
"$PYTHON" -c "
import sys
results = []
packages = [
    ('torch',          'torch'),
    ('torch_cluster',  'torch_cluster'),
    ('torch_scatter',  'torch_scatter'),
    ('torch_sparse',   'torch_sparse'),
    ('torch_geometric','torch_geometric'),
    ('matplotlib',     'matplotlib'),
]
for label, mod in packages:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'unknown')
        results.append(f'  OK  {label:<20} {ver}')
    except ImportError as e:
        results.append(f'  FAIL {label:<20} {e}')

for r in results:
    print(r)

# Confirm radius_graph is importable
try:
    from torch_cluster import radius_graph
    print('  OK  radius_graph importable')
except Exception as e:
    print(f'  FAIL radius_graph: {e}')
    sys.exit(1)

# CUDA availability
import torch
if torch.cuda.is_available():
    print(f'  OK  CUDA available — {torch.cuda.get_device_name(0)}')
    print(f'       VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('  WARN CUDA not available — profiling will run on CPU')
"

echo ""
echo "All dependencies installed. You can now run the stress test:"
echo ""
echo "  cd MPMC"
echo "  python stress_test.py --smoke          # quick validation (~60 s)"
echo "  python stress_test.py                  # full GPU stress sweep"
echo ""
