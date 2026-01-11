from .tag import LatestTag, Version
from .wallet import WalletManager

# Import start module from parent package for compatibility with web routes
import sys
import os
# Add neuron-lite to path so we can import start.py
_neuron_lite_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _neuron_lite_path not in sys.path:
    sys.path.insert(0, _neuron_lite_path)

try:
    import start
except ImportError:
    start = None
