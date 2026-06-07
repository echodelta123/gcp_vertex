"""
Shared test configuration.
Ensures the project root is on sys.path so demo modules can be imported cleanly.
"""
import sys
from pathlib import Path

# Add project root to path (one level up from tests/)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
