# tests/conftest.py

import os
import sys

# Project root is one level up from tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add project root to sys.path
sys.path.insert(0, PROJECT_ROOT)
