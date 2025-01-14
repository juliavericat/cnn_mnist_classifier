import os
import sys

# Root of the test folder
_TEST_ROOT = os.path.dirname(__file__)

# Root of the project (one level up from tests/)
_PROJECT_ROOT = os.path.abspath(os.path.join(_TEST_ROOT, ".."))

# Ensure src is in the Python path
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# Path to the data folder
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")
