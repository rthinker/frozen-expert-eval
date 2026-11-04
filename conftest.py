import sys
from pathlib import Path

# Add codebase root to sys.path so all tests can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent))

collect_ignore = ["third_party"]
