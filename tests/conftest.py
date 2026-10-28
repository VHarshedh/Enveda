import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BUILDERS = ROOT / "builders"
NOTEBOOKS = ROOT / "notebooks"

for p in (ROOT, SRC, BUILDERS):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
