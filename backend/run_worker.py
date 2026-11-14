"""
Start the queue worker (the GPU process) — works from ANY directory.

    python backend/run_worker.py
    python backend/run_worker.py --config config/config.yaml --once
"""

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from worker.main import main  # noqa: E402

if __name__ == "__main__":
    main()
