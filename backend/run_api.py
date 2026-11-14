"""
Start the VSR API (FastAPI + React UI) — works from ANY directory.

    python backend/run_api.py                # http://localhost:8000
    python backend/run_api.py --port 9000
"""

import argparse
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]


def main():
    parser = argparse.ArgumentParser(description="VSR API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    from api.main import app
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
