"""Local development entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from waam_rag.api.app import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
