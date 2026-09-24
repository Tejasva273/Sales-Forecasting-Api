"""Convenience entry point to run the API with uvicorn.

Usage::

    python run.py            # start the server on http://127.0.0.1:8000
    HOST=0.0.0.0 PORT=9000 python run.py

The FastAPI app itself lives in ``app.main:app`` so it can also be launched
directly with ``uvicorn app.main:app --reload``.
"""

from __future__ import annotations

import os

import uvicorn

from app.core.logging_config import configure_logging


def main() -> None:
    configure_logging()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("app.main:app", host=host, port=port, reload=reload_flag)


if __name__ == "__main__":
    main()
