"""Compatibility ASGI entry point for local hosts and Vercel discovery.

The deployable FastAPI application lives in ``api.index``. Keeping this
module as a thin re-export prevents platforms that discover ``app.py`` from
loading the legacy Gemini SDK implementation instead of the serverless app.
"""

from api.index import app

__all__ = ["app"]
