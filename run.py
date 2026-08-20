"""Trading Bot command-line entry point."""

from __future__ import annotations

import asyncio

from src.main import main


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        # Fallback for platforms where asyncio signal handlers are unavailable.
        raise SystemExit(0) from None
