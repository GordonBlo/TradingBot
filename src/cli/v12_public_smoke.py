"""Explicit, bounded PUBLIC engineering capture; no research acquisition."""
import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from src.microstructure.v10 import _canonical
from src.microstructure.v12 import PublicCollectorV2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='explicit public smoke authorization')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if not args.execute or not 0 < args.seconds <= 30:
        parser.error('--execute and 0 < --seconds <= 30 required')
    workspace = Path(__file__).resolve().parents[2]
    output = workspace / 'reports' / 'v12_phase2' / ('public_smoke_' + datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ'))
    certificate = asyncio.run(PublicCollectorV2(workspace).capture_engineering(output, seconds=args.seconds))
    print(_canonical({'directory': str(output), 'certificate': certificate}))


if __name__ == '__main__':
    main()
