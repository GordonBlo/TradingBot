"""Explicit PUBLIC engineering soak, capped at 900s; never acquisition authority."""
import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from src.microstructure.v10 import _canonical
from src.microstructure.v12 import PublicCollectorV2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engineering-soak', action='store_true')
    parser.add_argument('--duration-seconds', type=int, required=True)
    args = parser.parse_args()
    if not args.engineering_soak or not 0 < args.duration_seconds <= 900:
        parser.error('--engineering-soak and 0 < --duration-seconds <=900 required')
    workspace = Path(__file__).resolve().parents[2]
    directory = workspace / 'reports/v12_engineering' / datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    result = asyncio.run(PublicCollectorV2(workspace).capture_engineering_soak(
        directory, seconds=args.duration_seconds, engineering_authorized=True))
    print(_canonical({'directory': str(directory), 'classification': result['classification'],
                      'operational_checks': result['certificate']['checks']}))


if __name__ == '__main__':
    main()
