"""Engineering-data provenance guards, separate from frozen research rules."""

from pathlib import Path


CI_MARKER = "ci.provenance.json"
SMOKE_ROOT = Path("data/ci/v10_microstructure_smoke")


def reject_ci_research_path(path: Path) -> None:
    for candidate in (path.absolute(), path.resolve()):
        parts = tuple(part.lower() for part in candidate.parts)
        if any(parts[index:index + 2] == ("data", "ci") for index in range(len(parts) - 1)):
            raise ValueError("CI data root forbidden for V10 research readiness")
        if any((parent / CI_MARKER).exists() for parent in (candidate, *candidate.parents)):
            raise ValueError("CI_ONLY provenance forbidden for V10 research readiness")
