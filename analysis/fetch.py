"""Snapshot one tournament's questions for postmortem analysis.

Usage:
    uv run python analysis/fetch.py --tournament 33121
    uv run python analysis/fetch.py --tournament 33022 --out analysis_data/summer2026.json

Writes the raw post JSON (see metaculus_history.snapshot_tournament) so
analysis/report.py can derive metrics offline. Cache files are gitignored.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metaculus_history import snapshot_tournament


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tournament",
        required=True,
        type=int,
        help="Metaculus tournament id, e.g. 33121 for Fall 2026",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: analysis_data/<tournament>.json)",
    )
    args = parser.parse_args()

    out = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent.parent
        / "analysis_data"
        / f"{args.tournament}.json"
    )

    snapshot = snapshot_tournament(args.tournament)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2))

    statuses: dict[str, int] = {}
    for post in snapshot["posts"]:
        question = post.get("question") or {}
        status = question.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    print(f"Snapshot written to {out}")
    print(f"{snapshot['post_count']} posts, fetched at {snapshot['fetched_at']}")
    print(f"question statuses: {statuses}")


if __name__ == "__main__":
    main()