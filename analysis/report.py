"""Postmortem report from a tournament snapshot (analysis/fetch.py cache).

Usage:
    uv run python analysis/report.py --tournament 33022
    uv run python analysis/report.py --tournament 33121 --limit 10 --csv out.csv

Reads analysis_data/<tournament>.json and prints:
  - total spot peer score (the leaderboard number)
  - per-question score table, worst first
  - damage by question type
  - binary calibration bins (our final probability vs how often YES resolved)
  - forecast freshness (latency from question open; never-forecast count)
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LIMIT = 20


def _parse_time(value):
    if value is None or value == "":
        return None
    # Forecast history timestamps are Unix epochs; question times are ISO.
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, tz=datetime.UTC)
    try:
        return datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _fmt_time(value):
    moment = _parse_time(value)
    return moment.strftime("%Y-%m-%d %H:%M") if moment else "—"


def _earliest_forecast_time(my_forecasts):
    """Earliest of our forecasts on the question (history entries and latest)."""
    times = []
    for entry in my_forecasts.get("history") or []:
        times.append(_parse_time(entry.get("start_time")))
    times.append(_parse_time((my_forecasts.get("latest") or {}).get("start_time")))
    times = [t for t in times if t]
    return min(times) if times else None


def _fmt_num(value):
    if value is None:
        return "?"
    value = float(value)
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"


def _binary_p_yes(forecast_values):
    if not forecast_values:
        return None
    return forecast_values[1] if len(forecast_values) > 1 else forecast_values[0]


def _summarize_forecast(question, forecast_values):
    """One-line description of our final forecast for the score table."""
    if not forecast_values:
        return "no forecast"
    qtype = question.get("type")
    if qtype == "binary":
        return f"{_binary_p_yes(forecast_values):.0%} YES"
    if qtype == "multiple_choice":
        options = question.get("options") or []
        top = max(range(len(forecast_values)), key=lambda i: forecast_values[i])
        name = options[top] if top < len(options) else f"option {top}"
        return f"{forecast_values[top]:.0%} on {name}"
    # numeric / discrete: the calibration story lives in the tails (outside-range
    # mass), which is what cost Summer 2026 its points.
    cdf = forecast_values
    below, above = cdf[0], 1 - cdf[-1]
    scaling = question.get("scaling") or {}
    lo, hi = scaling.get("range_min"), scaling.get("range_max")
    range_txt = f"range [{_fmt_num(lo)}, {_fmt_num(hi)}]" if lo is not None else "no range"
    return f"{below:.1%} below, {above:.1%} above {range_txt}"


def _summarize_resolution(question):
    resolution = question.get("resolution")
    if resolution is None:
        return "unresolved"
    if question.get("type") == "multiple_choice" and isinstance(resolution, (int, float)):
        options = question.get("options") or []
        index = int(resolution)
        if 0 <= index < len(options):
            return str(options[index])
    return str(resolution)


def _miss_tag(question, forecast_values):
    """Heuristic cause tag for the worst rows; empty when nothing notable."""
    if not forecast_values:
        return ""
    qtype = question.get("type")
    if qtype == "binary":
        p_yes = _binary_p_yes(forecast_values)
        if str(question.get("resolution")).lower() in ("yes", "1", "true") and p_yes <= 0.10:
            return "EXTREME-NO-MISS"
        if str(question.get("resolution")).lower() in ("no", "0", "false") and p_yes >= 0.90:
            return "EXTREME-YES-MISS"
        return ""
    if qtype in ("numeric", "discrete"):
        # Out-of-range resolutions arrive as these strings, in-range ones as
        # numbers or numeric strings ("293426.0").
        resolution = question.get("resolution")
        if resolution in ("below_lower_bound", "above_upper_bound"):
            return "RESOLVED-OUTSIDE-RANGE"
        try:
            value = float(resolution)
        except (TypeError, ValueError):
            return ""
        scaling = question.get("scaling") or {}
        lo, hi = scaling.get("range_min"), scaling.get("range_max")
        if lo is not None and (value < lo or value > hi):
            return "RESOLVED-OUTSIDE-RANGE"
    return ""


def all_rows(snapshot):
    """Flatten each post into one analysis row (raw fields kept for the report)."""
    rows = []
    for post in snapshot.get("posts", []):
        question = post.get("question") or {}
        my_forecasts = question.get("my_forecasts") or {}
        latest = my_forecasts.get("latest") or {}
        forecast_values = latest.get("forecast_values")
        rows.append(
            {
                "post_id": post["id"],
                "title": (question.get("title") or post.get("title") or "")[:70],
                "type": question.get("type", "?"),
                "status": question.get("status", "?"),
                "forecast": _summarize_forecast(question, forecast_values),
                "resolution": _summarize_resolution(question),
                "spot_peer_score": (my_forecasts.get("score_data") or {}).get("spot_peer_score"),
                "baseline_score": (my_forecasts.get("score_data") or {}).get("baseline_score"),
                "forever_unforecast": not latest,
                "first_forecast": _earliest_forecast_time(my_forecasts),
                "open_time": _parse_time(question.get("open_time")),
                "close_time": _parse_time(
                    question.get("actual_close_time") or question.get("scheduled_close_time")
                ),
                "miss_tag": _miss_tag(question, forecast_values),
                "_binary_p_yes": (
                    _binary_p_yes(forecast_values) if question.get("type") == "binary" else None
                ),
                "_resolved_yes": str(question.get("resolution")).lower()
                in ("yes", "1", "true"),
            }
        )
    return rows


def scored_rows(rows):
    """Rows with an official score, worst first."""
    scored = [r for r in rows if r["spot_peer_score"] is not None]
    scored.sort(key=lambda r: r["spot_peer_score"])
    return scored


def type_breakdown(scored):
    """Total and count of spot peer score per question type."""
    by_type = {}
    for row in scored:
        bucket = by_type.setdefault(row["type"], {"count": 0, "total": 0.0})
        bucket["count"] += 1
        bucket["total"] += row["spot_peer_score"]
    return by_type


def binary_calibration(rows):
    """Resolution-based calibration: final p_yes in 10% bins vs YES rate.

    Each bin shows how many resolved binaries we put there and how often
    they actually resolved YES -- the direct check for overconfidence
    (predicting 70%+ and resolving YES far more often than 70%).
    """
    bins = [{"lo": lo, "n": 0, "yes": 0} for lo in range(0, 100, 10)]
    for row in rows:
        if row["type"] != "binary" or row["status"] != "resolved" or row["_binary_p_yes"] is None:
            continue
        index = min(int(row["_binary_p_yes"] * 100 // 10), 9)
        bins[index]["n"] += 1
        if row["_resolved_yes"]:
            bins[index]["yes"] += 1
    return bins


def freshness_stats(rows):
    """How long after open we first forecast; how many closed questions never
    got one (the per-question view of scheduled-run delivery)."""
    latencies = []
    never = 0
    for row in rows:
        if row["forever_unforecast"]:
            if row["status"] in ("closed", "resolved"):
                never += 1
            continue
        if row["first_forecast"] and row["open_time"]:
            latencies.append((row["first_forecast"] - row["open_time"]).total_seconds() / 3600)
    latencies.sort()
    return {
        "forecast": len(latencies),
        "never": never,
        "median_hours": latencies[len(latencies) // 2] if latencies else None,
        "within_24h": sum(1 for hours in latencies if hours <= 24),
    }


def render_report(snapshot, limit=DEFAULT_LIMIT):
    """The full postmortem as printable text."""
    rows = all_rows(snapshot)
    scored = scored_rows(rows)
    total = sum(r["spot_peer_score"] for r in scored)

    lines = []
    lines.append(
        f"Tournament {snapshot.get('tournament_id')}: {len(rows)} posts, "
        f"{len(scored)} scored, snapshot {snapshot.get('fetched_at')}"
    )
    lines.append(f"Total spot peer score (leaderboard number): {total:.1f}")
    lines.append("")

    lines.append(f"Worst {min(limit, len(scored))} questions:")
    lines.append(f"  {'score':>9}  {'type':<15}  {'tag':<21}  {'forecast':<52}  {'resolved':<18}  title")
    for row in scored[:limit]:
        lines.append(
            f"  {row['spot_peer_score']:>9.1f}  {row['type']:<15}  {row['miss_tag'] or '—':<21}  "
            f"{row['forecast'][:52]:<52}  {row['resolution'][:18]:<18}  {row['title']}"
        )
    lines.append("")

    lines.append("Damage by question type:")
    for qtype, bucket in sorted(type_breakdown(scored).items(), key=lambda kv: kv[1]["total"]):
        lines.append(
            f"  {qtype:<16} {bucket['count']:>4} questions   total {bucket['total']:>10.1f}   "
            f"per question {bucket['total'] / bucket['count']:>8.1f}"
        )
    lines.append("")

    lines.append("Binary calibration (final p_yes bin vs YES resolution rate):")
    for b in binary_calibration(rows):
        if b["n"]:
            rate = b["yes"] / b["n"]
            lines.append(f"  {b['lo']:>3}-{b['lo'] + 10}% : {b['n']:>3} questions, {rate:>5.0%} resolved YES")
    lines.append("")

    fresh = freshness_stats(rows)
    median = f"{fresh['median_hours']:.1f}h" if fresh["median_hours"] is not None else "—"
    lines.append(
        f"Freshness: {fresh['forecast']} forecast questions, median first-forecast latency "
        f"{median}, within 24h of open: {fresh['within_24h']}/{fresh['forecast']}; "
        f"{fresh['never']} closed/resolved questions never forecast"
    )
    return "\n".join(lines)


def export_csv(scored, path):
    """Write the full scored table (not just the worst rows) as CSV."""
    columns = [
        "post_id", "title", "type", "status", "spot_peer_score", "baseline_score",
        "forecast", "resolution", "miss_tag",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament", required=True, type=int, help="Tournament id; reads analysis_data/<id>.json")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Worst rows to show (default 20)")
    parser.add_argument("--csv", default=None, help="Write the full scored table to this CSV path")
    args = parser.parse_args()

    path = REPO_ROOT / "analysis_data" / f"{args.tournament}.json"
    if not path.exists():
        raise SystemExit(f"No snapshot at {path} -- run: uv run python analysis/fetch.py --tournament {args.tournament}")

    snapshot = json.loads(path.read_text())
    print(render_report(snapshot, limit=args.limit))

    if args.csv:
        export_csv(scored_rows(all_rows(snapshot)), args.csv)
        print(f"\nScored table written to {args.csv}")


if __name__ == "__main__":
    main()