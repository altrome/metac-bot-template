"""Historical tournament data for postmortem analysis.

`snapshot_tournament()` walks the Metaculus posts API for one tournament —
open, closed, and resolved questions alike — and stores the raw JSON of
each post's details. Raw passthrough is deliberate: resolutions, our
forecast history, the community prediction, and score objects all live in
that payload, so keeping it verbatim means new report metrics never
require re-querying the API.

Env vars: METACULUS_TOKEN (read via config, which loads .env).
"""

import datetime
import time

import requests

from config import METACULUS_TOKEN

API_BASE_URL = "https://www.metaculus.com/api"
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
PAGE_SIZE = 50
# Politeness gap between requests; snapshots run unattended. Metaculus sits
# behind Cloudflare, which challenges rapid-fire walkers -- the gap and the
# retry below keep a multi-hundred-post snapshot from tripping it.
DETAIL_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = (5, 15)

# One connection pool for the whole walk: keep-alive reuse looks less like
# a burst of independent clients and is faster than per-request sockets.
_session = requests.Session()


def _get_with_retry(url: str, params: dict | None = None):
    """GET with retry-and-backoff; raises the last error after MAX_RETRIES."""
    for attempt in range(MAX_RETRIES):
        response = _session.get(url, **AUTH_HEADERS, params=params)
        if response.ok:
            return response
        if attempt < MAX_RETRIES - 1:
            delay = RETRY_DELAYS_SECONDS[min(attempt, len(RETRY_DELAYS_SECONDS) - 1)]
            print(f"GET {url} failed ({response.status_code}); retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(response.text)


def get_posts_page(
    tournament_id: int | str, offset: int = 0, count: int = PAGE_SIZE
) -> dict:
    """One page of a tournament's posts (all statuses, binary through discrete)."""
    url_qparams = {
        "limit": count,
        "offset": offset,
        "forecast_type": "binary,multiple_choice,numeric,discrete",
        "tournaments": [tournament_id],
    }
    return _get_with_retry(f"{API_BASE_URL}/posts/", params=url_qparams).json()


def get_all_post_ids(tournament_id: int | str) -> list[int]:
    """Every post id in the tournament, across all pages, deduplicated.

    The list endpoint orders by a live rank, so a post can cross a page
    boundary between two fetches; collecting ids into a dict means it can
    then never be lost or double-counted.
    """
    ids: dict[int, None] = {}
    offset = 0
    while True:
        page = get_posts_page(tournament_id, offset=offset)
        before = len(ids)
        for post in page.get("results", []):
            ids[post["id"]] = None
        # The API's own pagination cursor, plus a stall guard for the
        # rank-reorder edge case (no new ids but next still set).
        if not page.get("next") or len(ids) == before:
            break
        offset += PAGE_SIZE
        time.sleep(DETAIL_DELAY_SECONDS)
    return list(ids)


def get_post_details(post_id: int) -> dict:
    """Full raw details of one post (question, forecasts, scores)."""
    return _get_with_retry(f"{API_BASE_URL}/posts/{post_id}/").json()


def snapshot_tournament(tournament_id: int | str) -> dict:
    """Snapshot every post of a tournament as raw API JSON.

    Returns {"tournament_id", "fetched_at", "post_count", "posts"} where
    posts[i] is the verbatim /posts/{id}/ payload.
    """
    post_ids = get_all_post_ids(tournament_id)
    print(f"Tournament {tournament_id}: fetching details for {len(post_ids)} posts", flush=True)
    posts = []
    for index, post_id in enumerate(post_ids):
        posts.append(get_post_details(post_id))
        if (index + 1) % 25 == 0 or index + 1 == len(post_ids):
            print(f"  fetched {index + 1}/{len(post_ids)} posts", flush=True)
        if index < len(post_ids) - 1:
            time.sleep(DETAIL_DELAY_SECONDS)
    return {
        "tournament_id": tournament_id,
        "fetched_at": datetime.datetime.now(datetime.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "post_count": len(posts),
        "posts": posts,
    }