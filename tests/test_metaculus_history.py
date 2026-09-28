"""Tests for metaculus_history — the tournament snapshot fetcher.

All HTTP is faked; snapshots of real tournaments are taken by the E2E
script (analysis/fetch.py), never by the test suite.
"""

from types import SimpleNamespace

import metaculus_history as mh


def _fake_session(pages, details, captured=None):
    """Patch mh._session: `pages` are list-endpoint payloads (in call order),
    `details` maps post id -> raw detail payload."""

    state = {"page": 0}

    def fake_get(url, headers=None, params=None):
        if captured is not None:
            captured.append((url, params))
        if url.rstrip("/").endswith("/posts"):
            page = pages[state["page"]]
            state["page"] += 1
            return SimpleNamespace(ok=True, json=lambda: page)
        post_id = int(url.rstrip("/").split("/")[-1])
        return SimpleNamespace(ok=True, json=lambda: details[post_id])

    return SimpleNamespace(get=fake_get)


PAGE_ONE = {
    "next": "https://www.metaculus.com/api/posts/?offset=50",
    "results": [{"id": 101, "question": {"status": "resolved"}}, {"id": 102}],
}
# Page two repeats post 101 (rank-reorder across a page boundary) plus a new one.
PAGE_TWO = {
    "next": None,
    "results": [{"id": 101, "question": {"status": "resolved"}}, {"id": 103}],
}
DETAILS = {
    101: {"id": 101, "question": {"id": 11, "status": "resolved", "resolution": 1.0}},
    102: {"id": 102, "question": {"id": 12, "status": "open"}},
    103: {"id": 103, "question": {"id": 13, "status": "closed"}},
}


def test_get_all_post_ids_paginates_and_dedupes(monkeypatch):
    monkeypatch.setattr(mh, "_session", _fake_session([PAGE_ONE, PAGE_TWO], DETAILS))
    monkeypatch.setattr(mh, "DETAIL_DELAY_SECONDS", 0)
    ids = mh.get_all_post_ids(33121)
    assert sorted(ids) == [101, 102, 103]


def test_snapshot_tournament_keeps_raw_posts(monkeypatch):
    monkeypatch.setattr(mh, "_session", _fake_session([PAGE_ONE, PAGE_TWO], DETAILS))
    monkeypatch.setattr(mh, "DETAIL_DELAY_SECONDS", 0)
    snapshot = mh.snapshot_tournament(33121)
    assert snapshot["tournament_id"] == 33121
    assert snapshot["post_count"] == 3
    assert snapshot["fetched_at"].endswith("Z")
    # Raw passthrough: the resolution payload arrives unmodified.
    by_id = {post["id"]: post for post in snapshot["posts"]}
    assert by_id[101]["question"]["resolution"] == 1.0
    assert by_id[103]["question"]["status"] == "closed"


def test_posts_page_requests_the_tournament(monkeypatch):
    captured = []
    monkeypatch.setattr(mh, "_session", _fake_session([PAGE_ONE], DETAILS, captured))
    mh.get_posts_page(33022)
    url, params = captured[0]
    assert url.endswith("/posts/")
    assert params["tournaments"] == [33022]
    assert "resolved" not in str(params)  # no status filter: all statuses


def test_stalled_pagination_stops(monkeypatch):
    # next stays set but no new ids ever appear: the stall guard must break.
    stall_page = {
        "next": "https://www.metaculus.com/api/posts/?offset=50",
        "results": [{"id": 101}],
    }
    monkeypatch.setattr(mh, "_session", _fake_session([stall_page, stall_page], DETAILS))
    monkeypatch.setattr(mh, "DETAIL_DELAY_SECONDS", 0)
    assert mh.get_all_post_ids(1) == [101]


def test_api_error_retries_then_raises(monkeypatch):
    monkeypatch.setattr(mh, "RETRY_DELAYS_SECONDS", (0, 0))
    attempts = []

    def failing_get(url, headers=None, params=None):
        attempts.append(url)
        return SimpleNamespace(ok=False, status_code=403, json=dict, text="401 unauthorized")

    monkeypatch.setattr(mh, "_session", SimpleNamespace(get=failing_get))
    try:
        mh.get_posts_page(33121)
        raise AssertionError("HTTP failure must raise")
    except RuntimeError as exc:
        assert "401" in str(exc)
    assert len(attempts) == mh.MAX_RETRIES, "must retry before giving up"