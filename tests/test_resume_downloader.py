"""Local tests for the resumable downloader (aiohttp ClientSession based).

Run with the heavy repo-wide addopts disabled, e.g.::

    pytest -o addopts="" tests/test_resume_downloader.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from resume_dl import ResumeError, ResumeInterrupted, download
from resume_dl.downloader import _default_state_path, strong_etag

pytestmark = pytest.mark.asyncio

_STATE_KEY = web.AppKey("state", dict)


def _scripted_app() -> web.Application:
    """A server whose responses are scripted per path and per hit."""
    app = web.Application()
    app[_STATE_KEY] = {"file": 0, "changed": 0, "wrong": 0, "eof": 0,
                    "short": 0, "fresh": 0, "weak": 0, "malformed": 0}

    async def file_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_STATE_KEY]["file"]
        request.app[_STATE_KEY]["file"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        if request.headers.get("Range") != "bytes=3-":
            # Fresh start (state invalidated): serve the full content.
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            await resp.prepare(request)
            await resp.write(b"abcdef")
            await resp.write_eof()
            return resp
        assert request.headers.get("If-Range") == '"v1"'
        resp = web.StreamResponse(
            status=206,
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": '"v1"',
            },
        )
        await resp.prepare(request)
        await resp.write(b"def")
        await resp.write_eof()
        return resp

    async def changed_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_STATE_KEY]["changed"]
        request.app[_STATE_KEY]["changed"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        resp = web.StreamResponse(
            headers={"Content-Length": "11", "ETag": '"v2"'}
        )
        await resp.prepare(request)
        await resp.write(b"brand-new!!")
        await resp.write_eof()
        return resp

    async def wrong_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_STATE_KEY]["wrong"]
        request.app[_STATE_KEY]["wrong"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        resp = web.StreamResponse(
            status=206,
            headers={
                "Content-Length": "6",
                "Content-Range": "bytes 0-5/6",
                "ETag": '"v1"',
            },
        )
        await resp.prepare(request)
        await resp.write(b"abcdef")
        await resp.write_eof()
        return resp

    async def eof_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_STATE_KEY]["eof"]
        request.app[_STATE_KEY]["eof"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        if phase == 1:
            resp = web.StreamResponse(
                status=206,
                headers={
                    "Content-Length": "6",
                    "Content-Range": "bytes 3-8/9",
                    "ETag": '"v1"',
                },
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"ghi")
            await resp.write_eof()
            return resp
        resp = web.StreamResponse(
            status=206,
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 6-8/9",
                "ETag": '"v1"',
            },
        )
        await resp.prepare(request)
        await resp.write(b"jkl")
        await resp.write_eof()
        return resp

    async def short_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_STATE_KEY]["short"]
        request.app[_STATE_KEY]["short"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        # 206 with the correct start (3), promised 6 bytes of the segment,
        # but only 5 arrive cleanly -> byte-count mismatch -> hard failure.
        resp = web.StreamResponse(
            status=206,
            headers={
                "Content-Length": "5",
                "Content-Range": "bytes 3-8/9",
                "ETag": '"v1"',
            },
        )
        await resp.prepare(request)
        await resp.write(b"abcde")
        await resp.write_eof()
        return resp

    async def fresh_handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            headers={"Content-Length": "6", "ETag": '"v1"'}
        )
        await resp.prepare(request)
        await resp.write(b"abcdef")
        await resp.write_eof()
        return resp

    async def weak_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_STATE_KEY]["weak"]
        request.app[_STATE_KEY]["weak"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": 'W/"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        assert request.headers.get("If-Range") is None
        resp = web.StreamResponse(
            status=206,
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": 'W/"v1"',
            },
        )
        await resp.prepare(request)
        await resp.write(b"def")
        await resp.write_eof()
        return resp

    async def malformed_handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=206,
            headers={
                "Content-Length": "6",
                "Content-Range": "not-a-range",
                "ETag": '"v1"',
            },
        )
        await resp.prepare(request)
        await resp.write(b"abcdef")
        await resp.write_eof()
        return resp

    app.router.add_get("/file", file_handler)
    app.router.add_get("/changed", changed_handler)
    app.router.add_get("/wrong", wrong_handler)
    app.router.add_get("/early-eof", eof_handler)
    app.router.add_get("/short", short_handler)
    app.router.add_get("/fresh", fresh_handler)
    app.router.add_get("/weak", weak_handler)
    app.router.add_get("/malformed", malformed_handler)
    return app


@pytest.fixture
async def client():
    async with TestServer(_scripted_app()) as server, TestClient(server) as c:
        yield c


async def test_resume_after_disconnect(client: TestClient, tmp_path: Path):
    target = tmp_path / "file.bin"
    with pytest.raises(ResumeInterrupted) as exc_info:
        await download(client, "file", target)
    assert exc_info.value.confirmed_bytes == 3
    assert exc_info.value.etag == '"v1"'
    state = _default_state_path(target)
    assert state.is_file()
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["bytes"] == 3
    assert saved["etag"] == '"v1"'
    temp = tmp_path / saved["temp"]
    assert temp.read_bytes() == b"abc"
    assert not target.exists()  # not published yet

    await download(client, "file", target)
    assert target.read_bytes() == b"abcdef"
    assert not state.exists()
    assert not list(tmp_path.glob("*.tmp"))


async def test_etag_change_returns_200_restarts_from_scratch(
    client: TestClient, tmp_path: Path
):
    target = tmp_path / "file.bin"
    with pytest.raises(ResumeInterrupted):
        await download(client, "changed", target)
    # Server ignores the range and answers 200 with brand new content.
    await download(client, "changed", target)
    assert target.read_bytes() == b"brand-new!!"
    assert not list(tmp_path.glob("*.tmp"))


async def test_wrong_range_start_fails_without_replacing_target(
    client: TestClient, tmp_path: Path
):
    target = tmp_path / "file.bin"
    target.write_bytes(b"ORIGINAL")  # previously published file
    with pytest.raises(ResumeInterrupted):
        await download(client, "wrong", target)
    with pytest.raises(ResumeError, match="Content-Range start"):
        await download(client, "wrong", target)
    assert target.read_bytes() == b"ORIGINAL"


async def test_early_eof_preserves_progress_and_target(
    client: TestClient, tmp_path: Path
):
    target = tmp_path / "file.bin"
    target.write_bytes(b"ORIGINAL2")
    with pytest.raises(ResumeInterrupted):
        await download(client, "early-eof", target)
    with pytest.raises(ResumeInterrupted) as exc_info:
        await download(client, "early-eof", target)
    assert exc_info.value.confirmed_bytes == 6
    assert target.read_bytes() == b"ORIGINAL2"
    state = _default_state_path(target)
    assert state.is_file()
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["bytes"] == 6
    # A third attempt completes the file.
    await download(client, "early-eof", target)
    assert target.read_bytes() == b"abcghijkl"


async def test_clean_short_read_is_hard_failure(
    client: TestClient, tmp_path: Path
):
    target = tmp_path / "file.bin"
    target.write_bytes(b"ORIGINAL")
    with pytest.raises(ResumeInterrupted):
        await download(client, "short", target)
    with pytest.raises(ResumeError, match="short read"):
        await download(client, "short", target)
    assert target.read_bytes() == b"ORIGINAL"
    # Temp truncated back to the confirmed length; state stays coherent.
    state = _default_state_path(target)
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["bytes"] == 3
    temp = tmp_path / saved["temp"]
    assert temp.stat().st_size == 3


async def test_fresh_200_download(client: TestClient, tmp_path: Path):
    target = tmp_path / "file.bin"
    await download(client, "fresh", target)
    assert target.read_bytes() == b"abcdef"


async def test_weak_etag_not_used_as_credential(
    client: TestClient, tmp_path: Path
):
    target = tmp_path / "file.bin"
    with pytest.raises(ResumeInterrupted):
        await download(client, "weak", target)
    state = _default_state_path(target)
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["etag"] is None  # weak ETag is not persisted as credential
    await download(client, "weak", target)
    assert target.read_bytes() == b"abcdef"


async def test_malformed_content_range_fails(client: TestClient, tmp_path: Path):
    target = tmp_path / "file.bin"
    target.write_bytes(b"ORIGINAL")
    # Build a valid resume state via a partial download, then repoint it at
    # the endpoint that answers with a malformed Content-Range.
    with pytest.raises(ResumeInterrupted):
        await download(client, "file", target)
    state = _default_state_path(target)
    saved = json.loads(state.read_text(encoding="utf-8"))
    saved["url"] = "malformed"
    state.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ResumeError, match="malformed Content-Range"):
        await download(client, "malformed", target)
    assert target.read_bytes() == b"ORIGINAL"


async def test_state_temp_correspondence_verified(
    client: TestClient, tmp_path: Path
):
    target = tmp_path / "file.bin"
    with pytest.raises(ResumeInterrupted):
        await download(client, "file", target)
    state = _default_state_path(target)
    saved = json.loads(state.read_text(encoding="utf-8"))
    temp = tmp_path / saved["temp"]
    assert temp.stat().st_size == 3

    # Tamper: shrink the temp file without updating the state.  The state is
    # now invalid -> fresh start -> no Range header -> the scripted server
    # serves the full 200 content (never appends to the old fragment).
    old_temp = temp
    temp.write_bytes(b"ab")
    await download(client, "file", target)
    assert target.read_bytes() == b"abcdef"
    # The old fragment must not have been appended to.
    assert old_temp.read_bytes() == b"ab"


async def test_strong_etag_parsing():
    assert strong_etag('"abc"') == '"abc"'
    assert strong_etag('W/"abc"') is None
    assert strong_etag('w/"abc"') is None
    assert strong_etag('"abc", "def"') == '"abc"'
    assert strong_etag("abc") is None
    assert strong_etag(None) is None
