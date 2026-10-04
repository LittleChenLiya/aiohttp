"""Focused tests for the resumable download component.

Only a local, deterministic aiohttp server fixture is used -- no network,
no Docker.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp
import pytest

from resumable_download import (
    ExhaustedRetries,
    InvalidRangeResponse,
    StateCorruptedError,
    download,
    is_strong_etag,
)
from resumable_download.downloader import (
    PART_SUFFIX,
    STATE_SUFFIX,
    state_path_for,
)
from resumable_download.testing import (
    ABCFILE,
    FRESHFILE,
    WEAKFILE,
    start_server,
)


@pytest.fixture
def dest(tmp_path: Path) -> Path:
    return tmp_path / "download.bin"


async def _start(scripts, **kwargs):
    runner, url, state = await start_server(scripts, **kwargs)
    return runner, url, state


def test_strong_etag_classifier() -> None:
    assert is_strong_etag('"abc"')
    assert is_strong_etag('"a\\"b"')
    assert not is_strong_etag(None)
    assert not is_strong_etag("")
    assert not is_strong_etag("abc")
    assert not is_strong_etag('"unterminated')
    assert not is_strong_etag('W/"abc"')
    assert not is_strong_etag('w/"abc"')
    assert not is_strong_etag('"bad\nctl"')


async def test_resume_after_drop_concatenates_abcdef(dest: Path) -> None:
    runner, url, state = await _start(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "range_or_full", "etag": '"v1"'},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            result = await download(session, url, dest, max_attempts=3)
        assert result == dest
        assert dest.read_bytes() == b"abcdef" == ABCFILE
        # Second request used Range + If-Range with the strong ETag.
        assert state["requests"][1]["range"] == "bytes=3-"
        assert state["requests"][1]["if_range"] == '"v1"'
        # Clean publication: no state or part files remain.
        assert not state_path_for(dest).exists()
        assert not list(dest.parent.glob(f"*{PART_SUFFIX}"))
    finally:
        await runner.cleanup()


async def test_etag_change_returns_fresh_content(dest: Path) -> None:
    runner, url, state = await _start(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "range_or_full", "etag": '"v2"', "body": FRESHFILE},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            await download(session, url, dest, max_attempts=3)
        assert dest.read_bytes() == FRESHFILE
        # The server got the stale ETag via If-Range, answered 200, and the
        # client started a new part file rather than appending.
        assert state["requests"][1]["if_range"] == '"v1"'
        assert not state_path_for(dest).exists()
    finally:
        await runner.cleanup()


async def test_wrong_range_start_fails_and_keeps_destination(dest: Path) -> None:
    dest.write_bytes(b"PUBLISHED")
    runner, url, _state = await _start(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "bad_range", "start": 99, "span": 3, "etag": '"v1"'},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(InvalidRangeResponse, match="starts at 99"):
                await download(session, url, dest, max_attempts=2)
        assert dest.read_bytes() == b"PUBLISHED"
        # Partial prefix remains on disk with matching state.
        sp = state_path_for(dest)
        record = json.loads(sp.read_text())
        assert record["offset"] == 3
        assert (dest.parent / record["part_name"]).stat().st_size == 3
    finally:
        await runner.cleanup()


async def test_premature_eof_exhausts_retries_and_keeps_destination(
    dest: Path,
) -> None:
    dest.write_bytes(b"PUBLISHED")
    runner, url, _state = await _start(
        [
            {"kind": "short_eof", "send": 2, "etag": '"v1"'},
            {"kind": "short_eof", "send": 2, "etag": '"v1"'},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ExhaustedRetries):
                await download(session, url, dest, max_attempts=2)
        assert dest.read_bytes() == b"PUBLISHED"
        sp = state_path_for(dest)
        record = json.loads(sp.read_text())
        assert record["etag"] == '"v1"'
        assert (dest.parent / record["part_name"]).stat().st_size == record["offset"]
    finally:
        await runner.cleanup()


async def test_weak_etag_is_not_a_recovery_credential(dest: Path) -> None:
    runner, url, state = await _start(
        [
            {"kind": "prefix_drop", "send": 3, "etag": 'W/"weak"'},
            {"kind": "full", "etag": 'W/"weak"', "body": WEAKFILE},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            await download(session, url, dest, max_attempts=3)
        assert dest.read_bytes() == WEAKFILE
        assert state["requests"][1]["range"] is None
        assert state["requests"][1]["if_range"] is None
        assert not state_path_for(dest).exists()
        # The unbacked fragment from the dropped weak response is removed.
        assert not list(dest.parent.glob(f"*{PART_SUFFIX}"))
    finally:
        await runner.cleanup()


async def test_lying_total_on_206_is_rejected(dest: Path) -> None:
    runner, url, _state = await _start(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "bad_total", "etag": '"v1"', "total": 4},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(InvalidRangeResponse, match="beyond total"):
                await download(session, url, dest, max_attempts=2)
        assert not dest.exists()
    finally:
        await runner.cleanup()


async def test_etag_mismatch_on_206_is_rejected(dest: Path) -> None:
    dest.write_bytes(b"PUBLISHED")
    runner, url, _state = await _start(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "etag_mismatch_206", "etag": '"v9"'},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(InvalidRangeResponse, match="differs"):
                await download(session, url, dest, max_attempts=2)
        assert dest.read_bytes() == b"PUBLISHED"
    finally:
        await runner.cleanup()


async def test_200_uses_a_new_part_file_not_append(dest: Path) -> None:
    runner, url, _state = await _start(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "range_or_full", "etag": '"v2"', "body": FRESHFILE},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            await download(session, url, dest, max_attempts=3)
        # Result is exactly the new 8-byte body, never "abc" + new bytes.
        assert dest.read_bytes() == FRESHFILE
    finally:
        await runner.cleanup()


def _tamper_state(dest: Path, **changes) -> tuple[Path, dict]:
    sp = state_path_for(dest)
    record = json.loads(sp.read_text())
    record.update(changes)
    sp.write_text(json.dumps(record))
    return sp, record


async def test_state_part_mismatch_offset_is_fatal(dest: Path) -> None:
    runner, url, _ = await _start(
        [{"kind": "prefix_drop", "send": 3, "etag": '"v1"'}]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ExhaustedRetries):
                await download(session, url, dest, max_attempts=1)
        sp, record = _tamper_state(dest, offset=999)
        async with aiohttp.ClientSession() as session:
            with pytest.raises(StateCorruptedError, match="offset 999"):
                await download(session, url, dest, max_attempts=1)
        assert sp.exists()  # corrupt state is never silently overwritten
        assert (dest.parent / record["part_name"]).exists()
    finally:
        await runner.cleanup()


async def test_state_pointing_at_missing_part_is_fatal(dest: Path) -> None:
    runner, url, _ = await _start(
        [{"kind": "prefix_drop", "send": 3, "etag": '"v1"'}]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ExhaustedRetries):
                await download(session, url, dest, max_attempts=1)
        sp, record = _tamper_state(dest)
        (dest.parent / record["part_name"]).unlink()
        async with aiohttp.ClientSession() as session:
            with pytest.raises(StateCorruptedError, match="missing"):
                await download(session, url, dest, max_attempts=1)
        assert sp.exists()
    finally:
        await runner.cleanup()


async def test_state_part_name_tamper_is_fatal(dest: Path) -> None:
    runner, url, _ = await _start(
        [{"kind": "prefix_drop", "send": 3, "etag": '"v1"'}]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ExhaustedRetries):
                await download(session, url, dest, max_attempts=1)
        sp = state_path_for(dest)
        record = json.loads(sp.read_text())
        imposter = dest.parent / (dest.name + ".imposter" + PART_SUFFIX)
        imposter.write_bytes(b"xx")
        record["part_name"] = imposter.name
        sp.write_text(json.dumps(record))
        async with aiohttp.ClientSession() as session:
            with pytest.raises(StateCorruptedError, match="offset"):
                await download(session, url, dest, max_attempts=1)
    finally:
        await runner.cleanup()


async def test_state_url_mismatch_is_fatal(dest: Path) -> None:
    runner, url, _ = await _start(
        [{"kind": "prefix_drop", "send": 3, "etag": '"v1"'}]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ExhaustedRetries):
                await download(session, url, dest, max_attempts=1)
        _tamper_state(dest, url=url + "/other")
        async with aiohttp.ClientSession() as session:
            with pytest.raises(StateCorruptedError, match="does not match"):
                await download(session, url, dest, max_attempts=1)
    finally:
        await runner.cleanup()


async def test_state_with_weak_etag_is_fatal(dest: Path) -> None:
    runner, url, _ = await _start(
        [{"kind": "prefix_drop", "send": 3, "etag": '"v1"'}]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ExhaustedRetries):
                await download(session, url, dest, max_attempts=1)
        _tamper_state(dest, etag='W/"v1"')
        async with aiohttp.ClientSession() as session:
            with pytest.raises(StateCorruptedError, match="strong ETag"):
                await download(session, url, dest, max_attempts=1)
    finally:
        await runner.cleanup()


async def test_malformed_state_file_is_fatal(dest: Path) -> None:
    runner, url, _ = await _start(
        [{"kind": "prefix_drop", "send": 3, "etag": '"v1"'}]
    )
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ExhaustedRetries):
                await download(session, url, dest, max_attempts=1)
        state_path_for(dest).write_text("{not json")
        async with aiohttp.ClientSession() as session:
            with pytest.raises(StateCorruptedError):
                await download(session, url, dest, max_attempts=1)
    finally:
        await runner.cleanup()


async def test_completed_download_cleans_state(dest: Path) -> None:
    runner, url, _ = await _start(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "range_or_full", "etag": '"v1"'},
        ]
    )
    try:
        async with aiohttp.ClientSession() as session:
            await download(session, url, dest, max_attempts=3)
        assert dest.read_bytes() == ABCFILE
        assert not list(dest.parent.glob(f"*{STATE_SUFFIX}*"))
    finally:
        await runner.cleanup()
