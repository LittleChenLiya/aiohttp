"""End-to-end demonstration of the resumable download component.

Runs entirely against a local aiohttp server on an ephemeral port and a temp
directory on the current drive; finishes in well under 90 seconds.

Scenarios
---------
1. Resume: the server first sends ``abc`` and hangs up, then answers the
   Range/If-Range request with bytes 3-5/6 (``def``). Final file:
   ``abcdef``.
2. ETag change: after a partial download the server changes its strong ETag
   and ignores the range; the 200 body is written to a fresh part file and
   replaces the destination.
3. Bad 206 range (Content-Range start mismatch): the download fails and the
   previously published destination file is untouched.
4. Premature EOF (declared length exceeds delivered bytes): same guarantee.

Run:  python resumable_demo/run_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp  # noqa: E402

from resumable_download import (  # noqa: E402
    ExhaustedRetries,
    InvalidRangeResponse,
    download,
)
from resumable_download.testing import (  # noqa: E402
    ABCFILE,
    FRESHFILE,
    start_server,
)


async def scenario_resume(workdir: Path) -> None:
    print("[1] resume: 'abc' + drop, then 206 bytes 3-5/6 ('def')")
    runner, url, _state = await start_server(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "range_or_full", "etag": '"v1"'},
        ]
    )
    try:
        dest = workdir / "resume.bin"
        async with aiohttp.ClientSession() as session:
            result = await download(session, url, dest, max_attempts=3)
        data = dest.read_bytes()
        print(f"    -> {result}: {data!r}")
        assert data == ABCFILE == b"abcdef"
        print("    OK: final file is abcdef")
    finally:
        await runner.cleanup()


async def scenario_etag_change(workdir: Path) -> None:
    print("[2] changed ETag -> 200 with brand-new content")
    runner, url, state = await start_server(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "range_or_full", "etag": '"v2"', "body": FRESHFILE},
        ]
    )
    try:
        dest = workdir / "changed.bin"
        async with aiohttp.ClientSession() as session:
            await download(session, url, dest, max_attempts=3)
        data = dest.read_bytes()
        print(f"    -> {data!r}; requests: {state['requests']}")
        assert data == FRESHFILE
        assert state["requests"][1]["if_range"] == '"v1"'
        print("    OK: stale prefix discarded, fresh 200 body published")
    finally:
        await runner.cleanup()


async def scenario_bad_range(workdir: Path) -> None:
    print("[3] wrong Content-Range start must not replace the destination")
    runner, url, _state = await start_server(
        [
            {"kind": "prefix_drop", "send": 3, "etag": '"v1"'},
            {"kind": "bad_range", "start": 99, "etag": '"v1"'},
        ]
    )
    try:
        dest = workdir / "badrange.bin"
        dest.write_bytes(b"PUBLISHED-ORIGINAL")
        async with aiohttp.ClientSession() as session:
            try:
                await download(session, url, dest, max_attempts=2)
            except InvalidRangeResponse as exc:
                print(f"    -> failed as expected: {exc}")
            else:  # pragma: no cover
                raise AssertionError("expected InvalidRangeResponse")
        assert dest.read_bytes() == b"PUBLISHED-ORIGINAL"
        print("    OK: published destination left intact")
    finally:
        await runner.cleanup()


async def scenario_premature_eof(workdir: Path) -> None:
    print("[4] premature EOF must not replace the destination")
    runner, url, _state = await start_server(
        [
            {"kind": "short_eof", "send": 2, "etag": '"v1"'},
            {"kind": "short_eof", "send": 2, "etag": '"v1"'},
        ]
    )
    try:
        dest = workdir / "shorteof.bin"
        dest.write_bytes(b"PUBLISHED-ORIGINAL")
        async with aiohttp.ClientSession() as session:
            try:
                await download(session, url, dest, max_attempts=2)
            except ExhaustedRetries as exc:
                print(f"    -> retries exhausted: {exc}")
            else:  # pragma: no cover
                raise AssertionError("expected ExhaustedRetries")
        assert dest.read_bytes() == b"PUBLISHED-ORIGINAL"
        print("    OK: destination intact; partial fragment kept on disk")
    finally:
        await runner.cleanup()


async def main() -> None:
    # The scripted server hangs connections up on purpose; quiet its logs.
    logging.getLogger("aiohttp").setLevel(logging.CRITICAL)
    with tempfile.TemporaryDirectory(prefix="rdemo-", dir=Path.cwd()) as tmp:
        workdir = Path(tmp)
        await scenario_resume(workdir)
        await scenario_etag_change(workdir)
        await scenario_bad_range(workdir)
        await scenario_premature_eof(workdir)
    print("\nall demo scenarios passed")


if __name__ == "__main__":
    asyncio.run(main())
