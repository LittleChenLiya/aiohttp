"""Demo: resumable download against an in-process aiohttp server.

Runs in well under 90 seconds.  Scenarios:

1. ``/file``      server sends ``abc`` then drops the connection; the retry
                  carries ``Range`` + ``If-Range`` and gets ``bytes 3-5/6``
                  with ``def`` -> the published file is ``abcdef``.
2. ``/changed``  the server ignores the range and answers ``200`` with brand
                  new content and a new ETag -> a new temp file is written
                  from byte zero; the old fragment is never appended to.
3. ``/wrong``    the server answers ``206`` with a range start that does not
                  match the local confirmed length -> hard failure; the
                  previously published target file is left untouched.
4. ``/early-eof`` the server answers ``206`` for the right range but drops
                  after three bytes -> progress is preserved and the target
                  file is left untouched; a third attempt completes the file.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resume_dl import ResumeError, ResumeInterrupted, download  # noqa: E402

_DEMO_DIR = Path(__file__).resolve().parent / ".demo_tmp"
_PHASES_KEY = web.AppKey("phases", dict)


def _scripted_app() -> web.Application:
    app = web.Application()
    phases = {"file": 0, "changed": 0, "wrong": 0, "eof": 0}
    app[_PHASES_KEY] = phases

    async def file_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_PHASES_KEY]["file"]
        request.app[_PHASES_KEY]["file"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        assert request.headers.get("Range") == "bytes=3-", request.headers
        assert request.headers.get("If-Range") == '"v1"', request.headers
        resp = web.StreamResponse(
            status=206,
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": '"v1"',
            }
        )
        await resp.prepare(request)
        await resp.write(b"def")
        await resp.write_eof()
        return resp

    async def changed_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_PHASES_KEY]["changed"]
        request.app[_PHASES_KEY]["changed"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        # Server ignores the range: brand new content, new ETag, 200.
        resp = web.StreamResponse(
            headers={"Content-Length": "11", "ETag": '"v2"'}
        )
        await resp.prepare(request)
        await resp.write(b"brand-new!!")
        await resp.write_eof()
        return resp

    async def wrong_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_PHASES_KEY]["wrong"]
        request.app[_PHASES_KEY]["wrong"] = phase + 1
        if phase == 0:
            resp = web.StreamResponse(
                headers={"Content-Length": "6", "ETag": '"v1"'}
            )
            resp.force_close()
            await resp.prepare(request)
            await resp.write(b"abc")
            await resp.write_eof()
            return resp
        # 206 but the range start does not match the local length (3).
        resp = web.StreamResponse(
            status=206,
            headers={
                "Content-Length": "6",
                "Content-Range": "bytes 0-5/6",
                "ETag": '"v1"',
            }
        )
        await resp.prepare(request)
        await resp.write(b"abcdef")
        await resp.write_eof()
        return resp

    async def eof_handler(request: web.Request) -> web.StreamResponse:
        phase = request.app[_PHASES_KEY]["eof"]
        request.app[_PHASES_KEY]["eof"] = phase + 1
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
            # Right start, promised 6 bytes of the segment, drop after 3.
            resp = web.StreamResponse(
                status=206,
                headers={
                    "Content-Length": "6",
                    "Content-Range": "bytes 3-8/9",
                    "ETag": '"v1"',
                }
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
            }
        )
        await resp.prepare(request)
        await resp.write(b"jkl")
        await resp.write_eof()
        return resp

    app.router.add_get("/file", file_handler)
    app.router.add_get("/changed", changed_handler)
    app.router.add_get("/wrong", wrong_handler)
    app.router.add_get("/early-eof", eof_handler)
    return app


async def _run() -> None:
    if _DEMO_DIR.exists():
        shutil.rmtree(_DEMO_DIR)
    _DEMO_DIR.mkdir(parents=True)

    async with TestServer(_scripted_app()) as server, TestClient(server) as client:
        base = "/"

        # Scenario 1: abc + drop, then 206 def -> abcdef.
        target = _DEMO_DIR / "file1.bin"
        try:
            await download(client, "file", target)
        except ResumeInterrupted as exc:
            print(f"[1] interrupted at {exc.confirmed_bytes} bytes, "
                  f"etag={exc.etag!r}; resuming")
        else:
            raise AssertionError("expected first attempt to be interrupted")
        state = target.with_name(target.name + ".resume.json")
        assert state.is_file(), "resume state must be preserved"
        import json as _json
        _state = _json.loads(state.read_text(encoding="utf-8"))
        _temp = target.parent / _state["temp"]
        assert _temp.read_bytes() == b"abc", "partial temp must hold abc"
        await download(client, "file", target)
        assert target.read_bytes() == b"abcdef", target.read_bytes()
        print("[1] resumed 206 -> published abcdef OK")

        # Scenario 2: server ignores range -> 200 with brand new content.
        target2 = _DEMO_DIR / "file2.bin"
        try:
            await download(client, "changed", target2)
        except ResumeInterrupted:
            pass
        await download(client, "changed", target2)
        assert target2.read_bytes() == b"brand-new!!", target2.read_bytes()
        assert not list(_DEMO_DIR.glob("*.tmp")), "no temp files may remain"
        print("[2] 200 restart -> brand-new content, no append OK")

        # Scenario 3: wrong range start -> hard failure, target untouched.
        target3 = _DEMO_DIR / "file3.bin"
        target3.write_bytes(b"ORIGINAL")  # previously published file
        try:
            await download(client, "wrong", target3)
        except ResumeInterrupted:
            pass
        try:
            await download(client, "wrong", target3)
        except ResumeError as exc:
            print(f"[3] hard failure as expected: {exc}")
        else:
            raise AssertionError("expected wrong-range failure")
        assert target3.read_bytes() == b"ORIGINAL", "target must be untouched"
        print("[3] wrong range -> target file untouched OK")

        # Scenario 4: early EOF on a 206 -> progress preserved, then complete.
        target4 = _DEMO_DIR / "file4.bin"
        target4.write_bytes(b"ORIGINAL2")
        try:
            await download(client, "early-eof", target4)
        except ResumeInterrupted:
            pass
        try:
            await download(client, "early-eof", target4)
        except ResumeInterrupted as exc:
            print(f"[4] early EOF after {exc.confirmed_bytes} bytes, "
                  f"etag={exc.etag!r}; target untouched")
        else:
            raise AssertionError("expected early-EOF interruption")
        assert target4.read_bytes() == b"ORIGINAL2", "target must be untouched"
        state4 = target4.with_name(target4.name + ".resume.json")
        assert state4.is_file(), "resume state must be preserved"
        await download(client, "early-eof", target4)
        assert target4.read_bytes() == b"abcghijkl", target4.read_bytes()
        print("[4] early EOF preserved progress -> resumed to full file OK")

    print("\nALL DEMO SCENARIOS PASSED")


def main() -> int:
    try:
        asyncio.run(_run())
    except AssertionError as exc:
        print(f"DEMO FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if _DEMO_DIR.exists():
            shutil.rmtree(_DEMO_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
