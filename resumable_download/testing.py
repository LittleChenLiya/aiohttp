"""Deterministic scripted HTTP server used by the demo and the pytest suite.

Not part of the component's public API. The server is configured with a list
of *scripts*; each request consumes the next script. A script is a tuple of
keyword arguments understood by :func:`scripted_handler_factory`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from aiohttp import web

ABCFILE = b"abcdef"
FRESHFILE = b"zzzzzzzz"
WEAKFILE = b"weak!!"


def _slice_range(
    body: bytes, request: web.Request
) -> tuple[int, int] | None:
    rng = request.headers.get("Range")
    if not rng or not rng.startswith("bytes="):
        return None
    spec = rng.removeprefix("bytes=")
    try:
        start_s, end_s = spec.split("-", 1)
        start = int(start_s)
        end = int(end_s) if end_s else len(body) - 1
    except ValueError:
        return None
    return start, end


def scripted_handler_factory(
    scripts: Sequence[dict],
    *,
    file_body: bytes = ABCFILE,
    strong_etag: str = '"v1"',
):
    """Build a request handler that replays *scripts* one per request.

    Script keys:

    * ``kind``: ``"prefix_drop"`` (send ``send`` bytes, hang up),
      ``"range_or_full"`` (honour Range/If-Range: 206 for a matching strong
      validator, else 200 with the current body/ETag), ``"full"`` (always 200),
      ``"bad_range"`` (reply 206 to *any* request, ignoring If-Range and
      shifting the start), ``"short_eof"`` (declare the full length but send
      fewer bytes, then hang up), ``"bad_total"`` (Content-Range total lies).
    * ``etag`` overrides the ETag for that response (use e.g. ``'W/"w"'``).
    * ``body`` overrides the entity body.
    """
    state = {"index": 0, "requests": []}

    async def handler(request: web.Request) -> web.StreamResponse:
        idx = state["index"]
        state["index"] += 1
        script = scripts[min(idx, len(scripts) - 1)]
        state["requests"].append(
            {
                "range": request.headers.get("Range"),
                "if_range": request.headers.get("If-Range"),
            }
        )

        kind = script.get("kind", "range_or_full")
        etag = script.get("etag", strong_etag)
        body = script.get("body", file_body)

        if kind == "prefix_drop":
            n = script.get("send", 3)
            resp = web.StreamResponse(
                status=200,
                headers={"Content-Length": str(len(body)), "ETag": etag},
            )
            await resp.prepare(request)
            await resp.write(body[:n])
            await resp.write(b"")
            # Hard close: the client sees a transport drop with unread bytes.
            request.transport.close()  # type: ignore[union-attr]
            return resp

        if kind == "short_eof":
            send = script.get("send", len(body) - 1)
            resp = web.StreamResponse(
                status=200,
                headers={"Content-Length": str(len(body)), "ETag": etag},
            )
            await resp.prepare(request)
            await resp.write(body[:send])
            request.transport.close()  # type: ignore[union-attr]
            return resp

        if kind == "full":
            return web.Response(body=body, headers={"ETag": etag})

        if kind == "bad_total":
            start, end = _slice_range(body, request) or (3, len(body) - 1)
            lie_total = script.get("total", len(body) + 10)
            chunk = body[start : end + 1]
            return web.Response(
                status=206,
                body=chunk,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{lie_total}",
                    "ETag": etag,
                },
            )

        if kind == "etag_mismatch_206":
            # Correct Content-Range for the request, but a different strong
            # ETag: the client must reject the recovery before appending.
            parsed = _slice_range(body, request) or (3, len(body) - 1)
            start, end = parsed
            return web.Response(
                status=206,
                body=body[start : end + 1],
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(body)}",
                    "ETag": etag,
                },
            )

        if kind == "bad_range":
            # Only misbehave for an actual Range request: the 206 claims a
            # start that differs from the client's offset. The body matches the
            # declared (wrong) range, so the failure is a validation error.
            if not request.headers.get("Range"):
                return web.Response(body=body, headers={"ETag": etag})
            start = script.get("start", 99)
            span = script.get("span", 3)
            end = start + span - 1
            chunk = (body * ((span // len(body)) + 1))[:span]
            return web.Response(
                status=206,
                body=chunk,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(body) + 100}",
                    "ETag": etag,
                },
            )

        # Default: RFC-style conditional range support.
        parsed = _slice_range(body, request)
        if parsed is not None:
            start, end = parsed
            validator = request.headers.get("If-Range")
            if validator == etag and not etag.startswith(("W/", "w/")):
                end = min(end, len(body) - 1)
                return web.Response(
                    status=206,
                    body=body[start : end + 1],
                    headers={
                        "Content-Range": (
                            f"bytes {start}-{end}/{len(body)}"
                        ),
                        "ETag": etag,
                    },
                )
            # Changed entity or weak validator: ignore the Range -> 200.
            return web.Response(body=body, headers={"ETag": etag})
        return web.Response(body=body, headers={"ETag": etag})

    handler.state = state  # type: ignore[attr-defined]
    return handler


async def start_server(scripts: Iterable[dict], **kwargs) -> tuple[web.AppRunner, str, dict]:
    """Start a scripted server on an ephemeral local port.

    Returns ``(runner, base_url, state)``; the caller must ``cleanup()`` the
    runner.
    """
    scripts = list(scripts)
    handler = scripted_handler_factory(scripts, **kwargs)
    app = web.Application()
    app.router.add_get("/file", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}/file", handler.state


def assert_no_published_file(path: Path) -> None:
    assert not path.exists(), f"destination {path} must not be created/replaced"
