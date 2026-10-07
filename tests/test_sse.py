"""Functional tests for the Server-Sent Events client helpers."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable

import pytest
from pytest_aiohttp import AiohttpClient

from aiohttp import ClientResponse, EventSource, web
from aiohttp.test_utils import TestClient

SSEClientFactory = Callable[
    [Iterable[bytes]], Awaitable[TestClient[web.Request, web.Application]]
]


@pytest.fixture
def close_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    """Count release()/close() calls made on ClientResponse."""
    counts = {"release": 0, "close": 0}
    orig_release = ClientResponse.release
    orig_close = ClientResponse.close

    def release(self: ClientResponse) -> None:
        counts["release"] += 1
        orig_release(self)

    def close(self: ClientResponse) -> None:
        counts["close"] += 1
        orig_close(self)

    monkeypatch.setattr(ClientResponse, "release", release)
    monkeypatch.setattr(ClientResponse, "close", close)
    return counts


@pytest.fixture
def make_sse_client(aiohttp_client: AiohttpClient) -> SSEClientFactory:
    """Build a test client whose server streams the given byte chunks."""

    async def factory(
        chunks: Iterable[bytes],
    ) -> TestClient[web.Request, web.Application]:
        async def handler(request: web.Request) -> web.StreamResponse:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for chunk in chunks:
                await resp.write(chunk)
                await asyncio.sleep(0)
            return resp

        app = web.Application()
        app.router.add_get("/", handler)
        return await aiohttp_client(app)

    return factory


async def test_multibyte_events_sent_byte_by_byte(
    make_sse_client: SSEClientFactory,
    close_counts: dict[str, int],
) -> None:
    """Multi-byte UTF-8 characters split into single-byte chunks reassemble."""
    payload = "data: 你好，\ndata: 世界\nid: 事件-1\nevent: 问候\n\n".encode("utf-8")
    client = await make_sse_client([payload[i : i + 1] for i in range(len(payload))])

    resp = await client.get("/")
    events = [event async for event in EventSource(resp)]

    assert len(events) == 1
    event = events[0]
    assert event.data == "你好，\n世界"
    assert event.id == "事件-1"
    assert event.event == "问候"
    assert event.retry is None
    # A fully consumed stream is released, not closed, exactly once.
    assert close_counts == {"release": 1, "close": 0}


async def test_comment_lines_are_skipped(
    make_sse_client: SSEClientFactory,
) -> None:
    client = await make_sse_client(
        [
            b": this is a comment\n",
            b":another comment\n",
            b"\n",  # blank line with no pending data dispatches nothing
            b"data: real\n",
            b": trailing comment\n",
            b"\n",
        ]
    )

    resp = await client.get("/")
    events = [event async for event in EventSource(resp)]

    assert len(events) == 1
    assert events[0].data == "real"
    assert events[0].event == "message"


async def test_empty_id_resets_last_event_id(
    make_sse_client: SSEClientFactory,
) -> None:
    client = await make_sse_client(
        [
            b"id: first\n",
            b"data: a\n\n",
            b"id:\n",
            b"data: b\n\n",
            b"data: c\n\n",
        ]
    )

    resp = await client.get("/")
    source = EventSource(resp)
    events = [event async for event in source]

    assert [event.data for event in events] == ["a", "b", "c"]
    assert [event.id for event in events] == ["first", "", ""]
    assert source.last_event_id == ""


async def test_invalid_retry_is_ignored(
    make_sse_client: SSEClientFactory,
) -> None:
    client = await make_sse_client(
        [
            b"retry: 3000\n",
            b"data: a\n\n",
            b"retry: abc\n",  # not a number
            b"retry: 12x\n",  # trailing junk
            b"retry:\n",  # empty
            b"retry: -5\n",  # not a bare decimal number
            "retry: ５００\n".encode("utf-8"),  # non-ASCII digits
            b"data: b\n\n",
        ]
    )

    resp = await client.get("/")
    source = EventSource(resp)
    events = [event async for event in source]

    assert [event.retry for event in events] == [3000, 3000]
    assert source.retry == 3000


async def test_id_containing_nul_is_ignored(
    make_sse_client: SSEClientFactory,
) -> None:
    client = await make_sse_client(
        [
            b"id: keep\n",
            b"id: has\x00nul\n",
            b"data: x\n\n",
        ]
    )

    resp = await client.get("/")
    source = EventSource(resp)
    events = [event async for event in source]

    assert [event.id for event in events] == ["keep"]
    assert source.last_event_id == "keep"


async def test_cr_lf_line_endings_split_across_chunks(
    make_sse_client: SSEClientFactory,
) -> None:
    client = await make_sse_client(
        [
            b"data: a\r",  # CR at the end of a chunk...
            b"\n",  # ...LF completes the CRLF pair
            b"\n",  # blank line dispatches the event
            b"data: b\r",  # lone CR: next chunk does not start with LF
            b"data: c\n\n",
            b"data: d\r\r\n",  # CR line ending, then CRLF blank line
        ]
    )

    resp = await client.get("/")
    events = [event async for event in EventSource(resp)]

    assert [event.data for event in events] == ["a", "b\nc", "d"]


async def test_trailing_partial_line_is_discarded(
    make_sse_client: SSEClientFactory,
    close_counts: dict[str, int],
) -> None:
    client = await make_sse_client(
        [
            b"data: complete\n\n",
            b"data: partial",  # no line ending, no dispatch blank line
        ]
    )

    resp = await client.get("/")
    events = [event async for event in EventSource(resp)]

    assert [event.data for event in events] == ["complete"]
    assert close_counts == {"release": 1, "close": 0}


async def test_line_too_long(
    make_sse_client: SSEClientFactory,
    close_counts: dict[str, int],
) -> None:
    client = await make_sse_client([b"data: " + b"x" * 100 + b"\n\n"])

    resp = await client.get("/")
    source = EventSource(resp, max_line_size=16)
    with pytest.raises(ValueError, match="longer than 16 characters"):
        async for _ in source:
            pass

    # A failed stream is closed, not released to the connection pool.
    assert close_counts == {"release": 0, "close": 1}


async def test_unterminated_line_too_long(
    make_sse_client: SSEClientFactory,
    close_counts: dict[str, int],
) -> None:
    client = await make_sse_client([b"data: " + b"x" * 100])  # no line ending

    resp = await client.get("/")
    source = EventSource(resp, max_line_size=16)
    with pytest.raises(ValueError, match="longer than 16 characters"):
        async for _ in source:
            pass

    assert close_counts == {"release": 0, "close": 1}


async def test_close_is_idempotent(
    make_sse_client: SSEClientFactory,
    close_counts: dict[str, int],
) -> None:
    client = await make_sse_client([b"data: a\n\n"])

    resp = await client.get("/")
    source = EventSource(resp)
    assert source.response is resp
    events = [event async for event in source]
    assert [event.data for event in events] == ["a"]

    source.close()
    source.close()
    assert close_counts == {"release": 1, "close": 0}


async def test_cancellation_closes_response(
    aiohttp_client: AiohttpClient,
    close_counts: dict[str, int],
) -> None:
    server_idle = asyncio.Event()
    finish = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"data: first\n\n")
        server_idle.set()
        await finish.wait()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    client = await aiohttp_client(app)

    resp = await client.get("/")
    source = EventSource(resp)
    first = await source.__anext__()
    assert first.data == "first"
    await server_idle.wait()

    task = asyncio.create_task(source.__anext__())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(StopAsyncIteration):
        await source.__anext__()

    finish.set()
    # A stream abandoned mid-flight is closed exactly once.
    assert close_counts == {"release": 0, "close": 1}


async def test_context_manager_closes_response_on_early_exit(
    make_sse_client: SSEClientFactory,
    close_counts: dict[str, int],
) -> None:
    client = await make_sse_client([b"data: a\n\n", b"data: b\n\n"])

    resp = await client.get("/")
    async with EventSource(resp) as source:
        event = await source.__anext__()
        assert event.data == "a"

    assert close_counts == {"release": 0, "close": 1}


async def test_bom_and_unknown_fields(
    make_sse_client: SSEClientFactory,
) -> None:
    client = await make_sse_client(
        [
            b"\xef",  # UTF-8 BOM, split across chunks, is stripped
            b"\xbb\xbf",
            b"unknown: field\n",
            b"datawithoutcolon\n",  # treated as field name with empty value
            b"data: x\n\n",
        ]
    )

    resp = await client.get("/")
    events = [event async for event in EventSource(resp)]

    assert [event.data for event in events] == ["x"]
