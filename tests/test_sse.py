import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import List

import pytest
from pytest_aiohttp import AiohttpClient

from aiohttp import (
    ClientResponse,
    EventSource,
    ServerSentEvent,
    SSEBufferOverflowError,
    web,
)
from aiohttp.test_utils import TestClient


@pytest.fixture
def count_releases(monkeypatch: pytest.MonkeyPatch) -> List[ClientResponse]:
    """Count the responses that effectively get released."""
    released: List[ClientResponse] = []
    original_release = ClientResponse.release

    def release_and_count(self: ClientResponse) -> None:
        if not self._released:
            released.append(self)
        original_release(self)

    monkeypatch.setattr(ClientResponse, "release", release_and_count)
    return released


def sse_handler(
    chunks: Sequence[bytes], delay: float = 0
) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for chunk in chunks:
            await resp.write(chunk)
            if delay:
                # Give the client a chance to read each chunk separately.
                await asyncio.sleep(delay)
        await resp.write_eof()
        return resp

    return handler


async def make_client(
    aiohttp_client: AiohttpClient, chunks: Sequence[bytes], delay: float = 0
) -> TestClient:
    app = web.Application()
    app.router.add_get("/events", sse_handler(chunks, delay))
    return await aiohttp_client(app)


async def read_all(source: EventSource) -> List[ServerSentEvent]:
    return [event async for event in source]


async def test_multibyte_events_sent_byte_by_byte(
    aiohttp_client: AiohttpClient,
) -> None:
    payload = "event: 问候\ndata: 你好，\ndata: 世界\nid: 事件-42\n\n".encode("utf-8")
    # The server sends the payload one byte at a time, splitting every
    # multi-byte UTF-8 sequence across chunk boundaries.
    chunks = [payload[i : i + 1] for i in range(len(payload))]
    client = await make_client(aiohttp_client, chunks, delay=0.001)

    async with client.get("/events") as resp:
        source = EventSource(resp)
        events = await read_all(source)
        last_event_id = source.last_event_id

    assert len(events) == 1
    event = events[0]
    assert event.data == "你好，\n世界"
    assert event.event == "问候"
    assert event.id == "事件-42"
    assert last_event_id == "事件-42"


async def test_comments_and_unknown_fields_skipped(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [
        b": comment one\n:comment two\n\n",
        b"fieldwithoutcolon\ndata: x\n\n",
    ]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        events = await read_all(EventSource(resp))

    assert events == [ServerSentEvent(data="x")]


async def test_event_without_data_is_not_dispatched(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [b"event: nothing\n\n", b"data: real\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        events = await read_all(EventSource(resp))

    assert events == [ServerSentEvent(data="real")]


async def test_event_type_and_space_stripping(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [
        b"event: greeting\ndata: hi\n\n",
        b"data:  two leading spaces\n\n",
        b"data:no space\n\n",
        b"data\n\n",
    ]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        events = await read_all(EventSource(resp))

    assert events == [
        ServerSentEvent(data="hi", event="greeting"),
        # Only a single leading space is stripped from the value.
        ServerSentEvent(data=" two leading spaces"),
        ServerSentEvent(data="no space"),
        ServerSentEvent(data=""),
    ]


async def test_empty_id_resets_last_event_id(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [b"id: 1\ndata: a\n\n", b"id:\ndata: b\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        source = EventSource(resp)
        events = await read_all(source)
        last_event_id = source.last_event_id

    assert [event.id for event in events] == ["1", ""]
    assert last_event_id == ""


async def test_retry_field_must_be_decimal_digits(
    aiohttp_client: AiohttpClient,
) -> None:
    payload = (
        "retry: 1500\n"
        "retry: abc\n"
        "retry: 1.5\n"
        "retry: -7\n"
        "retry: １２\n"  # non-ASCII digits do not count
        "data: x\n\n"
    ).encode("utf-8")
    chunks = [payload]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        source = EventSource(resp)
        events = await read_all(source)
        retry = source.retry

    assert events[0].retry == 1500
    assert retry == 1500


async def test_id_with_nul_character_is_ignored(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [b"id: keep\ndata: a\n\n", b"id: bad\x00id\ndata: b\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        source = EventSource(resp)
        events = await read_all(source)
        last_event_id = source.last_event_id

    assert [event.id for event in events] == ["keep", "keep"]
    assert last_event_id == "keep"


async def test_crlf_and_lf_split_across_chunks(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [
        b"data: a\r",  # CR at the end of a chunk ...
        b"\n",  # ... followed by the LF in the next chunk: a single CRLF
        b"data: b\r",  # CR at the end of a chunk ...
        b"data: c\r\n",  # ... not followed by LF: a lone CR terminator
        b"data: d\rdata: e\n",  # lone CR in the middle of a chunk
        b"\n",
    ]
    client = await make_client(aiohttp_client, chunks, delay=0.01)

    async with client.get("/events") as resp:
        events = await read_all(EventSource(resp))

    assert events == [ServerSentEvent(data="a\nb\nc\nd\ne")]


async def test_trailing_partial_line_is_discarded(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [b"data: complete\n\n", b"data: partial"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        events = await read_all(EventSource(resp))

    assert events == [ServerSentEvent(data="complete")]


async def test_trailing_event_without_blank_line_is_discarded(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [b"data: a\n\n", b"data: b\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        events = await read_all(EventSource(resp))

    assert events == [ServerSentEvent(data="a")]


async def test_leading_bom_is_stripped(aiohttp_client: AiohttpClient) -> None:
    chunks = [b"\xef\xbb\xbfdata: x\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        events = await read_all(EventSource(resp))

    assert events == [ServerSentEvent(data="x")]


async def test_several_events_in_a_single_chunk(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [b"data: a\n\ndata: b\n\ndata: c\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        events = await read_all(EventSource(resp))

    assert [event.data for event in events] == ["a", "b", "c"]


async def test_data_buffer_limit_releases_response(
    aiohttp_client: AiohttpClient,
    count_releases: List[ClientResponse],
) -> None:
    chunks = [b"data: " + b"x" * 100 + b"\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        source = EventSource(resp, max_event_size=16)
        with pytest.raises(SSEBufferOverflowError):
            await read_all(source)
        assert source.closed
        assert count_releases == [resp]


async def test_partial_line_limit_releases_response(
    aiohttp_client: AiohttpClient,
    count_releases: List[ClientResponse],
) -> None:
    chunks = [b"data: " + b"x" * 100]  # no line terminator at all
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        source = EventSource(resp, max_event_size=16)
        with pytest.raises(SSEBufferOverflowError):
            await read_all(source)
        assert count_releases == [resp]


async def test_invalid_max_event_size() -> None:
    with pytest.raises(ValueError, match="max_event_size"):
        EventSource(None, max_event_size=0)  # type: ignore[arg-type]


async def test_cancellation_releases_response(
    aiohttp_client: AiohttpClient,
    count_releases: List[ClientResponse],
) -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"data: first\n\n")
        with suppress(ConnectionResetError, asyncio.CancelledError):
            while True:
                await asyncio.sleep(0.01)
                await resp.write(b": ping\n\n")
        return resp

    app = web.Application()
    app.router.add_get("/events", handler)
    client = await aiohttp_client(app)

    async with client.get("/events") as resp:
        source = EventSource(resp)
        first = await source.__anext__()
        assert first.data == "first"

        pending = asyncio.create_task(source.__anext__())
        await asyncio.sleep(0.1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        assert source.closed
        assert count_releases == [resp]


async def test_eof_releases_response_exactly_once(
    aiohttp_client: AiohttpClient,
    count_releases: List[ClientResponse],
) -> None:
    chunks = [b"data: x\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        async with EventSource(resp) as source:
            events = await read_all(source)
            assert source.response is resp
        assert source.closed

    assert events == [ServerSentEvent(data="x")]
    assert count_releases == [resp]


async def test_iteration_after_close_stops(
    aiohttp_client: AiohttpClient,
) -> None:
    chunks = [b"data: x\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        source = EventSource(resp)
        await source.aclose()
        assert await read_all(source) == []


def test_event_repr_and_equality() -> None:
    event = ServerSentEvent("x")
    assert event == ServerSentEvent("x", "message", None, None)
    assert event != ServerSentEvent("y")
    assert event != "x"
    assert repr(event) == (
        "<ServerSentEvent(data='x', event='message', id=None, retry=None)>"
    )


def test_feed_handles_line_endings_split_across_chunks() -> None:
    source = EventSource(None)  # type: ignore[arg-type]
    source._feed("data: a\r")
    source._feed("\n")  # the LF of the split CRLF pair is swallowed
    source._feed("data: b\r")
    source._feed("data: c\n")  # lone CR at a chunk boundary ends the line
    source._feed("\n")
    assert list(source._events) == [ServerSentEvent(data="a\nb\nc")]


async def test_decoder_flush_overflow_releases_response(
    aiohttp_client: AiohttpClient,
    count_releases: List[ClientResponse],
) -> None:
    # The stream ends with an incomplete UTF-8 sequence; the replacement
    # char produced by the final decoder flush overflows the line buffer.
    chunks = [b"dat\xc3"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        source = EventSource(resp, max_event_size=3)
        with pytest.raises(SSEBufferOverflowError):
            await read_all(source)
        assert count_releases == [resp]


async def test_aclose_is_idempotent(
    aiohttp_client: AiohttpClient,
    count_releases: List[ClientResponse],
) -> None:
    chunks = [b"data: x\n\n"]
    client = await make_client(aiohttp_client, chunks)

    async with client.get("/events") as resp:
        source = EventSource(resp)
        await source.aclose()
        await source.aclose()
        assert source.closed

    assert count_releases == [resp]
