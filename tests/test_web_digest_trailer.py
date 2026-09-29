import asyncio
import base64
import hashlib
import zlib
from contextlib import suppress
from typing import Dict, Tuple

import pytest
from pytest_aiohttp import AiohttpServer

from aiohttp import web


def digest_header_value(data: bytes) -> str:
    """Expected RFC 9530 Digest trailer value for the given entity bytes."""
    return "sha-256=:" + base64.b64encode(hashlib.sha256(data).digest()).decode() + ":"


async def raw_request(
    host: str, port: int, request: bytes
) -> Tuple[bytes, Dict[str, str], bytes, Dict[str, str]]:
    """Send a raw HTTP request and parse a chunked response.

    Returns (status line, headers, entity body, trailers).
    """
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(request)
        await writer.drain()
        raw = await reader.read()
    finally:
        writer.close()
        with suppress(ConnectionError, asyncio.TimeoutError):
            await writer.wait_closed()

    head, _, rest = raw.partition(b"\r\n\r\n")
    header_lines = head.split(b"\r\n")
    status_line = header_lines[0]
    headers = {}
    for line in header_lines[1:]:
        name, _, value = line.partition(b":")
        headers[name.decode().strip().lower()] = value.decode().strip()

    body = b""
    trailers: Dict[str, str] = {}
    assert headers.get("transfer-encoding") == "chunked"
    while True:
        size_line, _, rest = rest.partition(b"\r\n")
        size = int(size_line, 16)
        if size == 0:
            trailer_block, _, _ = rest.partition(b"\r\n\r\n")
            for line in trailer_block.split(b"\r\n"):
                if line:
                    name, _, value = line.partition(b":")
                    trailers[name.decode().strip().lower()] = value.decode().strip()
            break
        body += rest[:size]
        rest = rest[size + 2 :]
    return status_line, headers, body, trailers


def get_request(path: str, accept_encoding: str = "identity") -> bytes:
    return (
        f"GET {path} HTTP/1.1\r\nHost: localhost\r\n"
        f"Accept-Encoding: {accept_encoding}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")


async def test_digest_trailer_chunked(aiohttp_server: AiohttpServer) -> None:
    """Digest trailer matches the SHA-256 of the transferred entity."""
    parts = [b"hello ", b"world", b"!" * 1000]

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest_trailer()
        await resp.prepare(request)
        for part in parts:
            await resp.write(part)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    status, headers, body, trailers = await raw_request(
        server.host, server.port, get_request("/")
    )
    assert status == b"HTTP/1.1 200 OK"
    assert headers["trailer"] == "Digest"
    assert body == b"".join(parts)
    assert trailers["digest"] == digest_header_value(body)


@pytest.mark.parametrize("encoding", ["deflate", "gzip"])
async def test_digest_trailer_after_compression(
    aiohttp_server: AiohttpServer, encoding: str
) -> None:
    """With compression enabled, the digest covers the compressed bytes."""
    payload = b"hello world " * 500

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_compression(web.ContentCoding(encoding))
        resp.enable_digest_trailer()
        await resp.prepare(request)
        await resp.write(payload[:1000])
        await resp.write(payload[1000:])
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    status, headers, body, trailers = await raw_request(
        server.host, server.port, get_request("/", encoding)
    )
    assert status == b"HTTP/1.1 200 OK"
    assert headers["content-encoding"] == encoding
    assert headers["trailer"] == "Digest"
    # The digest is computed over the compressed entity as transferred.
    assert trailers["digest"] == digest_header_value(body)
    if encoding == "gzip":
        assert zlib.decompress(body, zlib.MAX_WBITS | 16) == payload
    else:
        assert zlib.decompress(body) == payload


async def test_digest_trailer_empty_stream(aiohttp_server: AiohttpServer) -> None:
    """An empty chunked body still gets the digest of zero bytes."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest_trailer()
        await resp.prepare(request)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    status, headers, body, trailers = await raw_request(
        server.host, server.port, get_request("/")
    )
    assert status == b"HTTP/1.1 200 OK"
    assert body == b""
    assert trailers["digest"] == digest_header_value(b"")


async def test_digest_trailer_repeated_write_eof(aiohttp_server: AiohttpServer) -> None:
    """A second write_eof() is a no-op and does not duplicate the trailer."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest_trailer()
        await resp.prepare(request)
        await resp.write(b"data")
        await resp.write_eof()
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    status, headers, body, trailers = await raw_request(
        server.host, server.port, get_request("/")
    )
    assert status == b"HTTP/1.1 200 OK"
    assert body == b"data"
    assert trailers["digest"] == digest_header_value(b"data")


async def test_digest_trailer_not_sent_on_client_disconnect(
    aiohttp_server: AiohttpServer,
) -> None:
    """If the client goes away mid-body, no digest trailer is published."""
    first_chunk_sent = asyncio.Event()
    handler_done = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest_trailer()
        await resp.prepare(request)
        await resp.write(b"x" * 1024)
        first_chunk_sent.set()
        with suppress(ConnectionError, asyncio.CancelledError):
            for _ in range(1000):
                await resp.write(b"y" * 1024)
                await asyncio.sleep(0.01)
            await resp.write_eof()
        handler_done.set()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(get_request("/"))
    await writer.drain()
    await asyncio.wait_for(first_chunk_sent.wait(), 5)
    received = await reader.read(4096)
    # Disconnect in the middle of the body.
    writer.close()
    with suppress(ConnectionError):
        await writer.wait_closed()
    await asyncio.wait_for(handler_done.wait(), 5)

    assert b"Digest:" not in received


async def test_digest_trailer_not_sent_on_write_failure(
    aiohttp_server: AiohttpServer,
) -> None:
    """A failing write means the response never completes, so no trailer."""
    handler_done = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest_trailer()
        await resp.prepare(request)
        await resp.write(b"partial")
        # Simulate the connection dying before the final chunk.
        assert request.transport is not None
        request.transport.close()
        with suppress(ConnectionError):
            await resp.write(b"more data")
            await resp.write_eof()
        handler_done.set()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(get_request("/"))
    await writer.drain()
    received = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    with suppress(ConnectionError):
        await writer.wait_closed()
    await asyncio.wait_for(handler_done.wait(), 5)

    assert b"Digest:" not in received


async def test_digest_trailer_conflict_with_existing_trailer() -> None:
    """Enabling the digest trailer twice via the Trailer header fails."""
    resp = web.StreamResponse(headers={"Trailer": "Digest"})
    with pytest.raises(RuntimeError, match="already declares Digest"):
        resp.enable_digest_trailer()


async def test_digest_trailer_merges_with_existing_trailer(
    aiohttp_server: AiohttpServer,
) -> None:
    """A pre-existing Trailer header is kept and Digest is appended."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Trailer": "X-Custom"})
        resp.enable_digest_trailer()
        await resp.prepare(request)
        await resp.write(b"data")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    status, headers, body, trailers = await raw_request(
        server.host, server.port, get_request("/")
    )
    assert status == b"HTTP/1.1 200 OK"
    assert headers["trailer"] == "X-Custom, Digest"
    assert trailers["digest"] == digest_header_value(b"data")


async def test_digest_trailer_rejects_fixed_content_length() -> None:
    """A fixed Content-Length response cannot carry trailers."""
    resp = web.StreamResponse()
    resp.content_length = 10
    with pytest.raises(RuntimeError, match="incompatible with a fixed"):
        resp.enable_digest_trailer()


async def test_digest_trailer_rejects_after_prepare(
    aiohttp_server: AiohttpServer,
) -> None:
    """The digest trailer must be enabled before the response starts."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_chunked_encoding()
        await resp.prepare(request)
        with pytest.raises(RuntimeError, match="after the response"):
            resp.enable_digest_trailer()
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    status, headers, body, trailers = await raw_request(
        server.host, server.port, get_request("/")
    )
    assert status == b"HTTP/1.1 200 OK"
    assert "digest" not in trailers


async def test_digest_trailer_rejects_http10(aiohttp_server: AiohttpServer) -> None:
    """HTTP/1.0 has no chunked encoding, so the digest trailer is rejected."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest_trailer()
        await resp.prepare(request)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(b"GET / HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n")
    await writer.drain()
    raw = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    with suppress(ConnectionError):
        await writer.wait_closed()

    # prepare() raises RuntimeError because chunked encoding requires
    # HTTP/1.1; the request fails instead of silently dropping the trailer.
    assert b" 500 " in raw.split(b"\r\n")[0]
    assert b"Digest:" not in raw


async def test_digest_trailer_response_object(aiohttp_server: AiohttpServer) -> None:
    """web.Response with a fixed body can also carry the digest trailer."""

    async def handler(request: web.Request) -> web.Response:
        resp = web.Response(body=b"response body")
        resp.enable_digest_trailer()
        return resp

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    status, headers, body, trailers = await raw_request(
        server.host, server.port, get_request("/")
    )
    assert status == b"HTTP/1.1 200 OK"
    assert "content-length" not in headers
    assert body == b"response body"
    assert trailers["digest"] == digest_header_value(b"response body")
