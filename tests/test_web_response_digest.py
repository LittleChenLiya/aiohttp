"""End-to-end tests for the StreamResponse Digest trailer.

These tests talk to the server over a raw loopback socket and verify the
chunked framing, the announced Trailer header and the digest value by
hashing the exact entity bytes received on the wire.
"""

import asyncio
import base64
import gzip
import hashlib
import zlib
from contextlib import suppress

import pytest
from pytest_aiohttp import AiohttpRawServer

from aiohttp import web


def _parse_response(raw: bytes) -> tuple[bytes, dict[str, str], bytes, bytes]:
    head, _, rest = raw.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status_line = lines[0]
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(b":")
        headers[name.decode().strip().lower()] = value.decode().strip()
    entity = b""
    i = 0
    while True:
        j = rest.index(b"\r\n", i)
        size = int(rest[i:j], 16)
        i = j + 2
        if size == 0:
            end = rest.index(b"\r\n\r\n", i)
            trailer_block = rest[i:end]
            break
        entity += rest[i : i + size]
        i += size + 2
    trailers = {}
    for line in trailer_block.split(b"\r\n"):
        name, _, value = line.partition(b":")
        trailers[name.decode().strip().lower()] = value.decode().strip()
    return status_line, headers, entity, trailers


async def _raw_request(
    server: AiohttpRawServer,
    path: str = "/",
    extra_headers: str = "",
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection(server.host, server.port)
    request = f"GET {path} HTTP/1.1\r\nHost: example.com\r\n"
    if extra_headers:
        request += extra_headers.rstrip("\r\n") + "\r\n"
    writer.write((request + "\r\n").encode())
    await writer.drain()
    return reader, writer


async def _read_chunked_response(reader: asyncio.StreamReader) -> bytes:
    """Read until the chunked terminating CRLF after the trailers."""
    data = await reader.readuntil(b"\r\n\r\n")
    body_start = data.index(b"\r\n\r\n") + 4
    while data.count(b"\r\n\r\n", body_start) == 0:
        data += await reader.readuntil(b"\r\n\r\n")
    return data


async def test_digest_trailer_raw_loopback(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    payload = b"0123456789" * 1000

    async def handler(request: web.BaseRequest) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest()
        await resp.prepare(request)
        for offset in range(0, len(payload), 128):
            await resp.write(payload[offset : offset + 128])
        await resp.write_eof()
        return resp

    server = await aiohttp_raw_server(handler)
    reader, sock_writer = await _raw_request(server)
    try:
        raw = await _read_chunked_response(reader)
    finally:
        sock_writer.close()
        await sock_writer.wait_closed()

    status_line, headers, entity, trailers = _parse_response(raw)
    assert status_line.startswith(b"HTTP/1.1 200")
    assert headers["transfer-encoding"] == "chunked"
    assert headers["trailer"] == "Digest"
    assert entity == payload
    digest_value = trailers["digest"]
    assert digest_value.startswith("sha-256=:")
    received = base64.b64decode(digest_value[len("sha-256=:") : -1])
    assert received == hashlib.sha256(entity).digest()


@pytest.mark.parametrize("coding", ["gzip", "deflate"])
async def test_digest_trailer_after_compression_raw_loopback(
    aiohttp_raw_server: AiohttpRawServer, coding: str
) -> None:
    payload = b"abcdefghij" * 500

    async def handler(request: web.BaseRequest) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_compression(force=web.ContentCoding(coding))
        resp.enable_digest()
        await resp.prepare(request)
        await resp.write(payload)
        await resp.write_eof()
        return resp

    server = await aiohttp_raw_server(handler)
    reader, sock_writer = await _raw_request(
        server, extra_headers="Accept-Encoding: gzip, deflate\r\n"
    )
    try:
        raw = await _read_chunked_response(reader)
    finally:
        sock_writer.close()
        await sock_writer.wait_closed()

    status_line, headers, entity, trailers = _parse_response(raw)
    assert status_line.startswith(b"HTTP/1.1 200")
    assert headers["content-encoding"] == coding
    assert headers["trailer"] == "Digest"
    # The digest must match the *compressed* entity bytes received.
    digest_value = trailers["digest"]
    received = base64.b64decode(digest_value[len("sha-256=:") : -1])
    assert received == hashlib.sha256(entity).digest()
    decompressor = (
        gzip.decompress if coding == "gzip" else lambda data: zlib.decompress(data)
    )
    assert decompressor(entity) == payload


async def test_digest_trailer_empty_stream(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    async def handler(request: web.BaseRequest) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest()
        await resp.prepare(request)
        await resp.write_eof()
        return resp

    server = await aiohttp_raw_server(handler)
    reader, sock_writer = await _raw_request(server)
    try:
        raw = await _read_chunked_response(reader)
    finally:
        sock_writer.close()
        await sock_writer.wait_closed()

    _, headers, entity, trailers = _parse_response(raw)
    assert entity == b""
    assert headers["trailer"] == "Digest"
    received = base64.b64decode(trailers["digest"][len("sha-256=:") : -1])
    assert received == hashlib.sha256(b"").digest()


async def test_digest_trailer_not_emitted_on_client_abort(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    first_chunk_written = asyncio.Event()
    client_aborted = asyncio.Event()

    async def handler(request: web.BaseRequest) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.enable_digest()
        await resp.prepare(request)
        await resp.write(b"a" * 16)
        first_chunk_written.set()
        await client_aborted.wait()
        # Let the server observe the lost transport before writing again.
        await asyncio.sleep(0.1)
        with pytest.raises((ConnectionError, asyncio.CancelledError)):
            # A large write forces backpressure; against an aborted peer the
            # drain fails instead of completing the response.
            await resp.write(b"b" * (16 * 1024 * 1024))
            await resp.write_eof()
        return resp

    server = await aiohttp_raw_server(handler)
    reader, sock_writer = await asyncio.open_connection(
        server.host, server.port, limit=1024
    )
    sock_writer.write(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    await sock_writer.drain()
    received = bytearray()
    try:
        await first_chunk_written.wait()
        head = await reader.readuntil(b"\r\n\r\n")
        received.extend(head)
        assert b"Trailer: Digest" in head
        chunk_line = await reader.readline()
        assert chunk_line.strip() == b"10"
        received.extend(chunk_line)
        first_chunk = await reader.readexactly(16)
        assert first_chunk == b"a" * 16
        received.extend(first_chunk)
        received.extend(await reader.readline())
        # Abort the connection while the handler waits, then release it.
        sock_writer.transport.abort()
        client_aborted.set()
        # The peer is gone: reads terminate (possibly with an error).
        with suppress(Exception):
            while True:
                more = await asyncio.wait_for(reader.read(1024), timeout=1)
                if not more:
                    break
                received.extend(more)
    finally:
        with suppress(Exception):
            sock_writer.close()
            await sock_writer.wait_closed()

        # Give the handler time to notice the abort.
        await asyncio.sleep(0.2)

    # The announced trailer must never have arrived after the interruption.
    assert b"\r\n0\r\nDigest:" not in bytes(received)
