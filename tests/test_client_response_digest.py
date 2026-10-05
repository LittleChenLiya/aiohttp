"""Acceptance tests for opt-in response Content-Digest (sha-256) verification."""

import base64
import gzip
import hashlib
import socket
import threading
from collections import deque

import pytest

from aiohttp import ClientSession, ContentDigestError
from aiohttp.client_exceptions import ClientPayloadError
from aiohttp.client_reqrep import _parse_content_digest_sha256


def _digest_header(data: bytes) -> bytes:
    """Build a Content-Digest header line for the sha-256 of *data*."""
    digest = base64.b64encode(hashlib.sha256(data).digest())
    return b"Content-Digest: sha-256=:" + digest + b":\r\n"


def _http_response(
    body: bytes,
    *,
    content_length: bool = True,
    extra_headers: bytes = b"",
    digest_of: bytes | None = None,
) -> bytes:
    """Build a raw HTTP/1.1 response with a correct Content-Digest."""
    headers = b"HTTP/1.1 200 OK\r\n"
    if content_length:
        headers += b"Content-Length: %d\r\n" % len(body)
    headers += extra_headers
    if digest_of is not None:
        headers += _digest_header(digest_of)
    return headers + b"\r\n" + body


def _chunked_response(chunks: list[bytes], *, digest_of: bytes) -> bytes:
    body = b"".join(b"%x\r\n%s\r\n" % (len(c), c) for c in chunks) + b"0\r\n\r\n"
    return (
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
        + _digest_header(digest_of)
        + b"\r\n"
        + body
    )


class _Close:
    """Script sentinel: close the connection without (further) response."""


CLOSE = _Close()


class ScriptedServer:
    """A tiny scriptable HTTP/1.1 server running in background threads.

    Each request received (on any accepted connection) consumes the next
    script entry: bytes are sent verbatim, CLOSE closes the connection.
    """

    def __init__(self, script):
        self._script = deque(script)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(10)
        self.port = self._sock.getsockname()[1]
        self._closed = False
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(10)
        try:
            while self._script:
                if self._read_request(conn) is None:
                    return
                entry = self._script.popleft()
                if isinstance(entry, _Close):
                    return
                conn.sendall(entry)
        except OSError:
            pass
        finally:
            conn.close()

    @staticmethod
    def _read_request(conn: socket.socket) -> bytes | None:
        data = b""
        while b"\r\n\r\n" not in data:
            try:
                chunk = conn.recv(65536)
            except OSError:
                return None
            if not chunk:
                return None
            data += chunk
        return data

    def close(self) -> None:
        self._closed = True
        self._sock.close()


@pytest.fixture
def server_factory():
    servers = []

    def factory(script) -> ScriptedServer:
        server = ScriptedServer(script)
        servers.append(server)
        return server

    yield factory

    for server in servers:
        server.close()


BODY = b"Hello, Content-Digest! " * 40  # 960 bytes


async def test_digest_plain_ok(server_factory) -> None:
    server = server_factory([_http_response(BODY, digest_of=BODY)])
    async with ClientSession(verify_response_digest=True) as session:
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_chunked_ok(server_factory) -> None:
    chunks = [BODY[:100], BODY[100:500], BODY[500:]]
    server = server_factory([_chunked_response(chunks, digest_of=BODY)])
    async with ClientSession(verify_response_digest=True) as session:
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_gzip_ok(server_factory) -> None:
    compressed = gzip.compress(BODY)
    # The digest covers the bytes as received on the wire, i.e. the
    # still-compressed payload before automatic decompression.
    server = server_factory(
        [
            _http_response(
                compressed,
                extra_headers=b"Content-Encoding: gzip\r\n",
                digest_of=compressed,
            )
        ]
    )
    async with ClientSession(verify_response_digest=True) as session:
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_mismatch_then_next_request_ok(server_factory) -> None:
    corrupted = BODY[:-1] + bytes([BODY[-1] ^ 0xFF])
    server = server_factory(
        [
            _http_response(BODY, digest_of=corrupted),  # digest of altered body
            _http_response(BODY, digest_of=BODY),
        ]
    )
    async with ClientSession(verify_response_digest=True) as session:
        with pytest.raises(ContentDigestError):
            async with session.get(server.url) as resp:
                await resp.read()
        # The failed response must not corrupt the connection state: the
        # next request on the same session reads its response cleanly.
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_empty_body_ok(server_factory) -> None:
    server = server_factory([_http_response(b"", digest_of=b"")])
    async with ClientSession(verify_response_digest=True) as session:
        async with session.get(server.url) as resp:
            assert await resp.read() == b""


async def test_digest_empty_body_mismatch(server_factory) -> None:
    server = server_factory(
        [
            _http_response(b"", digest_of=b"not-empty"),
            _http_response(BODY, digest_of=BODY),
        ]
    )
    async with ClientSession(verify_response_digest=True) as session:
        with pytest.raises(ContentDigestError):
            async with session.get(server.url) as resp:
                await resp.read()
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_premature_close_then_next_request_ok(server_factory) -> None:
    # Server promises 1000 bytes but closes the connection after 100.
    partial = (
        b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n"
        + _digest_header(BODY)
        + b"\r\n"
        + BODY[:100]
    )
    server = server_factory([partial, CLOSE, _http_response(BODY, digest_of=BODY)])
    async with ClientSession(verify_response_digest=True) as session:
        with pytest.raises(ClientPayloadError):
            async with session.get(server.url) as resp:
                await resp.read()
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_malformed_header_then_next_request_ok(server_factory) -> None:
    bad = (
        b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n"
        b"Content-Digest: sha-256=!!!not-a-byte-sequence!!!\r\n\r\n" % len(BODY)
    ) + BODY
    server = server_factory([bad, _http_response(BODY, digest_of=BODY)])
    async with ClientSession(verify_response_digest=True) as session:
        with pytest.raises(ContentDigestError):
            async with session.get(server.url) as resp:
                await resp.read()
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_missing_header_skipped_by_default(server_factory) -> None:
    server = server_factory([_http_response(BODY)])
    async with ClientSession(verify_response_digest=True) as session:
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_missing_header_required(server_factory) -> None:
    server = server_factory([_http_response(BODY)])
    async with ClientSession(require_response_digest=True) as session:
        with pytest.raises(ContentDigestError):
            async with session.get(server.url) as resp:
                await resp.read()


async def test_digest_unsupported_algorithm_skipped(server_factory) -> None:
    headers = b"Content-Digest: sha-512=:" + base64.b64encode(b"x" * 64) + b":\r\n"
    server = server_factory([_http_response(BODY, extra_headers=headers)])
    async with ClientSession(verify_response_digest=True) as session:
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_not_verified_when_disabled(server_factory) -> None:
    corrupted = BODY[:-1] + bytes([BODY[-1] ^ 0xFF])
    server = server_factory([_http_response(BODY, digest_of=corrupted)])
    async with ClientSession() as session:  # verification not enabled
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_consecutive_requests_keep_alive(server_factory) -> None:
    other = b"second response body"
    server = server_factory(
        [
            _http_response(BODY, digest_of=BODY),
            _http_response(other, digest_of=other),
        ]
    )
    async with ClientSession(verify_response_digest=True) as session:
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY
        async with session.get(server.url) as resp:
            assert await resp.read() == other


async def test_digest_per_request_option(server_factory) -> None:
    corrupted = BODY[:-1] + bytes([BODY[-1] ^ 0xFF])
    server = server_factory(
        [
            _http_response(BODY, digest_of=corrupted),
            _http_response(BODY, digest_of=corrupted),
        ]
    )
    async with ClientSession() as session:
        # Enabled per-request: mismatch is detected.
        with pytest.raises(ContentDigestError):
            async with session.get(server.url, verify_response_digest=True) as resp:
                await resp.read()
        # Not enabled: mismatch is ignored.
        async with session.get(server.url) as resp:
            assert await resp.read() == BODY


async def test_digest_head_request_not_verified(server_factory) -> None:
    # A HEAD response carries no payload; Content-Digest describes the
    # (absent) representation and must not be checked against an empty body.
    headers = (
        b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n" % len(BODY)
        + _digest_header(BODY)
        + b"\r\n"
    )
    server = server_factory([headers])
    async with ClientSession(verify_response_digest=True) as session:
        async with session.head(server.url) as resp:
            assert await resp.read() == b""


def test_parse_content_digest_sha256_valid() -> None:
    digest = hashlib.sha256(BODY).digest()
    header = "sha-256=:%s:" % base64.b64encode(digest).decode()
    assert _parse_content_digest_sha256(header) == digest


def test_parse_content_digest_sha256_among_others() -> None:
    digest = hashlib.sha256(BODY).digest()
    header = "sha-512=:%s:, sha-256=:%s:" % (
        base64.b64encode(b"y" * 64).decode(),
        base64.b64encode(digest).decode(),
    )
    assert _parse_content_digest_sha256(header) == digest


def test_parse_content_digest_sha256_absent() -> None:
    assert _parse_content_digest_sha256("sha-512=:eHh4:") is None


@pytest.mark.parametrize(
    "header",
    [
        "sha-256=not-a-byte-sequence",
        "sha-256=:!!!invalid-base64!!!:",
        "sha-256=:%s:" % base64.b64encode(b"too-short").decode(),
        "sha-256",
    ],
)
def test_parse_content_digest_sha256_malformed(header: str) -> None:
    with pytest.raises(ValueError):
        _parse_content_digest_sha256(header)
