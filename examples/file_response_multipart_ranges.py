"""Demo of FileResponse multi-range (multipart/byteranges) support.

Starts a local loopback server serving a file containing "0123456789",
sends raw HTTP requests over a socket, and prints the exact request and
response headers (including a 416 for an unsatisfiable range).

Runs in a few seconds; no external services needed. From a source
checkout, run it against the in-tree aiohttp with:

    PYTHONPATH=. python examples/file_response_multipart_ranges.py
"""

import asyncio
import tempfile
from pathlib import Path

from aiohttp import web


async def show_request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    title: str,
    request_head: bytes,
    read_body: bool = True,
) -> None:
    print(f"=== {title} ===")
    print("--- request ---")
    print(request_head.decode().strip())
    writer.write(request_head)
    await writer.drain()

    response_head = await reader.readuntil(b"\r\n\r\n")
    print("--- response headers ---")
    print(response_head.decode().strip())
    headers = {}
    for line in response_head.decode().split("\r\n")[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    if read_body and "content-length" in headers:
        body = await reader.readexactly(int(headers["content-length"]))
        print("--- response body ---")
        print(repr(body))
    print()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.bin"
        path.write_bytes(b"0123456789")

        async def handler(request: web.Request) -> web.FileResponse:
            return web.FileResponse(path)

        app = web.Application()
        app.router.add_get("/", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:

            async def roundtrip(title: str, head: str, read_body: bool = True) -> None:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                try:
                    await show_request(reader, writer, title, head.encode(), read_body)
                finally:
                    writer.close()

            await roundtrip(
                "two ranges: bytes=0-1,8-",
                "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                "Range: bytes=0-1,8-\r\nConnection: close\r\n\r\n",
            )
            await roundtrip(
                "mergeable ranges: bytes=0-2,2-4",
                "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                "Range: bytes=0-2,2-4\r\nConnection: close\r\n\r\n",
            )
            await roundtrip(
                "unsatisfiable range: bytes=99-100 (expect 416)",
                "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                "Range: bytes=99-100\r\nConnection: close\r\n\r\n",
            )
            await roundtrip(
                "HEAD with two ranges (headers only)",
                "HEAD / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                "Range: bytes=0-1,8-\r\nConnection: close\r\n\r\n",
                read_body=False,
            )
            await roundtrip(
                "stale If-Range falls back to full 200",
                "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                'If-Range: "stale-etag"\r\n'
                "Range: bytes=0-1,8-\r\nConnection: close\r\n\r\n",
            )
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
