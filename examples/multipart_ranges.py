"""Demonstrate FileResponse support for multiple byte ranges.

Starts a loopback HTTP server serving a file containing ``0123456789``
and issues raw HTTP/1.1 requests against it, printing each request and
the raw response headers:

* ``bytes=0-1,8-``   -> 206 multipart/byteranges with parts "01" and "89"
* ``bytes=0-2,2-4``  -> 206 single range "01234" (adjacent ranges merged)
* ``bytes=99-100``   -> 416 Range Not Satisfiable
* HEAD + multi-range -> 206 with the same headers but no body
* If-Range mismatch  -> 200 full representation

Runs entirely on the loopback interface and finishes in a few seconds.
"""

import asyncio
import pathlib
import tempfile

from aiohttp import web


async def raw_request(port: int, request: bytes) -> None:
    """Send a raw HTTP request, printing it and the raw response."""
    print("=" * 72)
    print(">>> Request")
    print(request.decode().rstrip())
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    response = await asyncio.wait_for(reader.read(-1), timeout=10)
    writer.close()
    await writer.wait_closed()
    head, _, body = response.partition(b"\r\n\r\n")
    print("<<< Response headers")
    print(head.decode().rstrip())
    print(f"<<< Body ({len(body)} bytes)")
    print(body.decode(errors="replace"))


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "digits.txt"
        path.write_bytes(b"0123456789")

        async def handler(request: web.Request) -> web.FileResponse:
            return web.FileResponse(path)

        app = web.Application()
        app.router.add_get("/", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port = runner.addresses[0][:2]
        print(f"Serving {path} (10 bytes) on http://{host}:{port}")

        def request(method: str, *headers: str) -> bytes:
            lines = [f"{method} / HTTP/1.1", f"Host: {host}:{port}", *headers]
            lines.append("Connection: close")
            return ("\r\n".join(lines) + "\r\n\r\n").encode()

        try:
            await raw_request(port, request("GET", "Range: bytes=0-1,8-"))
            await raw_request(port, request("GET", "Range: bytes=0-2,2-4"))
            await raw_request(port, request("GET", "Range: bytes=99-100"))
            await raw_request(port, request("HEAD", "Range: bytes=0-1,8-"))
            await raw_request(
                port, request("GET", "Range: bytes=0-1,8-", 'If-Range: "stale"')
            )
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
