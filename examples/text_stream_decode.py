#!/usr/bin/env python3
"""Incrementally decode a response body into text with raw byte offsets.

The adapter decodes small byte chunks as they arrive (multi-byte characters
and the BOM may span chunk boundaries), reports the original byte offset of
every decoded piece, and never buffers the whole response.  In strict mode
an invalid sequence raises TextStreamDecodeError carrying the first raw
byte offset of the offending bytes, and the connection is released either
way.
"""

import asyncio

import aiohttp
from aiohttp import TextStreamDecodeError


async def fetch(session: aiohttp.ClientSession) -> None:
    async with session.get("http://httpbin.org/encoding/utf8") as resp:
        # Charset is determined from the response headers (defaulting to
        # UTF-8); pass encoding=... to override it explicitly.
        decoder = resp.text_decoder()
        print(f"decoding as {decoder.encoding}")
        async for chunk in decoder:
            # chunk.offset is the byte offset in the original encoded body
            print(f"[byte {chunk.offset:>6}] {chunk.text!r}")


async def fetch_strict(session: aiohttp.ClientSession) -> None:
    async with session.get("http://httpbin.org/encoding/utf8") as resp:
        try:
            async for chunk in resp.text_decoder(errors="strict"):
                pass
        except TextStreamDecodeError as exc:
            # the connection has already been released at this point
            print(f"invalid {exc.encoding} at raw byte offset {exc.stream_offset}")


async def go() -> None:
    async with aiohttp.ClientSession() as session:
        await fetch(session)
        await fetch_strict(session)


loop = asyncio.get_event_loop()
loop.run_until_complete(go())
loop.close()
