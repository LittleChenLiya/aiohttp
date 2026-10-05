"""Tests for aiohttp.text_decoder (incremental response text decoding)."""

import codecs

import pytest
from pytest_aiohttp import AiohttpClient

from aiohttp import web
from aiohttp.text_decoder import (
    ClientResponseTextDecoder,
    DecodedChunk,
    TextStreamDecodeError,
    TextStreamDecoder,
)

CAT = "猫".encode("utf-8")  # b"\xe7\x8c\xab" -- one char, three bytes


def test_multibyte_char_split_across_three_chunks() -> None:
    dec = TextStreamDecoder("utf-8")
    assert dec.feed(CAT[:1]) == []
    assert dec.feed(CAT[1:2]) == []
    assert dec.feed(CAT[2:]) == [DecodedChunk("猫", 0)]
    assert dec.finish() == []
    # only the incomplete character was ever buffered
    assert dec.buffered == 0


def test_offsets_track_raw_byte_positions() -> None:
    dec = TextStreamDecoder("utf-8")
    assert dec.feed(b"ab" + CAT) == [DecodedChunk("ab猫", 0)]
    assert dec.feed(b"cd") == [DecodedChunk("cd", 5)]
    assert dec.finish() == []


def test_invalid_byte_reports_first_raw_offset() -> None:
    dec = TextStreamDecoder("utf-8")
    dec.feed(b"ab" + CAT)  # 5 raw bytes consumed
    with pytest.raises(TextStreamDecodeError) as ctx:
        dec.feed(b"\xff")
    assert ctx.value.stream_offset == 5
    assert isinstance(ctx.value, UnicodeDecodeError)
    assert "offset 5" in str(ctx.value)


def test_invalid_byte_after_split_sequence_reports_sequence_start() -> None:
    dec = TextStreamDecoder("utf-8")
    dec.feed(b"x")
    assert dec.feed(CAT[:2]) == []  # lead bytes of an incomplete char
    with pytest.raises(TextStreamDecodeError) as ctx:
        dec.feed(b"\xff")  # invalid continuation: whole sequence is bad
    assert ctx.value.stream_offset == 1


def test_strict_mode_never_emits_replacement_characters() -> None:
    dec = TextStreamDecoder("utf-8")  # strict is the default
    with pytest.raises(TextStreamDecodeError):
        dec.feed(b"\x80abc")
    # non-strict replacement must be explicitly opted into
    dec = TextStreamDecoder("utf-8", errors="replace")
    assert dec.feed(b"\x80abc") == [DecodedChunk("�abc", 0)]


def test_bom_stripped_only_at_stream_start() -> None:
    bom = codecs.BOM_UTF8
    dec = TextStreamDecoder("utf-8")
    assert dec.feed(bom + CAT) == [DecodedChunk("猫", 3)]
    assert dec.finish() == []

    # A BOM-looking sequence later in the stream is ordinary text (U+FEFF).
    dec = TextStreamDecoder("utf-8")
    chunks = dec.feed(b"x" + bom + b"y")
    assert chunks == [DecodedChunk("x﻿y", 0)]
    assert dec.finish() == []


def test_bom_split_across_chunks_and_empty_chunks() -> None:
    bom = codecs.BOM_UTF8
    dec = TextStreamDecoder("utf-8")
    assert dec.feed(b"") == []  # empty chunk must not decide the BOM probe
    assert dec.feed(bom[:1]) == []
    assert dec.feed(b"") == []
    assert dec.feed(bom[1:]) == []
    assert dec.feed(CAT[:1]) == []
    assert dec.feed(CAT[1:]) == [DecodedChunk("猫", 3)]
    assert dec.finish() == []


def test_no_bom_decided_as_soon_as_possible() -> None:
    dec = TextStreamDecoder("utf-8")
    # first byte cannot extend any BOM -> decided immediately
    assert dec.feed(b"a") == [DecodedChunk("a", 0)]
    assert dec.finish() == []


def test_empty_stream() -> None:
    dec = TextStreamDecoder("utf-8")
    assert dec.feed(b"") == []
    assert dec.finish() == []


def test_buffer_holds_only_incomplete_character() -> None:
    dec = TextStreamDecoder("utf-8")
    for i in range(100):
        assert dec.feed(CAT) == [DecodedChunk("猫", i * 3)]
        assert dec.buffered == 0
    dec.feed(CAT[:2])
    assert dec.buffered == 2  # one incomplete character, nothing more
    assert dec.feed(CAT[2:]) == [DecodedChunk("猫", 100 * 3)]
    assert dec.buffered == 0
    assert dec.finish() == []


def test_eof_with_truncated_character_raises_with_offset() -> None:
    dec = TextStreamDecoder("utf-8")
    dec.feed(b"ab")
    dec.feed(CAT[:2])
    with pytest.raises(TextStreamDecodeError) as ctx:
        dec.finish()
    assert ctx.value.stream_offset == 2


def test_feed_after_finish_rejected() -> None:
    dec = TextStreamDecoder("utf-8")
    dec.finish()
    with pytest.raises(RuntimeError):
        dec.feed(b"x")
    with pytest.raises(RuntimeError):
        dec.finish()


def test_utf16_le_bom_and_split_code_unit() -> None:
    dec = TextStreamDecoder("utf-16-le")
    data = codecs.BOM_UTF16_LE + "猫".encode("utf-16-le")
    assert dec.feed(data[:3]) == []  # BOM + half a code unit
    assert dec.feed(data[3:]) == [DecodedChunk("猫", 2)]  # after the 2-byte BOM
    assert dec.finish() == []


async def test_response_text_decoder_roundtrip(aiohttp_client: AiohttpClient) -> None:
    body = codecs.BOM_UTF8 + "猫cat".encode("utf-8")

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="text/plain", charset="utf-8")

    app = web.Application()
    app.router.add_get("/", handler)
    client = await aiohttp_client(app)

    async with client.get("/") as resp:
        decoder = resp.text_decoder()
        assert isinstance(decoder, ClientResponseTextDecoder)
        assert decoder.encoding == "utf-8"
        chunks = [chunk async for chunk in decoder]
    assert "".join(c.text for c in chunks) == "猫cat"
    assert chunks[0].offset == 3  # first char sits right after the 3-byte BOM
    assert resp.closed


async def test_response_decode_failure_releases_reusable_connection(
    aiohttp_client: AiohttpClient,
) -> None:
    good = "ok".encode("utf-8")
    bad = "ab".encode("utf-8") + b"\xff"  # invalid utf-8 at raw offset 2

    async def bad_handler(request: web.Request) -> web.Response:
        return web.Response(body=bad, content_type="text/plain", charset="utf-8")

    async def good_handler(request: web.Request) -> web.Response:
        return web.Response(body=good, content_type="text/plain", charset="utf-8")

    app = web.Application()
    app.router.add_get("/bad", bad_handler)
    app.router.add_get("/good", good_handler)
    client = await aiohttp_client(app)
    connector = client.session.connector

    async with client.get("/bad") as resp:
        decoder = resp.text_decoder()
        with pytest.raises(TextStreamDecodeError) as ctx:
            async for _ in decoder:
                pass
        assert ctx.value.stream_offset == 2

    # the failure released the connection back to the pool
    (pool,) = connector._conns.values()
    assert len(pool) == 1
    released_protocol = pool[0][0]

    # ... and the very same connection can serve the next request
    async with client.get("/good") as resp2:
        assert await resp2.text() == "ok"
    (pool2,) = connector._conns.values()
    assert len(pool2) == 1
    assert pool2[0][0] is released_protocol


async def test_response_decoder_close_releases_connection(
    aiohttp_client: AiohttpClient,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"data", content_type="text/plain")

    app = web.Application()
    app.router.add_get("/", handler)
    client = await aiohttp_client(app)
    connector = client.session.connector

    async with client.get("/") as resp:
        decoder = resp.text_decoder()
        decoder.close()
        assert resp.closed
    assert sum(len(conns) for conns in connector._conns.values()) == 1
