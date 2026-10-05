"""Integration tests for incremental text decoding in ClientResponse."""

import asyncio
from unittest import mock

import pytest
from multidict import CIMultiDict
from yarl import URL

from aiohttp import ClientSession
from aiohttp.abc import AbstractStreamWriter
from aiohttp.client_reqrep import ClientResponse
from aiohttp.connector import Connection
from aiohttp.helpers import HeadersDictProxy, TimerNoop

from tests.test_client_response import WriterMock


@pytest.fixture
def session() -> mock.Mock:
    return mock.Mock()


def _make_chunked_content(
    loop: asyncio.AbstractEventLoop, chunks: list
) -> mock.Mock:
    content = mock.Mock()
    chunk_iter = iter(chunks)

    async def readchunk():
        try:
            data = next(chunk_iter)
        except StopIteration:
            return (b"", False)
        return (data, False)

    content.readchunk = readchunk
    return content


def _make_response(
    loop: asyncio.AbstractEventLoop,
    session: ClientSession,
    headers: dict,
    content: mock.Mock,
) -> ClientResponse:
    url = URL("http://def-cl-resp.org")
    response = ClientResponse(
        "get",
        url,
        writer=WriterMock(),
        continue100=None,
        timer=TimerNoop(),
        traces=[],
        loop=loop,
        session=session,
        request_headers=CIMultiDict(),
        original_url=url,
        stream_writer=mock.create_autospec(
            AbstractStreamWriter, spec_set=True, instance=True
        ),
    )
    response._headers = HeadersDictProxy(CIMultiDict(headers))
    response.content = content
    return response


async def test_text_incremental_multibyte_split_into_three_chunks(
    session: ClientSession,
) -> None:
    loop = asyncio.get_running_loop()
    data = "猫".encode("utf-8")
    chunks = [data[:1], data[1:2], data[2:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    result = await response.text()
    assert result == "猫"
    assert response._body == data
    assert response._connection is None


async def test_text_incremental_multiple_chars_across_chunks(
    session: ClientSession,
) -> None:
    loop = asyncio.get_running_loop()
    data = "你好世界".encode("utf-8")
    chunks = [data[:2], data[2:5], data[5:8], data[8:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    result = await response.text()
    assert result == "你好世界"


async def test_text_incremental_invalid_byte_reports_offset(
    session: ClientSession,
) -> None:
    loop = asyncio.get_running_loop()
    data = b"ab\xffcd"
    chunks = [data[:2], data[2:3], data[3:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    with pytest.raises(UnicodeDecodeError) as exc_info:
        await response.text()
    assert exc_info.value.start == 2
    assert exc_info.value.end == 3


async def test_text_incremental_invalid_byte_after_buffered_sequence(
    session: ClientSession,
) -> None:
    loop = asyncio.get_running_loop()
    data = b"\xe7\x8c\xff"
    chunks = [data[:2], data[2:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    with pytest.raises(UnicodeDecodeError) as exc_info:
        await response.text()
    assert exc_info.value.start == 0


async def test_text_incremental_failure_returns_connection(
    session: ClientSession,
) -> None:
    loop = asyncio.get_running_loop()
    data = b"ab\xffcd"
    chunks = [data[:2], data[2:3], data[3:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    connection = mock.Mock(spec=Connection)
    connection.protocol = mock.Mock()
    connection.protocol.upgraded = False
    response._connection = connection
    with pytest.raises(UnicodeDecodeError):
        await response.text()
    connection.release.assert_called_once()
    assert response._connection is None


async def test_text_incremental_bom_stripped_at_start(
    session: ClientSession,
) -> None:
    loop = asyncio.get_running_loop()
    bom = b"\xef\xbb\xbf"
    data = bom + "hello".encode("utf-8")
    chunks = [data[:2], data[2:5], data[5:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    result = await response.text()
    assert result == "hello"


async def test_text_incremental_bom_only_at_beginning(
    session: ClientSession,
) -> None:
    loop = asyncio.get_running_loop()
    bom = b"\xef\xbb\xbf"
    data = "hello".encode("utf-8") + bom + "world".encode("utf-8")
    chunks = [data[:5], data[5:8], data[8:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    result = await response.text()
    assert result == "hello﻿world"


async def test_text_incremental_empty_chunks(session: ClientSession) -> None:
    """Empty chunks at the decoder level do not affect decoding."""
    from aiohttp._text_decoder import _TextDecoder

    dec = _TextDecoder("utf-8")
    assert dec.decode(b"") == ""
    assert dec.decode(b"hello") == "hello"
    assert dec.decode(b"") == ""
    assert dec.decode(b"world") == "world"
    assert dec.decode(b"", final=True) == ""


async def test_text_incremental_eof_incomplete_sequence(
    session: ClientSession,
) -> None:
    loop = asyncio.get_running_loop()
    data = b"\xe7\x8c"
    chunks = [data[:1], data[1:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    with pytest.raises(UnicodeDecodeError) as exc_info:
        await response.text()
    assert exc_info.value.start == 0


async def test_text_incremental_empty_body(session: ClientSession) -> None:
    loop = asyncio.get_running_loop()
    content = _make_chunked_content(loop, [])
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    result = await response.text()
    assert result == ""
    assert response._body == b""


async def test_text_incremental_cp1251(session: ClientSession) -> None:
    loop = asyncio.get_running_loop()
    data = '{"тест": "пройден"}'.encode("cp1251")
    chunks = [data[:5], data[5:10], data[10:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "application/json;charset=cp1251"}, content
    )
    result = await response.text()
    assert result == '{"тест": "пройден"}'


async def test_text_incremental_errors_ignore(session: ClientSession) -> None:
    loop = asyncio.get_running_loop()
    data = b"ab\xffcd"
    chunks = [data[:2], data[2:3], data[3:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )
    result = await response.text(errors="ignore")
    assert result == "abcd"


async def test_text_incremental_buffer_does_not_accumulate_response(
    session: ClientSession,
) -> None:
    """The decoder buffer only holds incomplete characters, not the whole response."""
    loop = asyncio.get_running_loop()
    data = "你好世界".encode("utf-8")
    chunks = [data[:2], data[2:5], data[5:8], data[8:]]
    content = _make_chunked_content(loop, chunks)
    response = _make_response(
        loop, session, {"Content-Type": "text/plain;charset=utf-8"}, content
    )

    from aiohttp import _text_decoder

    original_decode = _text_decoder._TextDecoder.decode
    max_buffer = 0

    def tracking_decode(self, data, final=False):
        nonlocal max_buffer
        result = original_decode(self, data, final)
        max_buffer = max(max_buffer, len(self.buffer))
        return result

    _text_decoder._TextDecoder.decode = tracking_decode
    try:
        await response.text()
    finally:
        _text_decoder._TextDecoder.decode = original_decode
    # The buffer should never hold more than 3 bytes (max incomplete UTF-8 sequence).
    assert max_buffer <= 3
