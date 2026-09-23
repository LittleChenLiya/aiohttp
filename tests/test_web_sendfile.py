import asyncio
import contextlib
import io
from collections.abc import Iterator
from pathlib import Path
from stat import S_IFREG, S_IRUSR, S_IWUSR
from typing import Any
from unittest import mock

import pytest

from aiohttp import hdrs
from aiohttp.http_writer import StreamWriter
from aiohttp.test_utils import make_mocked_request
from aiohttp.web_fileresponse import (
    FileResponse,
    _clip_ranges,
    _merge_ranges,
    _parse_range_header,
)

MOCK_MODE = S_IFREG | S_IRUSR | S_IWUSR


def test_using_gzip_if_header_present_and_file_available(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    request = make_mocked_request(
        "GET",
        "http://python.org/logo.png",
        # Header uses some uppercase to ensure case-insensitive treatment
        headers={hdrs.ACCEPT_ENCODING: "GZip"},
    )

    gz_filepath = mock.create_autospec(Path, spec_set=True)
    gz_filepath.lstat.return_value.st_size = 1024
    gz_filepath.lstat.return_value.st_mtime_ns = 1603733507222449291
    gz_filepath.lstat.return_value.st_mode = MOCK_MODE

    filepath = mock.create_autospec(Path, spec_set=True)
    filepath.name = "logo.png"
    filepath.with_suffix.return_value = gz_filepath

    file_sender = FileResponse(filepath)
    file_sender._path = filepath
    file_sender._sendfile = mock.AsyncMock(return_value=None)  # type: ignore[method-assign]

    event_loop.run_until_complete(file_sender.prepare(request))

    assert not filepath.open.called
    assert gz_filepath.open.called


def test_gzip_if_header_not_present_and_file_available(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    request = make_mocked_request("GET", "http://python.org/logo.png", headers={})

    gz_filepath = mock.create_autospec(Path, spec_set=True)
    gz_filepath.lstat.return_value.st_size = 1024
    gz_filepath.lstat.return_value.st_mtime_ns = 1603733507222449291
    gz_filepath.lstat.return_value.st_mode = MOCK_MODE

    filepath = mock.create_autospec(Path, spec_set=True)
    filepath.name = "logo.png"
    filepath.with_suffix.return_value = gz_filepath
    filepath.stat.return_value.st_size = 1024
    filepath.stat.return_value.st_mtime_ns = 1603733507222449291
    filepath.stat.return_value.st_mode = MOCK_MODE

    file_sender = FileResponse(filepath)
    file_sender._path = filepath
    file_sender._sendfile = mock.AsyncMock(return_value=None)  # type: ignore[method-assign]

    event_loop.run_until_complete(file_sender.prepare(request))

    assert filepath.open.called
    assert not gz_filepath.open.called


def test_gzip_if_header_not_present_and_file_not_available(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    request = make_mocked_request("GET", "http://python.org/logo.png", headers={})

    gz_filepath = mock.create_autospec(Path, spec_set=True)
    gz_filepath.stat.side_effect = OSError(2, "No such file or directory")

    filepath = mock.create_autospec(Path, spec_set=True)
    filepath.name = "logo.png"
    filepath.with_suffix.return_value = gz_filepath
    filepath.stat.return_value.st_size = 1024
    filepath.stat.return_value.st_mtime_ns = 1603733507222449291
    filepath.stat.return_value.st_mode = MOCK_MODE

    file_sender = FileResponse(filepath)
    file_sender._path = filepath
    file_sender._sendfile = mock.AsyncMock(return_value=None)  # type: ignore[method-assign]

    event_loop.run_until_complete(file_sender.prepare(request))

    assert filepath.open.called
    assert not gz_filepath.open.called


def test_gzip_if_header_present_and_file_not_available(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    request = make_mocked_request(
        "GET", "http://python.org/logo.png", headers={hdrs.ACCEPT_ENCODING: "gzip"}
    )

    gz_filepath = mock.create_autospec(Path, spec_set=True)
    gz_filepath.lstat.side_effect = OSError(2, "No such file or directory")

    filepath = mock.create_autospec(Path, spec_set=True)
    filepath.name = "logo.png"
    filepath.with_suffix.return_value = gz_filepath
    filepath.stat.return_value.st_size = 1024
    filepath.stat.return_value.st_mtime_ns = 1603733507222449291
    filepath.stat.return_value.st_mode = MOCK_MODE

    file_sender = FileResponse(filepath)
    file_sender._path = filepath
    file_sender._sendfile = mock.AsyncMock(return_value=None)  # type: ignore[method-assign]

    event_loop.run_until_complete(file_sender.prepare(request))

    assert filepath.open.called
    assert not gz_filepath.open.called


def test_status_controlled_by_user(event_loop: asyncio.AbstractEventLoop) -> None:
    request = make_mocked_request("GET", "http://python.org/logo.png", headers={})

    filepath = mock.create_autospec(Path, spec_set=True)
    filepath.name = "logo.png"
    filepath.stat.return_value.st_size = 1024
    filepath.stat.return_value.st_mtime_ns = 1603733507222449291
    filepath.stat.return_value.st_mode = MOCK_MODE

    file_sender = FileResponse(filepath, status=203)
    file_sender._path = filepath
    file_sender._sendfile = mock.AsyncMock(return_value=None)  # type: ignore[method-assign]

    event_loop.run_until_complete(file_sender.prepare(request))

    assert file_sender._status == 203


async def test_file_response_sends_headers_immediately() -> None:
    """Test that FileResponse sends headers immediately (inherits from StreamResponse)."""
    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)

    request = make_mocked_request("GET", "http://python.org/logo.png", writer=writer)

    filepath = mock.create_autospec(Path, spec_set=True)
    filepath.name = "logo.png"
    filepath.stat.return_value.st_size = 1024
    filepath.stat.return_value.st_mtime_ns = 1603733507222449291
    filepath.stat.return_value.st_mode = MOCK_MODE

    file_sender = FileResponse(filepath)
    file_sender._path = filepath

    # FileResponse inherits from StreamResponse, so should send immediately
    assert file_sender._send_headers_immediately is True

    # Mock the actual transfer but let _sendfile call super().prepare() to write headers
    with (
        mock.patch.object(
            file_sender, "_sendfile_fallback", autospec=True, spec_set=True
        ) as sendfile_fallback,
        mock.patch("aiohttp.web_fileresponse.NOSENDFILE", True),
    ):
        sendfile_fallback.return_value = writer
        await file_sender.prepare(request)

    # Headers should be sent immediately
    writer.send_headers.assert_called_once()


async def test_sendfile_fallback_respects_count_boundary() -> None:
    """Regression test: _sendfile_fallback should not read beyond the requested count.

    Previously the first chunk used the full chunk_size even when count was smaller,
    and the loop subtracted chunk_size instead of the actual bytes read.  Both bugs
    could cause extra data to be sent when serving range requests.
    """
    file_data = b"A" * 100 + b"B" * 50  # 150 bytes total
    fobj = io.BytesIO(file_data)

    writer = mock.AsyncMock()
    written = bytearray()

    async def capture_write(data: bytes) -> None:
        written.extend(data)

    writer.write = capture_write
    writer.drain = mock.AsyncMock()

    file_sender = FileResponse("dummy.bin")
    file_sender._chunk_size = 64  # smaller than count to test multi-chunk

    # Request only the first 100 bytes (offset=0, count=100)
    await file_sender._sendfile_fallback(writer, fobj, offset=0, count=100)

    assert bytes(written) == b"A" * 100
    assert len(written) == 100


@pytest.mark.parametrize(
    ("header", "expected"),
    (
        ("bytes=0-1", [(0, 1)]),
        ("bytes=0-1,8-", [(0, 1), (8, None)]),
        ("bytes=-3", [(None, 3)]),
        ("bytes=0-1, 8-", [(0, 1), (8, None)]),  # optional whitespace
        ("BYTES=0-1", [(0, 1)]),  # case-insensitive ranges unit
    ),
)
def test_parse_range_header_valid(
    header: str, expected: list[tuple[int | None, int | None]]
) -> None:
    assert _parse_range_header(header, 16) == expected


@pytest.mark.parametrize(
    "header",
    (
        "blocks=0-10",  # wrong ranges unit
        "bytes=a-f",  # non-numeric
        "bytes=-",  # empty range spec
        "bytes=0--10",  # double dash
        "bytes=5-3",  # start after end
        "bytes=",  # no range spec at all
    ),
)
def test_parse_range_header_invalid(header: str) -> None:
    assert _parse_range_header(header, 16) is None


def test_parse_range_header_too_many() -> None:
    header = "bytes=" + ",".join("0-1" for _ in range(17))
    assert _parse_range_header(header, 16) is None
    assert _parse_range_header(header, 17) is not None


def test_clip_ranges() -> None:
    assert _clip_ranges([(0, 1), (8, None)], 10) == [(0, 1), (8, 9)]
    assert _clip_ranges([(None, 3)], 10) == [(7, 9)]
    assert _clip_ranges([(None, 100)], 10) == [(0, 9)]  # suffix larger than file
    assert _clip_ranges([(0, 99999)], 10) == [(0, 9)]  # clipped to file size
    assert _clip_ranges([(99, 100)], 10) == []  # unsatisfiable
    assert _clip_ranges([(None, 0)], 10) == []  # zero suffix is unsatisfiable
    assert _clip_ranges([(0, None)], 0) == []  # empty file


def test_merge_ranges() -> None:
    assert _merge_ranges([(8, 9), (0, 1)]) == [(0, 1), (8, 9)]  # sorted
    assert _merge_ranges([(0, 2), (2, 4)]) == [(0, 4)]  # adjacent merged
    assert _merge_ranges([(0, 3), (2, 4)]) == [(0, 4)]  # overlapping merged
    assert _merge_ranges([(0, 1), (3, 4)]) == [(0, 1), (3, 4)]  # gap kept


@contextlib.contextmanager
def _record_opened_files() -> Iterator[list[io.BufferedReader]]:
    opened: list[io.BufferedReader] = []
    real_open = Path.open

    def tracking_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        fobj = real_open(self, *args, **kwargs)
        opened.append(fobj)
        return fobj

    with mock.patch.object(Path, "open", tracking_open):
        yield opened


async def _wait_all_closed(opened: list[io.BufferedReader]) -> None:
    for _ in range(100):
        if all(fobj.closed for fobj in opened):
            return
        await asyncio.sleep(0.01)


def _make_multi_range_request(writer: StreamWriter) -> Any:
    return make_mocked_request(
        "GET",
        "http://python.org/digits.txt",
        headers={hdrs.RANGE: "bytes=0-1,8-"},
        writer=writer,
    )


@pytest.mark.parametrize(
    "exc", [ConnectionResetError("reset"), asyncio.CancelledError()]
)
async def test_multi_range_send_failure_closes_file(
    tmp_path: Path, exc: BaseException
) -> None:
    """A failed or cancelled multipart send still releases the file handle."""
    filepath = tmp_path / "digits.txt"
    filepath.write_bytes(b"0123456789")

    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)
    writer.write.side_effect = exc
    request = _make_multi_range_request(writer)

    file_sender = FileResponse(filepath)
    with _record_opened_files() as opened, pytest.raises(type(exc)):
        await file_sender.prepare(request)

    assert opened, "expected the file to be opened"
    await _wait_all_closed(opened)
    assert all(fobj.closed for fobj in opened)


async def test_multi_range_read_failure_closes_file(tmp_path: Path) -> None:
    """A read error during a multipart send still releases the file handle."""
    filepath = tmp_path / "digits.txt"
    filepath.write_bytes(b"0123456789")

    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)
    request = _make_multi_range_request(writer)

    file_sender = FileResponse(filepath)
    with (
        _record_opened_files() as opened,
        mock.patch.object(file_sender, "_seek_and_read", side_effect=OSError("boom")),
        pytest.raises(OSError),
    ):
        await file_sender.prepare(request)

    assert opened, "expected the file to be opened"
    await _wait_all_closed(opened)
    assert all(fobj.closed for fobj in opened)
