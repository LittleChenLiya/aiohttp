import asyncio
import io
import math
import os
import pathlib
import re
import secrets
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from enum import Enum, auto
from mimetypes import MimeTypes
from stat import S_ISREG
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Final, Optional

from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import DEFAULT_CHUNK_SIZE, ETAG_ANY, ETag, must_be_empty_body
from .typedefs import LooseHeaders, PathLike
from .web_exceptions import (
    HTTPForbidden,
    HTTPNotFound,
    HTTPNotModified,
    HTTPPartialContent,
    HTTPPreconditionFailed,
    HTTPRequestRangeNotSatisfiable,
)
from .web_response import StreamResponse

__all__ = ("FileResponse",)

if TYPE_CHECKING:
    from .web_request import BaseRequest

try:
    import aiofastnet
except ImportError:
    aiofastnet = None  # type: ignore[assignment]


_T_OnChunkSent = Optional[Callable[[bytes], Awaitable[None]]]


NOSENDFILE: Final[bool] = bool(os.environ.get("AIOHTTP_NOSENDFILE"))

CONTENT_TYPES: Final[MimeTypes] = MimeTypes()

# File extension to IANA encodings map that will be checked in the order defined.
ENCODING_EXTENSIONS = MappingProxyType(
    {ext: CONTENT_TYPES.encodings_map[ext] for ext in (".br", ".gz")}
)

FALLBACK_CONTENT_TYPE = "application/octet-stream"

# Maximum number of range items accepted in a single Range header.
# Requests with more items have their Range header ignored entirely.
MAX_RANGES: Final[int] = 16

_BYTE_RANGE_SET_RE: Final = re.compile(r"^bytes=(.*)$", re.ASCII | re.IGNORECASE)
_BYTE_RANGE_SPEC_RE: Final = re.compile(r"^(\d*)-(\d*)$", re.ASCII)

# Provide additional MIME type/extension pairs to be recognized.
# https://en.wikipedia.org/wiki/List_of_archive_formats#Compression_only
ADDITIONAL_CONTENT_TYPES = MappingProxyType(
    {
        "application/gzip": ".gz",
        "application/x-brotli": ".br",
        "application/x-bzip2": ".bz2",
        "application/x-compress": ".Z",
        "application/x-xz": ".xz",
    }
)


class _FileResponseResult(Enum):
    """The result of the file response."""

    SEND_FILE = auto()  # Ie a regular file to send
    NOT_ACCEPTABLE = auto()  # Ie a socket, or non-regular file
    PRE_CONDITION_FAILED = auto()  # Ie If-Match or If-None-Match failed
    NOT_MODIFIED = auto()  # 304 Not Modified


# Add custom pairs and clear the encodings map so guess_type ignores them.
CONTENT_TYPES.encodings_map.clear()
for content_type, extension in ADDITIONAL_CONTENT_TYPES.items():
    CONTENT_TYPES.add_type(content_type, extension)


_CLOSE_FUTURES: set[asyncio.Future[None]] = set()


def _parse_byte_ranges(
    range_header: str | None, file_size: int
) -> list[tuple[int, int]] | None:
    """Parse a Range header into sorted, merged (start, end) inclusive ranges.

    Return None if the header is absent, syntactically invalid, or has too
    many items (the Range header must be ignored and the full
    representation sent), an empty list if every range is unsatisfiable
    (416), or a non-empty list of (start, end) pairs clipped to file_size.
    """
    if range_header is None:
        return None
    match = _BYTE_RANGE_SET_RE.match(range_header.strip())
    if match is None:
        return None
    specs = match.group(1).split(",")
    if len(specs) > MAX_RANGES:
        return None
    parsed: list[tuple[int | None, int | None]] = []
    for spec in specs:
        spec_match = _BYTE_RANGE_SPEC_RE.match(spec.strip())
        if spec_match is None:
            return None
        first, last = spec_match.groups()
        if not first and not last:
            return None
        if first and last and int(first) > int(last):
            return None
        parsed.append((int(first) if first else None, int(last) if last else None))

    ranges: list[tuple[int, int]] = []
    for first, last in parsed:
        if first is None:
            # suffix-byte-range-spec: the last N bytes of the file
            assert last is not None
            if last == 0 or file_size == 0:
                continue
            ranges.append((max(file_size - last, 0), file_size - 1))
        else:
            if first >= file_size:
                # Unsatisfiable range; drop it.
                continue
            end = file_size - 1 if last is None else min(last, file_size - 1)
            ranges.append((first, end))

    if not ranges:
        return ranges
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            # Merge overlapping or adjacent ranges.
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


class FileResponse(StreamResponse):
    """A response object can be used to send files."""

    def __init__(
        self,
        path: PathLike,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        status: int = 200,
        reason: str | None = None,
        headers: LooseHeaders | None = None,
        text_charset: str | None = None,
    ) -> None:
        super().__init__(status=status, reason=reason, headers=headers)

        self._path = pathlib.Path(path)
        self._chunk_size = chunk_size
        if text_charset == "":
            raise ValueError("text_charset must not be an empty string")
        self._text_charset = text_charset

    def _seek_and_read(self, fobj: BinaryIO, offset: int, chunk_size: int) -> bytes:
        fobj.seek(offset)
        return fobj.read(chunk_size)

    async def _sendfile_fallback(
        self, writer: AbstractStreamWriter, fobj: BinaryIO, offset: int, count: int
    ) -> AbstractStreamWriter:
        # To keep memory usage low,fobj is transferred in chunks
        # controlled by the constructor's chunk_size argument.

        chunk_size = self._chunk_size
        loop = asyncio.get_running_loop()
        chunk = await loop.run_in_executor(
            None, self._seek_and_read, fobj, offset, min(chunk_size, count)
        )
        while chunk:
            await writer.write(chunk)
            count = count - len(chunk)
            if count <= 0:
                break
            chunk = await loop.run_in_executor(None, fobj.read, min(chunk_size, count))

        await writer.drain()
        return writer

    async def _sendfile(
        self, request: "BaseRequest", fobj: BinaryIO, offset: int, count: int
    ) -> AbstractStreamWriter:
        writer = await super().prepare(request)
        assert writer is not None

        if NOSENDFILE or self.compression:
            return await self._sendfile_fallback(writer, fobj, offset, count)

        loop = request._loop
        transport = request.transport
        if transport is None:
            raise ConnectionResetError("Connection lost")

        try:
            if aiofastnet is not None:
                await aiofastnet.sendfile(loop, transport, fobj, offset, count)
            else:
                await loop.sendfile(transport, fobj, offset, count)  # type: ignore[unreachable]
        except NotImplementedError:
            return await self._sendfile_fallback(writer, fobj, offset, count)

        await super().write_eof()
        return writer

    @staticmethod
    def _etag_match(etag_value: str, etags: tuple[ETag, ...], *, weak: bool) -> bool:
        if len(etags) == 1 and etags[0].value == ETAG_ANY:
            return True
        return any(
            etag.value == etag_value for etag in etags if weak or not etag.is_weak
        )

    async def _not_modified(
        self, request: "BaseRequest", etag_value: str, last_modified: float
    ) -> AbstractStreamWriter | None:
        self.set_status(HTTPNotModified.status_code)
        self._length_check = False
        self.etag = etag_value
        self.last_modified = last_modified
        # Delete any Content-Length headers provided by user. HTTP 304
        # should always have empty response body
        return await super().prepare(request)

    async def _precondition_failed(
        self, request: "BaseRequest"
    ) -> AbstractStreamWriter | None:
        self.set_status(HTTPPreconditionFailed.status_code)
        self.content_length = 0
        return await super().prepare(request)

    def _make_response(
        self, request: "BaseRequest", accept_encoding: str
    ) -> tuple[
        _FileResponseResult, io.BufferedReader | None, os.stat_result, str | None
    ]:
        """Return the response result, io object, stat result, and encoding.

        If an uncompressed file is returned, the encoding is set to
        :py:data:`None`.

        This method should be called from a thread executor
        since it calls os.stat which may block.
        """
        file_path, st, file_encoding = self._get_file_path_stat_encoding(
            accept_encoding
        )
        if not file_path:
            return _FileResponseResult.NOT_ACCEPTABLE, None, st, None

        etag_value = f"{st.st_mtime_ns:x}-{st.st_size:x}"

        # https://www.rfc-editor.org/rfc/rfc9110#section-13.1.1-2
        if (ifmatch := request.if_match) is not None and not self._etag_match(
            etag_value, ifmatch, weak=False
        ):
            return _FileResponseResult.PRE_CONDITION_FAILED, None, st, file_encoding

        if (
            (unmodsince := request.if_unmodified_since) is not None
            and ifmatch is None
            and st.st_mtime > unmodsince.timestamp()
        ):
            return _FileResponseResult.PRE_CONDITION_FAILED, None, st, file_encoding

        # https://www.rfc-editor.org/rfc/rfc9110#section-13.1.2-2
        if (ifnonematch := request.if_none_match) is not None and self._etag_match(
            etag_value, ifnonematch, weak=True
        ):
            return _FileResponseResult.NOT_MODIFIED, None, st, file_encoding

        if (
            (modsince := request.if_modified_since) is not None
            and ifnonematch is None
            and st.st_mtime <= modsince.timestamp()
        ):
            return _FileResponseResult.NOT_MODIFIED, None, st, file_encoding

        fobj = file_path.open("rb")
        with suppress(OSError):
            # fstat() may not be available on all platforms
            # Once we open the file, we want the fstat() to ensure
            # the file has not changed between the first stat()
            # and the open().
            st = os.stat(fobj.fileno())
        return _FileResponseResult.SEND_FILE, fobj, st, file_encoding

    def _get_file_path_stat_encoding(
        self, accept_encoding: str
    ) -> tuple[pathlib.Path | None, os.stat_result, str | None]:
        file_path = self._path
        for file_extension, file_encoding in ENCODING_EXTENSIONS.items():
            if file_encoding not in accept_encoding:
                continue

            compressed_path = file_path.with_suffix(file_path.suffix + file_extension)
            with suppress(OSError):
                # Do not follow symlinks and ignore any non-regular files.
                st = compressed_path.lstat()
                if S_ISREG(st.st_mode):
                    return compressed_path, st, file_encoding

        # Fallback to the uncompressed file
        st = file_path.stat()
        return file_path if S_ISREG(st.st_mode) else None, st, None

    async def prepare(self, request: "BaseRequest") -> AbstractStreamWriter | None:
        loop = asyncio.get_running_loop()
        # Encoding comparisons should be case-insensitive
        # https://www.rfc-editor.org/rfc/rfc9110#section-8.4.1
        accept_encoding = request.headers.get(hdrs.ACCEPT_ENCODING, "").lower()
        range_header = request.headers.get(hdrs.RANGE)
        if range_header is not None and "," in range_header:
            # Multi-range requests are served from the identity
            # representation: offsets into a compressed variant cannot be
            # mapped back onto the original file, and the ETag must be
            # generated for the representation being ranged over.
            accept_encoding = ""
        try:
            response_result, fobj, st, file_encoding = await loop.run_in_executor(
                None, self._make_response, request, accept_encoding
            )
        except PermissionError:
            self.set_status(HTTPForbidden.status_code)
            return await super().prepare(request)
        except OSError:
            # Most likely to be FileNotFoundError or OSError for circular
            # symlinks in python >= 3.13, so respond with 404.
            self.set_status(HTTPNotFound.status_code)
            return await super().prepare(request)

        # Forbid special files like sockets, pipes, devices, etc.
        if response_result is _FileResponseResult.NOT_ACCEPTABLE:
            self.set_status(HTTPForbidden.status_code)
            return await super().prepare(request)

        if response_result is _FileResponseResult.PRE_CONDITION_FAILED:
            return await self._precondition_failed(request)

        if response_result is _FileResponseResult.NOT_MODIFIED:
            etag_value = f"{st.st_mtime_ns:x}-{st.st_size:x}"
            last_modified = st.st_mtime
            return await self._not_modified(request, etag_value, last_modified)

        assert fobj is not None
        try:
            return await self._prepare_open_file(request, fobj, st, file_encoding)
        finally:
            # We do not await here because we do not want to wait
            # for the executor to finish before returning the response
            # so the connection can begin servicing another request
            # as soon as possible.
            close_future = loop.run_in_executor(None, fobj.close)
            # Hold a strong reference to the future to prevent it from being
            # garbage collected before it completes.
            _CLOSE_FUTURES.add(close_future)
            close_future.add_done_callback(_CLOSE_FUTURES.remove)

    async def _prepare_open_file(
        self,
        request: "BaseRequest",
        fobj: io.BufferedReader,
        st: os.stat_result,
        file_encoding: str | None,
    ) -> AbstractStreamWriter | None:
        status = self._status
        file_size: int = st.st_size
        file_mtime: float = st.st_mtime
        count: int = file_size
        start: int | None = None
        etag_value = f"{st.st_mtime_ns:x}-{st.st_size:x}"
        multipart_ranges: list[tuple[int, int]] | None = None

        # https://www.rfc-editor.org/info/rfc9110/#name-if-range
        if_range = request.if_range
        if isinstance(if_range, ETag):
            # https://www.rfc-editor.org/info/rfc9110/#section-13.1.5-12.1
            range_applies = not if_range.is_weak and if_range.value == etag_value
        elif if_range is not None:
            # https://www.rfc-editor.org/info/rfc9110/#section-13.1.5-10.2
            # Last-Modified is emitted as math.ceil(mtime), so the strong
            # comparison is against that same rounded value.
            range_applies = math.ceil(file_mtime) == if_range.timestamp()
        else:
            # A malformed validator can't match the current representation,
            # so only an absent header lets the Range through.
            range_applies = hdrs.IF_RANGE not in request.headers

        if range_applies:
            # If-Range header check:
            # condition = cached validator matches current representation.
            # return 206 if True else 200.
            # if False:
            #   Range header would not be processed, return 200
            # if True but Range header missing
            #   return 200
            ranges = _parse_byte_ranges(request.headers.get(hdrs.RANGE), file_size)
            if ranges is None:
                # The Range header is absent, syntactically invalid, or
                # contains too many items: ignore it and send the full
                # representation.
                pass
            elif not ranges:
                # https://tools.ietf.org/html/rfc7233:
                # A server generating a 416 (Range Not Satisfiable) response to
                # a byte-range request SHOULD send a Content-Range header field
                # with an unsatisfied-range value.
                # The complete-length in a 416 response indicates the current
                # length of the selected representation.
                #
                # Will do the same below. Many servers ignore this and do not
                # send a Content-Range header with HTTP 416
                self._headers[hdrs.CONTENT_RANGE] = f"bytes */{file_size}"
                self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                return await super().prepare(request)
            elif len(ranges) == 1:
                # A single satisfiable range keeps the simple 206 path.
                start, end = ranges[0]
                count = end - start + 1
                status = HTTPPartialContent.status_code
                self.set_status(status)
            else:
                multipart_ranges = ranges
                status = HTTPPartialContent.status_code
                self.set_status(status)

        # If the Content-Type header is not already set, guess it based on the
        # extension of the request path. The encoding returned by guess_type
        #  can be ignored since the map was cleared above.
        if hdrs.CONTENT_TYPE not in self._headers:
            if sys.version_info >= (3, 13):
                guesser = CONTENT_TYPES.guess_file_type
            else:
                guesser = CONTENT_TYPES.guess_type
            content_type = guesser(self._path)[0] or FALLBACK_CONTENT_TYPE
            self.content_type = content_type
            if self._text_charset is not None and content_type.startswith("text/"):
                self.charset = self._text_charset

        if file_encoding:
            self._headers[hdrs.CONTENT_ENCODING] = file_encoding
            self._headers[hdrs.VARY] = hdrs.ACCEPT_ENCODING
            # Disable compression if we are already sending
            # a compressed file since we don't want to double
            # compress.
            self._compression = False

        self.etag = etag_value
        self.last_modified = file_mtime
        self._headers[hdrs.ACCEPT_RANGES] = "bytes"

        if multipart_ranges is not None:
            return await self._prepare_multipart(
                request, fobj, multipart_ranges, file_size
            )

        self.content_length = count

        if status == HTTPPartialContent.status_code:
            real_start = start
            assert real_start is not None
            self._headers[hdrs.CONTENT_RANGE] = (
                f"bytes {real_start}-{real_start + count - 1}/{file_size}"
            )

        # If we are sending 0 bytes calling sendfile() will throw a ValueError
        if count == 0 or must_be_empty_body(request.method, status):
            return await super().prepare(request)

        # be aware that start could be None or int=0 here.
        offset = start or 0

        return await self._sendfile(request, fobj, offset, count)

    async def _prepare_multipart(
        self,
        request: "BaseRequest",
        fobj: io.BufferedReader,
        ranges: list[tuple[int, int]],
        file_size: int,
    ) -> AbstractStreamWriter | None:
        """Send multiple byte ranges as a multipart/byteranges response."""
        boundary = f"aiohttp-{secrets.token_hex(16)}"
        content_type = self._headers[hdrs.CONTENT_TYPE]
        self._headers[hdrs.CONTENT_TYPE] = f"multipart/byteranges; boundary={boundary}"

        part_headers: list[bytes] = []
        content_length = 0
        for range_start, range_end in ranges:
            part_header = (
                f"--{boundary}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Range: bytes {range_start}-{range_end}/{file_size}\r\n"
                f"\r\n"
            ).encode()
            part_headers.append(part_header)
            # 2 accounts for the CRLF terminating the part body.
            content_length += len(part_header) + (range_end - range_start + 1) + 2
        final_delimiter = f"--{boundary}--\r\n".encode()
        content_length += len(final_delimiter)
        self.content_length = content_length

        if must_be_empty_body(request.method, HTTPPartialContent.status_code):
            # HEAD responses carry the same headers but no body.
            return await super().prepare(request)

        writer = await super().prepare(request)
        assert writer is not None
        for (range_start, range_end), part_header in zip(ranges, part_headers):
            await writer.write(part_header)
            await self._sendfile_fallback(
                writer, fobj, range_start, range_end - range_start + 1
            )
            await writer.write(b"\r\n")
        await writer.write(final_delimiter)
        await writer.drain()
        return writer
