"""Incremental text decoding for client responses.

This module provides :class:`TextStreamDecoder`, a small state machine that
turns arbitrarily fragmented ``bytes`` chunks into text using a determined
charset, and :class:`ClientResponseTextDecoder`, an adapter that drives a
:class:`~aiohttp.ClientResponse` through it.

Design constraints:

* A BOM is only ever inspected/stripped at the very beginning of the stream.
* Multi-byte characters (and the BOM itself) may span chunk boundaries.
* In strict mode an invalid sequence raises :class:`TextStreamDecodeError`
  carrying the *first* raw byte offset of the offending sequence; no
  replacement characters are silently produced.
* The internal buffer only ever holds the bytes of one incomplete
  character (plus a few BOM-probe bytes at the start) -- never the whole
  response body.
* Both a decode failure and an explicit :meth:`ClientResponseTextDecoder.close`
  release the underlying response connection.
"""

import codecs
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple

if TYPE_CHECKING:
    from .client_reqrep import ClientResponse

__all__ = (
    "ClientResponseTextDecoder",
    "DecodedChunk",
    "TextStreamDecodeError",
    "TextStreamDecoder",
)

# Decode-error reasons that mean "the bytes so far are a valid prefix of a
# character, feed me more data" (as opposed to a genuinely invalid sequence).
_TRUNCATED_REASONS = frozenset(
    {
        "unexpected end of data",  # utf-8, utf-16 surrogate pairs, ...
        "incomplete multibyte sequence",  # gbk, big5, shift_jis, ...
        "truncated data",  # utf-16-le/be, utf-32-le/be odd byte counts
    }
)

# BOMs stripped at the start of the stream, keyed by normalized codec name.
# "utf-16"/"utf-32" (without endianness) are absent on purpose: those codecs
# already consume the BOM themselves.
_BOMS = {
    "utf-8": (codecs.BOM_UTF8,),
    "utf-16-le": (codecs.BOM_UTF16_LE,),
    "utf-16-be": (codecs.BOM_UTF16_BE,),
    "utf-32-le": (codecs.BOM_UTF32_LE,),
    "utf-32-be": (codecs.BOM_UTF32_BE,),
}


@dataclass(frozen=True)
class DecodedChunk:
    """A piece of decoded text and where it came from.

    ``offset`` is the absolute byte offset in the original (encoded) response
    body of the first byte that produced ``text`` -- BOM bytes included, so
    the first character after a 3-byte UTF-8 BOM has offset 3.
    """

    text: str
    offset: int


class TextStreamDecodeError(UnicodeDecodeError):
    """A strict-mode decode failure annotated with the raw stream offset.

    ``stream_offset`` is the absolute byte offset in the original response
    body of the first byte of the invalid sequence.  The inherited
    ``UnicodeDecodeError`` attributes describe the buffered fragment that
    was being decoded when the error was detected.
    """

    def __init__(self, exc: UnicodeDecodeError, base: int) -> None:
        super().__init__(exc.encoding, exc.object, exc.start, exc.end, exc.reason)
        self.stream_offset = base + exc.start

    def __str__(self) -> str:
        return f"{super().__str__()} (at stream byte offset {self.stream_offset})"


class TextStreamDecoder:
    """Incrementally decode byte chunks into text with raw byte offsets.

    Only the bytes of an incomplete character (and, at the start, a few
    BOM-probe bytes) are ever buffered; completed text is emitted
    immediately and never retained.
    """

    def __init__(self, encoding: str, errors: str = "strict") -> None:
        self._encoding = codecs.lookup(encoding).name
        self._errors = errors
        self._boms: Tuple[bytes, ...] = _BOMS.get(self._encoding, ())
        # Invariant: _pending holds at most a BOM probe (<= 4 bytes) or the
        # bytes of one incomplete character -- never a whole response.
        self._pending = bytearray()
        self._offset = 0  # absolute stream offset of _pending[0]
        self._bom_done = not self._boms
        self._finished = False

    @property
    def encoding(self) -> str:
        """The normalized name of the charset being decoded."""
        return self._encoding

    @property
    def buffered(self) -> int:
        """Number of raw bytes currently buffered (one char or BOM probe)."""
        return len(self._pending)

    def feed(self, data: bytes) -> List[DecodedChunk]:
        """Feed a chunk of raw bytes; return any completed text pieces.

        Empty chunks are legal and simply produce no output; they do not
        disturb an undecided BOM probe or an incomplete character.
        """
        if self._finished:
            raise RuntimeError("cannot feed a finished decoder")
        if data:
            self._pending += data
        if not self._bom_done:
            if self._awaiting_bom():
                return []
            self._resolve_bom()
        return self._decode_pending(final=False)

    def finish(self) -> List[DecodedChunk]:
        """Signal end-of-stream and flush.

        In strict mode, bytes still buffered at EOF (a truncated character)
        raise :class:`TextStreamDecodeError` pointing at their offset.
        """
        if self._finished:
            raise RuntimeError("decoder is already finished")
        if not self._bom_done:
            self._resolve_bom()
        try:
            return self._decode_pending(final=True)
        finally:
            self._finished = True

    def _awaiting_bom(self) -> bool:
        pending = bytes(self._pending)
        max_len = max(len(bom) for bom in self._boms)
        return len(pending) < max_len and any(
            bom.startswith(pending) for bom in self._boms
        )

    def _resolve_bom(self) -> None:
        pending = bytes(self._pending)
        for bom in self._boms:
            if pending.startswith(bom):
                del self._pending[: len(bom)]
                self._offset += len(bom)
                break
        self._bom_done = True

    def _decode_pending(self, final: bool) -> List[DecodedChunk]:
        out: List[DecodedChunk] = []
        while self._pending:
            try:
                text = bytes(self._pending).decode(self._encoding, "strict")
            except UnicodeDecodeError as exc:
                truncated = (
                    exc.end == len(exc.object) and exc.reason in _TRUNCATED_REASONS
                )
                if truncated and not final:
                    # Emit the complete prefix, keep the one incomplete
                    # character for the next chunk.
                    if exc.start:
                        head = bytes(self._pending[: exc.start]).decode(
                            self._encoding
                        )
                        out.append(DecodedChunk(head, self._offset))
                        self._offset += exc.start
                        del self._pending[: exc.start]
                    break
                if self._errors == "strict":
                    raise TextStreamDecodeError(exc, self._offset) from exc
                # Non-strict mode opted into replacement/ignore semantics.
                text = bytes(self._pending).decode(self._encoding, self._errors)
                out.append(DecodedChunk(text, self._offset))
                self._offset += len(self._pending)
                self._pending.clear()
            else:
                out.append(DecodedChunk(text, self._offset))
                self._offset += len(self._pending)
                self._pending.clear()
        return out


def _determine_encoding(response: "ClientResponse") -> str:
    """Determine the charset for a streaming response.

    Uses the same rules as :meth:`ClientResponse.get_encoding`; when the
    headers carry no charset information (and the body has not been read,
    so no content-based fallback is possible) UTF-8 is assumed.
    """
    try:
        return response.get_encoding()
    except RuntimeError:
        return "utf-8"


class ClientResponseTextDecoder:
    """Adapt a :class:`~aiohttp.ClientResponse` into a decoded-text stream.

    Iterating (or repeatedly calling :meth:`read`) yields :class:`DecodedChunk`
    pieces with their raw byte offsets.  The connection is released back to
    the pool on clean EOF, on a decode failure, and on :meth:`close`.
    """

    def __init__(
        self,
        response: "ClientResponse",
        encoding: Optional[str] = None,
        errors: str = "strict",
    ) -> None:
        self._response = response
        if encoding is None:
            encoding = _determine_encoding(response)
        self._decoder = TextStreamDecoder(encoding, errors)
        self._ready: Deque[DecodedChunk] = deque()
        self._eof = False

    @property
    def encoding(self) -> str:
        """The charset the response body is being decoded with."""
        return self._decoder.encoding

    def close(self) -> None:
        """Stop decoding and release the response connection."""
        self._eof = True
        self._ready.clear()
        self._response.release()

    async def read(self) -> Optional[DecodedChunk]:
        """Return the next decoded piece, or ``None`` at end of stream."""
        while not self._ready:
            if self._eof:
                return None
            try:
                data = await self._response.content.readany()
            except BaseException:
                self._response.close()
                raise
            if data:
                try:
                    self._ready.extend(self._decoder.feed(data))
                except TextStreamDecodeError:
                    self._response.release()
                    raise
            else:
                self._eof = True
                try:
                    self._ready.extend(self._decoder.finish())
                except TextStreamDecodeError:
                    self._response.release()
                    raise
                self._response.release()
        return self._ready.popleft()

    def __aiter__(self) -> "ClientResponseTextDecoder":
        return self

    async def __anext__(self) -> DecodedChunk:
        chunk = await self.read()
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    async def __aenter__(self) -> "ClientResponseTextDecoder":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.close()
