"""Incremental text decoding with raw byte offset tracking."""

import codecs


class _TextDecoder:
    """Incremental text decoder with raw byte offset tracking.

    Wraps a codecs incremental decoder and tracks the original byte offset
    of the buffered incomplete sequence. When a decode error occurs in
    strict mode, the error is raised with the first raw byte offset of the
    invalid sequence, so callers can pinpoint the bad byte in the original
    stream.
    """

    def __init__(self, encoding: str, errors: str = "strict") -> None:
        # utf-8-sig strips a BOM only at the start; a BOM mid-stream is
        # decoded as U+FEFF, matching the "BOM only at the beginning" rule.
        if encoding == "utf-8":
            encoding = "utf-8-sig"
        self._encoding = encoding
        self._errors = errors
        self._decoder = codecs.getincrementaldecoder(encoding)(errors)
        # Raw offset of the first byte currently held in the decoder's
        # internal buffer (incomplete multibyte sequence).
        self._buffer_offset = 0
        # Total raw bytes fed so far (offset of the next byte to feed).
        self._total_offset = 0
        # Some incremental decoders (e.g. cp1251, shift_jis) do not expose
        # their internal buffer. For those we maintain our own buffer and
        # use a stateless decoder.
        self._has_buffer = hasattr(self._decoder, "buffer")
        if not self._has_buffer:
            self._stateless_decoder = codecs.getdecoder(encoding)
            self._buffer = b""

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def buffer(self) -> bytes:
        """The buffered incomplete bytes."""
        if self._has_buffer:
            return self._decoder.buffer
        return self._buffer

    def decode(self, data: bytes, final: bool = False) -> str:
        """Decode a chunk of bytes, returning the decoded text.

        When ``final`` is true, flush any remaining bytes (raising if they
        form an incomplete sequence).
        """
        if not data and not final:
            return ""
        if self._has_buffer:
            return self._decode_incremental(data, final)
        return self._decode_stateless(data, final)

    def _decode_incremental(self, data: bytes, final: bool) -> str:
        old_buffer = self._decoder.buffer
        try:
            text = self._decoder.decode(data, final)
        except UnicodeDecodeError as exc:
            # exc.start is relative to (old_buffer + data); translate it
            # to the raw offset in the original stream.
            raw_offset = self._buffer_offset + exc.start
            raise UnicodeDecodeError(
                self._encoding,
                old_buffer + data,
                raw_offset,
                raw_offset + (exc.end - exc.start),
                exc.reason,
            ) from exc
        new_buffer = self._decoder.buffer
        consumed = len(old_buffer) + len(data) - len(new_buffer)
        self._buffer_offset += consumed
        self._total_offset += len(data)
        return text

    def _decode_stateless(self, data: bytes, final: bool) -> str:
        combined = self._buffer + data
        try:
            text, consumed = self._stateless_decoder(combined, self._errors)
        except UnicodeDecodeError as exc:
            if exc.reason == "incomplete multibyte sequence" and not final:
                # Buffer the incomplete sequence for the next chunk.
                valid_text, _ = self._stateless_decoder(
                    combined[: exc.start], self._errors
                )
                self._buffer = combined[exc.start :]
                self._buffer_offset += exc.start
                self._total_offset += len(data)
                return valid_text
            raw_offset = self._buffer_offset + exc.start
            raise UnicodeDecodeError(
                self._encoding,
                combined,
                raw_offset,
                raw_offset + (exc.end - exc.start),
                exc.reason,
            ) from exc
        self._buffer = b""
        self._buffer_offset += consumed
        self._total_offset += len(data)
        return text
