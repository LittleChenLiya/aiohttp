"""Tests for the incremental text decoder adapter."""

import pytest

from aiohttp._text_decoder import _TextDecoder


class TestTextDecoder:
    """Unit tests for _TextDecoder."""

    def test_multibyte_char_split_across_chunks(self) -> None:
        """A multibyte character split across chunks decodes to one char."""
        dec = _TextDecoder("utf-8")
        # 猫 = E7 8C AB in UTF-8
        assert dec.decode(b"\xe7\x8c") == ""
        assert dec.decode(b"\xab") == "猫"
        assert dec.buffer == b""

    def test_multibyte_char_split_into_three_chunks(self) -> None:
        """猫 split into three chunks still outputs one complete character."""
        dec = _TextDecoder("utf-8")
        assert dec.decode(b"\xe7") == ""
        assert dec.decode(b"\x8c") == ""
        assert dec.decode(b"\xab") == "猫"
        assert dec.buffer == b""

    def test_invalid_byte_reports_raw_offset(self) -> None:
        """An invalid byte in the middle reports its raw byte offset."""
        dec = _TextDecoder("utf-8")
        # "ab" then invalid 0xFF at offset 2
        with pytest.raises(UnicodeDecodeError) as exc_info:
            dec.decode(b"ab\xffcd")
        assert exc_info.value.start == 2
        assert exc_info.value.end == 3

    def test_invalid_byte_after_buffered_sequence(self) -> None:
        """Invalid byte after a buffered incomplete sequence reports offset 0."""
        dec = _TextDecoder("utf-8")
        dec.decode(b"\xe7\x8c")  # buffered incomplete sequence
        with pytest.raises(UnicodeDecodeError) as exc_info:
            dec.decode(b"\xff")
        # The invalid sequence starts at offset 0 (the buffered bytes).
        assert exc_info.value.start == 0

    def test_bom_stripped_at_start(self) -> None:
        """A UTF-8 BOM at the start is stripped."""
        dec = _TextDecoder("utf-8")
        # BOM = EF BB BF
        assert dec.decode(b"\xef\xbb\xbf") == ""
        assert dec.decode(b"hello") == "hello"

    def test_bom_split_across_chunks(self) -> None:
        """A BOM split across chunks is still stripped."""
        dec = _TextDecoder("utf-8")
        assert dec.decode(b"\xef\xbb") == ""
        assert dec.decode(b"\xbf") == ""
        assert dec.decode(b"hello") == "hello"

    def test_bom_only_at_beginning(self) -> None:
        """A BOM in the middle of the stream is decoded as U+FEFF, not stripped."""
        dec = _TextDecoder("utf-8")
        dec.decode(b"hello")
        # BOM in the middle
        result = dec.decode(b"\xef\xbb\xbf")
        assert result == "﻿"

    def test_empty_chunks(self) -> None:
        """Empty chunks return empty string and do not affect state."""
        dec = _TextDecoder("utf-8")
        assert dec.decode(b"") == ""
        assert dec.decode(b"hello") == "hello"
        assert dec.decode(b"") == ""
        assert dec.decode(b"world") == "world"

    def test_eof_flush_incomplete_sequence(self) -> None:
        """EOF with an incomplete sequence raises an error."""
        dec = _TextDecoder("utf-8")
        dec.decode(b"\xe7\x8c")  # incomplete
        with pytest.raises(UnicodeDecodeError) as exc_info:
            dec.decode(b"", final=True)
        assert exc_info.value.start == 0

    def test_eof_flush_complete_sequence(self) -> None:
        """EOF with complete bytes returns the remaining text."""
        dec = _TextDecoder("utf-8")
        dec.decode(b"hello ")
        assert dec.decode(b"", final=True) == ""
        dec2 = _TextDecoder("utf-8")
        dec2.decode(b"hello")
        assert dec2.decode(b"", final=True) == ""

    def test_buffer_only_keeps_incomplete_characters(self) -> None:
        """The buffer only holds incomplete multibyte sequences, not whole chunks."""
        dec = _TextDecoder("utf-8")
        dec.decode(b"hello")
        assert dec.buffer == b""
        dec.decode(b"\xe7\x8c")
        assert dec.buffer == b"\xe7\x8c"
        dec.decode(b"\xab")
        assert dec.buffer == b""

    def test_multiple_characters_across_chunks(self) -> None:
        """Multiple characters split across chunks decode correctly."""
        dec = _TextDecoder("utf-8")
        # "你好世界" split into chunks
        data = "你好世界".encode("utf-8")
        result = ""
        for i in range(0, len(data), 3):
            result += dec.decode(data[i : i + 3])
        result += dec.decode(b"", final=True)
        assert result == "你好世界"

    def test_cp1251_encoding(self) -> None:
        """Single-byte encodings like cp1251 decode correctly."""
        dec = _TextDecoder("cp1251")
        data = '{"тест": "пройден"}'.encode("cp1251")
        result = dec.decode(data[:5]) + dec.decode(data[5:])
        assert result == '{"тест": "пройден"}'

    def test_errors_ignore(self) -> None:
        """errors='ignore' skips invalid bytes without raising."""
        dec = _TextDecoder("utf-8", errors="ignore")
        result = dec.decode(b"ab\xffcd")
        assert result == "abcd"

    def test_errors_replace(self) -> None:
        """errors='replace' replaces invalid bytes with replacement chars."""
        dec = _TextDecoder("utf-8", errors="replace")
        result = dec.decode(b"ab\xffcd")
        assert result == "ab�cd"
