"""
Client-side helpers for consuming Server-Sent Events (SSE).

This module implements the event stream interpretation rules from the
WHATWG Server-Sent Events specification
(https://html.spec.whatwg.org/multipage/server-sent-events.html)
for a single HTTP response body.  It deliberately does *not* perform
automatic reconnection; it only parses the response it is given.
"""

import codecs
import dataclasses
from collections import deque
from types import TracebackType
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .client_reqrep import ClientResponse

__all__ = ("EventSource", "ServerSentEvent")


@dataclasses.dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """A single event dispatched from a ``text/event-stream`` response."""

    data: str
    """The event payload; multiple ``data`` lines are joined with a newline."""

    event: str = "message"
    """The event type, ``"message"`` unless an ``event`` field was present."""

    id: str = ""
    """The last event ID seen on the stream at dispatch time."""

    retry: Optional[int] = None
    """The last valid ``retry`` field seen on the stream, in milliseconds."""


class _SSEParser:
    """Incremental parser for the ``text/event-stream`` wire format."""

    def __init__(self, max_line_size: int) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = ""
        self._bom_checked = False
        self.events: deque[ServerSentEvent] = deque()
        self.last_event_id = ""
        self.retry: Optional[int] = None
        self._data: list[str] = []
        self._event_type = ""
        self._max_line_size = max_line_size

    def feed(self, chunk: bytes) -> None:
        self._append(self._decoder.decode(chunk), final=False)

    def feed_eof(self) -> None:
        self._append(self._decoder.decode(b"", True), final=True)
        # The stream ended in the middle of an event: the pending partial
        # line and the incomplete event are discarded, never dispatched.
        self._pending = ""
        self._data.clear()
        self._event_type = ""

    def _append(self, text: str, *, final: bool) -> None:
        if not self._bom_checked:
            if not text and not final:
                return
            self._bom_checked = True
            text = text.removeprefix("\ufeff")
        self._pending += text
        self._extract(final=final)

    def _extract(self, *, final: bool) -> None:
        buf = self._pending
        pos = 0
        i = 0
        n = len(buf)
        while i < n:
            ch = buf[i]
            if ch == "\n":
                self._process_line(buf[pos:i])
                i += 1
                pos = i
            elif ch == "\r":
                if i + 1 < n or final:
                    self._process_line(buf[pos:i])
                    i += 1
                    if i < n and buf[i] == "\n":
                        i += 1
                    pos = i
                else:
                    # A CR at the end of a chunk may be the first half of a
                    # CRLF pair split across chunks; wait for more data.
                    break
            else:
                i += 1
        self._pending = buf[pos:]
        if len(self._pending) > self._max_line_size:
            raise ValueError(
                f"SSE line is longer than {self._max_line_size} characters"
            )

    def _process_line(self, line: str) -> None:
        if len(line) > self._max_line_size:
            raise ValueError(
                f"SSE line is longer than {self._max_line_size} characters"
            )
        if not line:
            self._dispatch()
            return
        if line[0] == ":":
            return  # comment line
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event_type = value
        elif field == "id":
            # An id containing a NUL character is ignored; an empty id
            # resets the last event ID.
            if "\x00" not in value:
                self.last_event_id = value
        elif field == "retry":
            if value.isascii() and value.isdigit():
                self.retry = int(value)
        # Unknown fields are ignored.

    def _dispatch(self) -> None:
        data = self._data
        event_type = self._event_type
        self._data = []
        self._event_type = ""
        if not data:
            return
        self.events.append(
            ServerSentEvent(
                data="\n".join(data),
                event=event_type or "message",
                id=self.last_event_id,
                retry=self.retry,
            )
        )


class EventSource:
    """Iterate over the events of a ``text/event-stream`` response.

    The source parses a single response body; it does not attempt to
    reconnect when the stream ends.  Use it as an async iterator, and
    optionally as an async context manager to make sure the response is
    released when iteration stops early::

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                async with aiohttp.EventSource(resp) as events:
                    async for event in events:
                        print(event.data)

    When iteration finishes, is cancelled, or fails, the underlying
    response is released (fully consumed streams) or closed (streams
    abandoned mid-way) exactly once.
    """

    def __init__(
        self, response: "ClientResponse", *, max_line_size: int = 2**20
    ) -> None:
        self._response = response
        self._parser = _SSEParser(max_line_size)
        self._eof = False
        self._closed = False

    @property
    def response(self) -> "ClientResponse":
        """The response being parsed."""
        return self._response

    @property
    def last_event_id(self) -> str:
        """The last event ID seen on the stream so far."""
        return self._parser.last_event_id

    @property
    def retry(self) -> Optional[int]:
        """The last valid ``retry`` field seen on the stream, in milliseconds."""
        return self._parser.retry

    async def __aenter__(self) -> "EventSource":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __aiter__(self) -> "EventSource":
        return self

    async def __anext__(self) -> ServerSentEvent:
        parser = self._parser
        while not parser.events and not self._eof:
            if self._closed:
                raise StopAsyncIteration
            try:
                chunk = await self._response.content.readany()
                if chunk:
                    parser.feed(chunk)
                else:
                    self._eof = True
                    parser.feed_eof()
            except BaseException:
                self.close()
                raise
        if parser.events:
            return parser.events.popleft()
        self.close()
        raise StopAsyncIteration

    def close(self) -> None:
        """Stop iterating and release the underlying response."""
        if self._closed:
            return
        self._closed = True
        if self._eof:
            # The body was fully consumed; the connection can be reused.
            self._response.release()
        else:
            self._response.close()
