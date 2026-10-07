"""Client-side helpers for consuming Server-Sent Events (SSE)."""

import codecs
from collections import deque
from typing import TYPE_CHECKING, Deque, List, Optional

from .client_exceptions import ClientError

if TYPE_CHECKING:
    from .client_reqrep import ClientResponse

__all__ = (
    "EventSource",
    "ServerSentEvent",
    "SSEBufferOverflowError",
    "SSEError",
)

#: Default upper bound for a single event's buffered data, in characters.
DEFAULT_MAX_EVENT_SIZE = 1024 * 1024


class SSEError(ClientError):
    """Base class for Server-Sent Events processing errors."""


class SSEBufferOverflowError(SSEError):
    """A single event grew past the configured buffer limit."""


class ServerSentEvent:
    """A single event parsed from a Server-Sent Events stream."""

    __slots__ = ("data", "event", "id", "retry")

    def __init__(
        self,
        data: str,
        event: str = "message",
        id: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> None:
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry

    def __repr__(self) -> str:
        return (
            f"<ServerSentEvent(data={self.data!r}, event={self.event!r}, "
            f"id={self.id!r}, retry={self.retry!r})>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ServerSentEvent):
            return NotImplemented
        return (
            self.data == other.data
            and self.event == other.event
            and self.id == other.id
            and self.retry == other.retry
        )


class EventSource:
    r"""Iterate over Server-Sent Events parsed from a client response.

    The source performs a single pass over one response body; it never
    reconnects automatically.  The byte stream is decoded as UTF-8 and
    interpreted following the Server-Sent Events specification:

    * ``data`` fields are merged with ``"\n"`` separators,
    * ``event``, ``id`` and all-decimal ``retry`` fields are recognized,
    * comment lines (starting with ``":"``) are skipped,
    * an empty line dispatches the buffered event,
    * an empty ``id`` resets the last event ID,
    * an ``id`` containing a NUL character is ignored,
    * a partial trailing line at end of stream is discarded.

    Use it as an async iterator, optionally as an async context manager
    to guarantee the response is released::

        async with session.get(url) as resp:
            async with aiohttp.EventSource(resp) as source:
                async for event in source:
                    print(event.data)
    """

    def __init__(
        self,
        response: "ClientResponse",
        *,
        max_event_size: int = DEFAULT_MAX_EVENT_SIZE,
    ) -> None:
        if max_event_size <= 0:
            raise ValueError("max_event_size must be a positive integer")
        self._response = response
        self._max_event_size = max_event_size
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._last_event_id: Optional[str] = None
        self._retry: Optional[int] = None
        self._events: Deque[ServerSentEvent] = deque()
        self._line = ""  # pending partial line
        self._pending_cr = False  # previous feed ended with a CR
        self._started = False  # nothing decoded yet (BOM not stripped)
        self._data: List[str] = []
        self._data_size = 0
        self._event_type = ""
        self._eof = False
        self._closed = False

    @property
    def response(self) -> "ClientResponse":
        """The response this source is reading from."""
        return self._response

    @property
    def last_event_id(self) -> Optional[str]:
        """The last event ID seen so far, or ``None`` if never set."""
        return self._last_event_id

    @property
    def retry(self) -> Optional[int]:
        """The last valid ``retry`` field value, or ``None`` if never set."""
        return self._retry

    @property
    def closed(self) -> bool:
        """``True`` if the source has been closed explicitly."""
        return self._closed

    def __aiter__(self) -> "EventSource":
        return self

    async def __anext__(self) -> ServerSentEvent:
        while not self._events:
            if self._eof or self._closed:
                raise StopAsyncIteration
            try:
                chunk = await self._response.content.readany()
            except BaseException:
                await self.aclose()
                raise
            if chunk:
                try:
                    self._feed(self._decoder.decode(chunk))
                except BaseException:
                    await self.aclose()
                    raise
            else:
                self._eof = True
                try:
                    self._feed(self._decoder.decode(b"", final=True))
                except BaseException:
                    await self.aclose()
                    raise
                finally:
                    # Any pending data is discarded once the end of the
                    # stream is reached; the incomplete event is not
                    # dispatched.
                    self._discard_pending()
                self._response.release()
        return self._events.popleft()

    async def aclose(self) -> None:
        """Stop iterating and release the underlying response."""
        if not self._closed:
            self._closed = True
            self._response.release()

    async def __aenter__(self) -> "EventSource":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _feed(self, text: str) -> None:
        if not self._started:
            self._started = True
            if text.startswith("\ufeff"):  # strip a single leading BOM
                text = text[1:]
        if self._pending_cr:
            self._pending_cr = False
            if text.startswith("\n"):
                # A CRLF pair was split across two chunks.
                text = text[1:]
        if not text:
            return
        line = self._line
        start = 0
        pos = 0
        end = len(text)
        while pos < end:
            char = text[pos]
            if char == "\n" or char == "\r":
                self._process_line(line + text[start:pos])
                line = ""
                pos += 1
                if char == "\r":
                    if pos == end:
                        # A CR at the very end of the chunk: the LF of a
                        # CRLF pair may arrive with the next chunk.
                        self._pending_cr = True
                    elif text[pos] == "\n":
                        pos += 1
                start = pos
            else:
                pos += 1
        self._line = line + text[start:]
        if len(self._line) > self._max_event_size:
            raise SSEBufferOverflowError(
                f"Event buffer exceeded the limit of {self._max_event_size} "
                "characters"
            )

    def _process_line(self, line: str) -> None:
        if not line:
            self._dispatch_event()
            return
        if line.startswith(":"):
            return  # comment
        field, sep, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data_size += len(value) + 1
            if self._data_size > self._max_event_size:
                raise SSEBufferOverflowError(
                    f"Event buffer exceeded the limit of {self._max_event_size} "
                    "characters"
                )
            self._data.append(value)
        elif field == "event":
            self._event_type = value
        elif field == "id":
            if "\0" not in value:
                self._last_event_id = value
        elif field == "retry":
            if value.isascii() and value.isdigit():
                self._retry = int(value)

    def _dispatch_event(self) -> None:
        data = self._data
        event_type = self._event_type
        self._data = []
        self._data_size = 0
        self._event_type = ""
        if not data:
            return
        self._events.append(
            ServerSentEvent(
                data="\n".join(data),
                event=event_type or "message",
                id=self._last_event_id,
                retry=self._retry,
            )
        )

    def _discard_pending(self) -> None:
        self._line = ""
        self._pending_cr = False
        self._data = []
        self._data_size = 0
        self._event_type = ""
