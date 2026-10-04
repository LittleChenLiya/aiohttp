"""Resumable file downloader on top of :class:`aiohttp.ClientSession`.

Recovery model
--------------
* For every partial download a sibling JSON state file records the URL, the
  confirmed byte offset, the server's *strong* ETag and the exact part-file
  name it describes.
* Resume sends ``Range: bytes=<offset>-`` plus ``If-Range: "<etag>"``.
* Only strong ETags are valid recovery credentials; a weak ETag (``W/"..."``)
  is never persisted and never sent via ``If-Range``.
* A ``206`` response is validated before any byte is appended: its
  ``Content-Range`` must start at exactly the recorded offset and describe the
  expected total size; the response ETag (if any) must match.
* A ``200`` means "start over": a brand new uniquely-named part file is
  created; bytes are never appended to an old fragment.
* Data is streamed and flushed to the part file; on a transport drop the
  confirmed prefix (plus its ETag) stays on disk for the next attempt.
* When the transfer ends, the byte count must match the advertised total
  (Content-Range complete length, else Content-Length). Only then is the
  finished part atomically renamed onto the destination and the state file
  removed. Any failure leaves a previously published destination file
  untouched.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from collections.abc import Mapping

STATE_VERSION = 1
STATE_SUFFIX = ".dlstate"
PART_SUFFIX = ".part"

# Transient failures after which the confirmed prefix is kept and resumed.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    TimeoutError,
    OSError,
)

_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


class DownloadError(Exception):
    """Base class for downloader failures."""


class DownloadInterrupted(DownloadError):
    """A transport-level interruption with confirmed bytes preserved.

    The downloader retries interruptions internally; this class is part of the
    public exception hierarchy so callers can categorise transient failures
    (the chained ``__cause__`` of :class:`ExhaustedRetries` is the last
    underlying transport error).
    """


class InvalidRangeResponse(DownloadError):
    """The server's 206/Content-Range answer is inconsistent."""


class StateCorruptedError(DownloadError):
    """The state file and its part file no longer correspond."""


class ExhaustedRetries(DownloadError):
    """All attempts were used without obtaining a complete download."""


def is_strong_etag(etag: str | None) -> bool:
    """Return True only for a syntactically strong ETag.

    ``None``/empty values and weak validators (``W/"x"``) are rejected, as are
    malformed values. A quoted strong ETag may contain escaped characters.
    """
    if not etag or etag[0] in ("W", "w"):
        return False
    if not etag.startswith('"') or not etag.endswith('"') or len(etag) < 2:
        return False
    body = etag[1:-1]
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            if i + 1 >= len(body):
                return False
            i += 2
            continue
        # Opaque-text: control chars other than tab are forbidden.
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            return False
        i += 1
    return True


def state_path_for(dest: str | os.PathLike[str]) -> Path:
    """Canonical location of the state file for *dest*."""
    return Path(dest).with_name(Path(dest).name + STATE_SUFFIX)


def _new_token() -> str:
    return secrets.token_hex(8)


def _write_state(
    state_path: Path,
    *,
    url: str,
    offset: int,
    etag: str | None,
    part_name: str,
) -> None:
    record = {
        "version": STATE_VERSION,
        "url": url,
        "offset": offset,
        "etag": etag,
        "part_name": part_name,
    }
    tmp = state_path.with_name(state_path.name + f".tmp-{_new_token()}")
    tmp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    os.replace(tmp, state_path)


def _load_state(state_path: Path, dest: Path, url: str) -> dict[str, object] | None:
    """Strictly validate the state file together with its part file.

    Returns the record, or None when there is no state file. Raises
    :class:`StateCorruptedError` on any mismatch so a silently wrong resume can
    never happen.
    """
    if not state_path.exists():
        return None
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateCorruptedError(f"state file {state_path} is unreadable: {exc}") from exc

    if not isinstance(raw, dict):
        raise StateCorruptedError(f"state file {state_path} is not a JSON object")

    for key in ("version", "url", "offset", "etag", "part_name"):
        if key not in raw:
            raise StateCorruptedError(f"state file missing key {key!r}")

    if raw["version"] != STATE_VERSION:
        raise StateCorruptedError(f"unsupported state version {raw['version']!r}")
    if not isinstance(raw["url"], str) or raw["url"] != url:
        raise StateCorruptedError(
            f"state url {raw['url']!r} does not match request url {url!r}"
        )
    offset = raw["offset"]
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise StateCorruptedError(f"bad offset in state: {offset!r}")
    etag = raw["etag"]
    if etag is not None:
        if not isinstance(etag, str) or not is_strong_etag(etag):
            raise StateCorruptedError(f"state carries no strong ETag: {etag!r}")
    part_name = raw["part_name"]
    if not isinstance(part_name, str) or not part_name:
        raise StateCorruptedError(f"bad part_name in state: {part_name!r}")

    if state_path != state_path_for(dest):
        raise StateCorruptedError(
            f"state path {state_path} is not the canonical path for {dest}"
        )

    # The part file must live next to the destination and have the suffix.
    parent = dest.parent.resolve()
    part_path = (dest.parent / part_name).resolve()
    try:
        part_path.relative_to(parent)
    except ValueError as exc:
        raise StateCorruptedError(
            f"part file {part_name!r} escapes destination directory"
        ) from exc
    if part_path.name == dest.name or not part_path.name.endswith(PART_SUFFIX):
        raise StateCorruptedError(f"part file name is illegal: {part_name!r}")
    if not part_path.is_file():
        raise StateCorruptedError(f"part file {part_path} referenced by state is missing")
    actual_size = part_path.stat().st_size
    if actual_size != offset:
        raise StateCorruptedError(
            f"state offset {offset} != part file size {actual_size}"
        )
    return {"record": raw, "part_path": part_path, "offset": offset, "etag": etag}


def _remove_file(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


def _parse_content_range(value: str) -> tuple[int, int, int | None]:
    match = _CONTENT_RANGE_RE.match(value.strip())
    if not match:
        raise InvalidRangeResponse(f"malformed Content-Range: {value!r}")
    first, last, total = match.groups()
    first_i, last_i = int(first), int(last)
    if last_i < first_i:
        raise InvalidRangeResponse(f"inverted Content-Range: {value!r}")
    total_i: int | None = None if total == "*" else int(total)
    if total_i is not None and last_i >= total_i:
        raise InvalidRangeResponse(f"Content-Range end beyond total: {value!r}")
    return first_i, last_i, total_i



async def download(
    session: aiohttp.ClientSession,
    url: str,
    dest: str | os.PathLike[str],
    *,
    max_attempts: int = 5,
    headers: Mapping[str, str] | None = None,
    chunk_size: int = 64 * 1024,
) -> Path:
    """Download *url* into *dest*, resuming from disk state when possible.

    Returns the final destination path. Raises a subclass of
    :class:`DownloadError` on permanent failure or after *max_attempts*
    interrupted attempts.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    state_path = state_path_for(dest)
    last_error: BaseException | None = None

    for _attempt in range(1, max_attempts + 1):
        state = _load_state(state_path, dest, url)
        if state is not None:
            part_path: Path = state["part_path"]  # type: ignore[assignment]
            offset: int = state["offset"]  # type: ignore[assignment]
            etag: str | None = state["etag"]
        else:
            part_path = dest.with_name(f"{dest.name}.{_new_token()}{PART_SUFFIX}")
            offset = 0
            etag = None

        request_headers: dict[str, str] = dict(headers or ())
        resume = offset > 0 and etag is not None
        if resume:
            request_headers["Range"] = f"bytes={offset}-"
            request_headers["If-Range"] = etag  # type: ignore[assignment]

        try:
            async with session.get(url, headers=request_headers) as resp:
                if resp.status == 206:
                    if not resume:
                        raise InvalidRangeResponse(
                            "server returned 206 for a request without Range"
                        )
                    assert etag is not None
                    await _consume_partial(
                        resp,
                        part_path=part_path,
                        offset=offset,
                        etag=etag,
                        state_path=state_path,
                        url=url,
                        chunk_size=chunk_size,
                    )
                    await _publish(part_path, dest, state_path)
                    return dest

                if resp.status != 200:
                    body = await _read_error_body(resp)
                    raise DownloadError(
                        f"unexpected HTTP {resp.status} for {url}: {body[:200]!r}"
                    )

                # 200: the entity changed or the server ignores ranges.
                # Always write to a brand new, uniquely named part file from
                # offset zero -- bytes are never appended to the old fragment.
                fresh_part = dest.with_name(
                    f"{dest.name}.{_new_token()}{PART_SUFFIX}"
                )
                fresh_etag = resp.headers.get(aiohttp.hdrs.ETAG)
                fresh_strong = fresh_etag if is_strong_etag(fresh_etag) else None
                cl = resp.headers.get(aiohttp.hdrs.CONTENT_LENGTH)
                total: int | None = int(cl) if cl is not None and cl.isdigit() else None
                if fresh_strong is not None:
                    # State points at the new (empty) part file BEFORE the
                    # first byte; the old fragment is obsolete and removed.
                    _write_state(
                        state_path,
                        url=url,
                        offset=0,
                        etag=fresh_strong,
                        part_name=fresh_part.name,
                    )
                    if state is not None:
                        _remove_file(part_path)
                elif state is not None:
                    # A weak/missing ETag cannot back recovery: discard the old
                    # state and its fragment rather than resume on it.
                    _remove_file(state_path)
                    _remove_file(part_path)
                try:
                    written = await _stream_body(
                        resp,
                        part_path=fresh_part,
                        start=0,
                        state_path=state_path if fresh_strong else None,
                        url=url,
                        etag=fresh_strong,
                        chunk_size=chunk_size,
                        mode="xb",
                    )
                except _TRANSIENT_ERRORS as exc:
                    if fresh_strong is None:
                        _remove_file(fresh_part)
                    # Confirmed prefix (if any) stays on disk; retry.
                    last_error = exc
                    continue
                if total is not None and written != total:
                    raise InvalidRangeResponse(
                        f"body length {written} != Content-Length {total}"
                    )
                await _publish(fresh_part, dest, state_path)
                return dest
        except _TRANSIENT_ERRORS as exc:
            # Drop while establishing/reading a response: the state on disk
            # already describes the confirmed prefix, so just retry.
            last_error = exc
            continue

    raise ExhaustedRetries(
        f"download of {url} did not complete in {max_attempts} tries"
    ) from last_error


async def _consume_partial(
    resp: aiohttp.ClientResponse,
    *,
    part_path: Path,
    offset: int,
    etag: str,
    state_path: Path,
    url: str,
    chunk_size: int,
) -> None:
    cr_value = resp.headers.get(aiohttp.hdrs.CONTENT_RANGE)
    if cr_value is None:
        raise InvalidRangeResponse("206 response without Content-Range")
    start, last, total = _parse_content_range(cr_value)
    if start != offset:
        raise InvalidRangeResponse(
            f"Content-Range starts at {start}, expected offset {offset}"
        )
    if total is not None and total < offset:
        raise InvalidRangeResponse(
            f"Content-Range total {total} smaller than confirmed offset {offset}"
        )
    resp_etag = resp.headers.get(aiohttp.hdrs.ETAG)
    if resp_etag is not None and is_strong_etag(resp_etag) and resp_etag != etag:
        raise InvalidRangeResponse(
            f"206 ETag {resp_etag!r} differs from resume ETag {etag!r}"
        )

    promised = last - start + 1
    written = await _stream_body(
        resp,
        part_path=part_path,
        start=offset,
        state_path=state_path,
        url=url,
        etag=etag,
        chunk_size=chunk_size,
        mode="ab",
    )
    if written != promised:
        raise InvalidRangeResponse(
            f"206 delivered {written} bytes, Content-Range promised {promised}"
        )
    end = start + written
    if total is not None and end > total:
        raise InvalidRangeResponse(f"received {end} bytes, declared total is {total}")


async def _stream_body(
    resp: aiohttp.ClientResponse,
    *,
    part_path: Path,
    start: int,
    state_path: Path | None,
    url: str,
    etag: str | None,
    chunk_size: int,
    mode: str,
) -> int:
    """Stream the body to *part_path* and persist confirmed offsets.

    Transport errors propagate to the caller; state is only advanced after the
    corresponding bytes have been written and flushed.
    """
    written = 0
    with part_path.open(mode) as fh:
        async for chunk in resp.content.iter_chunked(chunk_size):
            if not chunk:
                continue
            fh.write(chunk)
            written += len(chunk)
            fh.flush()
            os.fsync(fh.fileno())
            if state_path is not None:
                _write_state(
                    state_path,
                    url=url,
                    offset=start + written,
                    etag=etag,
                    part_name=part_path.name,
                )
    return written


async def _publish(part_path: Path, dest: Path, state_path: Path) -> None:
    """Atomically replace *dest* with the completed part file."""
    os.replace(part_path, dest)
    _remove_file(state_path)
    for stale in dest.parent.glob(f"{dest.name}.*{PART_SUFFIX}"):
        _remove_file(stale)


async def _read_error_body(resp: aiohttp.ClientResponse) -> str:
    with suppress(Exception):
        raw = await resp.read()
        return raw.decode("utf-8", "replace")
    return ""
