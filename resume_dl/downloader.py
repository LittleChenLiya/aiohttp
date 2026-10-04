"""Resumable single-file downloader based on :class:`aiohttp.ClientSession`.

Resume rules
============

1. A partial download is resumable only when a *resume state* file exists
   next to the target and every cross-check passes: the state file is
   well-formed JSON (version 1), the temp file named in the state exists in
   the same directory, the temp file's actual byte size equals the recorded
   ``bytes`` count, the recorded ETag is a *strong* validator, and the
   recorded URL matches the requested URL.  Any failed check invalidates the
   state: the download starts over in a brand-new temp file.
2. When resuming, the request carries ``Range: bytes=<local-length>-`` and
   ``If-Range: <strong-etag>``.  A weak ETag (``W/"..."``) is never used as a
   resume credential; without a strong ETag no ``If-Range`` is sent.
3. A ``206 Partial Content`` response is only accepted when its
   ``Content-Range`` header is well-formed (``bytes <start>-<end>/<total>``),
   ``start`` equals the local confirmed byte count, and the number of bytes
   received equals ``end - start + 1`` (and equals ``total`` when given).
   The segment is appended to the existing temp file.
4. A ``200 OK`` response means the server ignored the range.  The content is
   written to a *new* temp file from byte zero; it is never appended to an
   old fragment.  On success the old temp file is removed.
5. If the connection drops mid-read
   (:class:`aiohttp.ClientPayloadError` /
   :class:`aiohttp.ServerDisconnectedError`), the bytes already received are
   flushed and the resume state (confirmed byte count + ETag) is persisted
   atomically.  :class:`ResumeInterrupted` is raised; call ``download`` again
   to resume.
6. A malformed ``Content-Range``, a wrong range start, or a clean short read
   (fewer bytes than promised without a connection drop) is a hard
   :class:`ResumeError`: nothing is published and the previously published
   target file is left untouched.
7. The target file is only replaced after the complete file has been written
   and flushed to disk, via an atomic ``os.replace``.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from aiohttp import (
    ClientError,
    ClientPayloadError,
    ClientSession,
    ServerDisconnectedError,
)

__all__ = [
    "ResumeError",
    "ResumeInterrupted",
    "download",
    "parse_content_range",
    "strong_etag",
]

_STATE_VERSION = 1
_CHUNK_SIZE = 64 * 1024
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


class ResumeError(Exception):
    """A non-resumable download failure; the published target is untouched."""


class ResumeInterrupted(ResumeError):
    """The connection dropped mid-read; progress is preserved for a retry."""

    def __init__(
        self,
        message: str,
        *,
        confirmed_bytes: int,
        etag: str | None,
        temp_path: Path,
        state_path: Path,
    ) -> None:
        super().__init__(message)
        self.confirmed_bytes = confirmed_bytes
        self.etag = etag
        self.temp_path = temp_path
        self.state_path = state_path


def strong_etag(value: str | None) -> str | None:
    """Return a strong ETag credential, or ``None`` for weak/malformed values.

    ``W/"abc"`` is weak and must not be used as an ``If-Range`` credential;
    ``"abc"`` is strong.  The first entity-tag of a comma-separated header is
    considered.
    """
    if not value:
        return None
    first = value.split(",", 1)[0].strip()
    if not first or first.startswith(("W/", "w/")):
        return None
    if len(first) >= 2 and first[0] == '"' and first[-1] == '"':
        return first
    return None


def parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    """Parse ``bytes <start>-<end>/<total>``; return ``None`` if malformed."""
    if not value:
        return None
    match = _CONTENT_RANGE_RE.match(value.strip())
    if not match:
        return None
    start_s, end_s, total_s = match.groups()
    start, end = int(start_s), int(end_s)
    if end < start:
        return None
    total = None if total_s == "*" else int(total_s)
    if total is not None and total < end + 1:
        return None
    return start, end, total


def _default_state_path(target: Path) -> Path:
    return target.with_name(target.name + ".resume.json")


def _new_temp_path(work_dir: Path, target: Path) -> Path:
    return work_dir / f"{target.name}.dl-{uuid.uuid4().hex}.tmp"


def _load_state(
    state_path: Path, work_dir: Path, url: str
) -> dict[str, Any] | None:
    """Validate the resume state together with its temp file.

    Returns ``None`` (invalid state -> fresh start) unless every check passes:
    well-formed JSON of the current version, temp basename, the temp file
    exists, its size equals ``bytes``, the ETag is strong or null, and the URL
    matches.
    """
    try:
        raw = state_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != _STATE_VERSION:
        return None
    temp_name = data.get("temp")
    bytes_count = data.get("bytes")
    etag = data.get("etag")
    if not isinstance(temp_name, str) or not temp_name:
        return None
    if Path(temp_name).name != temp_name or temp_name in (".", ".."):
        return None
    if not isinstance(bytes_count, int) or isinstance(bytes_count, bool):
        return None
    if bytes_count < 0:
        return None
    if etag is not None:
        if not isinstance(etag, str):
            return None
        if strong_etag(etag) is None:
            return None
    if data.get("url") != url:
        return None
    temp_path = work_dir / temp_name
    try:
        if not temp_path.is_file():
            return None
        if temp_path.stat().st_size != bytes_count:
            return None
    except OSError:
        return None
    return {
        "temp_path": temp_path,
        "bytes": bytes_count,
        "etag": etag if etag else None,
    }


def _save_state(
    state_path: Path,
    temp_path: Path,
    bytes_count: int,
    etag: str | None,
    url: str,
) -> None:
    data = {
        "version": _STATE_VERSION,
        "temp": temp_path.name,
        "bytes": bytes_count,
        "etag": etag,
        "url": url,
    }
    tmp_path = state_path.with_name(state_path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, state_path)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


async def _drain_to_file(
    resp: Any,
    file: Any,
    chunk_size: int,
    state_path: Path,
    temp_path: Path,
    already_confirmed: int,
    etag: str | None,
    url: str,
) -> int:
    """Stream the response body into ``file``; preserve progress on drop.

    Returns the number of bytes written in this attempt.  On a mid-read
    disconnect the confirmed total (``already_confirmed`` + received) is
    persisted and :class:`ResumeInterrupted` is raised.
    """
    received = 0
    try:
        async for chunk in resp.content.iter_chunked(chunk_size):
            file.write(chunk)
            received += len(chunk)
    except (ClientPayloadError, ServerDisconnectedError, ClientError) as exc:
        file.flush()
        os.fsync(file.fileno())
        _save_state(
            state_path,
            temp_path,
            already_confirmed + received,
            etag,
            url,
        )
        raise ResumeInterrupted(
            f"connection dropped mid-read after {received} bytes; "
            f"resume from byte {already_confirmed + received}",
            confirmed_bytes=already_confirmed + received,
            etag=etag,
            temp_path=temp_path,
            state_path=state_path,
        ) from exc
    file.flush()
    os.fsync(file.fileno())
    return received


async def download(
    session: ClientSession,
    url: str,
    target: str | Path,
    *,
    state_path: str | Path | None = None,
    chunk_size: int = _CHUNK_SIZE,
) -> Path:
    """Download ``url`` to ``target``, resuming a partial download when safe.

    Returns the target path.  On success the complete file is published
    atomically.  Raises :class:`ResumeInterrupted` when a retry can continue
    from preserved progress, :class:`ResumeError` on a hard failure (the
    published target is left untouched).
    """
    target = Path(target)
    work_dir = target.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    if state_path is None:
        state_path = _default_state_path(target)
    else:
        state_path = Path(state_path)

    resume = _load_state(state_path, work_dir, url)
    headers: dict[str, str] = {}
    if resume is not None:
        headers["Range"] = f"bytes={resume['bytes']}-"
        if resume["etag"]:
            headers["If-Range"] = resume["etag"]

    new_temp: Path | None = None
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 206:
                if resume is None:
                    raise ResumeError(
                        "server returned 206 without a prior resume state"
                    )
                content_range = parse_content_range(
                    resp.headers.get("Content-Range")
                )
                if content_range is None:
                    raise ResumeError(
                        f"malformed Content-Range header: "
                        f"{resp.headers.get('Content-Range')!r}"
                    )
                start, end, total = content_range
                if start != resume["bytes"]:
                    raise ResumeError(
                        f"Content-Range start {start} does not match local "
                        f"confirmed byte count {resume['bytes']}"
                    )
                expected = end - start + 1
                if expected <= 0:
                    raise ResumeError("empty Content-Range segment")
                etag = strong_etag(resp.headers.get("ETag")) or resume["etag"]
                with open(resume["temp_path"], "ab") as temp_file:
                    received = await _drain_to_file(
                        resp,
                        temp_file,
                        chunk_size,
                        state_path,
                        resume["temp_path"],
                        resume["bytes"],
                        etag,
                        url,
                    )
                if received != expected:
                    # Clean short read: hard failure.  Truncate the temp file
                    # back to the confirmed length so the resume state stays
                    # coherent; nothing is published.
                    with open(resume["temp_path"], "r+b") as temp_file:
                        temp_file.truncate(resume["bytes"])
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                    raise ResumeError(
                        f"short read: received {received} bytes, "
                        f"Content-Range promised {expected}"
                    )
                size = resume["temp_path"].stat().st_size
                if size != resume["bytes"] + expected:
                    raise ResumeError(
                        f"temp file size {size} does not match confirmed "
                        f"length {resume['bytes'] + expected}"
                    )
                if total is not None and size != total:
                    raise ResumeError(
                        f"temp file size {size} does not match Content-Range "
                        f"total {total}"
                    )
                os.replace(resume["temp_path"], target)
                _fsync_dir(work_dir)
                state_path.unlink(missing_ok=True)
                return target

            if resp.status == 200:
                # Full content (server ignored the range, or fresh start).
                # Write to a brand-new temp file from byte zero; never append.
                new_temp = _new_temp_path(work_dir, target)
                etag = strong_etag(resp.headers.get("ETag"))
                content_length = resp.content_length
                with open(new_temp, "wb") as temp_file:
                    received = await _drain_to_file(
                        resp,
                        temp_file,
                        chunk_size,
                        state_path,
                        new_temp,
                        0,
                        etag,
                        url,
                    )
                if content_length is not None and received != content_length:
                    raise ResumeError(
                        f"short read: received {received} bytes, "
                        f"Content-Length promised {content_length}"
                    )
                os.replace(new_temp, target)
                _fsync_dir(work_dir)
                state_path.unlink(missing_ok=True)
                if resume is not None:
                    resume["temp_path"].unlink(missing_ok=True)
                new_temp = None
                return target

            raise ResumeError(f"unexpected HTTP status {resp.status}")
    except ResumeInterrupted:
        raise
    except ResumeError:
        if new_temp is not None:
            new_temp.unlink(missing_ok=True)
        raise
