"""Small resumable HTTP download component built on aiohttp.ClientSession.

See ``resumable_download.downloader`` for the API and ``RECOVERY.rst``
alongside the demo script for the resume rules.
"""

from .downloader import (
    DownloadError,
    DownloadInterrupted,
    ExhaustedRetries,
    InvalidRangeResponse,
    StateCorruptedError,
    download,
    is_strong_etag,
)

__all__ = (
    "DownloadError",
    "DownloadInterrupted",
    "ExhaustedRetries",
    "InvalidRangeResponse",
    "StateCorruptedError",
    "download",
    "is_strong_etag",
)
