Resumable download recovery rules
=================================

The component lives in :mod:`resumable_download` and is built on
``aiohttp.ClientSession``. Entry point: ``download(session, url, dest)``.

Files on disk
-------------
For a destination ``D/name`` the component only ever touches three kinds of
paths, all next to the destination:

* ``D/name.<token>.part`` -- the current download fragment. A new random
  token is used whenever a download (re)starts from offset zero.
* ``D/name.dlstate`` -- one JSON object describing **exactly** that fragment:

  .. code-block:: json

     {
       "version": 1,
       "url": "http://host/file",
       "offset": 3,
       "etag": "\"v1\"",
       "part_name": "name.9f86...part"
     }

* ``D/name`` -- the published file. It is created or replaced only by an
  atomic ``os.replace`` of a fully validated fragment, then the state file is
  deleted. A previously published destination is never truncated, written or
  replaced on any failure.

State is written *before* its claim becomes true: the strong ETag state for a
fresh response is persisted before the first body byte, and while streaming,
bytes are ``flush``\ ed and ``fsync``\ ed before the offset is advanced.

When (and only when) a resume is attempted
------------------------------------------
On every attempt the state file and the part file are validated as a pair:

* JSON parses, version is known, all keys present with correct types;
* ``url`` equals the URL being requested;
* ``etag`` is present and a **strong** ETag (see below);
* ``part_name`` has the ``.part`` suffix, is not the destination name and
  stays inside the destination directory;
* the named part file exists and its exact size equals ``offset``.

Any mismatch raises ``StateCorruptedError`` and aborts -- the component never
"heals" state by guessing, truncating or redownloading behind the caller's
back. When a valid state with ``offset > 0`` exists, the request carries both
``Range: bytes=<offset>-`` and ``If-Range: <etag>``.

Strong vs weak ETags
--------------------
Only a quoted strong validator such as ``"v1"`` is a recovery credential.

* A weak validator (``W/"v1"``), a malformed ETag, or a missing ETag is never
  written to state and never sent in ``If-Range``.
* If the only ETag available for a response is weak/missing, a dropped
  connection leaves **no** resumable state: the unbacked fragment is deleted
  and the next attempt is a plain full GET.
* State already on disk whose ETag has been weakened (or a crafted state with
  a weak ETag) is rejected as corrupt.

Handling the response
---------------------
``206 Partial Content`` (only valid as an answer to a ranged, state-backed
request):

1. ``Content-Range`` must parse as ``bytes start-end/total``.
2. ``start`` must equal the state offset exactly -- a shifted range is a fatal
   ``InvalidRangeResponse``; nothing is appended.
3. ``total`` must not be smaller than the confirmed offset and ``end`` must be
   below ``total``.
4. A strong response ETag, if present, must equal the state ETag.
5. The fragment bytes are appended and counted. The delivered count must
   equal ``end - start + 1``; otherwise the response is invalid.
6. If the transfer ends exactly at ``total``, the fragment is published.
   If the connection drops earlier, the state now describes the confirmed
   prefix and the next attempt resumes from it.

``200 OK`` means "entity changed, or ranges are not honoured":

* A **brand new** ``.part`` file with a fresh random token is created; bytes
  are written from offset zero and are never appended to the old fragment.
* With a strong new ETag, the state is repointed at the new empty file first,
  and the old fragment is removed. With a weak/missing ETag, old state and
  fragment are discarded and the new run is not resumable.
* Completion requires the byte count to equal ``Content-Length`` (when
  advertised) before the atomic publish.

Other status codes are fatal ``DownloadError``; transport errors
(``ClientConnectionError``, ``ClientPayloadError`` such as a premature EOF,
timeouts, OS errors) consume one of up to ``max_attempts`` tries while keeping
the confirmed fragment; exhaustion raises ``ExhaustedRetries``.

Verification
------------
* Demo (local aiohttp server, finishes in seconds):
  ``python resumable_demo/run_demo.py``
* Focused tests:
  ``cd resumable_tests && python -m pytest test_resumable_download.py -o addopts=""``
