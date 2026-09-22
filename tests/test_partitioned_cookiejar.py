import asyncio
import json
from http.cookies import SimpleCookie
from pathlib import Path
from unittest import mock

import pytest
from pytest_aiohttp import AiohttpClient, AiohttpServer
from yarl import URL

from aiohttp import ClientSession, CookieJar, PartitionedCookieJar, web
from aiohttp.cookiejar import _normalize_partition_key


def test_partition_key_normalization() -> None:
    assert _normalize_partition_key("https://Example.COM:443/a/b?x=1") == (
        "https://example.com:443"
    )
    assert _normalize_partition_key("https://example.com") == (
        "https://example.com:443"
    )
    assert _normalize_partition_key(URL("http://example.com:80/path")) == (
        "http://example.com:80"
    )
    # Different explicit ports are different partitions.
    assert _normalize_partition_key("http://host:8080") == "http://host:8080"
    assert _normalize_partition_key("http://host:9090") == "http://host:9090"
    # IDNA is applied to unicode hosts.
    assert _normalize_partition_key("https://bücher.example") == (
        "https://xn--bcher-kva.example:443"
    )


@pytest.mark.parametrize(
    "bad", ["ftp://example.com", "/relative", "example.com", "https://"]
)
def test_partition_key_rejects_non_http_origin(bad: str) -> None:
    with pytest.raises(ValueError):
        _normalize_partition_key(bad)


def test_partition_key_rejects_credentials() -> None:
    with pytest.raises(ValueError):
        _normalize_partition_key("https://user:pw@example.com")


def test_partitions_isolate_same_identity() -> None:
    jar = PartitionedCookieJar()
    url = URL("https://example.com/")
    jar.update_cookies_from_headers(["sid=A"], url, "https://a.example")
    jar.update_cookies_from_headers(["sid=B"], url, "https://b.example")

    assert jar.filter_cookies(url, "https://a.example")["sid"].value == "A"
    assert jar.filter_cookies(url, "https://b.example")["sid"].value == "B"
    # Missing partition yields nothing; default partition is separate.
    assert "sid" not in jar.filter_cookies(url)
    assert len(jar) == 2


def test_default_partition_matches_plain_jar() -> None:
    plain = CookieJar()
    partitioned = PartitionedCookieJar()
    url = URL("https://example.com/")
    plain.update_cookies_from_headers(["sid=plain"], url)
    partitioned.update_cookies_from_headers(["sid=plain"], url)

    assert (
        partitioned.filter_cookies(url)["sid"].value
        == plain.filter_cookies(url)["sid"].value
    )


def test_partitioned_host_only_rules_apply_per_partition() -> None:
    jar = PartitionedCookieJar()
    p1 = "https://example.com"
    jar.update_cookies_from_headers(
        ["sid=A; Domain=example.com"], URL("https://example.com/"), p1
    )
    jar.update_cookies_from_headers(["sid=B"], URL("https://sub.example.com/"), p1)
    # Host-only B must not leak to the apex; domain A must reach the subdomain.
    sub = URL("https://sub.example.com/")
    sent = jar.filter_cookies(sub, p1)
    assert sent["sid"].value == "B"
    apex = jar.filter_cookies(URL("https://example.com/"), p1)
    assert apex["sid"].value == "A"
    keys = jar.host_only_cookies
    assert (
        "https://example.com:443\x00sub.example.com",
        "",
        "sid",
    ) in keys


def test_partitioned_secure_cookie_not_sent_over_http() -> None:
    jar = PartitionedCookieJar()
    p = "https://secure.com"
    jar.update_cookies_from_headers(["sid=A; Secure"], URL("https://secure.com/"), p)
    assert "sid" not in jar.filter_cookies(URL("http://secure.com/"), p)
    assert jar.filter_cookies(URL("https://secure.com/"), p)["sid"].value == "A"


def test_partitioned_expiration_keeps_other_partition() -> None:
    jar = PartitionedCookieJar()
    url = URL("https://example.com/")
    with mock.patch("aiohttp.cookiejar.time.time", return_value=1000.0):
        jar.update_cookies_from_headers(["sid=A; Max-Age=10"], url, "https://a.example")
        jar.update_cookies_from_headers(
            ["sid=B; Max-Age=1000"], url, "https://b.example"
        )
    assert len(jar) == 2
    with mock.patch("aiohttp.cookiejar.time.time", return_value=1020.0):
        assert "sid" not in jar.filter_cookies(url, "https://a.example")
        assert jar.filter_cookies(url, "https://b.example")["sid"].value == "B"
    assert len(jar) == 1


def test_clear_scoped_to_partition() -> None:
    jar = PartitionedCookieJar()
    url = URL("https://example.com/")
    jar.update_cookies_from_headers(["sid=A"], url, "https://a.example")
    jar.update_cookies_from_headers(["sid=B"], url, "https://b.example")
    jar.clear(partition_key="https://a.example")
    assert "sid" not in jar.filter_cookies(url, "https://a.example")
    assert jar.filter_cookies(url, "https://b.example")["sid"].value == "B"
    jar.clear()
    assert len(jar) == 0


def test_clear_domain_covers_all_partitions() -> None:
    jar = PartitionedCookieJar()
    url = URL("https://example.com/")
    jar.update_cookies_from_headers(["sid=A"], url, "https://a.example")
    jar.update_cookies_from_headers(["sid=B"], url, "https://b.example")
    jar.clear_domain("example.com")
    assert len(jar) == 0


def test_save_load_roundtrip_preserves_partitions_and_host_only(
    tmp_path: Path,
) -> None:
    jar = PartitionedCookieJar()
    jar.update_cookies_from_headers(
        ["sid=A"], URL("https://example.com/"), "https://a.example"
    )
    jar.update_cookies_from_headers(
        ["sid=B"], URL("https://example.com/"), "https://b.example"
    )
    jar.update_cookies_from_headers(["shared=S"], URL("https://example.com/"))
    file_path = tmp_path / "cookies.json"
    jar.save(file_path)

    payload = json.loads(file_path.read_text())
    assert payload["version"] == 1
    assert set(payload["partitions"]) == {
        "",
        "https://a.example:443",
        "https://b.example:443",
    }

    loaded = PartitionedCookieJar()
    loaded.load(file_path)
    url = URL("https://example.com/")
    assert loaded.filter_cookies(url, "https://a.example")["sid"].value == "A"
    assert loaded.filter_cookies(url, "https://b.example")["sid"].value == "B"
    assert loaded.filter_cookies(url)["shared"].value == "S"
    # Host-only scope survives persistence.
    assert (
        "https://a.example:443\x00example.com",
        "",
        "sid",
    ) in loaded.host_only_cookies
    assert (
        loaded.filter_cookies(URL("https://sub.example.com/"), "https://a.example")
        == SimpleCookie()
    )


def test_load_unsupported_version_fails_and_keeps_state(tmp_path: Path) -> None:
    jar = PartitionedCookieJar()
    jar.update_cookies_from_headers(
        ["sid=A"], URL("https://example.com/"), "https://a.example"
    )
    file_path = tmp_path / "bad.json"
    file_path.write_text(json.dumps({"version": 99, "partitions": {}}))
    with pytest.raises(ValueError, match="unsupported"):
        jar.load(file_path)
    assert (
        jar.filter_cookies(URL("https://example.com/"), "https://a.example")[
            "sid"
        ].value
        == "A"
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "{}",
        json.dumps({"version": 1}),
        json.dumps(
            {
                "version": 1,
                "partitions": {
                    "https://a.example:443": {"example.com|/": {"sid": {"key": "sid"}}}
                },
            }
        ),
        json.dumps({"version": 1, "partitions": {"ftp://bad:21": {}}}),
    ],
)
def test_load_corrupt_file_does_not_replace_state(tmp_path: Path, payload: str) -> None:
    jar = PartitionedCookieJar()
    jar.update_cookies_from_headers(
        ["sid=A"], URL("https://example.com/"), "https://a.example"
    )
    file_path = tmp_path / "corrupt.json"
    file_path.write_text(payload)
    with pytest.raises((ValueError, KeyError, TypeError, json.JSONDecodeError)):
        jar.load(file_path)
    assert (
        jar.filter_cookies(URL("https://example.com/"), "https://a.example")[
            "sid"
        ].value
        == "A"
    )


def test_plain_cookiejar_rejects_partition_key() -> None:
    plain = CookieJar()
    with pytest.raises(TypeError):
        plain.filter_cookies(URL("https://example.com/"), "https://a.example")
    with pytest.raises(TypeError):
        plain.update_cookies(
            SimpleCookie("sid=A"),
            URL("https://example.com/"),
            "https://a.example",
        )
    with pytest.raises(TypeError):
        plain.update_cookies_from_headers(
            ["sid=A"], URL("https://example.com/"), "https://a.example"
        )


async def _echo_cookie(request: web.Request) -> web.Response:
    return web.json_response({"cookie": request.headers.get("Cookie", "")})


async def _set_sid(request: web.Request) -> web.Response:
    value = request.query["v"]
    resp = web.Response(text="ok")
    resp.headers.add("Set-Cookie", f"sid={value}; Path=/")
    return resp


async def _redirect_to_echo(request: web.Request) -> web.Response:
    raise web.HTTPFound("/echo")


def _make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/set", _set_sid)
    app.router.add_get("/echo", _echo_cookie)
    app.router.add_get("/redirect", _redirect_to_echo)
    return app


async def test_loopback_partitions_echo_and_redirect(
    aiohttp_client: AiohttpClient,
) -> None:
    jar = PartitionedCookieJar(unsafe=True)
    client_a = await aiohttp_client(_make_app(), cookie_jar=jar)
    client_b = await aiohttp_client(_make_app(), cookie_jar=jar)
    p_a = f"http://127.0.0.1:{client_a.port}"
    p_b = f"http://127.0.0.1:{client_b.port}"

    await client_a.get("/set?v=A", partition_key=p_a)
    await client_b.get("/set?v=B", partition_key=p_b)

    # Same path/name, two partitions on two loopback origins.
    resp_a = await client_a.get("/echo", partition_key=p_a)
    resp_b = await client_b.get("/echo", partition_key=p_b)
    resp_none = await client_a.get("/echo")
    assert (await resp_a.json())["cookie"] == "sid=A"
    assert (await resp_b.json())["cookie"] == "sid=B"
    assert (await resp_none.json())["cookie"] == ""

    # Redirect must keep using the original request's partition.
    redirected = await client_a.get("/redirect", partition_key=p_a)
    assert (await redirected.json())["cookie"] == "sid=A"
    redirected_b = await client_b.get("/redirect", partition_key=p_b)
    assert (await redirected_b.json())["cookie"] == "sid=B"


async def test_loopback_partitions_interleaved_concurrent(
    aiohttp_client: AiohttpClient,
) -> None:
    jar = PartitionedCookieJar(unsafe=True)
    client_a = await aiohttp_client(_make_app(), cookie_jar=jar)
    client_b = await aiohttp_client(_make_app(), cookie_jar=jar)

    async def fetch(client: object, value: str) -> str:
        partition = f"http://127.0.0.1:{client.port}"
        await client.get(
            f"/set?v={value}",
            partition_key=partition,
        )
        resp = await client.get("/echo", partition_key=partition)
        return (await resp.json())["cookie"]

    pairs = [(client_a, "A"), (client_b, "B")] * 6
    results = await asyncio.gather(*(fetch(client, v) for client, v in pairs))
    assert results == ["sid=" + v for _, v in pairs]


@pytest.mark.skip_blockbuster
async def test_loopback_partitions_saved_and_reloaded(
    aiohttp_server: AiohttpServer, tmp_path: Path
) -> None:
    server = await aiohttp_server(_make_app())
    server = await aiohttp_server(_make_app())
    base_url = f"http://127.0.0.1:{server.port}"
    jar = PartitionedCookieJar(unsafe=True)
    async with ClientSession(cookie_jar=jar) as session:
        await session.get(f"{base_url}/set?v=A", partition_key=base_url)
    cookie_file = tmp_path / "cookies.json"
    jar.save(cookie_file)

    reloaded = PartitionedCookieJar(unsafe=True)
    reloaded.load(cookie_file)
    # Partition and host-only scope both survive persistence.
    assert reloaded.host_only_cookies == frozenset(
        {(f"{base_url}\x00127.0.0.1", "", "sid")}
    )
    async with ClientSession(cookie_jar=reloaded) as session:
        resp = await session.get(f"{base_url}/echo", partition_key=base_url)
        assert (await resp.json())["cookie"] == "sid=A"


async def test_cross_origin_redirect_keeps_original_partition(
    aiohttp_server: AiohttpServer,
) -> None:
    # Server A redirects to server B; the cookie set for partition A on B
    # must be stored back into partition A, and B's echo must see sid=A.
    server_b = await aiohttp_server(_make_app())

    async def redirect_to_b(request: web.Request) -> web.Response:
        raise web.HTTPFound(f"http://127.0.0.1:{server_b.port}/echo")

    app_a = web.Application()
    app_a.router.add_get("/redirect", redirect_to_b)
    server_a = await aiohttp_server(app_a)

    origin_a = f"http://127.0.0.1:{server_a.port}"
    origin_b = f"http://127.0.0.1:{server_b.port}"
    jar = PartitionedCookieJar(unsafe=True)
    # Seed sid=A into partition A, issued "by" origin B (host-only to B).
    jar.update_cookies_from_headers(["sid=A"], URL(origin_b + "/echo"), origin_a)
    async with ClientSession(cookie_jar=jar) as session:
        resp = await session.get(f"{origin_a}/redirect", partition_key=origin_a)
        assert (await resp.json())["cookie"] == "sid=A"
    # No cookie was written into the default or a target-derived partition.
    assert "sid" not in jar.filter_cookies(URL(origin_b + "/echo"))
    assert "sid" not in jar.filter_cookies(URL(origin_b + "/echo"), origin_b)
