"""Functional tests for the ClientSession redirect_policy hook."""

from __future__ import annotations  # TODO(PY311): Remove

import pytest
from pytest_aiohttp import AiohttpServer

from aiohttp import ClientSession, RedirectDecision, RedirectInfo, hdrs, web
from aiohttp.client_exceptions import InvalidUrlRedirectClientError
from aiohttp.test_utils import TestServer


def _server_url(server: TestServer, path: str = "/") -> str:
    return f"http://127.0.0.1:{server.port}{path}"


async def test_redirect_policy_allow_same_origin_deny_cross_port(
    aiohttp_server: AiohttpServer,
) -> None:
    server2_hits = 0

    async def target2(request: web.Request) -> web.Response:
        nonlocal server2_hits
        server2_hits += 1
        return web.Response(text="server2")

    app2 = web.Application()
    app2.router.add_get("/target", target2)
    server2 = await aiohttp_server(app2)

    async def ok(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def same_origin_redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/ok")

    async def cross_origin_redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound(_server_url(server2, "/target"))

    app1 = web.Application()
    app1.router.add_get("/ok", ok)
    app1.router.add_get("/same", same_origin_redirect)
    app1.router.add_get("/cross", cross_origin_redirect)
    server1 = await aiohttp_server(app1)

    calls = []

    def policy(info: RedirectInfo) -> RedirectDecision:
        calls.append(info)
        return RedirectDecision(allow=info.from_url.origin() == info.to_url.origin())

    async with ClientSession() as session:
        async with session.get(
            _server_url(server1, "/same"), redirect_policy=policy
        ) as resp:
            assert resp.status == 200
            assert await resp.text() == "ok"
            assert len(resp.history) == 1

        async with session.get(
            _server_url(server1, "/cross"), redirect_policy=policy
        ) as resp:
            # The denied redirect response is returned as the final response.
            assert resp.status == 302
            assert len(resp.history) == 0

    assert server2_hits == 0
    assert len(calls) == 2
    assert calls[0].redirect_count == 1
    assert calls[0].status == 302
    assert calls[1].to_url.port == server2.port


async def test_redirect_policy_302_post_method_rewrite(
    aiohttp_server: AiohttpServer,
) -> None:
    async def target(request: web.Request) -> web.Response:
        return web.Response(text=request.method)

    async def redirect(request: web.Request) -> web.Response:
        await request.read()
        raise web.HTTPFound("/target")

    app = web.Application()
    app.router.add_post("/redirect", redirect)
    app.router.add_route("*", "/target", target)
    server = await aiohttp_server(app)

    infos = []

    def policy(info: RedirectInfo) -> RedirectDecision:
        infos.append(info)
        return RedirectDecision()

    async with ClientSession() as session:
        async with session.post(
            _server_url(server, "/redirect"),
            data={"some": "data"},
            redirect_policy=policy,
        ) as resp:
            assert resp.status == 200
            assert await resp.text() == "GET"

    assert len(infos) == 1
    assert infos[0].status == 302
    assert infos[0].method == "GET"


async def test_redirect_policy_307_post_method_preserved(
    aiohttp_server: AiohttpServer,
) -> None:
    async def target(request: web.Request) -> web.Response:
        body = await request.read()
        return web.Response(text=f"{request.method}:{body.decode()}")

    async def redirect(request: web.Request) -> web.Response:
        await request.read()
        raise web.HTTPTemporaryRedirect("/target")

    app = web.Application()
    app.router.add_post("/redirect", redirect)
    app.router.add_post("/target", target)
    server = await aiohttp_server(app)

    infos = []

    def policy(info: RedirectInfo) -> RedirectDecision:
        infos.append(info)
        return RedirectDecision()

    async with ClientSession() as session:
        async with session.post(
            _server_url(server, "/redirect"),
            data=b"payload",
            redirect_policy=policy,
        ) as resp:
            assert resp.status == 200
            assert await resp.text() == "POST:payload"

    assert len(infos) == 1
    assert infos[0].status == 307
    assert infos[0].method == "POST"


async def test_redirect_policy_deny_keeps_connection_reusable(
    aiohttp_server: AiohttpServer,
) -> None:
    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/target")

    async def target(request: web.Request) -> web.Response:
        return web.Response(text="target")

    app = web.Application()
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/target", target)
    server = await aiohttp_server(app)

    def deny(info: RedirectInfo) -> RedirectDecision:
        return RedirectDecision(allow=False)

    async with ClientSession() as session:
        async with session.get(
            _server_url(server, "/redirect"), redirect_policy=deny
        ) as resp:
            assert resp.status == 302
        # The denied redirect response was released, so its connection is
        # back in the pool and can serve the next request.
        assert not session.connector._acquired
        async with session.get(_server_url(server, "/target")) as resp:
            assert resp.status == 200
            assert await resp.text() == "target"


async def test_redirect_policy_strip_headers(aiohttp_server: AiohttpServer) -> None:
    received = {}

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/target")

    async def target(request: web.Request) -> web.Response:
        received.update(request.headers)
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/target", target)
    server = await aiohttp_server(app)

    def policy(info: RedirectInfo) -> RedirectDecision:
        return RedirectDecision(strip_headers=("X-Secret",))

    async with ClientSession() as session:
        async with session.get(
            _server_url(server, "/redirect"),
            headers={"X-Secret": "1", "X-Keep": "2"},
            redirect_policy=policy,
        ) as resp:
            assert resp.status == 200

    assert "X-Secret" not in received
    assert received["X-Keep"] == "2"


async def test_redirect_policy_cannot_restore_stripped_authorization(
    aiohttp_server: AiohttpServer,
) -> None:
    received = {}

    async def target2(request: web.Request) -> web.Response:
        received.update(request.headers)
        return web.Response(text="ok")

    app2 = web.Application()
    app2.router.add_get("/target", target2)
    server2 = await aiohttp_server(app2)

    async def cross_origin_redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound(_server_url(server2, "/target"))

    app1 = web.Application()
    app1.router.add_get("/cross", cross_origin_redirect)
    server1 = await aiohttp_server(app1)

    def policy(info: RedirectInfo) -> RedirectDecision:
        # The policy allows the redirect, but the Authorization header
        # dropped by the library on the origin change must stay dropped.
        return RedirectDecision()

    async with ClientSession() as session:
        async with session.get(
            _server_url(server1, "/cross"),
            headers={hdrs.AUTHORIZATION: "Bearer token"},
            redirect_policy=policy,
        ) as resp:
            assert resp.status == 200

    assert hdrs.AUTHORIZATION not in received


async def test_redirect_policy_relative_location(
    aiohttp_server: AiohttpServer,
) -> None:
    async def redirect(request: web.Request) -> web.Response:
        return web.Response(status=301, headers={hdrs.LOCATION: "/target"})

    async def target(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/target", target)
    server = await aiohttp_server(app)

    infos = []

    def policy(info: RedirectInfo) -> None:
        # Returning None allows the redirect unchanged.
        infos.append(info)

    async with ClientSession() as session:
        async with session.get(
            _server_url(server, "/redirect"), redirect_policy=policy
        ) as resp:
            assert resp.status == 200

    assert len(infos) == 1
    assert infos[0].to_url.absolute
    assert infos[0].to_url.path == "/target"
    assert infos[0].from_url.path == "/redirect"


async def test_redirect_policy_not_called_for_invalid_redirect_url(
    aiohttp_server: AiohttpServer,
) -> None:
    async def redirect(request: web.Request) -> web.Response:
        return web.Response(status=301, headers={hdrs.LOCATION: "http://:/"})

    app = web.Application()
    app.router.add_get("/redirect", redirect)
    server = await aiohttp_server(app)

    called = False

    def policy(info: RedirectInfo) -> RedirectDecision:
        nonlocal called
        called = True
        return RedirectDecision()

    async with ClientSession() as session:
        with pytest.raises(InvalidUrlRedirectClientError):
            await session.get(_server_url(server, "/redirect"), redirect_policy=policy)

    assert not called


async def test_redirect_policy_exception_cancels_request(
    aiohttp_server: AiohttpServer,
) -> None:
    class CancelRedirect(Exception):
        pass

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/target")

    async def target(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/target", target)
    server = await aiohttp_server(app)

    def policy(info: RedirectInfo) -> RedirectDecision:
        raise CancelRedirect

    async with ClientSession() as session:
        with pytest.raises(CancelRedirect):
            await session.get(_server_url(server, "/redirect"), redirect_policy=policy)


async def test_redirect_policy_not_called_when_redirects_disabled(
    aiohttp_server: AiohttpServer,
) -> None:
    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/target")

    app = web.Application()
    app.router.add_get("/redirect", redirect)
    server = await aiohttp_server(app)

    called = False

    def policy(info: RedirectInfo) -> RedirectDecision:
        nonlocal called
        called = True
        return RedirectDecision()

    async with ClientSession() as session:
        async with session.get(
            _server_url(server, "/redirect"),
            allow_redirects=False,
            redirect_policy=policy,
        ) as resp:
            assert resp.status == 302

    assert not called


async def test_redirect_policy_session_level_default_and_override(
    aiohttp_server: AiohttpServer,
) -> None:
    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/target")

    async def target(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/target", target)
    server = await aiohttp_server(app)

    def deny(info: RedirectInfo) -> RedirectDecision:
        return RedirectDecision(allow=False)

    async with ClientSession(redirect_policy=deny) as session:
        async with session.get(_server_url(server, "/redirect")) as resp:
            assert resp.status == 302
        # A per-request policy overrides the session-level one.
        async with session.get(
            _server_url(server, "/redirect"), redirect_policy=None
        ) as resp:
            assert resp.status == 200
            assert await resp.text() == "ok"
