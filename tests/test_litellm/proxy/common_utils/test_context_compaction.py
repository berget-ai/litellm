import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import TypeAdapter

import litellm
from litellm import ModelResponse
from litellm.caching.caching import DualCache
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import is_request_body_safe
from litellm.proxy.common_utils.context_compaction import (
    ProxySummaryExecutor,
    _SummaryRequest,
    _summary_request,
    mark_summary_provider_completed,
    proxy_summary_executor_scope,
    summary_request_updates,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
    _request_stash,
    get_request_stash,
)
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.router_strategy.complexity_router.context_compaction import current_summary_executor
from litellm.types.utils import Usage

_BODY: Final = TypeAdapter(dict[str, object])
_MESSAGES: Final = [{"role": "user", "content": "Private older conversation"}]


@pytest.fixture(autouse=True)
def isolated_stash():
    token: Final = _request_stash.set(None)
    try:
        yield
    finally:
        _request_stash.reset(token)


def parent_request(app: FastAPI, api_key: str = "caller", root_path: str = "/gateway") -> Request:
    return Request({
        "type": "http", "method": "POST", "scheme": "https", "path": f"{root_path}/v1/messages",
        "root_path": root_path, "query_string": b"", "server": ("gateway.test", 443),
        "client": ("192.0.2.10", 2345), "app": app,
        "headers": [(b"authorization", f"Bearer {api_key}".encode()), (b"content-length", b"99999"),
                    (b"x-litellm-call-id", b"parent"), (b"litellm-disable-message-redaction", b"true"),
                    (b"x-app", b"cli"), (b"x-claude-code-agent-id", b"agent"),
                    (b"x-claude-code-session-id", b"session"), (b"x-litellm-model-id", b"target-id"),
                    (b"x-litellm-model", b"target")],
    })


def summary_app(
    limiter: _PROXY_MaxParallelRequestsHandler_v3,
    cache: DualCache,
    receive: Callable[[Request, Mapping[str, object]], Awaitable[ModelResponse]],
) -> FastAPI:
    app: Final = FastAPI()

    async def authorize(request: Request) -> UserAPIKeyAuth:
        body: Final = _BODY.validate_python(await request.json())
        assert is_request_body_safe(body, {}, None, str(body["model"]))
        if request.headers.get("authorization") != "Bearer caller" or body["model"] == "denied":
            raise HTTPException(403, "model denied")
        return UserAPIKeyAuth(api_key=hash_token("caller"), max_parallel_requests=1, rpm_limit=10)

    @app.post("/v1/chat/completions")
    async def complete(request: Request, auth: UserAPIKeyAuth = Depends(authorize)) -> dict[str, object]:
        body: Final = _BODY.validate_python(await request.json())
        data: Final = await add_litellm_data_to_request(
            body, request, auth, proxy_config=MagicMock(), general_settings={}
        )
        data["litellm_call_id"] = request.headers["x-litellm-call-id"]
        await limiter.async_pre_call_hook(auth, cache, data, "acompletion")
        response: Final = await receive(request, data)
        mark_summary_provider_completed()
        await limiter.async_post_call_success_hook(data, auth, response)
        return response.model_dump()

    return app


@pytest.mark.asyncio
async def test_authenticated_summary_isolated_private_and_scope_bound():
    cache: Final = DualCache()
    limiter: Final = _PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(cache))
    auth: Final = UserAPIKeyAuth(api_key=hash_token("caller"), max_parallel_requests=1)
    parent_data: Final = {
        "model": "target", "litellm_call_id": "parent", "turn_off_message_logging": True,
        "allowed_model_region": "eu", "user": "end-user", "litellm_session_id": "session",
        "metadata": {"tags": ["cost-center"]},
    }
    await limiter.async_pre_call_hook(auth, cache, parent_data, "acompletion")
    parent: Final = get_request_stash()
    acquisition: Final = parent.parallel_slot
    calls: Final = asyncio.Queue()

    async def receive(request: Request, data: Mapping[str, object]) -> ModelResponse:
        assert get_request_stash() is not parent
        assert parent.parallel_slot is acquisition
        assert request.client.host == "192.0.2.10"
        assert request.headers.get("litellm-disable-message-redaction") is None
        assert request.headers.get("x-app") is None
        assert request.headers.get("x-claude-code-agent-id") is None
        assert request.headers.get("x-litellm-model-id") is None
        assert request.headers.get("x-litellm-model") is None
        assert request.headers["x-claude-code-session-id"] == "session"
        assert data["litellm_call_id"] != "parent"
        assert data["turn_off_message_logging"] is True
        assert data["num_retries"] == 0
        assert data["max_retries"] == 0
        assert data["disable_fallbacks"] is True
        assert data["allowed_model_region"] == "eu"
        assert data["user"] == "end-user"
        assert data["litellm_session_id"] == "session"
        metadata: Final = _BODY.validate_python(data["metadata"])
        assert metadata[INTERNAL_CALL_ORIGIN_METADATA_KEY] == "context_compaction"
        assert "cost-center" in metadata["tags"]
        assert data["messages"] == _MESSAGES
        calls.put_nowait(data["model"])
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": "Summary"}, "finish_reason": "stop"}],
                             usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12))

    request: Final = parent_request(summary_app(limiter, cache, receive))
    assert current_summary_executor() is None
    with proxy_summary_executor_scope(request, parent_data, limiter):
        executor: Final = current_summary_executor()
        response: Final = await asyncio.create_task(executor("summary", _MESSAGES, 128, 10))
        assert response.choices[0].message.content == "Summary"
        with pytest.raises(litellm.APIError) as denied:
            await executor("denied", _MESSAGES, 128, 10)
        assert denied.value.status_code == 403
        assert calls.qsize() == 1
    assert current_summary_executor() is None
    assert get_request_stash() is parent
    assert parent.parallel_slot is acquisition
    with pytest.raises(ValueError, match="closed"):
        await executor("summary", _MESSAGES, 128, 10)
    await limiter.async_post_call_success_hook(parent_data, auth, ModelResponse())
    assert parent.parallel_slot is None


@pytest.mark.asyncio
async def test_real_proxy_auth_reentry_does_not_require_clientside_credentials_opt_in(monkeypatch):
    from litellm.proxy import proxy_server

    key: Final = "sk-test-context-compaction-master"
    router: Final = litellm.Router(model_list=[{
        "model_name": "summary",
        "litellm_params": {"model": "openai/gpt-4o-mini", "mock_response": "Summary"},
    }])
    monkeypatch.setattr(proxy_server, "master_key", key)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    request: Final = parent_request(proxy_server.app, api_key=key, root_path="")
    executor: Final = ProxySummaryExecutor(
        request, {"litellm_call_id": "parent", "turn_off_message_logging": True},
        proxy_server.proxy_logging_obj.max_parallel_request_limiter, None,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app)) as client:
        rejected: Final = await client.post(
            "https://gateway.test/v1/chat/completions",
            headers={"authorization": f"Bearer {key}"},
            json={"model": "summary", "messages": _MESSAGES, "turn_off_message_logging": True},
        )
    assert rejected.is_client_error
    assert "turn_off_message_logging" in rejected.text
    response: Final = await executor("summary", _MESSAGES, 128, 10)
    assert response.choices[0].message.content == "Summary"
    assert proxy_server.general_settings == {}


@pytest.mark.asyncio
async def test_real_auth_denied_summary_never_enters_inference(monkeypatch):
    from litellm.proxy import proxy_server

    key: Final = "sk-test-context-compaction-denied"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        hash_token(key), UserAPIKeyAuth(api_key=hash_token(key), models=["target"]), model_type=UserAPIKeyAuth
    )
    router: Final = litellm.Router(model_list=[{
        "model_name": "summary", "litellm_params": {"model": "openai/gpt-4o-mini", "mock_response": "Summary"},
    }])
    monkeypatch.setattr(proxy_server, "master_key", "sk-test-master")
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    inference: Final = AsyncMock(wraps=litellm.acompletion)
    monkeypatch.setattr(litellm, "acompletion", inference)
    executor: Final = ProxySummaryExecutor(
        parent_request(proxy_server.app, api_key=key, root_path=""), {"litellm_call_id": "parent"},
        proxy_server.proxy_logging_obj.max_parallel_request_limiter, None,
    )
    with pytest.raises(litellm.APIError) as denied:
        await executor("summary", _MESSAGES, 128, 10)
    assert denied.value.status_code in (401, 403)
    inference.assert_not_awaited()
    cached: Final = await cache.async_get_cache(hash_token(key), model_type=UserAPIKeyAuth)
    assert cached.models == ["target"]
    assert cached.spend == 0


@pytest.mark.parametrize("parent_redact", [False, True, None])
@pytest.mark.parametrize("recipient_redact", [False, True, None])
@pytest.mark.parametrize("global_redact", [False, True])
def test_summary_privacy_never_lowers_recipient_policy(
    parent_redact: bool | None,
    recipient_redact: bool | None,
    global_redact: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", global_redact)
    summary: Final = _SummaryRequest(call_id="parent", redact=parent_redact)
    token: Final = _summary_request.set(summary)
    recipient_data: Final = {"turn_off_message_logging": recipient_redact, "metadata": {}}
    try:
        updates: Final = summary_request_updates(
            parent_request(FastAPI()), UserAPIKeyAuth(),
            recipient_data, "metadata",
        )
        merged: Final = {**recipient_data, **updates}
        if parent_redact is True or recipient_redact is True or global_redact:
            assert updates["turn_off_message_logging"] is True
            assert merged["turn_off_message_logging"] is True
            assert summary.redact is True
        else:
            assert "turn_off_message_logging" not in updates
            assert merged["turn_off_message_logging"] is recipient_redact
    finally:
        _summary_request.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_summary_cancellation_stops_child_without_releasing_parent(cancel: bool):
    cache: Final = DualCache()
    limiter: Final = _PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(cache))
    auth: Final = UserAPIKeyAuth(api_key=hash_token("caller"), max_parallel_requests=1)
    parent_data: Final = {"model": "target", "litellm_call_id": "parent"}
    await limiter.async_pre_call_hook(auth, cache, parent_data, "acompletion")
    parent: Final = get_request_stash()
    acquisition: Final = parent.parallel_slot
    entered: Final = asyncio.Event()
    stopped: Final = asyncio.Event()

    async def receive(request: Request, data: Mapping[str, object]) -> ModelResponse:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        raise AssertionError("cancelled provider returned")

    executor: Final = ProxySummaryExecutor(
        parent_request(summary_app(limiter, cache, receive)), parent_data, limiter, parent
    )
    task: Final = asyncio.create_task(executor("summary", _MESSAGES, 128, 10 if cancel else 0.1))
    await asyncio.wait_for(entered.wait(), 5)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert stopped.is_set()
    assert get_request_stash() is parent
    assert parent.parallel_slot is acquisition
    assert not parent.reservation_released
    await limiter.async_post_call_success_hook(parent_data, auth, ModelResponse())
    assert parent.parallel_slot is None
