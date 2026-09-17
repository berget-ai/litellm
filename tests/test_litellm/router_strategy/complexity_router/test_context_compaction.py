import asyncio
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal

import pytest
from anyio.lowlevel import checkpoint
from pydantic import JsonValue, TypeAdapter

from litellm.litellm_core_utils.prompt_templates.compaction import CompactionHistory, split_compaction_history
from litellm.router_strategy.complexity_router.context_compaction import (
    MAX_SUMMARY_CALLS,
    CompactedRequest,
    CompactionFailure,
    CompactionState,
    ModelBudget,
    SummaryMemo,
    prepare_compaction,
)
from litellm.types.utils import Choices, Message, ModelResponse

_Format = Literal["chat", "anthropic", "responses"]
_TARGET: Final = ModelBudget("selected-deployment", input_limit=1_800, output_limit=80)
_SUMMARY: Final = ModelBudget("summary-deployment", input_limit=20_000, output_limit=32)
_MAPPING: Final = TypeAdapter[Mapping[str, object]](Mapping[str, object])
_SNAPSHOT: Final = TypeAdapter[Mapping[str, JsonValue]](Mapping[str, JsonValue])


def _json_mapping(value: object) -> Mapping[str, object]:
    return _MAPPING.validate_python(value)


def _serialize(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_mapping)


def _cost(payload: Mapping[str, object]) -> int:
    return len(_serialize(payload))


@dataclass(frozen=True)
class _Count:
    model: str
    payload: Mapping[str, object]
    tokens: int


@dataclass(frozen=True)
class _Counter:
    calls: SimpleQueue[_Count] = field(default_factory=SimpleQueue)
    cooperative: bool = False

    async def __call__(self, model: str, payload: Mapping[str, object]) -> int:
        if self.cooperative:
            await checkpoint()
        serialized: Final = _serialize(payload)
        tokens: Final = len(serialized)
        self.calls.put(_Count(model, _SNAPSHOT.validate_json(serialized), tokens))
        return tokens


@dataclass(frozen=True)
class _SummaryCall:
    model: str
    messages: tuple[Mapping[str, str], ...]
    max_tokens: int
    timeout: float


@dataclass(frozen=True)
class _Executor:
    summary: str | None = "historical fact = 73"
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"
    delay: float = 0
    fail: bool = False
    replies: tuple[str, ...] = ()
    calls: SimpleQueue[_SummaryCall] = field(default_factory=SimpleQueue)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def __call__(
        self, model: str, messages: Sequence[Mapping[str, str]], max_tokens: int, timeout: float
    ) -> ModelResponse:
        self.calls.put(_SummaryCall(model, tuple(dict(message) for message in messages), max_tokens, timeout))
        self.started.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("summary endpoint failed")
        content: Final = (
            self.replies[min(self.calls.qsize() - 1, len(self.replies) - 1)] if self.replies else self.summary
        )
        return ModelResponse(
            choices=[
                Choices(index=0, finish_reason=self.finish_reason, message=Message(role="assistant", content=content))
            ]
        )


def _state(timeout: float = 10) -> CompactionState:
    state: Final = CompactionState(timeout=timeout)
    state.arm("summary-group")
    return state


def _payload(wire: _Format = "chat") -> Mapping[str, object]:
    old: Final = "historical fact = 73; keep the identifier exactly. " * 100
    messages: Final = (
        (
            {"role": "user", "content": [{"type": "text", "text": old}]},
            {"role": "assistant", "content": [{"type": "text", "text": "old answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "current question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "prefill:"}]},
        )
        if wire == "anthropic"
        else (
            {"role": "user", "content": old},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current question"},
            {"role": "assistant", "content": "prefill:"},
        )
    )
    return {
        "input" if wire == "responses" else "messages": list(messages),
        "system": "request-level system instruction",
        "instructions": "request-level developer instructions",
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "max_output_tokens" if wire == "responses" else "max_tokens": 32,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ("chat", "anthropic", "responses"))
async def test_under_limit_returns_original_without_summary_calls(wire: _Format) -> None:
    payload: Final = _payload(wire)
    original: Final = deepcopy(payload)
    target: Final = ModelBudget(_TARGET.model, input_limit=_cost(payload) + 1_000, output_limit=_TARGET.output_limit)
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await prepare_compaction(payload, target, _state(), (_SUMMARY,), executor, counter)
    assert result is None
    assert executor.calls.empty()
    assert counter.calls.get_nowait() == _Count(target.model, original, _cost(original))
    assert counter.calls.empty()
    assert payload == original


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ("chat", "anthropic", "responses"))
async def test_overflow_recounts_complete_request_and_preserves_active_tail(wire: _Format) -> None:
    payload: Final = _payload(wire)
    original: Final = deepcopy(payload)
    history: Final = split_compaction_history(payload)
    assert isinstance(history, CompactionHistory)
    executor: Final = _Executor()
    counter: Final = _Counter()
    state: Final = _state()
    result: Final = await prepare_compaction(payload, _TARGET, state, (_SUMMARY,), executor, counter)
    assert isinstance(result, CompactedRequest)
    assert result.field == history.field
    assert result.value == history.rewrite("historical fact = 73")
    assert state.calls == executor.calls.qsize() == 1
    call: Final = executor.calls.get_nowait()
    assert call.model == "summary-group"
    assert call.messages[1]["content"] == history.history_text
    assert call.max_tokens <= _SUMMARY.output_limit
    assert 0 < call.timeout <= 10
    counts: Final = tuple(counter.calls.get_nowait() for _ in range(counter.calls.qsize()))
    assert counts[0].payload == original
    assert counts[-1].model == _TARGET.model
    assert counts[-1].payload == {**original, history.field: result.value}
    assert counts[-1].tokens <= _TARGET.input_budget(32)
    assert all(
        count.payload["tools"] == payload["tools"]
        and count.payload["system"] == payload["system"]
        and count.payload["instructions"] == payload["instructions"]
        for count in counts
        if count.model == _TARGET.model
    )
    assert payload == original


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("max_tokens", "max_completion_tokens", "max_output_tokens"))
async def test_output_reservation_changes_whether_compaction_is_required(key: str) -> None:
    base: Final = {"messages": [{"role": "user", "content": "old " * 100}, {"role": "user", "content": "new"}]}
    target: Final = ModelBudget("selected", input_limit=_cost(base) + 200, output_limit=250)
    low_executor: Final = _Executor()
    high_executor: Final = _Executor()
    low: Final = await prepare_compaction({**base, key: 10}, target, _state(), (_SUMMARY,), low_executor, _Counter())
    high: Final = await prepare_compaction({**base, key: 200}, target, _state(), (_SUMMARY,), high_executor, _Counter())
    assert low is None
    assert low_executor.calls.empty()
    assert isinstance(high, CompactedRequest)
    assert high_executor.calls.qsize() == 1
    assert _cost({**base, key: 200, high.field: high.value}) <= target.input_budget(200)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", (0, -1, True, "32", 81))
async def test_invalid_output_reservation_fails_before_spending(value: object) -> None:
    executor: Final = _Executor()
    result: Final = await prepare_compaction(
        {**_payload(), "max_tokens": value}, _TARGET, _state(), (_SUMMARY,), executor, _Counter()
    )
    assert isinstance(result, CompactionFailure)
    assert "output-token" in result.message
    assert executor.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("protected", ("system", "instructions", "tools"))
async def test_oversized_preserved_fields_fail_before_spending(protected: str) -> None:
    value: Final = (
        [{"name": "lookup", "description": "tool specification " * 1_000}]
        if protected == "tools"
        else "instructions " * 1_000
    )
    payload: Final = {**_payload(), protected: value}
    original: Final = deepcopy(payload)
    executor: Final = _Executor()
    result: Final = await prepare_compaction(payload, _TARGET, _state(), (_SUMMARY,), executor, _Counter())
    assert isinstance(result, CompactionFailure)
    assert "preserved" in result.message
    assert executor.calls.empty()
    assert payload == original


@pytest.mark.asyncio
async def test_chunking_accounts_for_all_history_and_every_summary_deployment() -> None:
    payload: Final = _payload()
    history: Final = split_compaction_history(payload)
    assert isinstance(history, CompactionHistory)
    budgets: Final = (_SUMMARY, ModelBudget("smaller-summary-deployment", input_limit=1_200, output_limit=32))
    executor: Final = _Executor(summary="73")
    state: Final = _state()
    result: Final = await prepare_compaction(payload, _TARGET, state, budgets, executor, _Counter())
    assert isinstance(result, CompactedRequest)
    calls: Final = tuple(executor.calls.get_nowait() for _ in range(executor.calls.qsize()))
    assert 1 < len(calls) == state.calls <= MAX_SUMMARY_CALLS
    assert "".join(call.messages[1]["content"] for call in calls) == history.history_text
    assert all(
        _cost({"messages": call.messages}) <= budget.input_budget(call.max_tokens)
        for call in calls
        for budget in budgets
    )
    assert result.value == history.rewrite("\n\n".join("73" for _ in calls))


@pytest.mark.asyncio
async def test_call_budget_stops_before_another_paid_chunk() -> None:
    summary: Final = ModelBudget("small-summary", input_limit=600, output_limit=32)
    executor: Final = _Executor(summary="73")
    state: Final = _state()
    result: Final = await prepare_compaction(_payload(), _TARGET, state, (summary,), executor, _Counter())
    assert isinstance(result, CompactionFailure)
    assert "call budget" in result.message
    assert executor.calls.qsize() == state.calls == MAX_SUMMARY_CALLS
    assert state.memo is None


@pytest.mark.asyncio
async def test_expired_deadline_does_not_start_a_summary() -> None:
    executor: Final = _Executor()
    result: Final = await prepare_compaction(_payload(), _TARGET, _state(timeout=0), (_SUMMARY,), executor, _Counter())
    assert isinstance(result, CompactionFailure)
    assert "time" in result.message
    assert executor.calls.empty()


@pytest.mark.asyncio
async def test_deadline_cancels_an_inflight_summary() -> None:
    executor: Final = _Executor(delay=5)
    state: Final = _state(timeout=0.15)
    result: Final = await asyncio.wait_for(
        prepare_compaction(_payload(), _TARGET, state, (_SUMMARY,), executor, _Counter()), timeout=1
    )
    assert isinstance(result, CompactionFailure)
    assert "time" in result.message
    assert state.calls == executor.calls.qsize() == 1
    assert state.memo is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", (None, "", " \n\t"))
async def test_empty_summary_is_an_explicit_failure(content: str | None) -> None:
    executor: Final = _Executor(summary=content)
    state: Final = _state()
    result: Final = await prepare_compaction(_payload(), _TARGET, state, (_SUMMARY,), executor, _Counter())
    assert isinstance(result, CompactionFailure)
    assert "empty summary" in result.message
    assert state.calls == 1
    assert state.memo is None


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ("length", "tool_calls"))
async def test_truncated_or_tool_call_summary_is_not_dispatched(finish: Literal["length", "tool_calls"]) -> None:
    executor: Final = _Executor(finish_reason=finish)
    state: Final = _state()
    result: Final = await prepare_compaction(_payload(), _TARGET, state, (_SUMMARY,), executor, _Counter())
    assert isinstance(result, CompactionFailure)
    assert "incomplete summary" in result.message
    assert state.calls == 1
    assert state.memo is None


@pytest.mark.asyncio
async def test_summary_failure_is_sticky_without_charging_again() -> None:
    executor: Final = _Executor(fail=True)
    state: Final = _state()
    first: Final = await prepare_compaction(_payload(), _TARGET, state, (_SUMMARY,), executor, _Counter())
    second: Final = await prepare_compaction(_payload(), _TARGET, state, (_SUMMARY,), executor, _Counter())
    assert isinstance(first, CompactionFailure)
    assert second == first
    assert executor.calls.qsize() == state.calls == 1


@pytest.mark.asyncio
async def test_matching_retry_reuses_summary_without_a_second_charge() -> None:
    payload: Final = _payload()
    executor: Final = _Executor()
    counter: Final = _Counter()
    state: Final = _state()
    first: Final = await prepare_compaction(payload, _TARGET, state, (_SUMMARY,), executor, counter)
    second: Final = await prepare_compaction(payload, _TARGET, state, (_SUMMARY,), executor, counter)
    assert isinstance(first, CompactedRequest)
    assert second == first
    assert executor.calls.qsize() == state.calls == 1
    counts: Final = tuple(counter.calls.get_nowait() for _ in range(counter.calls.qsize()))
    assert sum(count.model == _TARGET.model and count.payload == payload for count in counts) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_first", (False, True))
async def test_larger_fallback_uses_original_even_after_compaction_failure(failed_first: bool) -> None:
    payload: Final = _payload()
    executor: Final = _Executor(fail=failed_first)
    state: Final = _state()
    first: Final = await prepare_compaction(payload, _TARGET, state, (_SUMMARY,), executor, _Counter())
    assert isinstance(first, CompactionFailure if failed_first else CompactedRequest)
    counter: Final = _Counter()
    larger: Final = ModelBudget("larger-fallback", input_limit=_cost(payload) + 1_000, output_limit=80)
    second: Final = await prepare_compaction(payload, larger, state, (_SUMMARY,), executor, counter)
    assert second is None
    assert executor.calls.qsize() == state.calls == 1
    assert counter.calls.get_nowait().payload == payload
    assert counter.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ("previous_response_id", "conversation", "item_reference", "encrypted_content"))
async def test_opaque_history_fails_without_summary_calls(reference: str) -> None:
    payload: Final[Mapping[str, object]] = (
        {**_payload("responses"), reference: "server-side-history"}
        if reference in ("previous_response_id", "conversation")
        else {
            "input": [
                {"role": "user", "content": "old " * 2_000},
                {"type": "item_reference", "id": "opaque"}
                if reference == "item_reference"
                else {
                    "type": "reasoning",
                    "encrypted_content": "opaque",
                    "summary": [{"type": "summary_text", "text": "visible summary"}],
                },
                {"role": "user", "content": "new"},
            ],
            "max_output_tokens": 32,
        }
    )
    executor: Final = _Executor()
    result: Final = await prepare_compaction(payload, _TARGET, _state(), (_SUMMARY,), executor, _Counter())
    assert isinstance(result, CompactionFailure)
    assert executor.calls.empty()


@pytest.mark.asyncio
async def test_summary_that_cannot_shrink_to_fit_fails_explicitly() -> None:
    executor: Final = _Executor(summary="still too long " * 300)
    state: Final = _state()
    result: Final = await prepare_compaction(_payload(), _TARGET, state, (_SUMMARY,), executor, _Counter())
    assert isinstance(result, CompactionFailure)
    assert "could not reduce" in result.message
    assert state.calls == executor.calls.qsize() == 2
    assert state.memo is None


@pytest.mark.asyncio
async def test_original_fits_without_a_valid_summary_configuration() -> None:
    payload: Final = _payload()
    target: Final = ModelBudget("selected", input_limit=_cost(payload) + 1_000, output_limit=80)
    executor: Final = _Executor()
    result: Final = await prepare_compaction(payload, target, CompactionState(), (), executor, _Counter())
    assert result is None
    assert executor.calls.empty()


@pytest.mark.asyncio
async def test_parent_timeout_shortens_the_deadline_and_cannot_be_extended() -> None:
    executor: Final = _Executor(delay=5)
    state: Final = _state(timeout=10)
    initial: Final = state.deadline
    state.limit_timeout(0.15)
    limited: Final = state.deadline
    assert limited < initial
    state.limit_timeout(10)
    assert state.deadline == limited
    result: Final = await asyncio.wait_for(
        prepare_compaction(_payload(), _TARGET, state, (_SUMMARY,), executor, _Counter()), timeout=1
    )
    assert isinstance(result, CompactionFailure)
    assert "time" in result.message
    assert state.calls == executor.calls.qsize() == 1


@pytest.mark.asyncio
async def test_oversized_summary_is_reduced_then_recounted_before_dispatch() -> None:
    executor: Final = _Executor(replies=("still too long " * 300, "73"))
    state: Final = _state()
    counter: Final = _Counter()
    payload: Final = _payload()
    history: Final = split_compaction_history(payload)
    assert isinstance(history, CompactionHistory)
    result: Final = await prepare_compaction(payload, _TARGET, state, (_SUMMARY,), executor, counter)
    assert isinstance(result, CompactedRequest)
    assert result.value == history.rewrite("73")
    assert state.calls == executor.calls.qsize() == 2
    assert state.memo is not None and state.memo.summary == "73"
    counts: Final = tuple(counter.calls.get_nowait() for _ in range(counter.calls.qsize()))
    assert counts[-1].model == _TARGET.model
    assert counts[-1].tokens <= _TARGET.input_budget(32)


@pytest.mark.asyncio
async def test_changed_history_cannot_reuse_a_paid_summary() -> None:
    first_payload: Final = {
        "messages": [{"role": "user", "content": "first history " * 400}, {"role": "user", "content": "current"}],
        "max_tokens": 32,
    }
    second_payload: Final = {
        "messages": [{"role": "user", "content": "changed history " * 400}, {"role": "user", "content": "current"}],
        "max_tokens": 32,
    }
    executor: Final = _Executor()
    state: Final = _state()
    first: Final = await prepare_compaction(first_payload, _TARGET, state, (_SUMMARY,), executor, _Counter())
    second: Final = await prepare_compaction(second_payload, _TARGET, state, (_SUMMARY,), executor, _Counter())
    assert isinstance(first, CompactedRequest)
    assert isinstance(second, CompactedRequest)
    assert state.calls == executor.calls.qsize() == 2
    calls: Final = tuple(executor.calls.get_nowait() for _ in range(executor.calls.qsize()))
    assert calls[0].messages[1]["content"] != calls[1].messages[1]["content"]
    assert "changed history" in calls[1].messages[1]["content"]


@pytest.mark.asyncio
async def test_expired_state_still_allows_a_fitting_original_request() -> None:
    payload: Final = _payload()
    target: Final = ModelBudget("larger-fallback", input_limit=_cost(payload) + 1_000, output_limit=80)
    state: Final = _state(timeout=0)
    executor: Final = _Executor(fail=True)
    assert state.remaining() == 0
    result: Final = await prepare_compaction(payload, target, state, (_SUMMARY,), executor, _Counter(cooperative=True))
    assert result is None
    assert executor.calls.empty()
    assert state.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_failure", (False, True))
async def test_expired_state_reuses_a_fitting_paid_summary_without_another_call(cached_failure: bool) -> None:
    payload: Final = _payload()
    history: Final = split_compaction_history(payload)
    assert isinstance(history, CompactionHistory)
    state: Final = _state(timeout=0)
    state.remember(SummaryMemo(model="summary-group", history=history.history_text, summary="73"))
    if cached_failure:
        state.failed(CompactionFailure("Prior reduction exhausted its time budget"))
    executor: Final = _Executor(fail=True)
    assert state.remaining() == 0
    result: Final = await prepare_compaction(payload, _TARGET, state, (_SUMMARY,), executor, _Counter(cooperative=True))
    assert isinstance(result, CompactedRequest)
    assert result.value == history.rewrite("73")
    assert executor.calls.empty()
    assert state.calls == 0


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_and_stops_summary_work() -> None:
    executor: Final = _Executor(delay=5)
    state: Final = _state()
    task: Final = asyncio.create_task(prepare_compaction(_payload(), _TARGET, state, (_SUMMARY,), executor, _Counter()))
    await asyncio.wait_for(executor.started.wait(), timeout=1)
    cancelled: Final = task.cancel()
    assert cancelled
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert state.calls == executor.calls.qsize() == 1
    assert state.memo is None
    assert state.failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "opaque"),
    (
        ("input", {"type": "item_reference", "id": "unavailable"}),
        ("input", {"type": "reasoning", "summary": [], "encrypted_content": "ciphertext"}),
        ("input", {"role": "assistant", "content": "answer", "reasoning_content": "uncounted reasoning"}),
        ("input", {"role": "assistant", "content": [{"type": "refusal", "refusal": "uncounted text"}]}),
        (
            "input",
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "bytes", "format": "wav"}}]},
        ),
        (
            "messages",
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "text", "signature": "signed"}]},
        ),
        (
            "messages",
            {
                "role": "assistant",
                "content": "answer",
                "thinking_blocks": [{"type": "thinking", "thinking": "text", "signature": "signed"}],
            },
        ),
        ("messages", {"role": "assistant", "content": "answer", "reasoning_content": {"signature": "signed"}}),
        (
            "messages",
            {"role": "assistant", "content": "answer", "provider_specific_fields": {"compaction_blocks": ["opaque"]}},
        ),
        (
            "messages",
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "pending",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{}",
                            "provider_specific_fields": {"thought_signature": "signed"},
                        },
                    }
                ],
            },
        ),
    ),
)
@pytest.mark.parametrize("with_prefix", (False, True))
async def test_opaque_replay_state_fails_before_original_count(
    field: str, opaque: Mapping[str, object], with_prefix: bool
) -> None:
    payload: Final = {
        field: [
            *(
                [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}]
                if with_prefix
                else []
            ),
            {"role": "user", "content": "current"},
            opaque,
        ]
    }
    original: Final = deepcopy(payload)
    target: Final = ModelBudget("fitting-original", input_limit=_cost(payload) + 1_000, output_limit=80)
    executor: Final = _Executor()
    counter: Final = _Counter()
    state: Final = _state()
    result: Final = await prepare_compaction(payload, target, state, (_SUMMARY,), executor, counter)
    assert isinstance(result, CompactionFailure)
    assert counter.calls.empty()
    assert executor.calls.empty()
    assert state.calls == 0
    assert payload == original


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("previous_response_id", "conversation"))
async def test_server_side_history_fails_before_counting_short_input(key: str) -> None:
    payload: Final = {"input": "short current question", key: "unavailable"}
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await prepare_compaction(payload, _TARGET, _state(), (_SUMMARY,), executor, counter)
    assert isinstance(result, CompactionFailure)
    assert "unavailable" in result.message
    assert counter.calls.empty()
    assert executor.calls.empty()


@pytest.mark.asyncio
async def test_plain_responses_input_fits_without_a_compactable_prefix() -> None:
    payload: Final = {"input": "short current question", "instructions": "rules", "max_output_tokens": 32}
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await prepare_compaction(payload, _TARGET, _state(), (), executor, counter)
    assert result is None
    assert counter.calls.get_nowait().payload == payload
    assert counter.calls.empty()
    assert executor.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("messages", "input", "system", "tools", "max_tokens"))
@pytest.mark.parametrize("identical", (False, True))
async def test_extra_body_protected_collisions_fail_even_if_identical(key: str, identical: bool) -> None:
    original: Final = _payload("responses" if key == "input" else "chat")
    payload: Final = {**original, "extra_body": {key: original[key] if identical else "replacement"}}
    target: Final = ModelBudget("fitting-original", input_limit=_cost(payload) + 1_000, output_limit=80)
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await prepare_compaction(payload, target, _state(), (_SUMMARY,), executor, counter)
    assert isinstance(result, CompactionFailure)
    assert "extra_body" in result.message
    assert counter.calls.empty()
    assert executor.calls.empty()
    assert payload == {**original, "extra_body": {key: original[key] if identical else "replacement"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", (False, True))
async def test_extra_body_benign_provider_flag_remains_unchanged(overflow: bool) -> None:
    payload: Final = {**_payload(), "extra_body": {"provider_feature_enabled": True}}
    target: Final = (
        _TARGET if overflow else ModelBudget("fitting-original", input_limit=_cost(payload) + 1_000, output_limit=80)
    )
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await prepare_compaction(payload, target, _state(), (_SUMMARY,), executor, counter)
    assert isinstance(result, CompactedRequest) if overflow else result is None
    counts: Final = tuple(counter.calls.get_nowait() for _ in range(counter.calls.qsize()))
    assert counts[-1].payload["extra_body"] == payload["extra_body"]
    assert executor.calls.qsize() == int(overflow)


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("system", "tools"))
async def test_summary_defaults_can_exhaust_capacity_before_any_paid_call(key: str) -> None:
    value: Final = (
        "summary instructions " * 2_000
        if key == "system"
        else [{"name": "lookup", "description": "summary tool " * 2_000}]
    )
    defaults: Final = MappingProxyType({key: value})
    budget: Final = ModelBudget(_SUMMARY.model, _SUMMARY.input_limit, _SUMMARY.output_limit, defaults)
    control_executor: Final = _Executor()
    control: Final = await prepare_compaction(_payload(), _TARGET, _state(), (_SUMMARY,), control_executor, _Counter())
    assert isinstance(control, CompactedRequest)
    executor: Final = _Executor()
    counter: Final = _Counter()
    state: Final = _state()
    result: Final = await prepare_compaction(_payload(), _TARGET, state, (budget,), executor, counter)
    assert isinstance(result, CompactionFailure)
    assert "summary instructions" in result.message
    assert executor.calls.empty()
    assert state.calls == 0
    counts: Final = tuple(counter.calls.get_nowait() for _ in range(counter.calls.qsize()))
    summary_counts: Final = tuple(count for count in counts if count.model == budget.model)
    assert summary_counts
    assert all(
        count.payload[key] == value and count.payload["max_tokens"] == budget.output_limit for count in summary_counts
    )


@pytest.mark.asyncio
async def test_summary_explicit_messages_and_output_cap_override_defaults() -> None:
    defaults: Final = MappingProxyType(
        {
            "messages": [{"role": "user", "content": "ignored default history " * 2_000}],
            "max_tokens": _SUMMARY.output_limit * 10,
            "system": "summary system instructions",
            "extra_body": {"provider_feature_enabled": True},
        }
    )
    original: Final = _serialize(defaults)
    budget: Final = ModelBudget(_SUMMARY.model, _SUMMARY.input_limit, _SUMMARY.output_limit, defaults)
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await prepare_compaction(_payload(), _TARGET, _state(), (budget,), executor, counter)
    assert isinstance(result, CompactedRequest)
    assert executor.calls.qsize() == 1
    call: Final = executor.calls.get_nowait()
    counts: Final = tuple(counter.calls.get_nowait() for _ in range(counter.calls.qsize()))
    summary_counts: Final = tuple(count for count in counts if count.model == budget.model)
    assert summary_counts
    assert all(count.payload["messages"] == [dict(message) for message in call.messages] for count in summary_counts)
    assert all(count.payload["max_tokens"] == call.max_tokens == budget.output_limit for count in summary_counts)
    assert all(
        count.payload["system"] == defaults["system"] and count.payload["extra_body"] == defaults["extra_body"]
        for count in summary_counts
    )
    assert _serialize(defaults) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("messages", "input", "system", "tools", "max_tokens"))
async def test_summary_protected_extra_body_is_rejected_before_summary_count(key: str) -> None:
    defaults: Final = MappingProxyType({"extra_body": {key: "replacement"}})
    budget: Final = ModelBudget(_SUMMARY.model, _SUMMARY.input_limit, _SUMMARY.output_limit, defaults)
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await prepare_compaction(_payload(), _TARGET, _state(), (budget,), executor, counter)
    assert isinstance(result, CompactionFailure)
    assert "extra_body" in result.message
    assert executor.calls.empty()
    counts: Final = tuple(counter.calls.get_nowait() for _ in range(counter.calls.qsize()))
    assert all(count.model == _TARGET.model for count in counts)


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("max_output_tokens", "max_completion_tokens"))
@pytest.mark.parametrize("matching", (False, True))
async def test_summary_alternate_output_limit_must_match_explicit_cap(key: str, matching: bool) -> None:
    defaults: Final = MappingProxyType({key: _SUMMARY.output_limit if matching else _SUMMARY.output_limit // 2})
    budget: Final = ModelBudget(_SUMMARY.model, _SUMMARY.input_limit, _SUMMARY.output_limit, defaults)
    executor: Final = _Executor()
    result: Final = await prepare_compaction(_payload(), _TARGET, _state(), (budget,), executor, _Counter())
    assert isinstance(result, CompactedRequest) if matching else isinstance(result, CompactionFailure)
    assert executor.calls.qsize() == int(matching)


@pytest.mark.asyncio
async def test_summary_higher_priority_matching_cap_cannot_hide_conflicting_alias() -> None:
    defaults: Final = MappingProxyType(
        {"max_output_tokens": _SUMMARY.output_limit, "max_completion_tokens": _SUMMARY.output_limit // 2}
    )
    budget: Final = ModelBudget(_SUMMARY.model, _SUMMARY.input_limit, _SUMMARY.output_limit, defaults)
    executor: Final = _Executor()
    result: Final = await prepare_compaction(_payload(), _TARGET, _state(), (budget,), executor, _Counter())
    assert isinstance(result, CompactionFailure)
    assert executor.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_body", ("invalid body", ["invalid body"], {1: "invalid key"}))
async def test_malformed_extra_body_fails_before_counting(extra_body: object) -> None:
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await prepare_compaction(
        {**_payload(), "extra_body": extra_body}, _TARGET, _state(), (_SUMMARY,), executor, counter
    )
    assert isinstance(result, CompactionFailure)
    assert result.message == "Context compaction requires extra_body to be a JSON object"
    assert executor.calls.empty()
    assert counter.calls.empty()
