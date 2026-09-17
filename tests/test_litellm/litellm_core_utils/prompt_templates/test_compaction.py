import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Final, Literal

import pytest

from litellm.litellm_core_utils.prompt_templates.compaction import (
    CompactionHistory,
    CompactionHistoryError,
    split_compaction_history,
    validate_compaction_input,
)
from litellm.litellm_core_utils.token_counter import token_counter
from litellm.types.llms.openai import ChatCompletionAssistantMessage, ChatCompletionUserMessage

_Format = Literal["chat", "anthropic", "responses"]
_FORMATS: Final = ("chat", "anthropic", "responses")


def _message(wire: _Format, role: str, text: str) -> Mapping[str, object]:
    if wire == "chat":
        return {"role": role, "content": text}
    return {
        **({"type": "message"} if wire == "responses" else {}),
        "role": role,
        "content": [
            {
                "type": ("output_text" if role == "assistant" else "input_text") if wire == "responses" else "text",
                "text": text,
            }
        ],
    }


def _exchange(wire: _Format, identifier: str) -> tuple[Mapping[str, object], ...]:
    if wire == "chat":
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": identifier,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"fact":"old"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": identifier, "content": "historical fact = 73"},
        )
    if wire == "anthropic":
        return (
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": identifier, "name": "lookup", "input": {"fact": "old"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": identifier,
                        "content": [{"type": "text", "text": "historical fact = 73"}],
                    }
                ],
            },
        )
    return (
        {
            "type": "function_call",
            "id": f"item_{identifier}",
            "call_id": identifier,
            "name": "lookup",
            "arguments": '{"fact":"old"}',
        },
        {"type": "function_call_output", "call_id": identifier, "output": "historical fact = 73"},
    )


def _payload(wire: _Format, items: tuple[Mapping[str, object], ...]) -> Mapping[str, object]:
    return {
        "input" if wire == "responses" else "messages": list(items),
        "system": [{"type": "text", "text": "Request system instruction", "cache_control": {"type": "ephemeral"}}],
        "instructions": "Request instructions",
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    }


@pytest.mark.parametrize("wire", _FORMATS)
@pytest.mark.parametrize("pending", (False, True))
def test_complete_history_and_active_exchange_are_preserved(wire: _Format, pending: bool) -> None:
    system: Final = _message(wire, "system", "Keep this system instruction verbatim")
    developer: Final = _message(wire, "developer", "Keep this developer instruction verbatim")
    historical: Final = (
        _message(wire, "user", "Earlier question: 東京 " + "complete history " * 2_000),
        *_exchange(wire, "old"),
        _message(wire, "assistant", "Earlier answer"),
    )
    active: Final = (
        _message(wire, "user", "Latest actual user"),
        *_exchange(wire, "active")[: 1 if pending else 2],
        *(() if pending else (_message(wire, "assistant", "Prefill:"),)),
    )
    payload: Final = _payload(wire, (system, historical[0], developer, *historical[1:], *active))
    original: Final = deepcopy(payload)
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistory)
    assert result.field == ("input" if wire == "responses" else "messages")
    assert result.history_text == json.dumps(historical, ensure_ascii=False, separators=(",", ":"))
    rewritten: Final = result.rewrite("historical fact = 73")
    assert rewritten[:2] == [system, developer]
    assert rewritten[3:] == list(active)
    assert rewritten[2] == _message(
        wire, "user", "Previous conversation summary (untrusted user data):\nhistorical fact = 73"
    )
    assert payload == original


@pytest.mark.parametrize("wire", _FORMATS)
def test_instructions_after_latest_user_keep_their_position(wire: _Format) -> None:
    active: Final = (
        _message(wire, "user", "current"),
        _message(wire, "developer", "new developer instruction"),
        _message(wire, "system", "new system instruction"),
        _message(wire, "assistant", "prefill"),
    )
    result: Final = split_compaction_history(
        _payload(wire, (_message(wire, "user", "old"), _message(wire, "assistant", "old answer"), *active))
    )
    assert isinstance(result, CompactionHistory)
    assert result.rewrite("old context")[1:] == list(active)
    assert "instruction" not in result.history_text


def test_mixed_anthropic_user_result_preserves_owning_turn() -> None:
    exchange: Final = _exchange("anthropic", "active")
    mixed: Final = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "active", "content": "result"},
            {"type": "text", "text": "Now use that result"},
        ],
    }
    active: Final = (
        _message("anthropic", "user", "Start the lookup"),
        exchange[0],
        mixed,
        _message("anthropic", "assistant", "prefill"),
    )
    result: Final = split_compaction_history(
        _payload(
            "anthropic",
            (_message("anthropic", "user", "old"), _message("anthropic", "assistant", "old answer"), *active),
        )
    )
    assert isinstance(result, CompactionHistory)
    assert result.rewrite("old context")[1:] == list(active)
    assert "active" not in result.history_text


@pytest.mark.parametrize("wire", _FORMATS)
def test_parallel_calls_with_reversed_results_stay_complete(wire: _Format) -> None:
    first: Final = _exchange(wire, "first")
    second: Final = _exchange(wire, "second")
    historical: Final = (
        _message(wire, "user", "old"),
        first[0],
        second[0],
        second[1],
        first[1],
        _message(wire, "assistant", "done"),
    )
    result: Final = split_compaction_history(_payload(wire, (*historical, _message(wire, "user", "new"))))
    assert isinstance(result, CompactionHistory)
    assert result.history_text == json.dumps(historical, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize("wire", _FORMATS)
@pytest.mark.parametrize("malformation", ("orphan", "reversed", "duplicate_call", "duplicate_result", "mismatched"))
def test_invalid_tool_ids_fail_closed(wire: _Format, malformation: str) -> None:
    exchange: Final = _exchange(wire, "known")
    invalid: Final = {
        "orphan": exchange[1:],
        "reversed": tuple(reversed(exchange)),
        "duplicate_call": (exchange[0], *exchange),
        "duplicate_result": (*exchange, exchange[1]),
        "mismatched": (exchange[0], _exchange(wire, "unknown")[1]),
    }[malformation]
    payload: Final = _payload(wire, (_message(wire, "user", "old"), *invalid, _message(wire, "user", "new")))
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistoryError)
    assert result.message
    assert validate_compaction_input(payload) == result


@pytest.mark.parametrize(
    ("wire", "media"),
    (
        ("chat", "image"),
        ("chat", "image_url"),
        ("chat", "file"),
        ("chat", "document"),
        ("anthropic", "image"),
        ("anthropic", "document"),
        ("responses", "input_image"),
        ("responses", "input_file"),
    ),
)
def test_media_is_preserved_only_in_the_active_tail(wire: _Format, media: str) -> None:
    media_item: Final = {"role": "user", "content": [{"type": media, "source": {"data": "opaque media bytes"}}]}
    old: Final = (_message(wire, "user", "old text"), _message(wire, "assistant", "old answer"))
    active_result: Final = split_compaction_history(_payload(wire, (*old, media_item)))
    assert isinstance(active_result, CompactionHistory)
    assert active_result.rewrite("summary")[1:] == [media_item]
    historical_result: Final = split_compaction_history(
        _payload(wire, (media_item, *old, _message(wire, "user", "new")))
    )
    assert isinstance(historical_result, CompactionHistoryError)
    assert "media" in historical_result.message


@pytest.mark.parametrize(
    "opaque",
    (
        {"type": "item_reference", "id": "ref_1"},
        {"type": "reasoning", "encrypted_content": "ciphertext", "summary": []},
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "ciphertext"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "text", "signature": "signed_state"}]},
        {"role": "assistant", "content": [{"type": "compaction", "content": "opaque"}]},
    ),
)
@pytest.mark.parametrize("in_tail", (False, True))
def test_opaque_state_fails_even_in_preserved_tail(opaque: Mapping[str, object], in_tail: bool) -> None:
    old: Final = (_message("responses", "user", "old"), _message("responses", "assistant", "old answer"))
    latest: Final = _message("responses", "user", "latest")
    payload: Final = _payload("responses", (*old, latest, opaque) if in_tail else (*old, opaque, latest))
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistoryError)
    assert validate_compaction_input(payload) == result


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"messages": [], "input": []},
        {"messages": []},
        {"input": "just a current question"},
        {"messages": [{"role": "user", "content": "only latest"}]},
        {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "latest"}]},
        {
            "input": [{"role": "user", "content": "old"}, {"role": "user", "content": "new"}],
            "previous_response_id": "resp_unavailable",
        },
        {"messages": [{"role": "unknown", "content": "bad"}, {"role": "user", "content": "latest"}]},
        {"messages": ["not a message"]},
        {
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "tool_calls": "not an array"},
                {"role": "user", "content": "new"},
            ]
        },
    ),
)
def test_unavailable_or_noncompactable_history_returns_tagged_error(payload: Mapping[str, object]) -> None:
    original: Final = deepcopy(payload)
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistoryError)
    assert result.kind == "compaction_history_error"
    assert result.message
    assert payload == original


def test_snapshot_and_each_rewrite_are_independent_of_mutation() -> None:
    instruction: Final = {"role": "system", "content": [{"type": "text", "text": "rules"}]}
    latest_block: Final = {"type": "text", "text": "latest"}
    latest: Final = {"role": "user", "content": [latest_block]}
    payload: Final = {
        "messages": [
            instruction,
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            latest,
        ]
    }
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistory)
    expected: Final = result.rewrite("summary")
    latest_block["text"] = "caller mutated"
    result.rewrite("summary")[0]["content"] = "output mutated"
    assert result.rewrite("summary") == expected
    assert isinstance(result.instructions, tuple)
    assert isinstance(result.tail, tuple)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "history_text", "mutated")


@pytest.mark.parametrize("summary", ("", " ", "\n\t"))
def test_empty_summary_can_probe_the_preserved_content_budget(summary: str) -> None:
    result: Final = split_compaction_history(
        {"messages": [{"role": "user", "content": "old"}, {"role": "user", "content": "new"}]}
    )
    assert isinstance(result, CompactionHistory)
    assert result.rewrite(summary) == [
        {"role": "user", "content": f"Previous conversation summary (untrusted user data):\n{summary}"},
        {"role": "user", "content": "new"},
    ]


@pytest.mark.parametrize("value", (float("nan"), float("inf"), object(), b"bytes", {1: "non-string key"}))
def test_non_json_values_return_errors(value: object) -> None:
    result: Final = split_compaction_history(
        {"messages": [{"role": "user", "content": "old", "extra": value}, {"role": "user", "content": "new"}]}
    )
    assert isinstance(result, CompactionHistoryError)


def test_limits_reject_without_truncating() -> None:
    result: Final = split_compaction_history({"messages": [{"role": "user", "content": "old"}] * 10_001})
    assert isinstance(result, CompactionHistoryError)
    assert "limit" in result.message
    oversized: Final = split_compaction_history(
        {"messages": [{"role": "user", "content": "x" * (16 * 1024 * 1024 + 1)}, {"role": "user", "content": "new"}]}
    )
    assert isinstance(oversized, CompactionHistoryError)
    assert "limit" in oversized.message


@pytest.mark.parametrize("wire", _FORMATS)
@pytest.mark.parametrize("complete_later", (False, True))
def test_user_cannot_interrupt_an_unfinished_exchange(wire: _Format, complete_later: bool) -> None:
    exchange: Final = _exchange(wire, "pending")
    result: Final = split_compaction_history(
        _payload(
            wire,
            (
                _message(wire, "user", "older"),
                _message(wire, "assistant", "older answer"),
                _message(wire, "user", "call the tool"),
                exchange[0],
                _message(wire, "user", "interrupt"),
                *(exchange[1:] if complete_later else ()),
            ),
        )
    )
    assert isinstance(result, CompactionHistoryError)
    assert "interrupts" in result.message


@pytest.mark.parametrize("wire", ("anthropic", "responses"))
def test_media_inside_old_tool_result_is_rejected(wire: _Format) -> None:
    exchange: Final = _exchange(wire, "media")
    media: Final = [{"type": "image", "source": {"data": "bytes"}}]
    output: Final = (
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "media", "content": media}]}
        if wire == "anthropic"
        else {"type": "function_call_output", "call_id": "media", "output": media}
    )
    result: Final = split_compaction_history(
        _payload(wire, (_message(wire, "user", "old"), exchange[0], output, _message(wire, "user", "new")))
    )
    assert isinstance(result, CompactionHistoryError)
    assert "media" in result.message


def test_nonfinite_numbers_in_preserved_tail_are_rejected() -> None:
    result: Final = split_compaction_history(
        {
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "user", "content": "new", "metadata": {"value": float("nan")}},
            ]
        }
    )
    assert isinstance(result, CompactionHistoryError)


def _nested(depth: int) -> object:
    return "text" if depth == 0 else {"nested": _nested(depth - 1)}


def test_nested_and_node_limits_are_enforced() -> None:
    deep: Final = split_compaction_history(
        {"messages": [{"role": "user", "content": "old", "metadata": _nested(65)}, {"role": "user", "content": "new"}]}
    )
    assert isinstance(deep, CompactionHistoryError)
    assert "nesting limit" in deep.message
    many_nodes: Final = split_compaction_history(
        {
            "messages": [
                {"role": "user", "content": "old", "metadata": [0] * 100_001},
                {"role": "user", "content": "new"},
            ]
        }
    )
    assert isinstance(many_nodes, CompactionHistoryError)
    assert "size limit" in many_nodes.message


@pytest.mark.parametrize("wire", _FORMATS)
def test_plain_string_messages_keep_their_wire_style(wire: _Format) -> None:
    result: Final = split_compaction_history(
        _payload(
            wire,
            (
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "new"},
            ),
        )
    )
    assert isinstance(result, CompactionHistory)
    assert result.rewrite("summary")[0] == {
        "role": "user",
        "content": "Previous conversation summary (untrusted user data):\nsummary",
    }


@pytest.mark.parametrize("wire", _FORMATS)
def test_preflight_accepts_single_turn_with_instructions_and_prefill(wire: _Format) -> None:
    payload: Final = _payload(
        wire,
        (_message(wire, "system", "rules"), _message(wire, "user", "current"), _message(wire, "assistant", "prefill")),
    )
    original: Final = deepcopy(payload)
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistoryError)
    assert "no compactable prefix" in result.message
    assert payload == original


def test_preflight_accepts_plain_responses_input_without_inspecting_request_context() -> None:
    context: Final = object()
    payload: Final = {"input": "plain current question", "instructions": "rules", "_private_context": context}
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistoryError)
    assert "no compactable prefix" in result.message
    assert payload["_private_context"] is context


@pytest.mark.parametrize("key", ("previous_response_id", "conversation"))
def test_preflight_rejects_unavailable_server_history_for_short_input(key: str) -> None:
    payload: Final = {"input": "current", key: "unavailable_history"}
    result: Final = validate_compaction_input(payload)
    assert isinstance(result, CompactionHistoryError)
    assert "unavailable" in result.message
    assert split_compaction_history(payload) == result


@pytest.mark.parametrize(
    "extension",
    (
        {"thinking_blocks": [{"type": "thinking", "thinking": "private", "signature": "signed"}]},
        {"thinking_blocks": [{"type": "redacted_thinking", "data": "ciphertext"}]},
        {"reasoning_items": [{"type": "reasoning", "encrypted_content": "ciphertext"}]},
        {"reasoning_details": [{"type": "reasoning.encrypted", "data": "ciphertext"}]},
        {"reasoning_content": [{"type": "thinking", "thinking": "structured state"}]},
        {"provider_specific_fields": {"compaction_blocks": [{"type": "compaction", "content": "state"}]}},
        {"audio": {"id": "audio_unavailable"}},
        {"future_provider_state": {"data": "unclassified"}},
    ),
)
@pytest.mark.parametrize("in_tail", (False, True))
def test_shared_inventory_rejects_opaque_message_extensions(extension: Mapping[str, object], in_tail: bool) -> None:
    assistant: Final = {"role": "assistant", "content": "answer", **extension}
    latest: Final = {"role": "user", "content": "latest"}
    payload: Final = {
        "messages": [{"role": "user", "content": "old"}, *((latest, assistant) if in_tail else (assistant, latest))]
    }
    original: Final = deepcopy(payload)
    result: Final = validate_compaction_input(payload)
    assert isinstance(result, CompactionHistoryError)
    assert split_compaction_history(payload) == result
    assert payload == original


@pytest.mark.parametrize("provider_state", ({"thought_signature": "signed"}, {"other_replay_state": "opaque"}))
def test_nested_tool_function_provider_state_is_rejected(provider_state: Mapping[str, object]) -> None:
    call: Final = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "active",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{}",
                    "provider_specific_fields": provider_state,
                },
            }
        ],
    }
    payload: Final = {"messages": [{"role": "user", "content": "old"}, {"role": "user", "content": "current"}, call]}
    result: Final = validate_compaction_input(payload)
    assert isinstance(result, CompactionHistoryError)
    assert "provider_specific_fields" in result.message
    assert split_compaction_history(payload) == result


def test_countable_reasoning_text_and_inert_empty_extensions_are_preserved() -> None:
    assistant: Final[Mapping[str, object]] = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "readable reasoning",
        "thinking_blocks": [],
        "reasoning_items": None,
        "provider_specific_fields": {},
        "function_call": None,
    }
    payload: Final = {
        "messages": [{"role": "user", "content": "old"}, assistant, {"role": "user", "content": "current"}, assistant]
    }
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistory)
    assert "readable reasoning" in result.history_text
    assert result.rewrite("summary")[-1] == assistant


def test_tool_input_data_is_not_mistaken_for_provider_replay_state() -> None:
    data: Final = {"encrypted_content": "user data", "signature": "ordinary field", "thinking_blocks": ["literal"]}
    call: Final = {"role": "assistant", "content": [{"type": "tool_use", "id": "old", "name": "lookup", "input": data}]}
    payload: Final = {
        "messages": [
            {"role": "user", "content": "old", "metadata": data},
            call,
            _exchange("anthropic", "old")[1],
            {"role": "user", "content": "current"},
        ]
    }
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistory)
    assert json.dumps(data, separators=(",", ":")) in result.history_text


@pytest.mark.parametrize(
    ("field", "block"),
    (
        ("messages", {"type": "image_url", "image_url": {"url": "https://example.test/image.png", "detail": "low"}}),
        (
            "messages",
            {
                "type": "document",
                "title": "notes",
                "context": "background",
                "source": {"type": "text", "media_type": "text/plain", "data": "readable document"},
            },
        ),
        ("input", {"type": "input_image", "image_url": "https://example.test/image.png"}),
    ),
)
def test_countable_media_is_valid_before_fit_but_not_summarizable(field: str, block: Mapping[str, object]) -> None:
    media: Final = {"role": "user", "content": [block]}
    payload: Final = {
        field: [media, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "current"}]
    }
    assert validate_compaction_input({field: [media]}) is None
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistoryError)
    assert "media" in result.message


@pytest.mark.parametrize(
    "block",
    (
        {"type": "input_file", "file_id": "file_unavailable"},
        {"type": "file", "file": {"file_id": "file_unavailable"}},
        {"type": "image", "source": {"type": "file", "file_id": "file_unavailable"}},
    ),
)
def test_media_references_are_not_treated_as_countable_local_content(block: Mapping[str, object]) -> None:
    field: Final = "input" if block["type"] == "input_file" else "messages"
    result: Final = validate_compaction_input({field: [{"role": "user", "content": [block]}]})
    assert isinstance(result, CompactionHistoryError)
    assert "file_id" in result.message


@pytest.mark.parametrize(
    "block",
    (
        {"type": "refusal", "refusal": "refusal text"},
        {"type": "thinking", "thinking": "reasoning text"},
        {"type": "input_audio", "input_audio": {"data": "bytes", "format": "wav"}},
        {"type": "document", "source": {"type": "text", "data": "uncounted document"}},
    ),
)
def test_responses_blocks_dropped_by_count_transform_are_rejected(block: Mapping[str, object]) -> None:
    payload: Final = {"input": [{"role": "user", "content": "current"}, {"role": "assistant", "content": [block]}]}
    result: Final = validate_compaction_input(payload)
    assert isinstance(result, CompactionHistoryError)
    assert "Unsupported" in result.message
    assert split_compaction_history(payload) == result


@pytest.mark.parametrize("source", ("content", "summary"))
def test_responses_reasoning_text_is_preserved_without_opaque_replay_state(source: str) -> None:
    reasoning: Final = {
        "type": "reasoning",
        source: [{"type": "summary_text" if source == "summary" else "reasoning_text", "text": "readable reasoning"}],
    }
    payload: Final = {
        "input": [{"role": "user", "content": "old"}, reasoning, {"role": "user", "content": "current"}, reasoning]
    }
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistory)
    assert "readable reasoning" in result.history_text
    assert result.rewrite("summary")[-1] == reasoning


def test_chat_reasoning_content_is_fully_counted_by_shared_token_counter() -> None:
    model: Final = "openai/compaction-counter-test"
    short: Final = "counted reasoning step"
    long: Final = " ".join((short,) * 128)

    def messages(text: str, plain: bool = False) -> tuple[ChatCompletionUserMessage, ChatCompletionAssistantMessage]:
        return (
            ChatCompletionUserMessage(role="user", content="current"),
            ChatCompletionAssistantMessage(
                role="assistant", content=text if plain else None, reasoning_content=None if plain else text
            ),
        )

    assert validate_compaction_input({"messages": list(messages(long))}) is None
    long_reasoning: Final = token_counter(model=model, messages=messages(long))
    short_reasoning: Final = token_counter(model=model, messages=messages(short))
    long_plain: Final = token_counter(model=model, messages=messages(long, plain=True))
    short_plain: Final = token_counter(model=model, messages=messages(short, plain=True))
    assert long_reasoning - short_reasoning == long_plain - short_plain > 0


@pytest.mark.parametrize("in_tail", (False, True))
def test_responses_message_reasoning_content_is_rejected_as_uncounted(in_tail: bool) -> None:
    assistant: Final = {"role": "assistant", "content": "answer", "reasoning_content": "uncounted reasoning"}
    latest: Final = {"role": "user", "content": "current"}
    payload: Final = {
        "input": [{"role": "user", "content": "old"}, *((latest, assistant) if in_tail else (assistant, latest))]
    }
    result: Final = validate_compaction_input(payload)
    assert isinstance(result, CompactionHistoryError)
    assert "reasoning_content" in result.message
    assert split_compaction_history(payload) == result
