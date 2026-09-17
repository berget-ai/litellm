from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import accumulate, islice
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

_FrozenJSON: TypeAlias = str | int | float | bool | None | tuple["_FrozenJSON", ...] | Mapping[str, "_FrozenJSON"]
_Item: TypeAlias = Mapping[str, _FrozenJSON]
_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_MAX_ITEMS: Final = 10_000
_MAX_NODES: Final = 100_000
_MAX_TEXT: Final = 16 * 1024 * 1024
_MAX_DEPTH: Final = 64
_FieldKind: TypeAlias = Literal["text", "data", "opaque"]
_ContentContext: TypeAlias = Literal["messages", "input", "function_output", "reasoning"]
_CONTENT_TYPES: Final[Mapping[_ContentContext, frozenset[str]]] = MappingProxyType(
    {
        "messages": frozenset(("text", "thinking", "image", "image_url", "file", "document")),
        "input": frozenset(
            ("text", "input_text", "output_text", "summary_text", "reasoning_text", "input_image", "input_file")
        ),
        "function_output": frozenset(("text", "input_text", "output_text", "input_image", "image_url")),
        "reasoning": frozenset(("text", "input_text", "output_text", "summary_text", "reasoning_text")),
    }
)
_OPAQUE_FIELDS: Final = (
    "encrypted_content",
    "signature",
    "thought_signature",
    "thinking_blocks",
    "reasoning_items",
    "reasoning_details",
    "provider_specific_fields",
    "audio",
    "function_call",
    "file_id",
    "file_url",
)


def _field_inventory(text: tuple[str, ...], data: tuple[str, ...]) -> Mapping[str, _FieldKind]:
    groups: Final[tuple[tuple[_FieldKind, tuple[str, ...]], ...]] = (
        ("text", text),
        ("data", data),
        ("opaque", _OPAQUE_FIELDS),
    )
    return MappingProxyType({name: kind for kind, names in groups for name in names})


_MESSAGE_FIELDS: Final[Mapping[Literal["messages", "input"], Mapping[str, _FieldKind]]] = MappingProxyType(
    {
        "messages": _field_inventory(
            ("type", "role", "id", "status", "name", "refusal", "reasoning_content", "tool_call_id"),
            ("content", "tool_calls", "cache_control", "annotations", "metadata"),
        ),
        "input": _field_inventory(
            ("type", "role", "id", "status"),
            ("content", "cache_control", "annotations", "metadata"),
        ),
    }
)
_FUNCTION_FIELDS: Final = _field_inventory(
    ("type", "id", "status", "call_id", "name", "arguments", "namespace"),
    ("output",),
)
_CHAT_CALL_FIELDS: Final = _field_inventory(("id", "type"), ("function", "index"))
_CHAT_FUNCTION_FIELDS: Final = _field_inventory(("name", "arguments"), ())
_REASONING_FIELDS: Final = _field_inventory(("type", "id", "status"), ("summary", "content"))
_TEXT_FIELDS: Final = _field_inventory(
    ("type", "text"),
    ("cache_control", "annotations", "citations"),
)
_THINKING_FIELDS: Final = _field_inventory(("type", "thinking"), ("cache_control",))
_TOOL_USE_FIELDS: Final = _field_inventory(("type", "id", "name"), ("input", "cache_control"))
_TOOL_RESULT_FIELDS: Final = _field_inventory(("type", "tool_use_id"), ("content", "is_error", "cache_control"))
_MEDIA_FIELDS: Final = _field_inventory(
    ("type", "detail", "file_data", "filename", "data", "mime_type", "title", "context"),
    ("source", "image_url", "file", "input_audio", "video_url", "cache_control", "citations"),
)
_MEDIA_SOURCE_FIELDS: Final = _field_inventory(
    ("type", "url", "data", "media_type", "mime_type", "detail", "format", "filename", "file_data"),
    ("content",),
)
_MEDIA_TYPES: Final = frozenset(("image", "image_url", "input_image", "file", "input_file", "document"))


@dataclass(frozen=True)
class CompactionHistoryError:
    message: str
    kind: Literal["compaction_history_error"] = "compaction_history_error"


@dataclass(frozen=True)
class CompactionHistory:
    field: Literal["messages", "input"]
    instructions: tuple[_Item, ...]
    tail: tuple[_Item, ...]
    history_text: str

    def rewrite(
        self, summary: str
    ) -> list[dict[str, object]]:  # mutable-ok: public contract returns fresh, caller-owned wire JSON
        """Return fresh wire items; an empty summary supports protected-budget probes."""
        if len(summary) > _MAX_TEXT:
            raise ValueError("Compaction summary exceeds the history text limit")
        latest_user: Final = next(item for item in reversed(self.tail) if _genuine_user(item))
        label: Final = f"Previous conversation summary (untrusted user data):\n{summary}"
        content: Final[_FrozenJSON] = (
            (MappingProxyType({"type": "input_text" if self.field == "input" else "text", "text": label}),)
            if isinstance(latest_user.get("content"), tuple)
            else label
        )
        message: Final[_Item] = MappingProxyType(
            {
                **(
                    MappingProxyType({"type": "message"})
                    if latest_user.get("type") == "message"
                    else MappingProxyType({})
                ),
                "role": "user",
                "content": content,
            }
        )
        return [  # mutable-ok: provider wire messages require a fresh JSON array
            _thaw_item(item) for item in (*self.instructions, message, *self.tail)
        ]


class _InvalidHistory(ValueError):
    pass


@dataclass(frozen=True)
class _ToolEvent:
    identifier: str
    family: Literal["chat", "anthropic", "responses"]
    result: bool
    index: int


@dataclass(frozen=True)
class _CheckedItem:
    item: _Item
    events: tuple[_ToolEvent, ...]
    text_only: bool


@dataclass(frozen=True)
class _CheckedHistory:
    field: Literal["messages", "input"]
    items: tuple[_CheckedItem, ...]
    starts: tuple[int, ...]


def _freeze(value: JsonValue) -> _FrozenJSON:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: _FrozenJSON) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}  # mutable-ok: JSON serialization requires objects
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]  # mutable-ok: wire JSON arrays must not alias immutable snapshots
    return value


def _thaw_item(item: _Item) -> dict[str, object]:  # mutable-ok: individual wire messages are caller-owned JSON objects
    return {key: _thaw(value) for key, value in item.items()}  # mutable-ok: materialize one independent wire object


def _measure(value: JsonValue, depth: int = 0) -> Iterator[int]:
    if depth > _MAX_DEPTH:
        raise _InvalidHistory("History exceeds the nesting limit")
    if isinstance(value, float) and not math.isfinite(value):
        raise _InvalidHistory("History requires finite JSON numbers")
    yield len(value) if isinstance(value, str) else 0
    if isinstance(value, dict):
        for key, child in value.items():
            yield len(key)
            yield from _measure(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            yield from _measure(child, depth + 1)


def _mapping(value: _FrozenJSON) -> _Item:
    if not isinstance(value, Mapping):
        raise _InvalidHistory("History items and content blocks must be objects")
    return value


def _string(item: _Item, key: str) -> str:
    value: Final = item.get(key)
    if not isinstance(value, str) or not value:
        raise _InvalidHistory(f"History requires a nonempty {key}")
    return value


def _blocks(value: _FrozenJSON) -> tuple[_Item, ...]:
    if not isinstance(value, tuple):
        raise _InvalidHistory("History content blocks must be an array")
    return tuple(_mapping(block) for block in value)


def _check_field(name: str, value: _FrozenJSON, kind: _FieldKind | None) -> None:
    if kind is None:
        raise _InvalidHistory(f"Unsupported history field: {name}")
    if kind == "text" and value is not None and not isinstance(value, str):
        raise _InvalidHistory(f"History field {name} must be text")
    if kind == "opaque" and value is not None and not (isinstance(value, (Mapping, tuple)) and not value):
        raise _InvalidHistory(f"Opaque provider state in {name} cannot be counted or compacted")


def _check_fields(item: _Item, inventory: Mapping[str, _FieldKind]) -> None:
    for name, value in item.items():
        _check_field(name, value, inventory.get(name))


def _text_only(value: _FrozenJSON, context: _ContentContext = "messages") -> bool:
    if value is None or isinstance(value, str):
        return True
    blocks: Final = _blocks(value)
    return all(tuple(_text_block(block, context) for block in blocks))


def _text_block(block: _Item, context: _ContentContext) -> bool:
    kind: Final = block.get("type")
    if not isinstance(kind, str) or kind not in _CONTENT_TYPES[context]:
        raise _InvalidHistory(f"Unsupported history content block or media for {context}")
    if kind not in _MEDIA_TYPES:
        _check_fields(block, _THINKING_FIELDS if kind == "thinking" else _TEXT_FIELDS)
        key: Final = "thinking" if kind == "thinking" else "text"
        if not isinstance(block.get(key), str):
            raise _InvalidHistory("Text history blocks require text")
        return True
    _check_fields(block, _MEDIA_FIELDS)
    for source in (block.get(name) for name in ("source", "image_url", "file")):
        if isinstance(source, Mapping):
            _check_fields(source, _MEDIA_SOURCE_FIELDS)
            _text_only(source.get("content"))
        elif source is not None and not isinstance(source, str):
            raise _InvalidHistory("Media sources must be text or objects")
    return False


def _genuine_user(item: _Item) -> bool:
    if item.get("role") != "user":
        return False
    content: Final = item.get("content")
    return isinstance(content, str) or (
        isinstance(content, tuple) and any(_mapping(block).get("type") != "tool_result" for block in content)
    )


def _chat_call(value: _FrozenJSON, index: int) -> _ToolEvent:
    call: Final = _mapping(value)
    _check_fields(call, _CHAT_CALL_FIELDS)
    function: Final = _mapping(call.get("function"))
    _check_fields(function, _CHAT_FUNCTION_FIELDS)
    if call.get("type") != "function":
        raise _InvalidHistory("Unsupported Chat tool call")
    _string(function, "name")
    _string(function, "arguments")
    return _ToolEvent(_string(call, "id"), "chat", False, index)


def _anthropic_block(
    block: _Item, role: _FrozenJSON, index: int, field: Literal["messages", "input"]
) -> tuple[_ToolEvent | None, bool]:
    kind: Final = block.get("type")
    if kind == "tool_use":
        _check_fields(block, _TOOL_USE_FIELDS)
        if role != "assistant":
            raise _InvalidHistory("Anthropic tool_use must belong to an assistant")
        _string(block, "name")
        _mapping(block.get("input"))
        return _ToolEvent(_string(block, "id"), "anthropic", False, index), True
    if kind == "tool_result":
        _check_fields(block, _TOOL_RESULT_FIELDS)
        if role != "user":
            raise _InvalidHistory("Anthropic tool_result must belong to a user")
        return _ToolEvent(_string(block, "tool_use_id"), "anthropic", True, index), _text_only(block.get("content"))
    return None, _text_block(block, field)


def _inspect(item: _Item, index: int, field: Literal["messages", "input"]) -> _CheckedItem:
    kind: Final = item.get("type")
    role: Final = item.get("role")
    if kind in ("function_call", "function_call_output"):
        _check_fields(item, _FUNCTION_FIELDS)
        if field != "input" or role is not None:
            raise _InvalidHistory("Responses function items require input without a role")
        if kind == "function_call":
            _string(item, "name")
            _string(item, "arguments")
        elif item.get("output") is None:
            raise _InvalidHistory("Responses tool results require output")
        event: Final = _ToolEvent(_string(item, "call_id"), "responses", kind == "function_call_output", index)
        return _CheckedItem(item, (event,), _text_only(item.get("output"), "function_output"))
    if kind == "reasoning" and field == "input":
        _check_fields(item, _REASONING_FIELDS)
        return _CheckedItem(
            item, (), all((_text_only(item.get("summary"), "reasoning"), _text_only(item.get("content"), "reasoning")))
        )
    if kind not in (None, "message") or role not in ("system", "developer", "user", "assistant", "tool"):
        raise _InvalidHistory("Unsupported history item or role")
    _check_fields(item, _MESSAGE_FIELDS[field])
    if item.get("function_call") is not None or role == "tool" and field == "input":
        raise _InvalidHistory("Unsupported legacy or mixed-format tool history")
    content: Final = item.get("content")
    checked_blocks: Final = (
        tuple(_anthropic_block(block, role, index, field) for block in _blocks(content))
        if isinstance(content, tuple)
        else ()
    )
    if content is not None and not isinstance(content, (str, tuple)):
        raise _InvalidHistory("Message content must be text or content blocks")
    calls: Final = item.get("tool_calls")
    if calls is not None and (role != "assistant" or field != "messages" or not isinstance(calls, tuple)):
        raise _InvalidHistory("Chat tool_calls require an assistant message and an array")
    events: Final = (
        *(_chat_call(call, index) for call in (calls if isinstance(calls, tuple) else ())),
        *(event for event, _ in checked_blocks if event is not None),
        *((_ToolEvent(_string(item, "tool_call_id"), "chat", True, index),) if role == "tool" else ()),
    )
    if field == "input" and any(event.family != "responses" for event in events):
        raise _InvalidHistory("Mixed-format tool history is unsupported")
    return _CheckedItem(item, events, all(text_only for _, text_only in checked_blocks))


def _checked_history(payload: Mapping[str, object]) -> _CheckedHistory:
    if payload.get("previous_response_id") is not None or payload.get("conversation") is not None:
        raise _InvalidHistory("previous_response_id or conversation history is unavailable for compaction")
    if ("messages" in payload) == ("input" in payload):
        raise _InvalidHistory("Compaction requires exactly one of messages or input")
    field: Final[Literal["messages", "input"]] = "messages" if "messages" in payload else "input"
    data: Final = _JSON.validate_python(payload[field], strict=True)
    if not (field == "input" and isinstance(data, str)) and (
        not isinstance(data, list) or not data or len(data) > _MAX_ITEMS
    ):
        raise _InvalidHistory("Compaction requires a nonempty history array within the item limit")
    sizes: Final = tuple(islice(_measure(data), _MAX_NODES + 1))
    if len(sizes) > _MAX_NODES or sum(sizes) > _MAX_TEXT:
        raise _InvalidHistory("History exceeds the compaction size limit")
    messages: Final = (
        (MappingProxyType({"role": "user", "content": data}),)
        if isinstance(data, str)
        else tuple(_mapping(_freeze(item)) for item in data)
    )
    items: Final = tuple(_inspect(item, index, field) for index, item in enumerate(messages))
    events: Final = tuple(event for item in items for event in item.events)
    if len(frozenset(event.family for event in events)) > 1:
        raise _InvalidHistory("Mixed-format tool history is unsupported")
    calls: Final = MappingProxyType({event.identifier: event for event in events if not event.result})
    results: Final = MappingProxyType({event.identifier: event for event in events if event.result})
    if len(calls) + len(results) != len(events):
        raise _InvalidHistory("Duplicate tool call or result IDs")
    if any(
        identifier not in calls or calls[identifier].family != result.family or calls[identifier].index >= result.index
        for identifier, result in results.items()
    ):
        raise _InvalidHistory("Tool results must match an earlier call ID in the same format")
    user_counts: Final = tuple(accumulate((_genuine_user(item.item) for item in items), initial=0))
    if any(
        user_counts[results[identifier].index if identifier in results else len(items)] != user_counts[call.index + 1]
        for identifier, call in calls.items()
    ):
        raise _InvalidHistory("A new user turn interrupts an incomplete tool exchange")
    open_before: Final = tuple(
        accumulate((sum(-1 if event.result else 1 for event in item.events) for item in items), initial=0)
    )
    starts: Final = tuple(
        index for index, item in enumerate(items) if _genuine_user(item.item) and open_before[index] == 0
    )
    return _CheckedHistory(field, items, starts)


def _split(payload: Mapping[str, object]) -> CompactionHistory:
    checked: Final = _checked_history(payload)
    items: Final = checked.items
    if not checked.starts:
        raise _InvalidHistory("History has no genuine user turn at a complete tool boundary")
    start: Final = checked.starts[-1]
    prefix: Final = tuple(item for item in items[:start] if item.item.get("role") not in ("system", "developer"))
    if not prefix:
        raise _InvalidHistory("History has no compactable prefix")
    if any(not item.text_only for item in prefix):
        raise _InvalidHistory("Unsupported media in summarized history")
    history_text: Final = json.dumps(
        tuple(_thaw_item(item.item) for item in prefix), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    if len(history_text) > _MAX_TEXT:
        raise _InvalidHistory("Serialized history exceeds the compaction size limit")
    return CompactionHistory(
        field=checked.field,
        instructions=tuple(item.item for item in items[:start] if item.item.get("role") in ("system", "developer")),
        tail=tuple(item.item for item in items[start:]),
        history_text=history_text,
    )


def _history_error(error: ValueError | RecursionError) -> CompactionHistoryError:
    return CompactionHistoryError(
        str(error) if isinstance(error, _InvalidHistory) else "History must contain bounded JSON data"
    )


def validate_compaction_input(payload: Mapping[str, object]) -> CompactionHistoryError | None:
    """Validate countable local history without requiring an older compactable prefix."""
    try:
        _checked_history(payload)
        return None
    except (ValidationError, ValueError, RecursionError) as error:
        return _history_error(error)


def split_compaction_history(payload: Mapping[str, object]) -> CompactionHistory | CompactionHistoryError:
    """Snapshot complete JSON history, retaining instructions and the active user/tool turn.

    Limits are 10,000 items, 100,000 JSON nodes, 64 nesting levels and 16 Mi characters.
    Request-level instructions, system and tools stay with the caller's original request.
    """
    try:
        return _split(payload)
    except (ValidationError, ValueError, RecursionError) as error:
        return _history_error(error)
