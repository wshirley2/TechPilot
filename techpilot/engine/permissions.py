"""Reusable permission protocol for tool side effects.

The runtime owns every :class:`PermissionRequest`.  A prompt may only return a
decision; it never returns replacement tool arguments.  This keeps the pending
tool call authoritative while an interactive approval is in progress.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4


class PermissionAction(str, Enum):
    """The three deterministic outcomes produced by a permission policy."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionEffect(str, Enum):
    """Observable effect used by Chat and Goal tool permissions."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    DELEGATE = "delegate"
    UNKNOWN = "unknown"


class PermissionGrantScope(str, Enum):
    """How long an interactive allow decision may be reused."""

    ONCE = "once"
    SESSION = "session"
    PREFIX = "prefix"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """One immutable, runtime-owned request presented to a permission policy."""

    tool_call_id: str
    tool_name: str
    effect: PermissionEffect
    normalized_arguments: Mapping[str, Any]
    reason: str
    scope: str
    trusted_preview: str | None = None
    source_snapshot: str | None = None
    command_tokens: tuple[str, ...] = ()
    command_prefix: tuple[str, ...] = ()
    request_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        # A prompt receives this object.  Freeze a deep copy so it cannot mutate
        # the exact arguments the runtime will execute after approval.
        object.__setattr__(self, "normalized_arguments", _freeze(dict(self.normalized_arguments)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for future event persistence."""

        return {
            "request_id": self.request_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "effect": self.effect.value,
            "normalized_arguments": _thaw(self.normalized_arguments),
            "reason": self.reason,
            "scope": self.scope,
            "trusted_preview": self.trusted_preview,
            "source_snapshot": self.source_snapshot,
            "command_tokens": list(self.command_tokens),
            "command_prefix": list(self.command_prefix),
        }


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """A policy or user decision for one PermissionRequest."""

    action: PermissionAction
    reason: str
    grant_scope: PermissionGrantScope = PermissionGrantScope.ONCE
    prompted: bool = False

    @classmethod
    def allow(
        cls,
        reason: str,
        grant_scope: PermissionGrantScope = PermissionGrantScope.ONCE,
    ) -> PermissionDecision:
        return cls(PermissionAction.ALLOW, reason, grant_scope)

    @classmethod
    def ask(cls, reason: str) -> PermissionDecision:
        return cls(PermissionAction.ASK, reason)

    @classmethod
    def deny(cls, reason: str) -> PermissionDecision:
        return cls(PermissionAction.DENY, reason)


class PermissionPolicy(Protocol):
    """Pure code policy used before any interactive prompt."""

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        """Classify one request as ALLOW, ASK, or DENY."""


class PermissionPrompt(Protocol):
    """Interactive adapter that may answer an ASK decision."""

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        """Return ALLOW or DENY without changing the request."""


class DenyPermissionPrompt:
    """Safe fallback for runtimes that have no interactive user interface."""

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        del request
        return PermissionDecision.deny("No interactive permission prompt is available")


class PermissionManager:
    """Apply code policy, interactive decisions, and current-session grants."""

    def __init__(self, policy: PermissionPolicy, prompt: PermissionPrompt | None = None):
        self.policy = policy
        self.prompt = prompt or DenyPermissionPrompt()
        self._session_grants: set[tuple[str, str, str]] = set()
        self._command_prefix_grants: set[tuple[str, ...]] = set()
        self._lock = threading.RLock()

    def authorize(
        self,
        request: PermissionRequest,
        *,
        force_prompt: bool = False,
    ) -> PermissionDecision:
        """Resolve one request without ever accepting replacement arguments.

        ``force_prompt`` is used after a source snapshot changes.  It bypasses
        an older session grant so the newly generated preview is explicitly
        reviewed again.
        """

        policy_decision = self.policy.decide(request)
        if policy_decision.action is not PermissionAction.ASK:
            return policy_decision

        with self._lock:
            if not force_prompt:
                granted = self._matching_grant(request)
                if granted is not None:
                    return granted

            user_decision = self.prompt.decide(request)
            prompted = not isinstance(self.prompt, DenyPermissionPrompt)
            if user_decision.action is PermissionAction.ASK:
                return PermissionDecision.deny("Permission prompt returned an unresolved ASK decision")
            if user_decision.action is PermissionAction.DENY:
                return PermissionDecision(
                    user_decision.action,
                    user_decision.reason,
                    user_decision.grant_scope,
                    prompted=prompted,
                )

            if user_decision.grant_scope is PermissionGrantScope.SESSION:
                self._session_grants.add(self._session_key(request))
            elif user_decision.grant_scope is PermissionGrantScope.PREFIX:
                if not request.command_prefix:
                    return PermissionDecision.deny("This request does not support a command-prefix grant")
                self._command_prefix_grants.add(request.command_prefix)
            return PermissionDecision(
                user_decision.action,
                user_decision.reason,
                user_decision.grant_scope,
                prompted=prompted,
            )

    def clear_session_grants(self) -> None:
        """Drop all grants when the owning Chat session ends."""

        with self._lock:
            self._session_grants.clear()
            self._command_prefix_grants.clear()

    def _matching_grant(self, request: PermissionRequest) -> PermissionDecision | None:
        if self._session_key(request) in self._session_grants:
            return PermissionDecision.allow(
                "Allowed by a current-session grant",
                PermissionGrantScope.SESSION,
            )
        for prefix in self._command_prefix_grants:
            if request.command_prefix and request.command_tokens[: len(prefix)] == prefix:
                return PermissionDecision.allow(
                    f"Allowed by command-prefix grant: {' '.join(prefix)}",
                    PermissionGrantScope.PREFIX,
                )
        return None

    @staticmethod
    def _session_key(request: PermissionRequest) -> tuple[str, str, str]:
        return request.tool_name, request.effect.value, request.scope


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | frozenset):
        return [_thaw(item) for item in value]
    return value
