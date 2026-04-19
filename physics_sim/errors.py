"""Shared exception helpers for contract and lifecycle errors."""

from __future__ import annotations

from collections.abc import Iterable


class PhysicsSimError(Exception):
    """Base class for project-specific exceptions."""


class PhysicsSimConfigurationError(ValueError, PhysicsSimError):
    """Raised when configuration values violate a declared contract."""


class PhysicsSimLifecycleError(RuntimeError, PhysicsSimError):
    """Raised when backend/pipeline lifecycle order is violated."""


def lifecycle_error(
    *,
    owner: str,
    operation: str,
    expected: str,
    detail: str | None = None,
) -> PhysicsSimLifecycleError:
    """Build a contextual lifecycle error with stable message shape."""
    msg = (
        f"[E_LIFECYCLE] owner={owner} operation={operation} "
        f"expected={expected}"
    )
    if detail:
        msg = f"{msg} detail={detail}"
    return PhysicsSimLifecycleError(msg)


def configuration_error(
    *,
    owner: str,
    operation: str,
    expected: str,
    detail: str | None = None,
) -> PhysicsSimConfigurationError:
    """Build a contextual configuration error with stable message shape."""
    msg = (
        f"[E_CONFIG] owner={owner} operation={operation} "
        f"expected={expected}"
    )
    if detail:
        msg = f"{msg} detail={detail}"
    return PhysicsSimConfigurationError(msg)


def unknown_registry_error(
    *,
    registry: str,
    key: str,
    available: Iterable[str],
) -> PhysicsSimConfigurationError:
    """Build a contextual unknown-registry-key error."""
    choices = ",".join(sorted(available))
    return PhysicsSimConfigurationError(
        f"[E_REGISTRY_UNKNOWN] registry={registry} key={key!r} "
        f"available=[{choices}]"
    )

