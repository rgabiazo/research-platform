"""Generic planning-only publish-back helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_env_value
from .paths import to_workspace_relative

PUBLISH_BACK_POLICIES = ("never", "if_absent", "overwrite")


def build_publish_back_scaffold(
    *,
    workspace_root: str | Path,
    run_root: str | Path,
    publish_back: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = publish_back or {}
    items = payload.get("items")
    if items is None:
        items = []
    if not isinstance(items, list):
        raise ValueError("publish_back.items must be a list.")
    scaffold = {
        "default_policy": resolve_publish_back_policy(payload.get("default_policy"), default="never"),
        "plan_path": to_workspace_relative(Path(run_root) / "publish-back-plan.yaml", workspace_root),
    }
    if items:
        scaffold["items"] = items
    return scaffold


def build_publish_back_plan(
    *,
    workspace_root: str | Path,
    run_root: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    resolved_workspace_root = Path(workspace_root).resolve()
    resolved_run_root = Path(run_root).resolve()
    scaffold = build_publish_back_scaffold(
        workspace_root=resolved_workspace_root,
        run_root=resolved_run_root,
        publish_back=_expect_mapping(manifest.get("publish_back"), "publish_back"),
    )
    planned_items: list[dict[str, Any]] = []
    for index, item in enumerate(scaffold.get("items", []), start=1):
        entry = _expect_mapping(item, f"publish_back.items[{index}]")
        source_value = entry.get("source")
        destination_value = entry.get("destination")
        if source_value in (None, ""):
            raise ValueError(f"publish_back.items[{index}] must define source.")
        if destination_value in (None, ""):
            raise ValueError(f"publish_back.items[{index}] must define destination.")

        source_path = resolve_publish_back_path(workspace_root=resolved_workspace_root, value=source_value)
        destination_path = resolve_publish_back_path(workspace_root=resolved_workspace_root, value=destination_value)
        if not _is_within(source_path, resolved_run_root):
            raise ValueError(f"Publish-back source must stay under the run root: {source_value}")
        if _is_within(destination_path, resolved_run_root):
            raise ValueError(f"Publish-back destination must not point back into the run root: {destination_value}")
        if not source_path.exists():
            planned_items.append(
                {
                    "source": _display_path(source_path, workspace_root=resolved_workspace_root),
                    "destination": _display_path(destination_path, workspace_root=resolved_workspace_root),
                    "policy": resolve_publish_back_policy(entry.get("policy"), default=scaffold["default_policy"]),
                    "action": "skip",
                    "destination_exists": destination_path.exists(),
                    "reason": "source missing",
                    "source_exists": False,
                }
            )
            continue

        policy = resolve_publish_back_policy(entry.get("policy"), default=scaffold["default_policy"])
        destination_exists = destination_path.exists()
        action, reason = _plan_action(policy=policy, destination_exists=destination_exists)
        planned_items.append(
            {
                "source": _display_path(source_path, workspace_root=resolved_workspace_root),
                "destination": _display_path(destination_path, workspace_root=resolved_workspace_root),
                "policy": policy,
                "action": action,
                "destination_exists": destination_exists,
                "reason": reason,
                "source_exists": True,
            }
        )

    actionable_count = sum(1 for item in planned_items if item["action"] != "skip")
    plan = {
        "run_id": manifest["run_id"],
        "generated_at": _timestamp(),
        "default_policy": scaffold["default_policy"],
        "summary": {
            "total_items": len(planned_items),
            "actionable_items": actionable_count,
        },
    }
    if planned_items:
        plan["items"] = planned_items
    return plan


def resolve_publish_back_policy(value: Any, *, default: str | None = None) -> str:
    candidate = default if value in (None, "") else str(value).strip()
    if not candidate:
        raise ValueError("Publish-back policy must not be empty.")
    if candidate not in PUBLISH_BACK_POLICIES:
        allowed = ", ".join(PUBLISH_BACK_POLICIES)
        raise ValueError(f"Unsupported publish-back policy: {candidate}. Expected one of: {allowed}.")
    return candidate


def resolve_publish_back_path(*, workspace_root: str | Path, value: str | Path) -> Path:
    resolved_value = resolve_env_value(value)
    if resolved_value is None:
        raise ValueError("Publish-back path must not be empty.")
    candidate = Path(resolved_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(workspace_root).resolve() / candidate).resolve()


def _expect_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _display_path(path: Path, *, workspace_root: Path) -> str:
    return to_workspace_relative(path, workspace_root)


def _plan_action(*, policy: str, destination_exists: bool) -> tuple[str, str]:
    if policy == "never":
        return "skip", "policy=never"
    if policy == "if_absent":
        if destination_exists:
            return "skip", "destination exists"
        return "copy", "destination absent"
    if destination_exists:
        return "overwrite", "destination exists"
    return "copy", "destination absent"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
