from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import pytest

import research_platform.neuro.mvpa.runtime_transaction as transaction
from research_platform.neuro.mvpa.runtime_transaction import (
    MANIFEST_RELATIVE_PATH,
    MvpaRuntimeOutputSpec,
    MvpaRuntimeTransactionError,
    execute_mvpa_runtime_transaction,
    plan_mvpa_runtime_transaction,
    runtime_output_specs,
)


def _write_complete_outputs(
    staging_root: Path,
    *,
    representation_kind: str = "prepared_features",
    extra: bool = False,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for spec in runtime_output_specs(representation_kind):
        if spec.relative_path == MANIFEST_RELATIVE_PATH:
            continue
        path = staging_root / spec.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if spec.content_type == "tsv":
            path.write_text("value\n1\n", encoding="utf-8", newline="\n")
            columns = ["value"]
        else:
            payload: dict[str, Any] = {"kind": spec.name}
            if spec.relative_path.endswith("/provenance.json"):
                payload.update(
                    {
                        "output_paths": {spec.name: spec.relative_path},
                        "row_counts": {spec.name: 1},
                    }
                )
            path.write_text(
                json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            columns = []
        artifacts.append(
            {
                "name": spec.name,
                "relative_path": spec.relative_path,
                "row_count": 1,
                "columns": columns,
            }
        )
    output_paths = {
        str(artifact["name"]): str(artifact["relative_path"])
        for artifact in artifacts
    }
    row_counts = {
        str(artifact["name"]): int(artifact["row_count"])
        for artifact in artifacts
    }
    for artifact in artifacts:
        relative_path = str(artifact["relative_path"])
        if not relative_path.endswith("/provenance.json"):
            continue
        path = staging_root / relative_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["output_paths"] = output_paths
        payload["row_counts"] = row_counts
        path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if extra:
        (staging_root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    return {"synthetic_writer": {"artifacts": artifacts}}


def _manifest_payload() -> dict[str, Any]:
    return {
        "project": "project-neutral",
        "mvpa_set": "prepared-runtime",
        "source_reference": "root_ref:mvpa_inputs/patterns.tsv",
        "source_sha256": "a" * 64,
        "warnings": [],
    }


def _tree_digest(root: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _transaction_remnants(parent: Path, name: str) -> list[Path]:
    return sorted(parent.glob(f".{name}.*")) + sorted(parent.glob(f".{name}.claim"))


def test_inventory_is_representation_specific_and_complete() -> None:
    image = runtime_output_specs("image")
    prepared = runtime_output_specs("prepared_features")

    assert len(image) == 14
    assert len(prepared) == 14
    assert image[-1].relative_path == MANIFEST_RELATIVE_PATH
    assert prepared[-1].relative_path == MANIFEST_RELATIVE_PATH
    assert image[0].relative_path == "neuro/pattern-extraction/patterns.tsv"
    assert prepared[0].relative_path == "neuro/pattern-materialization/patterns.tsv"
    assert {item.relative_path for item in image[4:]} == {
        item.relative_path for item in prepared[4:]
    }


def test_plan_is_read_only_and_rejects_existing_and_symlinked_destinations(
    tmp_path: Path,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "mvpa" / "run-a"
    before = tuple(tmp_path.rglob("*"))

    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    assert plan.valid
    assert tuple(tmp_path.rglob("*")) == before
    assert not final_root.parent.exists()

    final_root.parent.mkdir(parents=True)
    final_root.mkdir()
    collision = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )
    assert not collision.valid
    assert collision.collision_paths == ("runtime_root",)

    final_root.rmdir()
    final_root.parent.rmdir()
    link = named_root / "linked"
    link.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    unsafe = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=link / "run-a",
        representation_kind="prepared_features",
    )
    assert not unsafe.valid
    assert any("symbolic link" in error for error in unsafe.errors)


def test_plan_rejects_outside_root_and_special_file_parent(tmp_path: Path) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()

    outside = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=tmp_path / "outside" / "run-a",
        representation_kind="prepared_features",
    )
    assert not outside.valid
    assert any("beneath its configured named root" in error for error in outside.errors)

    special_parent = named_root / "special-parent"
    special_parent.write_text("not a directory\n", encoding="utf-8")
    special = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=special_parent / "run-a",
        representation_kind="prepared_features",
    )
    assert not special.valid
    assert any("parent is not a directory" in error for error in special.errors)


@pytest.mark.parametrize(
    "output_spec",
    runtime_output_specs("prepared_features"),
    ids=lambda spec: spec.name,
)
def test_every_fixed_output_collision_fails_before_writer_invocation(
    tmp_path: Path,
    output_spec: MvpaRuntimeOutputSpec,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "run-a"
    target = final_root / output_spec.relative_path
    target.parent.mkdir(parents=True)
    target.write_text("sentinel\n", encoding="utf-8")
    writer_called = False

    def writer(staging_root: Path) -> dict[str, Any]:
        nonlocal writer_called
        writer_called = True
        return _write_complete_outputs(staging_root)

    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )
    assert not plan.valid
    assert "runtime_root" in plan.collision_paths
    assert output_spec.name in plan.collision_paths
    with pytest.raises(MvpaRuntimeTransactionError, match="already exists"):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=writer,
            manifest_payload=_manifest_payload(),
        )

    assert not writer_called
    assert target.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize("alias_path", ("same/path.tsv", "SAME/PATH.tsv"))
def test_plan_rejects_duplicate_or_case_aliased_output_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_path: str,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    outputs = (
        MvpaRuntimeOutputSpec("first", "same/path.tsv", "tsv"),
        MvpaRuntimeOutputSpec("second", alias_path, "tsv"),
    )
    monkeypatch.setattr(transaction, "runtime_output_specs", lambda _kind: outputs)

    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=named_root / "run-a",
        representation_kind="prepared_features",
    )

    assert not plan.valid
    assert any("alias one destination" in error for error in plan.errors)


def test_plan_rejects_platform_without_atomic_no_replace_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "run-a"
    before = tuple(tmp_path.rglob("*"))
    monkeypatch.setattr(transaction.sys, "platform", "unsupported-posix")

    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    assert not plan.valid
    assert any("atomic no-replace" in error for error in plan.errors)
    assert tuple(tmp_path.rglob("*")) == before
    with pytest.raises(MvpaRuntimeTransactionError, match="atomic no-replace"):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=_write_complete_outputs,
            manifest_payload=_manifest_payload(),
        )
    assert not final_root.exists()


def test_success_promotes_exact_inventory_and_portable_hash_manifest(tmp_path: Path) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "mvpa" / "run-a"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    result = execute_mvpa_runtime_transaction(
        plan,
        write_outputs=_write_complete_outputs,
        manifest_payload=_manifest_payload(),
    )

    assert result.executed
    assert result.recovery_path is None
    files = sorted(path.relative_to(final_root).as_posix() for path in final_root.rglob("*") if path.is_file())
    assert files == sorted(spec.relative_path for spec in runtime_output_specs("prepared_features"))
    manifest_text = (final_root / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert str(tmp_path) not in manifest_text
    assert manifest["status"] == "succeeded"
    assert manifest["errors"] == []
    assert len(manifest["outputs"]) == 13
    for output in manifest["outputs"]:
        path = final_root / output["relative_path"]
        assert sha256(path.read_bytes()).hexdigest() == output["sha256"]
        assert output["row_count"] == 1
    assert not _transaction_remnants(final_root.parent, final_root.name)


def test_provenance_validation_scopes_repeated_artifact_names_by_relative_path(
    tmp_path: Path,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "run-a"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    def repeated_names(staging_root: Path) -> dict[str, Any]:
        flat = _write_complete_outputs(staging_root)["synthetic_writer"]["artifacts"]
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for artifact in flat:
            relative_path = str(artifact["relative_path"])
            writer = relative_path.rsplit("/", 1)[0]
            generic_name = relative_path.rsplit("/", 1)[1].split(".", 1)[0]
            artifact["name"] = generic_name
            grouped.setdefault(writer, {"artifacts": []})["artifacts"].append(artifact)
        for record in grouped.values():
            provenance = next(
                item
                for item in record["artifacts"]
                if str(item["relative_path"]).endswith("provenance.json")
            )
            path = staging_root / str(provenance["relative_path"])
            output_paths = {
                str(item["name"]): str(item["relative_path"])
                for item in record["artifacts"]
            }
            row_counts = {
                str(item["name"]): int(item["row_count"])
                for item in record["artifacts"]
            }
            path.write_text(
                json.dumps(
                    {"output_paths": output_paths, "row_counts": row_counts},
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return grouped

    result = execute_mvpa_runtime_transaction(
        plan,
        write_outputs=repeated_names,
        manifest_payload=_manifest_payload(),
    )

    assert result.executed
    assert (final_root / MANIFEST_RELATIVE_PATH).is_file()


@pytest.mark.parametrize(
    ("defect", "error_match"),
    (
        ("omitted_path", "complete writer artifact set"),
        ("wrong_path", "output-path relationship"),
        ("wrong_count", "row count"),
    ),
)
def test_provenance_requires_complete_writer_relationships(
    tmp_path: Path,
    defect: str,
    error_match: str,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / defect
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    def malformed_provenance(staging_root: Path) -> dict[str, Any]:
        records = _write_complete_outputs(staging_root)
        path = staging_root / "neuro/pattern-materialization/provenance.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        target_name = "neuro_materialized_patterns_tsv"
        if defect == "omitted_path":
            payload["output_paths"].pop(target_name)
        elif defect == "wrong_path":
            payload["output_paths"][target_name] = (
                "analysis/prepared-patterns/rows.tsv"
            )
        else:
            payload["row_counts"][target_name] += 1
        path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return records

    with pytest.raises(ValueError, match=error_match):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=malformed_provenance,
            manifest_payload=_manifest_payload(),
        )

    assert not final_root.exists()
    assert not _transaction_remnants(named_root, final_root.name)


def test_independent_roots_are_byte_deterministic_and_same_root_rerun_is_safe(
    tmp_path: Path,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    roots = (named_root / "run-a", named_root / "run-b")
    for root in roots:
        plan = plan_mvpa_runtime_transaction(
            named_root=named_root,
            final_root=root,
            representation_kind="prepared_features",
        )
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=_write_complete_outputs,
            manifest_payload=_manifest_payload(),
        )
    assert _tree_digest(roots[0]) == _tree_digest(roots[1])

    before = _tree_digest(roots[0])
    collision = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=roots[0],
        representation_kind="prepared_features",
    )
    with pytest.raises(MvpaRuntimeTransactionError, match="already exists"):
        execute_mvpa_runtime_transaction(
            collision,
            write_outputs=_write_complete_outputs,
            manifest_payload=_manifest_payload(),
        )
    assert _tree_digest(roots[0]) == before


@pytest.mark.parametrize("failure", ("writer", "validation", "interrupt"))
def test_pre_promotion_failures_leave_no_destination_or_transaction_tree(
    tmp_path: Path,
    failure: str,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "nested" / "run-a"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    def fail(staging_root: Path) -> dict[str, Any]:
        if failure == "writer":
            (staging_root / "partial.tsv").write_text("value\n1\n", encoding="utf-8")
            raise RuntimeError("writer failed")
        if failure == "interrupt":
            (staging_root / "partial.tsv").write_text("value\n1\n", encoding="utf-8")
            raise KeyboardInterrupt()
        return _write_complete_outputs(staging_root, extra=True)

    error = KeyboardInterrupt if failure == "interrupt" else (RuntimeError if failure == "writer" else ValueError)
    with pytest.raises(error):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=fail,
            manifest_payload=_manifest_payload(),
        )

    assert not final_root.exists()
    assert not final_root.parent.exists()
    assert not _transaction_remnants(named_root, final_root.name)


def test_promotion_failure_cleans_staging_and_concurrent_destination_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "run-a"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    def concurrent_claim(staging: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel.txt").write_text("concurrent\n", encoding="utf-8")
        raise FileExistsError("concurrent claim")

    monkeypatch.setattr(transaction, "_promote_staging_tree", concurrent_claim)
    with pytest.raises(FileExistsError, match="concurrent claim"):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=_write_complete_outputs,
            manifest_payload=_manifest_payload(),
        )

    assert (final_root / "sentinel.txt").read_text(encoding="utf-8") == "concurrent\n"
    assert not _transaction_remnants(named_root, final_root.name)


def test_interruption_immediately_after_promotion_rolls_back_owned_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "run-a"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )
    real_promote = transaction._promote_staging_tree

    def promote_then_interrupt(staging: Path, destination: Path) -> None:
        real_promote(staging, destination)
        raise KeyboardInterrupt()

    monkeypatch.setattr(transaction, "_promote_staging_tree", promote_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=_write_complete_outputs,
            manifest_payload=_manifest_payload(),
        )

    assert not final_root.exists()
    assert not _transaction_remnants(named_root, final_root.name)


def test_partial_parent_creation_failure_removes_owned_parent_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "level-one" / "level-two" / "run-a"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )
    real_mkdir = Path.mkdir
    writer_called = False

    def fail_second_parent(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == final_root.parent:
            raise PermissionError("injected parent creation failure")
        real_mkdir(path, *args, **kwargs)

    def writer(staging_root: Path) -> dict[str, Any]:
        nonlocal writer_called
        writer_called = True
        return _write_complete_outputs(staging_root)

    monkeypatch.setattr(Path, "mkdir", fail_second_parent)
    with pytest.raises(PermissionError, match="injected parent creation failure"):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=writer,
            manifest_payload=_manifest_payload(),
        )

    assert not writer_called
    assert not (named_root / "level-one").exists()
    assert not final_root.exists()
    assert not _transaction_remnants(named_root, final_root.name)


def test_cleanup_failure_reports_recoverable_owned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "run-a"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )
    owned_staging: list[Path] = []
    real_rmtree = transaction.shutil.rmtree

    def fail_writer(staging_root: Path) -> dict[str, Any]:
        owned_staging.append(staging_root)
        (staging_root / "recoverable.txt").write_text(
            "recoverable\n",
            encoding="utf-8",
        )
        raise RuntimeError("injected writer failure")

    def refuse_owned_cleanup(path: str | Path, *args: Any, **kwargs: Any) -> None:
        candidate = Path(path)
        if owned_staging and candidate == owned_staging[0]:
            raise OSError("injected cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(transaction.shutil, "rmtree", refuse_owned_cleanup)
    with pytest.raises(
        MvpaRuntimeTransactionError,
        match="recoverable transaction path",
    ) as raised:
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=fail_writer,
            manifest_payload=_manifest_payload(),
        )

    assert len(owned_staging) == 1
    assert raised.value.recovery_path == owned_staging[0]
    assert (owned_staging[0] / "recoverable.txt").read_text(encoding="utf-8") == (
        "recoverable\n"
    )
    assert not final_root.exists()


def test_atomic_promotion_refuses_a_concurrently_created_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "run-a"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )
    real_promote = transaction._promote_staging_tree

    def create_empty_destination_then_promote(staging: Path, destination: Path) -> None:
        destination.mkdir()
        real_promote(staging, destination)

    monkeypatch.setattr(
        transaction,
        "_promote_staging_tree",
        create_empty_destination_then_promote,
    )
    with pytest.raises(MvpaRuntimeTransactionError, match="claimed concurrently"):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=_write_complete_outputs,
            manifest_payload=_manifest_payload(),
        )

    assert final_root.is_dir()
    assert tuple(final_root.iterdir()) == ()
    assert not _transaction_remnants(named_root, final_root.name)


def test_existing_foreign_transaction_claim_is_never_removed(tmp_path: Path) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "run-a"
    foreign_claim = named_root / ".run-a.claim"
    foreign_claim.mkdir()
    sentinel = foreign_claim / "owner.txt"
    sentinel.write_text("foreign-owner\n", encoding="utf-8")
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    with pytest.raises(MvpaRuntimeTransactionError, match="already claims"):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=_write_complete_outputs,
            manifest_payload=_manifest_payload(),
        )

    assert not final_root.exists()
    assert sentinel.read_text(encoding="utf-8") == "foreign-owner\n"
    assert foreign_claim.is_dir()
    assert tuple(
        path
        for path in _transaction_remnants(named_root, final_root.name)
        if path != foreign_claim
    ) == ()


def test_manifest_rejects_local_paths_and_writer_count_mismatches(tmp_path: Path) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()

    unsafe_root = named_root / "unsafe"
    unsafe_plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=unsafe_root,
        representation_kind="prepared_features",
    )
    with pytest.raises(ValueError, match="non-portable"):
        execute_mvpa_runtime_transaction(
            unsafe_plan,
            write_outputs=_write_complete_outputs,
            manifest_payload={**_manifest_payload(), "input": "/home/alice/patterns.tsv"},
        )
    assert not unsafe_root.exists()

    count_root = named_root / "count-mismatch"
    count_plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=count_root,
        representation_kind="prepared_features",
    )

    def wrong_count(staging_root: Path) -> dict[str, Any]:
        records = _write_complete_outputs(staging_root)
        records["synthetic_writer"]["artifacts"][0]["row_count"] = 2
        return records

    with pytest.raises(ValueError, match="row count"):
        execute_mvpa_runtime_transaction(
            count_plan,
            write_outputs=wrong_count,
            manifest_payload=_manifest_payload(),
        )
    assert not count_root.exists()


def test_prepared_portable_provenance_rejects_embedded_local_paths(tmp_path: Path) -> None:
    named_root = tmp_path / "artifacts"
    named_root.mkdir()
    final_root = named_root / "unsafe-provenance"
    plan = plan_mvpa_runtime_transaction(
        named_root=named_root,
        final_root=final_root,
        representation_kind="prepared_features",
    )

    def unsafe_provenance(staging_root: Path) -> dict[str, Any]:
        records = _write_complete_outputs(staging_root)
        path = staging_root / "neuro/pattern-materialization/provenance.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["input_provenance"] = {
            "command": "loader --input=/home/alice/private.tsv"
        }
        path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return records

    with pytest.raises(ValueError, match="Portable MVPA runtime provenance"):
        execute_mvpa_runtime_transaction(
            plan,
            write_outputs=unsafe_provenance,
            manifest_payload=_manifest_payload(),
        )

    assert not final_root.exists()
    assert not _transaction_remnants(named_root, final_root.name)
