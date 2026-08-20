from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os

import pytest

import research_platform.core.tabular_output_transaction as transaction
from research_platform.core.tabular_output_transaction import (
    FINAL_OUTPUT_DIRECTORY,
    SCHEMA_VERSION,
    TRANSACTION_MANIFEST_NAME,
    OutputSpec,
    TabularOutputTransactionError,
    build_transaction_plan,
    cleanup_owned_staging,
    create_owned_staging,
    output_specs_from_plan,
    preflight_transaction_root,
    promote_staging_no_replace,
    read_owned_regular_file,
    seal_staged_transaction,
    transaction_staging_entries,
    validate_committed_transaction,
    validate_sealed_transaction,
    validate_staged_outputs,
)


PLAN_IDENTITY = {
    "schema_version": "research_platform.core.run_plan_identity.v1",
    "sha256": "a" * 64,
}


def _specs() -> tuple[OutputSpec, ...]:
    return (
        OutputSpec("metadata", "metadata.json", "json"),
        OutputSpec("features", "tables/features.tsv", "tsv", ("record_id", "value")),
    )


def _write_outputs(root: Path) -> dict[str, bytes]:
    payloads = {
        "metadata.json": b'{"feature_columns":["value"],"target":"binary_target"}\n',
        "tables/features.tsv": b"record_id\tvalue\nrow-01\t1.5\nrow-02\t2.5\n",
    }
    for relative_path, data in payloads.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return payloads


def _seal(run_root: Path):
    staging = create_owned_staging(run_root)
    payloads = _write_outputs(staging.path)
    manifest = seal_staged_transaction(
        staging,
        outputs=_specs(),
        run_id="run-a",
        workflow_action="train",
        workflow_target="model",
        plan_identity_schema=PLAN_IDENTITY["schema_version"],
        plan_identity_sha256=PLAN_IDENTITY["sha256"],
    )
    return staging, payloads, manifest


def test_plan_is_deterministic_portable_and_roundtrips_specs() -> None:
    plan = build_transaction_plan(
        run_id="run-a",
        workflow_action="train",
        workflow_target="model",
        outputs=_specs(),
    )

    assert plan == build_transaction_plan(
        run_id="run-a",
        workflow_action="train",
        workflow_target="model",
        outputs=_specs(),
    )
    assert plan["schema_version"] == SCHEMA_VERSION
    assert plan["final_output_directory"] == "outputs"
    assert plan["transaction_manifest"] == "outputs/transaction-manifest.json"
    assert plan["existing_output"] == "fail"
    parsed = output_specs_from_plan(plan)
    assert [(item.logical_name, item.relative_path, item.content_type) for item in parsed] == [
        ("metadata", "metadata.json", "json"),
        ("features", "tables/features.tsv", "tsv"),
    ]
    assert all(not Path(item["relative_path"]).is_absolute() for item in plan["outputs"])


@pytest.mark.parametrize(
    "mutator",
    (
        lambda plan: plan.update({"run_id": "../unsafe"}),
        lambda plan: plan.update({"workflow": {"action": "", "target": "model"}}),
        lambda plan: plan.update({"existing_output": "replace"}),
        lambda plan: plan.update({"unexpected": True}),
    ),
)
def test_plan_parser_rejects_unsafe_or_malformed_identity(mutator) -> None:
    plan = build_transaction_plan(
        run_id="run-a",
        workflow_action="train",
        workflow_target="model",
        outputs=_specs(),
    )
    mutator(plan)
    with pytest.raises(TabularOutputTransactionError):
        output_specs_from_plan(plan)


@pytest.mark.parametrize(
    "specs",
    (
        (),
        (OutputSpec("first", "../escape.json", "json"),),
        (OutputSpec("first", "/absolute.json", "json"),),
        (OutputSpec("first", "nested\\file.json", "json"),),
        (OutputSpec("first", TRANSACTION_MANIFEST_NAME, "json"),),
        (
            OutputSpec("first", "same.json", "json"),
            OutputSpec("second", "SAME.JSON", "json"),
        ),
    ),
)
def test_plan_rejects_missing_unsafe_reserved_or_aliased_outputs(specs) -> None:
    with pytest.raises(TabularOutputTransactionError):
        build_transaction_plan(
            run_id="run-a",
            workflow_action="train",
            workflow_target="model",
            outputs=specs,
        )


def test_preflight_is_read_only_and_rejects_final_or_staging_residue(tmp_path: Path) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    before = tuple(run_root.iterdir())

    preflight_transaction_root(run_root)
    assert tuple(run_root.iterdir()) == before

    residue = run_root / ".outputs.foreign.staging"
    residue.mkdir()
    with pytest.raises(TabularOutputTransactionError, match="recovery evidence") as error:
        preflight_transaction_root(run_root)
    assert error.value.recovery_path == residue
    residue.rmdir()

    final_root = run_root / FINAL_OUTPUT_DIRECTORY
    final_root.mkdir()
    with pytest.raises(TabularOutputTransactionError, match="already exists"):
        preflight_transaction_root(run_root)


def test_create_staging_is_exclusive_owned_and_cleanup_is_inode_guarded(tmp_path: Path) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging = create_owned_staging(run_root)

    assert staging.path.parent == run_root
    assert transaction_staging_entries(run_root) == (staging.path,)
    identity = os.lstat(staging.path)
    assert (identity.st_dev, identity.st_ino) == staging.filesystem_identity

    moved = run_root / "owned-moved"
    staging.path.rename(moved)
    staging.path.mkdir()
    sentinel = staging.path / "sentinel.txt"
    sentinel.write_text("foreign\n", encoding="utf-8")
    with pytest.raises(TabularOutputTransactionError, match="foreign entry"):
        cleanup_owned_staging(staging)
    assert sentinel.read_text(encoding="utf-8") == "foreign\n"


def test_owned_cleanup_removes_complete_owned_tree(tmp_path: Path) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging = create_owned_staging(run_root)
    _write_outputs(staging.path)

    cleanup_owned_staging(staging)

    assert not staging.path.exists()
    assert transaction_staging_entries(run_root) == ()


def test_safe_reader_returns_all_bytes_across_multiple_chunks(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    data = (b"0123456789abcdef" * (1024 * 80)) + b"tail"
    assert len(data) > 1024 * 1024
    (root / "large.json").write_bytes(data)

    assert read_owned_regular_file(root, "large.json") == data

    (root / "link.json").symlink_to(root / "large.json")
    with pytest.raises(TabularOutputTransactionError, match="opened safely"):
        read_owned_regular_file(root, "link.json")


def test_validation_records_exact_hash_size_rows_and_columns(tmp_path: Path) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging = create_owned_staging(run_root)
    payloads = _write_outputs(staging.path)

    records = validate_staged_outputs(staging, _specs())

    by_name = {record.logical_name: record for record in records}
    assert by_name["metadata"].byte_size == len(payloads["metadata.json"])
    assert by_name["metadata"].sha256 == sha256(payloads["metadata.json"]).hexdigest()
    assert by_name["features"].columns == ("record_id", "value")
    assert by_name["features"].row_count == 2
    assert by_name["features"].sha256 == sha256(
        payloads["tables/features.tsv"]
    ).hexdigest()


@pytest.mark.parametrize(
    ("defect", "error_match"),
    (
        ("missing", "inventory"),
        ("extra", "inventory"),
        ("bad_json", "strict finite JSON"),
        ("nonfinite_json", "strict finite JSON"),
        ("bad_tsv_width", "row width"),
        ("nonfinite_tsv", "non-finite"),
        ("symlink", "symbolic links"),
        ("special", "special filesystem entry"),
    ),
)
def test_validation_rejects_incomplete_malformed_or_unsafe_inventory(
    tmp_path: Path,
    defect: str,
    error_match: str,
) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging = create_owned_staging(run_root)
    _write_outputs(staging.path)
    if defect == "missing":
        (staging.path / "metadata.json").unlink()
    elif defect == "extra":
        (staging.path / "extra.txt").write_text("extra\n", encoding="utf-8")
    elif defect == "bad_json":
        (staging.path / "metadata.json").write_text("{bad}\n", encoding="utf-8")
    elif defect == "nonfinite_json":
        (staging.path / "metadata.json").write_text('{"value":NaN}\n', encoding="utf-8")
    elif defect == "bad_tsv_width":
        (staging.path / "tables/features.tsv").write_text(
            "record_id\tvalue\nrow-01\n", encoding="utf-8"
        )
    elif defect == "nonfinite_tsv":
        (staging.path / "tables/features.tsv").write_text(
            "record_id\tvalue\nrow-01\tNaN\n", encoding="utf-8"
        )
    elif defect == "symlink":
        target = staging.path / "metadata-real.json"
        (staging.path / "metadata.json").rename(target)
        (staging.path / "metadata.json").symlink_to(target)
    else:
        special = staging.path / "socket-entry"
        os.mkfifo(special)
        with pytest.raises(TabularOutputTransactionError, match=error_match):
            validate_staged_outputs(staging, _specs())
        return

    with pytest.raises(TabularOutputTransactionError, match=error_match):
        validate_staged_outputs(staging, _specs())


def test_seal_writes_portable_deterministic_manifest_and_flushes(tmp_path: Path) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging, payloads, manifest = _seal(run_root)

    parsed, manifest_digest = validate_sealed_transaction(
        staging,
        outputs=_specs(),
        expected_manifest=manifest,
    )
    manifest_bytes = (staging.path / TRANSACTION_MANIFEST_NAME).read_bytes()
    assert parsed == manifest
    assert manifest_digest == sha256(manifest_bytes).hexdigest()
    assert str(tmp_path) not in manifest_bytes.decode("utf-8")
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["plan_identity"] == PLAN_IDENTITY
    for record in manifest["outputs"]:
        assert record["sha256"] == sha256(payloads[record["relative_path"]]).hexdigest()
        assert record["byte_size"] == len(payloads[record["relative_path"]])


def test_sealed_validation_detects_manifest_or_output_mutation(tmp_path: Path) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging, _, manifest = _seal(run_root)
    (staging.path / "metadata.json").write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(TabularOutputTransactionError, match="exact output bytes"):
        validate_sealed_transaction(
            staging,
            outputs=_specs(),
            expected_manifest=manifest,
        )


def test_committed_validation_rechecks_exact_identity_inventory_and_bytes(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging, _, expected_manifest = _seal(run_root)
    final_root = promote_staging_no_replace(staging, run_root / FINAL_OUTPUT_DIRECTORY)

    manifest, manifest_digest, records = validate_committed_transaction(
        final_root,
        outputs=_specs(),
        expected_run_id="run-a",
        expected_workflow_action="train",
        expected_workflow_target="model",
        expected_plan_identity=PLAN_IDENTITY,
    )

    assert manifest == expected_manifest
    assert manifest_digest == sha256(
        (final_root / TRANSACTION_MANIFEST_NAME).read_bytes()
    ).hexdigest()
    assert [record.logical_name for record in records] == ["metadata", "features"]

    (final_root / "metadata.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(TabularOutputTransactionError, match="exact current output bytes"):
        validate_committed_transaction(
            final_root,
            outputs=_specs(),
            expected_run_id="run-a",
            expected_workflow_action="train",
            expected_workflow_target="model",
            expected_plan_identity=PLAN_IDENTITY,
        )


@pytest.mark.parametrize(
    ("run_id", "action", "target", "plan_identity", "error_match"),
    (
        ("run-b", "train", "model", PLAN_IDENTITY, "different source run"),
        ("run-a", "preprocess", "model", PLAN_IDENTITY, "different workflow"),
        ("run-a", "train", "tabular", PLAN_IDENTITY, "different workflow"),
        (
            "run-a",
            "train",
            "model",
            {"schema_version": PLAN_IDENTITY["schema_version"], "sha256": "b" * 64},
            "reviewed plan identity",
        ),
    ),
)
def test_committed_validation_rejects_identity_mismatch(
    tmp_path: Path,
    run_id: str,
    action: str,
    target: str,
    plan_identity: dict[str, str],
    error_match: str,
) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging, _, _ = _seal(run_root)
    final_root = promote_staging_no_replace(staging, run_root / FINAL_OUTPUT_DIRECTORY)

    with pytest.raises(TabularOutputTransactionError, match=error_match):
        validate_committed_transaction(
            final_root,
            outputs=_specs(),
            expected_run_id=run_id,
            expected_workflow_action=action,
            expected_workflow_target=target,
            expected_plan_identity=plan_identity,
        )


@pytest.mark.parametrize("defect", ("extra", "symlink"))
def test_committed_validation_rejects_extra_or_symlinked_output(
    tmp_path: Path,
    defect: str,
) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging, _, _ = _seal(run_root)
    final_root = promote_staging_no_replace(staging, run_root / FINAL_OUTPUT_DIRECTORY)
    if defect == "extra":
        (final_root / "extra.txt").write_text("extra\n", encoding="utf-8")
        error_match = "inventory"
    else:
        original = final_root / "metadata-original.json"
        (final_root / "metadata.json").rename(original)
        (final_root / "metadata.json").symlink_to(original)
        error_match = "symbolic links"

    with pytest.raises(TabularOutputTransactionError, match=error_match):
        validate_committed_transaction(
            final_root,
            outputs=_specs(),
            expected_run_id="run-a",
            expected_workflow_action="train",
            expected_workflow_target="model",
            expected_plan_identity=PLAN_IDENTITY,
        )


def test_promotion_is_atomic_no_replace_and_preserves_collision(tmp_path: Path) -> None:
    first_root = tmp_path / "run-a"
    first_root.mkdir()
    staging, _, _ = _seal(first_root)
    final_root = promote_staging_no_replace(staging, first_root / FINAL_OUTPUT_DIRECTORY)
    assert final_root.is_dir()
    assert not staging.path.exists()

    second_root = tmp_path / "run-b"
    second_root.mkdir()
    second, _, _ = _seal(second_root)
    collision = second_root / FINAL_OUTPUT_DIRECTORY
    collision.mkdir()
    sentinel = collision / "sentinel.txt"
    sentinel.write_text("foreign\n", encoding="utf-8")
    with pytest.raises(TabularOutputTransactionError, match="claimed"):
        promote_staging_no_replace(second, collision)
    assert sentinel.read_text(encoding="utf-8") == "foreign\n"
    assert second.path.is_dir()


def test_promotion_fails_closed_when_atomic_primitive_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging, _, _ = _seal(run_root)
    monkeypatch.setattr(
        transaction,
        "atomic_no_replace_support_error",
        lambda: "atomic no-replace unavailable",
    )

    with pytest.raises(TabularOutputTransactionError, match="unavailable"):
        promote_staging_no_replace(staging, run_root / FINAL_OUTPUT_DIRECTORY)

    assert staging.path.is_dir()
    assert not (run_root / FINAL_OUTPUT_DIRECTORY).exists()


def test_atomic_promotion_collision_preserves_concurrent_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging, _, _ = _seal(run_root)
    final_root = run_root / FINAL_OUTPUT_DIRECTORY

    def concurrent_claim(_source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel.txt").write_text("foreign\n", encoding="utf-8")
        raise TabularOutputTransactionError("claimed concurrently")

    monkeypatch.setattr(transaction, "_atomic_no_replace_directory", concurrent_claim)

    with pytest.raises(TabularOutputTransactionError, match="concurrently"):
        promote_staging_no_replace(staging, final_root)

    assert (final_root / "sentinel.txt").read_text(encoding="utf-8") == "foreign\n"
    assert staging.path.is_dir()


def test_post_promotion_fsync_failure_reports_committed_recovery_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging, _, _ = _seal(run_root)
    final_root = run_root / FINAL_OUTPUT_DIRECTORY
    real_fsync_directory = transaction._fsync_directory

    def fail_parent_flush(path: Path) -> None:
        if path == run_root:
            raise OSError("injected parent fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(transaction, "_fsync_directory", fail_parent_flush)

    with pytest.raises(TabularOutputTransactionError, match="durability is uncertain") as error:
        promote_staging_no_replace(staging, final_root)

    assert error.value.promotion_committed
    assert error.value.recovery_path == final_root
    assert final_root.is_dir()
    assert not staging.path.exists()


def test_cleanup_failure_reports_owned_recovery_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging = create_owned_staging(run_root)
    _write_outputs(staging.path)
    monkeypatch.setattr(
        transaction.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("injected cleanup failure")),
    )

    with pytest.raises(TabularOutputTransactionError, match="requires recovery") as error:
        cleanup_owned_staging(staging)

    assert error.value.recovery_path == staging.path
    assert staging.path.is_dir()


def test_seal_flushes_files_and_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run-a"
    run_root.mkdir()
    staging = create_owned_staging(run_root)
    _write_outputs(staging.path)
    real_fsync = transaction.os.fsync
    calls: list[int] = []

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(transaction.os, "fsync", record_fsync)

    seal_staged_transaction(
        staging,
        outputs=_specs(),
        run_id="run-a",
        workflow_action="train",
        workflow_target="model",
        plan_identity_schema=PLAN_IDENTITY["schema_version"],
        plan_identity_sha256=PLAN_IDENTITY["sha256"],
    )

    assert len(calls) >= 6  # manifest write, three files, nested directory, staging root
