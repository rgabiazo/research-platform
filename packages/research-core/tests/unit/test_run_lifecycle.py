from __future__ import annotations

import os
from pathlib import Path
import threading

import pytest

from research_platform.core.run_lifecycle import (
    PLAN_IDENTITY_SCHEMA,
    RunLifecycleError,
    acquire_execution_claim,
    build_plan_identity,
    claim_path,
    path_entry_exists,
    resolved_run_path,
    validate_run_id,
    verify_plan_identity,
)


def _manifest(
    *,
    mode: str = "plan",
    dry_run: bool = True,
    created_at: str = "2026-07-21T00:00:00+00:00",
) -> dict[str, object]:
    return {
        "run_id": "unit-safe",
        "created_at": created_at,
        "slice": "tabular",
        "workflow": {"action": "train", "target": "model"},
        "project": {"name": "project-example"},
        "batch": {
            "name": "toy_binary_logreg",
            "selected_row": {
                "feature_table": "project-pilot-tabular/toy_features.tsv",
                "target_column": "binary_target",
            },
        },
        "predictor_contract": {
            "feature_columns": ["feature_a", "feature_b"],
            "feature_count": 2,
            "target_column": "binary_target",
        },
        "resources": {"cpus": 1, "ram_gb": 4, "threads": 1, "n_jobs": 1},
        "execution": {
            "mode": mode,
            "dry_run": dry_run,
            "command": ["bash", "artifacts/runs/unit-safe/execute.sh"],
        },
    }


@pytest.mark.parametrize(
    "run_id",
    (
        "",
        " ",
        " unit",
        "unit ",
        ".",
        "..",
        "/absolute",
        "nested/run",
        "nested\\run",
        "../escape",
        "unit\x00name",
        "unit\nname",
        "unit\x7fname",
        "-leading-hyphen",
        ".hidden",
        "unit:name",
        "unit.",
        "x" * 249,
    ),
)
def test_validate_run_id_rejects_unsafe_values_without_normalizing(run_id: str) -> None:
    with pytest.raises(RunLifecycleError):
        validate_run_id(run_id)


@pytest.mark.parametrize(
    "run_id",
    ("unit", "unit-01", "unit_01", "20260721T120000Z-local-train-model", "A.b-c_2"),
)
def test_validate_run_id_preserves_safe_values_exactly(run_id: str) -> None:
    assert validate_run_id(run_id) == run_id


def test_resolved_run_and_claim_paths_are_contained_siblings(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"

    run_root = resolved_run_path(artifacts_root, "unit-safe")

    assert run_root == artifacts_root.resolve() / "runs" / "unit-safe"
    assert claim_path(artifacts_root, "unit-safe") == run_root.parent / ".unit-safe.claim"
    assert not artifacts_root.exists()


def test_reuse_error_is_actionable_without_offering_mutation() -> None:
    message = str(RunLifecycleError.for_reuse("unit-safe", "the run root already exists"))

    assert "unit-safe" in message
    assert "Inspect the existing run and choose a new run id" in message
    assert "there is no overwrite, resume, retry, replace, or force option" in message


def test_path_entry_exists_detects_broken_symlinks(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    broken = tmp_path / "broken"
    broken.symlink_to(missing)

    assert path_entry_exists(broken)
    assert not path_entry_exists(missing)


@pytest.mark.parametrize("run_id", ("../escape", "nested/run", "nested\\run", "unit\x00name"))
def test_unsafe_claim_ids_create_no_filesystem_entry(tmp_path: Path, run_id: str) -> None:
    artifacts_root = tmp_path / "artifacts"

    with pytest.raises(RunLifecycleError):
        acquire_execution_claim(artifacts_root, run_id)

    assert not artifacts_root.exists()


def test_plan_identity_is_deterministic_and_normalizes_authorization_only() -> None:
    script = b"#!/usr/bin/env bash\nset -euo pipefail\n"
    planned = _manifest(mode="plan", dry_run=True, created_at="first")
    planned["plan_identity"] = {"ignored": True}
    local = _manifest(mode="local", dry_run=False, created_at="second")

    first = build_plan_identity(planned, script)
    second = build_plan_identity(local, script)

    assert first == second
    assert first["schema_version"] == PLAN_IDENTITY_SCHEMA
    assert first["files"]["execute.sh"]["size_bytes"] == len(script)
    assert len(first["sha256"]) == 64


def test_plan_identity_is_independent_of_mapping_insertion_order() -> None:
    original = _manifest()
    reversed_manifest = dict(reversed(list(original.items())))

    assert build_plan_identity(original, b"execute\n") == build_plan_identity(
        reversed_manifest,
        b"execute\n",
    )


def test_extra_file_mapping_order_is_not_semantic() -> None:
    first = build_plan_identity(
        _manifest(),
        b"execute\n",
        extra_files={"stage": b"stage\n", "submit": b"submit\n"},
    )
    second = build_plan_identity(
        _manifest(),
        b"execute\n",
        extra_files={"submit": b"submit\n", "stage": b"stage\n"},
    )

    assert first == second


@pytest.mark.parametrize(
    "mutation",
    (
        lambda manifest: manifest["workflow"].update(action="preprocess"),
        lambda manifest: manifest["project"].update(name="project-other"),
        lambda manifest: manifest["batch"].update(name="other-batch"),
        lambda manifest: manifest["batch"]["selected_row"].update(target_column="other_target"),
        lambda manifest: manifest["predictor_contract"].update(feature_columns=["feature_b", "feature_a"]),
        lambda manifest: manifest["resources"].update(cpus=2),
        lambda manifest: manifest["execution"].update(command=["bash", "different.sh"]),
    ),
)
def test_plan_identity_changes_for_semantic_manifest_drift(mutation: object) -> None:
    original = _manifest()
    changed = _manifest()
    mutation(changed)  # type: ignore[operator]

    assert build_plan_identity(original, b"execute\n") != build_plan_identity(
        changed,
        b"execute\n",
    )


def test_plan_identity_binds_execute_slurm_and_extra_file_bytes() -> None:
    manifest = _manifest(mode="slurm", dry_run=False)
    base = build_plan_identity(
        manifest,
        b"execute\n",
        b"sbatch\n",
        extra_files={"hpc/stage-plan.yaml": b"stage\n"},
    )

    assert base != build_plan_identity(
        manifest,
        b"changed\n",
        b"sbatch\n",
        extra_files={"hpc/stage-plan.yaml": b"stage\n"},
    )
    assert base != build_plan_identity(
        manifest,
        b"execute\n",
        b"changed\n",
        extra_files={"hpc/stage-plan.yaml": b"stage\n"},
    )
    assert base != build_plan_identity(
        manifest,
        b"execute\n",
        b"sbatch\n",
        extra_files={"hpc/stage-plan.yaml": b"changed\n"},
    )
    assert sorted(base["files"]) == ["execute.sh", "hpc/stage-plan.yaml", "submit.sbatch"]


def test_plan_identity_rejects_nonbytes_and_duplicate_reviewed_labels() -> None:
    with pytest.raises(RunLifecycleError, match="must be supplied as bytes"):
        build_plan_identity(_manifest(), "not-bytes")  # type: ignore[arg-type]
    with pytest.raises(RunLifecycleError, match="Duplicate reviewed-file label"):
        build_plan_identity(
            _manifest(),
            b"execute\n",
            extra_files={"execute.sh": b"duplicate\n"},
        )


def test_verify_plan_identity_accepts_exact_reviewed_plan() -> None:
    manifest = _manifest()
    identity = build_plan_identity(
        manifest,
        b"execute\n",
        extra_files={"submission-plan.json": b"{}\n"},
    )
    manifest["plan_identity"] = identity

    verified = verify_plan_identity(
        manifest,
        b"execute\n",
        extra_files={"submission-plan.json": b"{}\n"},
    )

    assert verified == identity
    assert verified is not identity


def test_verify_plan_identity_accepts_reviewed_plan_to_local_execute_transition() -> None:
    planned = _manifest(mode="plan", dry_run=True)
    planned_identity = build_plan_identity(planned, b"execute\n")
    local_execute = _manifest(mode="local", dry_run=False, created_at="later")
    local_execute["plan_identity"] = planned_identity

    assert verify_plan_identity(local_execute, b"execute\n") == planned_identity


@pytest.mark.parametrize(
    "stored",
    (
        None,
        {},
        {"schema_version": "other", "sha256": "0" * 64, "files": {}},
        {"schema_version": PLAN_IDENTITY_SCHEMA, "sha256": "invalid", "files": {}},
        {
            "schema_version": PLAN_IDENTITY_SCHEMA,
            "sha256": "0" * 64,
            "files": {},
            "unexpected": True,
        },
    ),
)
def test_verify_plan_identity_rejects_missing_or_malformed_identity(stored: object) -> None:
    manifest = _manifest()
    if stored is not None:
        manifest["plan_identity"] = stored

    with pytest.raises(RunLifecycleError):
        verify_plan_identity(manifest, b"execute\n")


def test_verify_plan_identity_rejects_script_and_extra_file_drift() -> None:
    manifest = _manifest()
    manifest["plan_identity"] = build_plan_identity(
        manifest,
        b"execute\n",
        extra_files={"stage": b"original\n"},
    )

    with pytest.raises(RunLifecycleError, match="does not match"):
        verify_plan_identity(
            manifest,
            b"changed\n",
            extra_files={"stage": b"original\n"},
        )
    with pytest.raises(RunLifecycleError, match="does not match"):
        verify_plan_identity(
            manifest,
            b"execute\n",
            extra_files={"stage": b"changed\n"},
        )


def test_verify_plan_identity_rejects_slurm_script_drift() -> None:
    manifest = _manifest(mode="slurm", dry_run=False)
    manifest["plan_identity"] = build_plan_identity(
        manifest,
        b"execute\n",
        b"sbatch\n",
    )

    with pytest.raises(RunLifecycleError, match="does not match"):
        verify_plan_identity(manifest, b"execute\n", b"changed\n")


def test_execution_claim_is_exclusive_and_owned_release_is_idempotent(tmp_path: Path) -> None:
    claim = acquire_execution_claim(tmp_path / "artifacts", "unit-safe")

    assert claim.path.is_dir()
    assert claim.filesystem_identity == (
        os.lstat(claim.path).st_dev,
        os.lstat(claim.path).st_ino,
    )
    with pytest.raises(RunLifecycleError, match="already has an execution claim"):
        acquire_execution_claim(tmp_path / "artifacts", "unit-safe")

    claim.release()
    claim.release()
    assert claim.released
    assert not path_entry_exists(claim.path)


def test_foreign_claim_and_sentinel_are_never_removed(tmp_path: Path) -> None:
    path = claim_path(tmp_path / "artifacts", "unit-safe")
    path.mkdir(parents=True)
    sentinel = path / "owner.txt"
    sentinel.write_text("foreign\n", encoding="utf-8")

    with pytest.raises(RunLifecycleError, match="already has an execution claim"):
        acquire_execution_claim(tmp_path / "artifacts", "unit-safe")

    assert sentinel.read_text(encoding="utf-8") == "foreign\n"


def test_nonempty_owned_claim_is_preserved_as_recovery_evidence(tmp_path: Path) -> None:
    claim = acquire_execution_claim(tmp_path / "artifacts", "unit-safe")
    sentinel = claim.path / "recovery.txt"
    sentinel.write_text("retain\n", encoding="utf-8")

    with pytest.raises(RunLifecycleError, match="not an empty removable directory"):
        claim.release()

    assert sentinel.read_text(encoding="utf-8") == "retain\n"
    assert not claim.released


def test_replaced_claim_identity_is_preserved(tmp_path: Path) -> None:
    claim = acquire_execution_claim(tmp_path / "artifacts", "unit-safe")
    original = claim.path.with_name(f"{claim.path.name}.original")
    claim.path.rename(original)
    claim.path.mkdir()
    sentinel = claim.path / "owner.txt"
    sentinel.write_text("replacement\n", encoding="utf-8")

    with pytest.raises(RunLifecycleError, match="replaced and will not be removed"):
        claim.release()

    assert sentinel.read_text(encoding="utf-8") == "replacement\n"
    assert original.is_dir()


def test_concurrent_claimants_have_exactly_one_owner_without_timing_sleeps(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    start = threading.Barrier(3)
    release_owner = threading.Event()
    owner_acquired = threading.Event()
    loser_finished = threading.Event()
    lock = threading.Lock()
    results: list[tuple[str, object]] = []

    def attempt() -> None:
        start.wait(timeout=5)
        try:
            claim = acquire_execution_claim(artifacts_root, "unit-safe")
        except RunLifecycleError as exc:
            with lock:
                results.append(("loser", exc))
            loser_finished.set()
            return
        with lock:
            results.append(("owner", claim.filesystem_identity))
        owner_acquired.set()
        assert release_owner.wait(timeout=5)
        claim.release()

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    assert owner_acquired.wait(timeout=5)
    assert loser_finished.wait(timeout=5)
    release_owner.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert [kind for kind, _ in results].count("owner") == 1
    assert [kind for kind, _ in results].count("loser") == 1
    assert not path_entry_exists(claim_path(artifacts_root, "unit-safe"))
