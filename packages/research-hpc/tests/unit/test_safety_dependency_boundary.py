"""Static dependency and behavior boundaries for H2 safety authority."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from research_platform.hpc import safety


REPO_ROOT = Path(__file__).resolve().parents[4]
SAFETY_ROOT = (
    REPO_ROOT
    / "packages"
    / "research-hpc"
    / "src"
    / "research_platform"
    / "hpc"
    / "safety"
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _resolved_import_targets(
    path: Path,
    *,
    source: str | None = None,
) -> set[str]:
    tree = ast.parse(
        path.read_text(encoding="utf-8") if source is None else source,
        filename=str(path),
    )
    src_root = next(parent for parent in path.parents if parent.name == "src")
    package = path.relative_to(src_root).parts[:-1]
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            retained = len(package) - (node.level - 1)
            if retained < 0:
                continue
            base_parts = (*package[:retained], *(node.module or "").split("."))
            base = ".".join(part for part in base_parts if part)
        else:
            base = node.module or ""
        if base:
            targets.add(base)
        targets.update(
            f"{base}.{alias.name}" if base else alias.name
            for alias in node.names
            if alias.name != "*"
        )
    return targets


def _resolved_call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = (
                    f"{node.module}.{alias.name}"
                )

    def dotted_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            if prefix is not None:
                return f"{prefix}.{node.attr}"
        return None

    return {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (name := dotted_name(node.func)) is not None
    }


def _fenced_block_after(
    text: str,
    marker: str,
    *,
    language: str = "text",
) -> str:
    marker_index = text.index(marker)
    opening_fence = f"```{language}\n"
    fence_index = text.index(opening_fence, marker_index) + len(opening_fence)
    fence_end = text.index("\n```", fence_index)
    return text[fence_index:fence_end]


def _normalized_text(text: str) -> str:
    prose_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        while stripped.startswith(">"):
            stripped = stripped[1:].lstrip()
        prose_lines.append(stripped)
    return " ".join(" ".join(prose_lines).replace("**", "").split())


class SafetyDependencyBoundaryTests(unittest.TestCase):
    def test_package_exports_implemented_h2a_h2b_h2c1_authority(self) -> None:
        exports = set(safety.__all__)
        implemented = {
            "canonical_json_bytes",
            "parse_canonical_json_bytes",
            "domain_separated_sha256",
            "PortableRelativePath",
            "require_distinct_file_paths",
            "TrustedRoot",
            "open_trusted_root",
            "RegularFileRecord",
            "RegularFileInventory",
            "scan_regular_file_inventory",
            "DescriptorRetirementObservation",
            "DescriptorRetirementIdentity",
            "DescriptorRetirementRecord",
            "DescriptorRetirementEvidence",
            "DescriptorRetirementError",
            "PublicationState",
            "StagingState",
            "StagingCleanupState",
            "PublicationResult",
            "StagingCleanupResult",
            "PublicationEntryIdentity",
            "NamespaceEvidence",
            "PublicationRecoveryEvidence",
            "StagingCleanupRecoveryEvidence",
            "StagedFileHandle",
            "StagedDirectoryHandle",
            "StagingLifecycleError",
            "StagingAuthorityError",
            "PublicationError",
            "PublicationValidationError",
            "PublicationCapabilityError",
            "PublicationCollisionError",
            "StagingAdmissionError",
            "PublicationDurabilityError",
            "PublicationOutcomeUncertainError",
            "PublicationNamespaceConflictError",
            "PublicationNamespaceUncertainError",
            "StagingCleanupError",
            "open_exclusive_staged_file",
            "open_exclusive_staged_directory",
            "publish_completed_file",
            "publish_completed_directory",
            "cleanup_owned_staging",
        }
        self.assertTrue(implemented <= exports)
        for public_name in implemented:
            with self.subTest(public_name=public_name):
                self.assertTrue(hasattr(safety, public_name))

        for reserved in (
            "atomic_publish",
            "ClaimHandle",
            "ClaimRecord",
            "ReceiptEnvelope",
            "PUBLICATION_STAGING_DOMAIN",
            "RENAME_NOREPLACE",
            "RENAME_EXCL",
            "build_claim_record",
            "acquire_exclusive_claim",
            "release_exclusive_claim",
        ):
            with self.subTest(reserved=reserved):
                self.assertNotIn(reserved, exports)
                self.assertFalse(hasattr(safety, reserved))

    def test_h2a_h2b_h2c1_modules_do_not_import_other_domains_or_remote_stacks(
        self,
    ) -> None:
        forbidden_prefixes = (
            "research_platform.core",
            "research_platform.neuro",
            "ops",
            "subprocess",
            "socket",
            "asyncio",
            "urllib",
            "http",
            "requests",
            "paramiko",
            "random",
            "time",
            "datetime",
            "shutil",
        )
        modules = sorted(SAFETY_ROOT.glob("*.py"))
        self.assertEqual(
            {path.name for path in modules},
            {
                "__init__.py",
                "canonical.py",
                "inventory.py",
                "paths.py",
                "publication.py",
            },
        )
        for path in modules:
            with self.subTest(path=path):
                imports = _imports(path)
                self.assertFalse(
                    {
                        name
                        for name in imports
                        if any(
                            name == prefix or name.startswith(f"{prefix}.")
                            for prefix in forbidden_prefixes
                        )
                    }
                )

    def test_existing_domain_transactions_do_not_import_h2_safety(self) -> None:
        production_files = tuple(
            path
            for path in sorted(
                REPO_ROOT.glob("packages/*/src/**/*.py")
            )
            if not path.is_relative_to(SAFETY_ROOT)
        )
        self.assertGreater(len(production_files), 3)
        for path in production_files:
            with self.subTest(path=path):
                self.assertFalse(
                    {
                        target
                        for target in _resolved_import_targets(path)
                        if target == "research_platform.hpc.safety"
                        or target.startswith(
                            "research_platform.hpc.safety."
                        )
                    },
                )

    def test_dependency_scan_resolves_equivalent_safety_imports(self) -> None:
        synthetic_module = (
            REPO_ROOT
            / "packages"
            / "research-hpc"
            / "src"
            / "research_platform"
            / "hpc"
            / "synthetic_boundary.py"
        )
        spellings = (
            "import research_platform.hpc.safety",
            "from research_platform.hpc.safety import canonical_json_bytes",
            "from research_platform.hpc import safety",
            "from .safety import canonical_json_bytes",
            "from . import safety",
        )
        for source in spellings:
            with self.subTest(source=source):
                targets = _resolved_import_targets(
                    synthetic_module,
                    source=source,
                )
                self.assertTrue(
                    any(
                        target == "research_platform.hpc.safety"
                        or target.startswith(
                            "research_platform.hpc.safety."
                        )
                        for target in targets
                    )
                )

    def test_inventory_has_no_path_following_or_remote_fallback(self) -> None:
        source = (SAFETY_ROOT / "inventory.py").read_text(encoding="utf-8")
        for forbidden in (
            "_ROOT_CREATION_TOKEN",
            "_from_validated",
            "Path.resolve",
            "realpath",
            "os.walk",
            "Path.rglob",
            "/proc/self/fd",
            "os.environ",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        for reserved_module in (
            "atomic.py",
            "claims.py",
            "receipts.py",
        ):
            with self.subTest(reserved_module=reserved_module):
                self.assertFalse((SAFETY_ROOT / reserved_module).exists())

    def test_publication_has_no_forbidden_fallback_or_remote_dependency(
        self,
    ) -> None:
        publication = SAFETY_ROOT / "publication.py"
        self.assertTrue(publication.is_file())
        source = publication.read_text(encoding="utf-8")
        for forbidden in (
            "Path.resolve",
            "realpath",
            "/proc/self/fd",
            "os.environ",
            "SYS_rename",
            "SYS_renameat",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        calls = _resolved_call_names(publication)
        forbidden_calls = {
            "os.rename",
            "os.renames",
            "os.replace",
            "os.system",
            "os.getenv",
            "os.putenv",
            "os.unsetenv",
            "os.walk",
            "os.path.realpath",
            "pathlib.Path.resolve",
            "pathlib.Path.rglob",
            "shutil.copy",
            "shutil.copy2",
            "shutil.copyfile",
            "shutil.copytree",
            "shutil.move",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.Popen",
            "subprocess.run",
        }
        self.assertFalse(calls & forbidden_calls)
        self.assertFalse(
            {
                name
                for name in calls
                if name == "syscall" or name.endswith(".syscall")
            }
        )

    def test_standard_library_tests_do_not_pin_a_macos_temporary_parent(self) -> None:
        unit_root = REPO_ROOT / "packages" / "research-hpc" / "tests" / "unit"
        forbidden_parent = "/" + "private" + "/tmp"
        for path in sorted(unit_root.glob("test_safety_*.py")):
            with self.subTest(path=path):
                tree = ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                )
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    function = node.func
                    if not (
                        isinstance(function, ast.Attribute)
                        and function.attr == "TemporaryDirectory"
                    ):
                        continue
                    for keyword in node.keywords:
                        if (
                            keyword.arg == "dir"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value == forbidden_parent
                        ):
                            self.fail(f"macOS-only temporary parent in {path}")

    def test_h2b_and_future_h2d_contracts_are_frozen_in_the_adr(self) -> None:
        adr = (
            REPO_ROOT
            / "docs"
            / "decisions"
            / "ADR-0023-hpc-safety-primitives.md"
        ).read_text(encoding="utf-8")
        normalized_adr = _normalized_text(adr)

        trusted_root_contract = (
            "H2b owns a context-managed trusted-root opener",
            "absolute, lexically normalized path",
            "component-by-component from `/`",
            "directory descriptors and no-follow behavior",
            "symlink or non-directory at any root or ancestor component",
            "root descriptor and its device/inode identity",
            "`open_trusted_root()` is a context-manager factory",
            "calling it opens no filesystem descriptor",
            "Direct `TrustedRoot` construction is unsupported and rejected",
            "pins the validated descriptor until explicit "
            "`TrustedRoot.close()` or context exit",
            "descriptor cannot claim root/ancestor-path validation",
            "explicitly documented as the trust boundary",
        )
        receipt_dependency_contract = (
            "`family_version` is a positive integer from 1 through "
            "`2**31 - 1`",
            "`relation` is exactly `prior` or `prerequisite`",
            "family version numerically",
            "digest-algorithm UTF-8 bytes",
            "digest-value ASCII bytes",
            "only once across the entire list regardless of relation",
        )
        for statement in trusted_root_contract + receipt_dependency_contract:
            with self.subTest(statement=statement):
                self.assertIn(_normalized_text(statement), normalized_adr)

    def test_h2c1_descriptor_retirement_contract_is_frozen(self) -> None:
        adr = (
            REPO_ROOT
            / "docs"
            / "decisions"
            / "ADR-0023-hpc-safety-primitives.md"
        ).read_text(encoding="utf-8")
        normalized_adr = _normalized_text(adr)

        self.assertEqual(
            _fenced_block_after(
                adr,
                "The exact descriptor-retirement observation ABI is:",
                language="python",
            ),
            "class DescriptorRetirementObservation(str, Enum):\n"
            '    CLOSED = "closed"\n'
            '    ALREADY_ABSENT = "already_absent"\n'
            '    FOREIGN_PRESERVED = "foreign_preserved"\n'
            '    UNINSPECTABLE = "uninspectable"\n'
            "    CLOSE_OUTCOME_UNCERTAIN = "
            '"close_outcome_uncertain"',
        )
        self.assertEqual(
            _fenced_block_after(
                adr,
                "The exact immutable descriptor-retirement evidence schema",
            ),
            "DescriptorRetirementIdentity:\n"
            "    device: int\n"
            "    inode: int\n"
            "    entry_type: Literal[\n"
            '        "regular_file",\n'
            '        "directory",\n'
            '        "symlink",\n'
            '        "fifo",\n'
            '        "socket",\n'
            '        "character_device",\n'
            '        "block_device",\n'
            '        "other",\n'
            "    ]\n"
            "    owner_uid: int\n"
            "\n"
            "DescriptorRetirementRecord:\n"
            "    ordinal: int\n"
            "    role: Literal[\n"
            '        "traversal_entry",\n'
            '        "traversal_directory",\n'
            '        "traversal_parent",\n'
            '        "operation_staging",\n'
            '        "operation_parent",\n'
            '        "handle_staging",\n'
            '        "handle_parent",\n'
            "    ]\n"
            "    observation: DescriptorRetirementObservation\n"
            "    close_attempted: bool\n"
            "    admitted_identity: "
            "DescriptorRetirementIdentity | None\n"
            "    observed_identity: "
            "DescriptorRetirementIdentity | None\n"
            "    error_errno: int | None\n"
            "\n"
            "DescriptorRetirementEvidence:\n"
            "    records: tuple[DescriptorRetirementRecord, ...]",
        )
        self.assertEqual(
            _fenced_block_after(
                adr,
                "The seven descriptor roles have these exact "
                "acquisition-purpose meanings:",
            ),
            "traversal_entry:\n"
            "    transient operation-local descriptor opened for one "
            "included regular-file leaf beneath a staged directory during "
            "population, validation, flushing, or cleanup\n"
            "traversal_directory:\n"
            "    transient operation-local descriptor opened for one "
            "descendant directory beneath the staged root during population, "
            "validation, flushing, or cleanup; excludes the top-level "
            "staged-root descriptor\n"
            "traversal_parent:\n"
            "    transient operation-local descriptor acquired specifically "
            "as the stable immediate-parent authority for creating, "
            "verifying, or removing a descendant traversal entry during "
            "population, validation, flushing, or cleanup; excludes "
            "publication-parent descriptors\n"
            "operation_staging:\n"
            "    fresh operation-local descriptor reopened for the top-level "
            "staged file or staged-directory root during publication or "
            "cleanup; distinct from the descriptor retained from admission\n"
            "operation_parent:\n"
            "    fresh operation-local descriptor for the top-level "
            "publication parent used by a terminal publication or cleanup "
            "operation\n"
            "handle_staging:\n"
            "    staging file/root descriptor acquired during staging "
            "admission and intended for the public handle; it receives this "
            "role immediately upon acquisition, remains provisional before "
            "yield, is retained by the handle after successful admission, "
            "and keeps this role if pre-yield admission fails\n"
            "handle_parent:\n"
            "    publication-parent descriptor acquired during staging "
            "admission and intended for the public handle; it receives this "
            "role immediately upon acquisition, remains provisional before "
            "yield, is retained by the handle after successful admission, "
            "and keeps this role if pre-yield admission fails",
        )
        self.assertEqual(
            _fenced_block_after(
                adr,
                "The two `FOREIGN_PRESERVED` branches are exactly:",
            ),
            "private_generation_mismatch:\n"
            "    observation = FOREIGN_PRESERVED\n"
            "    close_attempted = False\n"
            "    admitted_identity = locally_snapshotted_admitted_identity\n"
            "    observed_identity = None\n"
            "    error_errno = None\n"
            "\n"
            "matching_generation_unequal_stable_identity:\n"
            "    observation = FOREIGN_PRESERVED\n"
            "    close_attempted = False\n"
            "    admitted_identity = non-null\n"
            "    observed_identity = non-null\n"
            "    observed_identity != admitted_identity\n"
            "    error_errno = None",
        )
        self.assertEqual(
            _fenced_block_after(
                adr,
                "The two private `NOT_COMMITTED` retirement-batch origins "
                "are exactly:",
            ),
            "seal_time_collision:\n"
            "    public_state: NOT_COMMITTED\n"
            "    handle_retirement_batch: pending\n"
            "    immutable_ledger: preserved\n"
            "    cleanup_authorization: preserved\n"
            "\n"
            "terminal_publication_proven_not_committed:\n"
            "    public_state: NOT_COMMITTED\n"
            "    handle_retirement_batch: consumed\n"
            "    operation_retirement_batch: consumed\n"
            "    consumption_precedes: PublicationError delivery\n"
            "    immutable_ledger: preserved\n"
            "    cleanup_authorization: preserved",
        )
        self.assertEqual(
            _fenced_block_after(
                adr,
                "The only valid `DescriptorRetirementError` cross-field "
                "tuples are:",
            ),
            "otherwise_successful_publication:\n"
            '    operation: "publish"\n'
            "    state: PUBLISHED\n"
            "    terminal_result: exact PublicationResult\n"
            "    terminal_result.state: COMMITTED_DURABLE\n"
            "\n"
            "otherwise_successful_cleanup:\n"
            '    operation: "cleanup"\n'
            "    state: DISCARDED\n"
            "    terminal_result: exact StagingCleanupResult\n"
            "    terminal_result.state: DISCARDED_DURABLE\n"
            "\n"
            "population_or_sealing:\n"
            '    operation: Literal["write", "mkdir", "write_file", "seal"]\n'
            "    state: RETIRED\n"
            "    terminal_result: None\n"
            "\n"
            "context_exit:\n"
            '    operation: "context_exit"\n'
            "    state: Literal[RETIRED, NOT_COMMITTED]\n"
            "    terminal_result: None\n"
            "\n"
            "finalization:\n"
            '    operation: "finalization"\n'
            "    state: RETIRED\n"
            "    terminal_result: None",
        )
        self.assertEqual(
            _fenced_block_after(
                adr,
                "The pending seal-time-collision `NOT_COMMITTED` context-exit "
                "matrix is exactly:",
            ),
            "anomaly_free_without_body_exception:\n"
            "    returns: normally\n"
            "    state: NOT_COMMITTED\n"
            "    immutable_ledger: preserved\n"
            "    cleanup_authorization: preserved\n"
            "    pending_handle_retirement_batch: consumed_once\n"
            "\n"
            "retirement_anomaly_without_body_exception:\n"
            "    error: DescriptorRetirementError\n"
            "    error.state: NOT_COMMITTED\n"
            '    error.operation: "context_exit"\n'
            "    error.terminal_result: None\n"
            "    state: NOT_COMMITTED\n"
            "    immutable_ledger: preserved\n"
            "    cleanup_authorization: preserved\n"
            "    cleanup: not_performed\n"
            "    retirement_retry: forbidden\n"
            "\n"
            "retirement_anomaly_with_body_exception:\n"
            "    composition: "
            "provenance_aware_attachment_or_ordered_BaseExceptionGroup\n"
            "    body_exception: primary\n"
            "    state: NOT_COMMITTED\n"
            "    immutable_ledger: preserved\n"
            "    cleanup_authorization: preserved\n"
            "    cleanup: not_performed\n"
            "    retirement_retry: forbidden",
        )
        self.assertEqual(
            _fenced_block_after(
                adr,
                "The already-consumed terminal-publication `NOT_COMMITTED`",
            ),
            "without_body_exception:\n"
            "    returns: normally\n"
            "    descriptor_inspection: not_performed\n"
            "    close: not_performed\n"
            "    retirement_attempt: not_performed\n"
            "\n"
            "with_body_exception:\n"
            "    propagation: body_exception_unchanged\n"
            "    descriptor_inspection: not_performed\n"
            "    close: not_performed\n"
            "    retirement_attempt: not_performed",
        )
        self.assertEqual(
            _fenced_block_after(
                adr,
                "Descriptor retirement proceeds in this exact order:",
            ),
            "1. establish and preserve the filesystem outcome and lifecycle "
            "state\n"
            "2. under the private authority lock, snapshot each owned "
            "descriptor's role, admitted identity, and private "
            "per-acquisition generation\n"
            "3. before inspection or close, detach every raw descriptor "
            "slot, permanently consume its retirement attempt, and "
            "compare-and-remove only its matching ownership-registry "
            "generation\n"
            "4. process every independently owned descriptor in the frozen "
            "deterministic role order and continue after anomalies\n"
            "5. generation mismatch -> FOREIGN_PRESERVED with "
            "observed_identity=None; do not inspect or close\n"
            "6. matching generation without an admitted identity -> "
            "UNINSPECTABLE; otherwise perform one pre-close fstat\n"
            "7. pre-close EBADF -> ALREADY_ABSENT; do not close\n"
            "8. other failure to obtain a stable comparison -> "
            "UNINSPECTABLE; do not close\n"
            "9. matching generation plus unequal stable identity -> "
            "FOREIGN_PRESERVED; do not close\n"
            "10. exact generation and equal identity -> invoke close exactly "
            "once\n"
            "11. close returns -> CLOSED\n"
            "12. close raises -> CLOSE_OUTCOME_UNCERTAIN\n"
            "13. never retry close\n"
            "14. never inspect or act on that descriptor number after close "
            "returns or raises\n"
            "15. context exit and finalization cannot retry consumed "
            "retirement\n"
            "16. continue retiring every other independently owned "
            "descriptor after an anomaly\n"
            "17. the exact live TrustedRoot is borrowed and is never closed, "
            "retired, or adopted by H2c1",
        )

        prose_contract = (
            "The five descriptor-retirement names in that complete frozen "
            "surface are implemented and exported by this working snapshot",
            "Hosted acceptance remains pending until the eventual commit's "
            "CI succeeds",
            "One evidence value is complete for one retirement batch",
            "Descriptors already verifiably retired before that batch are "
            "excluded",
            "Records are immutable and ordered by the frozen role order above",
            "each role occurs at most once",
            "Every owned descriptor maps to exactly one role",
            "one descriptor can never generate multiple records",
            "Role is fixed by acquisition purpose and never changes because "
            "the descriptor is later used for another check",
            "During staged-directory population, `write_file()` uses "
            "`traversal_entry` for the descriptor returned by exclusive "
            "child-file creation",
            "`mkdir()` uses `traversal_directory` for the no-follow descriptor "
            "opened for the newly created descendant directory",
            "either operation uses `traversal_parent` for a separately "
            "acquired stable immediate-parent descriptor",
            "Population roles are assigned at descriptor acquisition "
            "according to acquisition purpose, before successful ledger "
            "admission is known",
            "Role assignment alone does not prove successful ledger inclusion "
            "or authorize cleanup of a newly created namespace entry",
            "This population coverage broadens only the acquisition-purpose "
            "meaning of the three existing traversal roles",
            "It adds no descriptor role, role ordering, public state, enum "
            "member, property, result or evidence field, API, lifecycle "
            "transition, or descriptor bound",
            "A top-level operation descriptor retains its `operation_*` role "
            "even when used to begin traversal",
            "A descendant directory serving as the parent of its children "
            "remains `traversal_directory` unless a separate descriptor was "
            "acquired specifically as `traversal_parent`",
            "Absent roles emit no record",
            "one retirement batch never owns two descriptors with the same "
            "role",
            "Role is determined by acquisition purpose, not by whether a "
            "public handle was ultimately yielded",
            "Descriptor-retirement evidence for a failed pre-yield admission "
            "may therefore contain `handle_staging`, `handle_parent`, or both",
            "Those provisional admission descriptors never use "
            "`operation_staging`, `operation_parent`, or a traversal role",
            "Each still maps to exactly one role and one record",
            "the frozen role order, not acquisition order, determines their "
            "evidence-tuple order",
            "existing admission/abandonment bound of two descriptors is "
            "unchanged",
            "No public handle is returned after any admission failure",
            "When stable admission fails after this invocation created the "
            "reservation, the exact `StagingAdmissionError` remains primary",
            "other pre-yield failures preserve their already frozen exact "
            "subtype",
            "A role cannot be reused until the prior descriptor authority for "
            "that role is consumed",
            "`ordinal` is an exact zero-based integer equal to the record's "
            "tuple position",
            "Evidence contains from one through seven records",
            "at least one record is not `CLOSED`",
            "Evidence created after a filesystem outcome is already proven "
            "contains at most four records",
            "`operation_staging`, `operation_parent`, `handle_staging`, and "
            "`handle_parent`",
            "Every numeric field is exact `int`, never `bool`",
            "Device, inode, and UID range from `0` through `2**64 - 1`",
            "`error_errno` is `None` or an exact integer from `1` through "
            "`2**31 - 1`",
            "Direct-construction bypass, hostile subclasses, mutation, "
            "copying, deep-copying, and serialization are rejected",
            "exposes no raw descriptor, host path, credential, mutable "
            "mapping, unbounded collection, or free-form operating-system "
            "text",
            "`CLOSED` requires `close_attempted=True`, a non-null admitted "
            "identity, an observed identity equal to it, and "
            "`error_errno=None`",
            "`ALREADY_ABSENT` requires `close_attempted=False`, no observed "
            "identity, and the pre-close `EBADF` as its bounded errno",
            "Its admitted identity may be `None` only for a pre-admission "
            "descriptor that never obtained stable identity",
            "`FOREIGN_PRESERVED` is valid only through one of the two mutually "
            "exclusive branches frozen below",
            "For a private-generation mismatch, `admitted_identity` equals "
            "the locally snapshotted admitted identity and may be `None` only "
            "for a pre-admission descriptor that never obtained stable "
            "identity",
            "The descriptor is neither inspected nor closed",
            "For a matching generation plus unequal stable `fstat` identity, "
            "both identities are non-null and unequal, and the descriptor is "
            "not closed",
            "If the generation matches but no admitted identity exists, or if "
            "no stable comparison can be made, the record is "
            "`UNINSPECTABLE`, never the unequal-identity branch",
            "`UNINSPECTABLE` requires `close_attempted=False` and no observed "
            "identity",
            "An `OSError` contributes its bounded errno; a non-`OSError` "
            "contributes `None`",
            "`CLOSE_OUTCOME_UNCERTAIN` requires `close_attempted=True`, a "
            "non-null admitted identity, and an equal pre-close observed "
            "identity. An `OSError` contributes its bounded errno; a "
            "non-`OSError` contributes `None`",
            "retirement-evidence field is `None` when no retirement anomaly "
            "occurred",
            "`DescriptorRetirementError` destination and staging are exact "
            "one-component `PortableRelativePath` values",
            "No other combination is valid",
            "`terminal_result` is an exact `PublicationResult` if and only if "
            "the otherwise-successful publication tuple applies",
            "it is an exact `StagingCleanupResult` if and only if the "
            "otherwise-successful cleanup tuple applies",
            "otherwise it is `None`",
            "Population or sealing uses the standalone form only when the "
            "operation otherwise had no primary H2c1 error",
            "Every pre-yield admission failure remains an exact "
            "`StagingAdmissionError`",
            "There is no standalone admission-stage "
            "`DescriptorRetirementError`",
            "A `publish` operation cannot carry `NOT_COMMITTED`, uncertainty, "
            "or a null result through `DescriptorRetirementError`",
            "A `cleanup` operation cannot carry cleanup uncertainty through "
            "`DescriptorRetirementError`",
            "`context_exit` plus `NOT_COMMITTED` tuple is valid only for the "
            "seal-time-collision origin while its private handle-retirement "
            "batch is still pending",
            "An already-consumed terminal-publication origin cannot create a "
            "new `DescriptorRetirementError` at context exit",
            "its exact runtime type is one of `StagingLifecycleError`, "
            "`StagingAuthorityError`, `PublicationError`, "
            "`PublicationValidationError`, `PublicationCapabilityError`, "
            "`PublicationCollisionError`, `StagingAdmissionError`, "
            "`PublicationDurabilityError`, "
            "`PublicationOutcomeUncertainError`, "
            "`PublicationNamespaceConflictError`, "
            "`PublicationNamespaceUncertainError`, or "
            "`StagingCleanupError`, not a hostile subclass",
            "H2c1's private internal error allocator created it",
            "its private non-exported provenance binds it to the exact staging "
            "handle or provisional staging context currently being retired",
            "its `retirement_evidence` field is still `None`",
            "the private provenance remains inaccessible through the "
            "supported public API and cannot itself authorize filesystem "
            "mutation",
            "populates the internal backing slot exactly once",
            "Callers cannot assign the slot, and no second attachment is "
            "permitted",
            "re-raises the same error object, and preserves its exact subtype "
            "and all existing evidence",
            "Otherwise H2c1 does not mutate the body exception",
            "first member of one ordered `BaseExceptionGroup`",
            "`DescriptorRetirementError` as the second member",
            "an error from another handle or context, a caller-created exact "
            "H2c1 error lacking matching provenance, a hostile subclass, an "
            "error whose retirement field is already non-null, and every "
            "arbitrary non-H2c1 `BaseException`",
            "Earlier retirement evidence is never overwritten, merged, or "
            "discarded",
            "Logical H2c1 authority becomes permanently unavailable before "
            "kernel cleanup",
            "A generation mismatch is foreign state and is preserved without "
            "descriptor inspection and with `observed_identity=None`",
            "A matching generation without an admitted identity, or without "
            "a stable comparison, is `UNINSPECTABLE`",
            "Any exception from `close`, including `EBADF`, `EINTR`, or "
            "`EIO`, is an uncertain close outcome",
            "close is never blindly retried",
            "no post-close operation uses that descriptor number",
            "All remaining independently owned descriptors are processed "
            "after one anomaly",
            "admission or abandonment owns at most two descriptors",
            "proven publication has at most three descriptors",
            "proven cleanup has at most four descriptors",
            "nested traversal or cleanup has at most seven simultaneously "
            "owned H2c1 descriptors",
            "native rename owns no descriptor",
            "exact live `TrustedRoot` is borrowed throughout",
            "Private acquisition generations detect H2c-managed same-inode "
            "descriptor reuse",
            "POSIX metadata cannot distinguish unrestricted external "
            "same-number, same-inode ABA",
            "unrestricted raw descriptor manipulation remains outside the "
            "frozen threat boundary",
            "Detectably foreign descriptors are preserved",
            "physical descriptor lifetime uncertain, but logical H2c1 "
            "authority remains permanently retired",
            "never rewrites an already established `PublicationState`, "
            "`StagingCleanupState`, or `StagingState`",
            "durable publication remains `COMMITTED_DURABLE` and the handle "
            "remains `PUBLISHED`",
            "committed durability uncertainty remains unchanged and "
            "`PUBLISHED`",
            "namespace conflict or namespace uncertainty remains its original "
            "exact subtype and `PUBLISHED`",
            "commit-outcome uncertainty remains unchanged and `RETIRED`",
            "durable cleanup remains `DISCARDED_DURABLE` and the handle "
            "remains `DISCARDED`",
            "cleanup durability or outcome uncertainty retains its already "
            "frozen outcome and lifecycle",
            "proven `NOT_COMMITTED` outcome remains `NOT_COMMITTED`",
            "A matching internally created exact H2c1 exception with empty "
            "retirement evidence remains the same catchable subtype",
            "its publication, cleanup, collision, durability, namespace, and "
            "recovery evidence remains unchanged",
            "retirement evidence attaches orthogonally",
            "otherwise-normal durable publication raises "
            "`DescriptorRetirementError` carrying the exact "
            "`PublicationResult`",
            "otherwise-normal durable cleanup raises the same error carrying "
            "its exact `StagingCleanupResult`",
            "return only after every required owned descriptor has been "
            "successfully and verifiably retired",
            "A population or sealing retirement anomaly with no primary H2c1 "
            "error uses the standalone `DescriptorRetirementError` tuple "
            "frozen above",
            "Normal abandonment with a retirement anomaly uses "
            '`operation="context_exit"` and `state=RETIRED`',
            "Pre-yield admission never raises a standalone "
            "`DescriptorRetirementError`",
            "its exact `StagingAdmissionError` remains primary and receives "
            "retirement evidence only through matching private provenance",
            "A matching internally created exact H2c1 exception with empty "
            "retirement evidence remains the same catchable subtype",
            "Every exception that fails the private provenance conditions "
            "remains unmodified",
            "Multiple descriptor anomalies aggregate into one bounded "
            "retirement-evidence value",
            "Python's unraisable-exception mechanism",
            "After either private origin's retirement batch is consumed, a "
            "proven `NOT_COMMITTED` handle retains its single cleanup "
            "authorization",
            "Later cleanup opens fresh descriptors through the same exact "
            "live `TrustedRoot`",
            "revalidates the complete immutable ledger and namespace",
            "never reuses, revives, retries, or adopts retired descriptor "
            "state",
            "Publication cannot be retried",
            "admitted handle descriptor batch remains pending, and the "
            "collision error is delivered without consuming that batch",
            "context exit before cleanup runs the frozen three-branch "
            "pending-batch matrix below",
            "`cleanup_owned_staging()` passes its pre-attempt lifecycle and "
            "bound-root admission before context exit",
            "consumes and detaches the pending handle batch exactly once "
            "before opening fresh cleanup-operation descriptors",
            "pre-attempt lifecycle or authority rejection leaves the batch "
            "pending and consumes no cleanup authorization",
            "terminal publication path consumes the complete handle and "
            "operation retirement batch before delivering its "
            "`PublicationError`",
            "any retirement anomaly attaches to that exact publication error "
            "under the frozen private-provenance rule",
            "Context exit performs no descriptor inspection, close, or "
            "retirement attempt for that already-consumed batch",
            "Without a body exception it returns normally",
            "with a body exception it propagates that exception unchanged",
            "Every private `NOT_COMMITTED` retirement batch changes privately "
            "and irreversibly from `pending` to `consumed`",
            "no batch may be consumed twice",
            "marker is private, immutable to callers, non-exported, "
            "nonserializable, and carries no filesystem authority",
            "adds no public state, property, enum, result field, evidence "
            "field, or API",
            "Both origins expose only the existing public `NOT_COMMITTED` "
            "lifecycle state and preserve identical ledger and cleanup "
            "authorization",
            "Context exit consults only private lifecycle provenance and "
            "never guesses from descriptor numbers",
            "Descriptor-number reuse cannot recreate a consumed batch",
            "Cleanup always uses fresh descriptors through the same exact "
            "live `TrustedRoot`",
            "Cleanup authorization is consumed when cleanup begins",
            "failed or partial cleanup permanently retires the handle",
            "`OPEN` or `SEALED` abandonment preserves staging, enters "
            "`RETIRED`, and consumes retirement once",
            "An anomaly-free exit with a body exception propagates that body "
            "exception unchanged",
            "the provenance-aware attachment or grouping rule preserves the "
            "body exception as primary",
            "A `NOT_COMMITTED` context exit never silently suppresses a "
            "retirement anomaly",
            "It performs no cleanup and permits no descriptor-retirement "
            "retry",
            "`NOT_COMMITTED` context exit consults only private lifecycle "
            "provenance",
            "pending seal-time-collision batch follows the exact three-branch "
            "matrix above",
            "while consuming that batch exactly once",
            "already-consumed terminal-publication batch follows the exact "
            "no-second-retirement behavior above",
            "`PUBLISHED` and `DISCARDED` context exit performs no second "
            "publication, cleanup, descriptor inspection, or close attempt",
            "`RETIRED` context exit is a no-op",
            "garbage collection never publishes, deletes, or retries "
            "descriptor close",
            "finalization permanently detaches remaining logical authority",
            "collection of a `NOT_COMMITTED` handle loses its cleanup "
            "authorization",
            "descriptor-number reuse cannot revive a handle or trigger "
            "another close",
        )
        for statement in prose_contract:
            with self.subTest(statement=statement):
                self.assertIn(_normalized_text(statement), normalized_adr)

        incompatible_contract = (
            "a generation mismatch may carry a non-null observed identity",
            "one descriptor may occupy multiple retirement roles",
            "role may change because the descriptor is later used for another "
            "check",
            "a standalone admission-stage `DescriptorRetirementError`",
            "`publish` may carry `NOT_COMMITTED` through "
            "`DescriptorRetirementError`",
            "`cleanup` may carry cleanup uncertainty through "
            "`DescriptorRetirementError`",
            "any existing H2c1 exception receives retirement evidence",
            "a caller-created H2c1 error receives retirement evidence",
            "an unrelated H2c1 error receives retirement evidence",
            "a hostile subclass receives retirement evidence",
            "existing retirement evidence may be overwritten",
            "`NOT_COMMITTED` retirement failure returns normally",
            "a provisional admission descriptor has no valid retirement role",
            "a provisional admission descriptor uses an operation or "
            "traversal role",
            "terminal-publication retirement is attempted again at context "
            "exit",
            "a pending seal-time retirement batch may be silently skipped",
            "the pending or consumed retirement marker is exposed through the "
            "supported public API",
            "staging file/root descriptor retained by the public handle from "
            "successful staging admission",
            "publication-parent descriptor retained by the public handle from "
            "successful staging admission",
            "included regular-file leaf beneath a staged directory during "
            "validation, flushing, or cleanup",
            "descendant directory beneath the staged root during validation, "
            "flushing, or cleanup",
            "stable immediate-parent authority for verifying or removing a "
            "descendant traversal entry",
        )
        for statement in incompatible_contract:
            with self.subTest(incompatible=statement):
                self.assertNotIn(_normalized_text(statement), normalized_adr)
        self.assertNotIn('"staging_admission",', adr)

    def test_h2c1_is_implemented_while_h2c2_is_deferred(self) -> None:
        adr = (
            REPO_ROOT
            / "docs"
            / "decisions"
            / "ADR-0023-hpc-safety-primitives.md"
        ).read_text(encoding="utf-8")
        headline_adr = (
            REPO_ROOT
            / "docs"
            / "decisions"
            / "ADR-0022-headline-hpc-execution-contract.md"
        ).read_text(encoding="utf-8")
        normalized_adr = _normalized_text(adr)

        prose_contract = (
            "H2c1 atomic no-replace publication implemented by this working "
            "snapshot",
            "Hosted H2c1 acceptance and H2c2-H2d remain pending",
            "freezes the four top-level H2 boundaries and the detailed H2a, "
            "H2b, H2d, and H2c1 contracts",
            "Detailed H2c2 semantics remain separately pending",
            "implemented by this working snapshot; hosted acceptance remains "
            "pending until the eventual commit's CI succeeds",
            "H2c2 — exclusive claims",
            "a separately reviewed pending gate",
            "detailed API, acquisition, release, durability, and recovery "
            "semantics are frozen only after H2c1 is implemented and "
            "hosted-CI-green",
            "H2c remains pending and incomplete until both H2c1 and H2c2 pass",
            "H2a -> H2b -> H2c -> H2d -> H3 order",
            "H2d and H3 remain blocked",
            "no runtime workflow consumes the implemented H2c1 primitives",
            "exact live `TrustedRoot` representing the publication or claim "
            "parent",
            "owned by the effective user, provide owner read, write, and "
            "search access, not be group- or other-writable",
            "Every H2c guarantee that relies on exclusive mutation "
            "authority—not only deletion—requires a caller-controlled parent "
            "and cooperative writers",
            "malicious or uncooperative same-UID process",
            "ACL-authorized writer",
            "changing staged content, adding case aliases, replacing entries, "
            "or manipulating cleanup targets",
            "detects observable drift but does not claim prevention or "
            "complete detection",
            "Native exact-name no-replace behavior is supplied by the "
            "operating-system publication syscall",
            "The exact live `TrustedRoot` supplied to "
            "`publish_completed_file`, `publish_completed_directory`, or "
            "`cleanup_owned_staging` must be the same authority bound to the "
            "staging handle",
            "Every authority and portable-path argument, and every "
            "staging-handle argument, uses exact-type admission; hostile "
            "subclasses are rejected",
            "Every handle is exact and opaque",
            "Every result, identity, and evidence value is exact, immutable, "
            "and read-only",
            "These properties are observational only",
            "Direct construction, subclass admission, copying, and "
            "serialization are rejected",
            "failure objects are catchable through the frozen hierarchy",
            "transport fields are read-only",
            "Every chunk must be exact `bytes`",
            "handle partial writes and `EINTR`, reject zero progress",
            "total file sizes within the signed-64 range",
            "bounded by the frozen H2a/H2b item and portable-path limits",
            "Every supplied path is an exact `PortableRelativePath`",
            "directory parents are created only by an explicit `mkdir`",
            "No raw descriptor or host path is exposed",
            "No caller-created entry is adopted",
            "no callback or boolean may assert ownership",
            "regular files as exact mode `0600` or `0700` and directories as "
            "exact mode `0700`",
            "does not trust `umask`",
            "changes only the owner execute bit",
            "never auto-publish or auto-delete through context exit or "
            "garbage collection",
            "return private context-manager objects",
            "Their creation is lazy: calling either factory opens or creates "
            "nothing",
            "yield a handle only after stable staging admission",
            "first authoritative destination and case-alias scan occurs at "
            "`seal`",
            "deterministic reservation already exists before this invocation "
            "creates anything",
            "`PublicationCollisionError` with `NOT_COMMITTED`, the validated "
            "destination, and `evidence=None`",
            "performs no mutation",
            "stable admission fails after this invocation creates the "
            "reservation but before yielding",
            "preserves the entry, permanently retires the provisional "
            "authority, returns no handle",
            "raises `StagingAdmissionError`",
            "may carry `evidence=None` only when no stable provisional "
            "identity could be obtained",
            "Non-null pre-yield admission evidence records destination and "
            "namespace observations as `not_attempted`",
            "parent `fsync` as `not_attempted`",
            "seal-time scan finds an existing exact destination or "
            "differently spelled case alias",
            "handle enters `NOT_COMMITTED`",
            "handle retains its immutable ledger and one cleanup "
            "authorization",
            "admitted handle descriptor batch remains pending",
            "collision error is delivered without consuming that batch",
            "Normal or exceptional context exit from `OPEN` or `SEALED` is "
            "abandonment",
            "runs the frozen descriptor-retirement protocol once, enters "
            "`RETIRED`, and preserves the staging entry as recovery evidence",
            "Context exit from `NOT_COMMITTED` follows the private origin "
            "partition above",
            "runs retirement only for a pending seal-time batch",
            "performs no second retirement for a terminal-publication batch "
            "already consumed",
            "Finalization may perform only the same best-effort descriptor "
            "retirement",
            "Correctness never depends on garbage collection",
            "Every terminal publish or cleanup path retires all live "
            "descriptor authority owned by the staging handle",
            "`PUBLISHED` and `DISCARDED` remain the truthful lifecycle labels "
            "after that descriptor retirement",
            "publication attempt proven `NOT_COMMITTED` logically retires its "
            "live descriptor authority and permanently loses publication "
            "authority",
            "retains only its immutable staging identity and ledger as "
            "authorization for the one permitted cleanup attempt",
            "cleanup reopens and revalidates the entry descriptor-relatively "
            "through the same exact `TrustedRoot`",
            "proven commit remains `PUBLISHED` after descriptor retirement",
            "namespace verification remains uncertain",
            "proven durable cleanup remains `DISCARDED`",
            "Abandoned `OPEN` or `SEALED` handles",
            "staging authority becomes foreign, contradictory, or "
            "uninspectable",
            "commit-outcome-uncertain handles",
            "all failed or non-durable cleanup handles are permanently "
            "`RETIRED`",
            "proven `NOT_COMMITTED` handle retains only its one cleanup "
            "authorization",
            "Context exit after any terminal operation performs no second "
            "publish, cleanup, or other filesystem mutation or second "
            "descriptor-retirement attempt",
            "terminal retirement batch detaches every owned descriptor slot "
            "and consumes every retirement attempt",
            "context exit has no residual descriptor authority to inspect or "
            "close",
            "Context exit after a terminal operation performs no second "
            "publish, cleanup, filesystem mutation, descriptor inspection, "
            "close, or retirement attempt",
            "terminal result or failure is delivered only after the "
            "retirement batch has consumed every owned descriptor attempt",
            "terminal publication proven `NOT_COMMITTED` is the sole "
            "exception among terminal publication and cleanup outcomes",
            "complete terminal retirement batch is consumed, but its one "
            "cleanup authorization remains",
            "seal-time collision is not a terminal publication",
            "admitted handle batch remains pending until an admitted cleanup, "
            "context exit, or finalization consumes it",
            "Finalization follows the already frozen unraisable-error and "
            "cleanup-authorization-loss rules",
            "Neither a terminal nor a retired handle can be revived by reuse "
            "of a former descriptor number",
            "Exclusive creation enters `OPEN`",
            "successful writes and directory creation remain `OPEN`",
            "successful sealing enters `SEALED`",
            "publication attempt proven not committed enters `NOT_COMMITTED`",
            "proven commit enters `PUBLISHED`",
            "successful durable cleanup enters `DISCARDED`",
            "foreign, replaced, contradictory, abandoned, or uninspectable "
            "authority enters `RETIRED`",
            "Cleanup failure or partial cleanup also enters `RETIRED`",
            "Population and sealing operations are permitted only in `OPEN`",
            "sealing ends all content mutation",
            "Publication is permitted only from `SEALED`",
            "each handle gets at most one publication attempt",
            "`NOT_COMMITTED` cannot publish again but remains eligible for "
            "cleanup",
            "`PUBLISHED`, `DISCARDED`, and `RETIRED` are terminal",
            "Repeated publication or cleanup after a terminal state is "
            "rejected without filesystem mutation",
            "invalid for the current lifecycle state raises "
            "`StagingLifecycleError` before filesystem access",
            "consumes no publication or cleanup attempt",
            "leaves the handle unchanged",
            "already `PUBLISHED` handle therefore remains and reports "
            "`PUBLISHED`, never `NOT_COMMITTED`",
            "Exact-type violations remain `TypeError`",
            "bound-root validation precedes lifecycle admission and does not "
            "rewrite lifecycle state",
            "different, closed, retired, or nonmatching exact `TrustedRoot`",
            "raises `StagingAuthorityError` before filesystem access",
            "`StagingLifecycleError` and `StagingAuthorityError` are "
            "pre-attempt invocation-admission rejections",
            "not publication or cleanup outcome failures",
            "outside the post-admission publication-evidence rule",
            "No terminal operation is silently retried",
            "Before rename, the staging entry is the cooperative case-alias "
            "reservation",
            "reservation key is the lowercase ASCII destination leaf encoded "
            "as exact ASCII bytes",
            "`Foo` and `foo` use the same cooperative reservation",
            "first authoritative descriptor-relative parent enumeration "
            "occurs during `seal`, while the reservation is held",
            "successful native rename atomically moves the staged inode to "
            "the destination",
            "staging leaf is then expected to be absent",
            "destination becomes the cooperative namespace authority",
            "Another compliant contender may subsequently recreate the "
            "staging leaf",
            "Post-publication validation checks the destination identity and "
            "alias set; it does not require the old staging name to remain "
            "present",
            "foreign newly created staging entry present at the first "
            "authoritative post-call source observation",
            "does not change a commit already proven by the original inode at "
            "the destination",
            "complete `.rp-stage-v1-` prefix is reserved for H2c internals",
            "destination whose lowercase ASCII spelling begins with that "
            "prefix is rejected",
            "Canonical inventory equality alone cannot detect replacement by "
            "a different inode containing identical bytes",
            "noncanonical internal ledger for every staged file and directory",
            "device, inode, entry type, link count, effective ownership, "
            "complete relevant mode, size, and stability timestamps",
            "Before and after every file or directory flush",
            "compares the opened descriptor with the ledger",
            "compares the still-named parent entry with the same ledger",
            "Same-content inode replacement",
            "canonical equality is never used alone as mutation authority",
            "H2c v1 uses standard successful `fsync` semantics on Linux and "
            "macOS",
            "Every protocol-required regular-file, staged-directory, and "
            "publication-parent `fsync` returned successfully, with required "
            "identity checks also passing",
            "required identity checks are the destination, pinned "
            "publication-parent, and flush-target checks",
            "Post-commit source-reservation and case-alias observations are "
            "namespace verification transported separately",
            "already proven durability state remains `COMMITTED_DURABLE`",
            "not a promise against physical power loss, controller or "
            "drive-cache behavior, dishonest storage firmware, "
            "network-filesystem behavior",
            "H2c v1 does not require or invoke `F_FULLFSYNC`",
            "deepest-first directory flushes, excluding the staged root",
            "separate ledger comparison before and after the staged-root "
            "flush",
            "identical canonical reinventory and an identical safety ledger",
            "A `PublicationResult` exists only with `COMMITTED_DURABLE`",
            "destination identity equals the admitted staging identity",
            "A `StagingCleanupResult` exists only with `DISCARDED_DURABLE`",
            "discarded identity is the admitted staging identity proven "
            "removed",
            "All numeric fields require exact `int`, never `bool`",
            "Device, inode, and owner identities are from `0` through "
            "`2**64 - 1`",
            "link count is from `1` through `2**63 - 1`",
            "complete mode is from `0` through `0o177777`",
            "size is from `0` through `2**63 - 1`",
            "observed foreign identities use the complete finite entry-type "
            "list without opening a symlink or special entry",
            "`native_errno` is `None` or an exact integer from `1` through "
            "`2**31 - 1`",
            "`remaining_expected_entries` is `None` unless the original "
            "staging root and known expected ledger residue are proven",
            "exact integer from `0` through the frozen 100,000-item bound",
            "`not_attempted`, absent, or uninspectable observations carry "
            "`None`",
            "`not_attempted` is permitted only in pre-yield "
            "`StagingAdmissionError` evidence",
            "every post-admission terminal publication evidence forbids it",
            "For publication evidence, `exact` requires the observed identity "
            "to equal the staging identity",
            "`absent`, `uninspectable`, and `contradictory` require `None`",
            "stable `foreign` or `replaced` observations require a non-null "
            "identity unequal to the staging identity",
            "For cleanup evidence, `exact` likewise requires the observed "
            "root to equal the staging identity",
            "`malformed` carries the stable observed identity when "
            "descriptor-relative metadata is available",
            "Root identity and membership state are represented only by "
            "`exact`, `owned_partial`, `absent`, `foreign`, `replaced`, "
            "`contradictory`, `malformed`, or `uninspectable`",
            "Sibling case aliases are represented only by "
            "`NamespaceEvidence`",
            "original unchanged root with a conflicting alias uses root "
            "observation `exact` plus conflict namespace evidence",
            "owned partial root with a conflicting alias uses "
            "`owned_partial` plus conflict namespace evidence",
            "foreign, replaced, malformed, or uninspectable root remains "
            "represented independently",
            "`owned_partial` carries the current observed root identity",
            "stable root authority is equality of device, inode, entry type, "
            "and owner UID",
            "mode must remain exactly equal to the admitted safe staging-root "
            "mode",
            "cleanup authorizes no `chmod`",
            "Only link count, size, and applicable stability metadata may "
            "differ",
            "only when the difference is explained by removals this cleanup "
            "attempt has already proven",
            "Every remaining entry must be an expected ledger member",
            "every removed entry must be proven absent",
            "`NamespaceEvidence` is immutable and bounded",
            "`not_attempted`: empty alias tuple, count `None`, and "
            "`aliases_complete` exact `False`",
            "`no_conflict`: empty alias tuple, count `0`, and "
            "`aliases_complete` exact `True`",
            "`complete_conflict`: a nonempty canonical alias tuple of at most "
            "100,000 entries, an exact count equal to its length, and "
            "`aliases_complete` exact `True`",
            "`bounded_conflict`: exactly 100,000 canonical alias entries",
            "count `None`, and `aliases_complete` exact `False`",
            "`uninspectable`: only the zero through 100,000 safely observed "
            "canonical alias entries, count `None`, and `aliases_complete` "
            "exact `False`",
            "Every alias is an exact one-component `PortableRelativePath`",
            "exact-spelling unique, different from the destination or staging "
            "spelling",
            "lowercase-ASCII-equivalent to it, and sorted by portable-path "
            "bytes",
            "no raw descriptor, host path, credential, caller- or OS-supplied "
            "free-form text, mutable mapping, or unbounded collection",
            "staging identity is distinct from later observations",
            "Pre-admission validation, capability, or reservation-collision "
            "failures may carry `evidence=None`",
            "failed staging admission may carry `None` only when no stable "
            "provisional identity could be obtained",
            "Destination is non-null once exact `PortableRelativePath` "
            "validation succeeds",
            "before that validation it may be `None`",
            "Every post-admission publication failure carries non-null "
            "bounded recovery evidence and the exact destination",
            "Validated destination identity is exposed through "
            "`evidence.observed_destination_identity`",
            "exception messages are not authoritative evidence",
            "Terminal lifecycle rejection is represented by "
            "`StagingLifecycleError`, not by a publication or cleanup outcome",
            "`StagingAdmissionError` always carries `NOT_COMMITTED`",
            "`entry_may_remain` is an exact `bool`",
            "`PublicationValidationError`, `PublicationCapabilityError`, and "
            "`PublicationCollisionError` carry `NOT_COMMITTED`",
            "capability and collision remain distinguishable",
            "`PublicationDurabilityError` carries `NOT_COMMITTED` for a "
            "proven precommit sync failure",
            "`COMMITTED_DURABILITY_UNCERTAIN` after a proven commit",
            "`PublicationOutcomeUncertainError` carries "
            "`COMMIT_OUTCOME_UNCERTAIN`",
            "`PublicationNamespaceConflictError` carries only "
            "`COMMITTED_DURABLE` or `COMMITTED_DURABILITY_UNCERTAIN`",
            "`PublicationNamespaceUncertainError` carries only the same two "
            "committed states",
            "any proven-commit post-commit source/namespace verification "
            "anomaly from matrix case 1b or 1c",
            "may therefore carry `no_conflict` alias evidence alongside "
            "anomalous source evidence",
            "Commit/durability identity and namespace-scan completeness are "
            "transported separately",
            "Namespace failure never changes a proven commit to "
            "`COMMIT_OUTCOME_UNCERTAIN`",
            "normal `PublicationResult` requires both `COMMITTED_DURABLE` and "
            "complete `no_conflict` namespace evidence",
            "with no source or namespace anomaly",
            "Every `StagingCleanupError` carries exactly one of "
            "`NOT_DISCARDED`, `DISCARDED_DURABILITY_UNCERTAIN`, or "
            "`DISCARD_OUTCOME_UNCERTAIN`",
            "Only durable success returns normally",
            "no uncertain or non-durable outcome is converted into a normal "
            "result",
            "Publication outcome observations begin with the first anchored "
            "observation after the native publication call returns or fails",
            "Pre-call ledger, source, destination, parent, and alias checks "
            "are preconditions",
            "not the temporal baseline for post-call `foreign` versus "
            "`replaced`",
            "Pre-call identity drift fails before the native call",
            "is not labeled as a post-call outcome",
            "`absent` means the name is authoritatively absent",
            "`foreign` means a stable unequal identity is present at the first "
            "authoritative post-call outcome observation of that name",
            "`replaced` means identity drift between required post-call "
            "outcome observations after the first post-call state was "
            "recorded",
            "Destination identity has the highest priority",
            "destination carrying the original staged identity proves commit",
            "both names authoritatively carry the original single-link "
            "identity is contradictory",
            "foreign staging reservation present at the first authoritative "
            "post-call source observation does not erase a commit",
            "first post-call observation records source absence or another "
            "stable state and a later required outcome observation sees a "
            "different entry",
            "later observation is `replaced`",
            "When the native call may have committed, H2c1 attempts "
            "publication-parent `fsync`",
            "Commit evidence and sync evidence are carried separately",
            "differently spelled alias appearing after the original staged "
            "identity reaches the destination does not make the rename "
            "occurrence uncertain",
            "Known complete or bounded conflict raises "
            "`PublicationNamespaceConflictError`",
            "post-commit namespace scan that cannot be completed reliably",
            "raises `PublicationNamespaceUncertainError`",
            "Each carries its bounded namespace evidence and the proven "
            "publication state",
            "Neither entry is rolled back",
            "Useful staged files and directories are expected to be nonempty",
            "exact unchanged staged regular file",
            "exact staged directory tree matching its complete owned ledger",
            "rejects every unexpected or additional entry",
            "every expected entry already absent before this cleanup proves "
            "its removal",
            "every replaced, special, unsafe-mode, or otherwise malformed "
            "membership state",
            "every conflicting or incomplete namespace observation",
            "deletes only expected files and directories "
            "descriptor-relatively, deepest-first",
            "immediate identity revalidation before every removal",
            "stops at the first anomaly",
            "flushes every affected directory and the publication parent",
            "Only `DISCARDED_DURABLE` returns normally",
            "`NOT_DISCARDED` applies only when the root observation is "
            "`exact` or `owned_partial`",
            "stable root authority is proven",
            "every remaining member is known expected ledger residue",
            "every removed member is proven absent",
            "alias enumeration is complete and conflict-free",
            "mere existence of the staging leaf is not ownership evidence",
            "Cleanup `malformed` includes the original stable root containing "
            "any unexpected or additional entry",
            "expected entry already absent before this cleanup proves its "
            "removal",
            "special entry, or an unsafe mode or membership state",
            "same-named but replaced, foreign, contradictory, malformed, or "
            "uninspectable root also maps to "
            "`DISCARD_OUTCOME_UNCERTAIN`",
            "Simultaneously observed sibling aliases remain encoded only in "
            "the separate namespace evidence",
            "staging root proven removed followed by successful "
            "publication-parent `fsync` is `DISCARDED_DURABLE`",
            "proven removal followed by parent-`fsync` failure is "
            "`DISCARDED_DURABILITY_UNCERTAIN`",
            "Known partial expected residue under the proven original root "
            "remains `NOT_DISCARDED` only through the `owned_partial` "
            "observation",
            "Nonempty or incomplete alias evidence, or uninspectable alias "
            "enumeration, maps to `DISCARD_OUTCOME_UNCERTAIN`",
            "Only complete `no_conflict` namespace evidence permits "
            "`NOT_DISCARDED`",
            "Any non-durable or failed cleanup permanently retires the handle",
            "There is no cleanup retry or handle reconstruction",
            "immutable canonical claim record",
            "Descriptor-relative exclusive `mkdir` remains the atomic "
            "ownership point",
            "exact owner, record, claim-parent, and claim-directory identity",
            "missing, partial, malformed, stale, foreign, replaced, aliased, "
            "or nonempty claim blocks",
            "no automatic stale deletion, adoption, renewal, replacement, "
            "expiry, or forced cleanup",
            "Deterministic competing-claim tests use barriers rather than "
            "timing sleeps",
            "Detailed H2c2 acquisition, release, durability, tombstone, retry, "
            "and recovery semantics remain pending",
            "claim acquisition depends on H2c1's final publication and result "
            "authority",
            "known partial release must be distinguished from a genuinely "
            "uncertain release",
            "parent durability failure after `rmdir` requires a separately "
            "frozen cross-process recovery or tombstone decision",
            "full device/inode/type/link/mode fingerprint",
            "exact builder and acquisition signatures and canonical digest "
            "serialization require the focused H2c2 design gate",
            "does not freeze H2c2 public API names",
            "H2c2 remains pending, unimplemented, and unexported",
            "H2c1 no-replace publication is implemented by this working "
            "snapshot",
            "H2c2 exclusive claims and H2d receipt envelopes/complete H2 "
            "acceptance remain pending",
            "H2c is incomplete until both of its internal gates are committed "
            "and hosted-CI-green",
            "H2 is therefore incomplete and H3 remains blocked",
            "None of the implemented H2a/H2b/H2c1 foundations provides "
            "runtime provisioning",
            "claim promotion, export, tagging, or release publication",
            "No fake-remote or live-cluster validation has occurred",
            "The implemented H2c1 publication matrix injects failure before "
            "write",
            "Hosted Linux and macOS acceptance remains pending until the "
            "eventual commit's CI succeeds",
            "H2a, H2b, and H2c1 create no remote or workflow capability and "
            "change no current runtime",
        )
        for statement in prose_contract:
            with self.subTest(statement=statement):
                self.assertIn(_normalized_text(statement), normalized_adr)

        self.assertLess(
            adr.index("### H2c1 no-replace publication contract"),
            adr.index("### H2c2 exclusive-claim contract"),
        )
        self.assertLess(
            adr.index("### H2c2 exclusive-claim contract"),
            adr.index("### H2d receipt-envelope foundation"),
        )
        exact_fenced_contracts = {
            "The H2c1 module-level entry points are exactly:": (
                "open_exclusive_staged_file\n"
                "open_exclusive_staged_directory\n"
                "publish_completed_file\n"
                "publish_completed_directory\n"
                "cleanup_owned_staging"
            ),
            "The public authority, value, and failure names are "
            "exactly:": (
                "PublicationState\n"
                "StagingState\n"
                "StagingCleanupState\n"
                "PublicationResult\n"
                "StagingCleanupResult\n"
                "PublicationEntryIdentity\n"
                "NamespaceEvidence\n"
                "PublicationRecoveryEvidence\n"
                "StagingCleanupRecoveryEvidence\n"
                "StagedFileHandle\n"
                "StagedDirectoryHandle\n"
                "StagingLifecycleError\n"
                "StagingAuthorityError\n"
                "PublicationError\n"
                "PublicationValidationError\n"
                "PublicationCapabilityError\n"
                "PublicationCollisionError\n"
                "StagingAdmissionError\n"
                "PublicationDurabilityError\n"
                "PublicationOutcomeUncertainError\n"
                "PublicationNamespaceConflictError\n"
                "PublicationNamespaceUncertainError\n"
                "StagingCleanupError\n"
                "DescriptorRetirementObservation\n"
                "DescriptorRetirementIdentity\n"
                "DescriptorRetirementRecord\n"
                "DescriptorRetirementEvidence\n"
                "DescriptorRetirementError"
            ),
            "The exact read-only lifecycle observations are:": (
                "StagedFileHandle.state -> StagingState\n"
                "StagedDirectoryHandle.state -> StagingState"
            ),
            "The handles expose only": (
                "StagedFileHandle.write(chunk: bytes) -> None\n"
                "StagedFileHandle.seal(*, executable: bool = False) -> None\n"
                "\n"
                "StagedDirectoryHandle.mkdir("
                "path: PortableRelativePath) -> None\n"
                "StagedDirectoryHandle.write_file(\n"
                "    path: PortableRelativePath,\n"
                "    chunks: Iterable[bytes],\n"
                "    *,\n"
                "    executable: bool = False,\n"
                ") -> None\n"
                "StagedDirectoryHandle.seal(*, scope: str) -> None"
            ),
            "Context entry performs, in order:": (
                "1. validate exact arguments, the parent authority, and "
                "required capabilities\n"
                "2. attempt exclusive creation of the deterministic staging "
                "reservation\n"
                "3. verify the opened and still-named staging identity\n"
                "4. yield a handle only after stable staging admission"
            ),
            "One pinned publication parent contains both direct sibling leaf "
            "entries:": (
                "<publication-parent>/<destination-leaf>\n"
                "<publication-parent>/.rp-stage-v1-<alias-digest>"
            ),
            "The staging lifecycle states are:": (
                "OPEN\n"
                "SEALED\n"
                "NOT_COMMITTED\n"
                "PUBLISHED\n"
                "DISCARDED\n"
                "RETIRED"
            ),
            "Cleanup is permitted only from:": (
                "OPEN\nSEALED\nNOT_COMMITTED"
            ),
            "The alias digest is the H2a domain-separated SHA-256:": (
                "SHA256(\n"
                "  PUBLICATION_STAGING_DOMAIN\n"
                "  || uint64be(len(reservation_key))\n"
                "  || reservation_key\n"
                ")"
            ),
            "The only H2c1 native backends and flags are:": (
                "Linux:\n"
                "int renameat2(\n"
                "  int olddirfd,\n"
                "  const char *oldpath,\n"
                "  int newdirfd,\n"
                "  const char *newpath,\n"
                "  unsigned int flags\n"
                ")\n"
                "RENAME_NOREPLACE = 1\n"
                "\n"
                "macOS:\n"
                "int renameatx_np(\n"
                "  int fromfd,\n"
                "  const char *from,\n"
                "  int tofd,\n"
                "  const char *to,\n"
                "  unsigned int flags\n"
                ")\n"
                "RENAME_EXCL = 4"
            ),
            "The exact immutable normal-result fields are:": (
                "PublicationResult:\n"
                "    state: PublicationState\n"
                "    destination: PortableRelativePath\n"
                "    destination_identity: PublicationEntryIdentity\n"
                "    namespace_evidence: NamespaceEvidence\n"
                "\n"
                "StagingCleanupResult:\n"
                "    state: StagingCleanupState\n"
                "    staging: PortableRelativePath\n"
                "    discarded_identity: PublicationEntryIdentity\n"
                "    namespace_evidence: NamespaceEvidence"
            ),
            "The exact immutable entry-identity and bounded-evidence fields "
            "are:": (
                "PublicationEntryIdentity:\n"
                "    device: int\n"
                "    inode: int\n"
                '    entry_type: Literal["regular_file", "directory", '
                '"symlink", "fifo", "socket", "character_device", '
                '"block_device", "other"]\n'
                "    link_count: int\n"
                "    owner_uid: int\n"
                "    mode: int\n"
                "    size_bytes: int\n"
                "\n"
                "NamespaceEvidence:\n"
                '    namespace_observation: Literal["not_attempted", '
                '"no_conflict", "complete_conflict", "bounded_conflict", '
                '"uninspectable"]\n'
                "    conflicting_aliases: "
                "tuple[PortableRelativePath, ...]\n"
                "    conflicting_alias_count: int | None\n"
                "    aliases_complete: bool\n"
                "\n"
                "PublicationRecoveryEvidence:\n"
                "    staging_identity: PublicationEntryIdentity\n"
                '    source_observation: Literal["not_attempted", "exact", '
                '"absent", "foreign", "replaced", "contradictory", '
                '"uninspectable"]\n'
                "    observed_source_identity: "
                "PublicationEntryIdentity | None\n"
                '    destination_observation: Literal["not_attempted", '
                '"exact", "absent", "foreign", "replaced", '
                '"contradictory", "uninspectable"]\n'
                "    observed_destination_identity: "
                "PublicationEntryIdentity | None\n"
                "    namespace_evidence: NamespaceEvidence\n"
                '    parent_fsync: Literal["not_attempted", "succeeded", '
                '"failed", "uncertain"]\n'
                "    native_errno: int | None\n"
                "\n"
                "StagingCleanupRecoveryEvidence:\n"
                "    staging_identity: PublicationEntryIdentity\n"
                '    root_observation: Literal["exact", "owned_partial", '
                '"absent", "foreign", "replaced", "contradictory", '
                '"malformed", "uninspectable"]\n'
                "    observed_root_identity: "
                "PublicationEntryIdentity | None\n"
                "    remaining_expected_entries: int | None\n"
                "    namespace_evidence: NamespaceEvidence\n"
                '    parent_fsync: Literal["not_attempted", "succeeded", '
                '"failed", "uncertain"]\n'
                "    native_errno: int | None"
            ),
            "The exact catchable failure hierarchy is:": (
                "StagingLifecycleError -> Exception\n"
                "StagingAuthorityError -> Exception\n"
                "PublicationError -> Exception\n"
                "PublicationValidationError -> PublicationError\n"
                "PublicationCapabilityError -> PublicationError\n"
                "PublicationCollisionError -> PublicationError\n"
                "StagingAdmissionError -> PublicationError\n"
                "PublicationDurabilityError -> PublicationError\n"
                "PublicationOutcomeUncertainError -> PublicationError\n"
                "PublicationNamespaceConflictError -> PublicationError\n"
                "PublicationNamespaceUncertainError -> PublicationError\n"
                "StagingCleanupError -> Exception\n"
                "DescriptorRetirementError -> Exception"
            ),
            "The exact read-only failure transport fields are:": (
                "StagingLifecycleError:\n"
                "    state: StagingState\n"
                '    operation: Literal["write", "mkdir", "write_file", '
                '"seal", "publish", "cleanup"]\n'
                "    retirement_evidence: "
                "DescriptorRetirementEvidence | None\n"
                "\n"
                "StagingAuthorityError:\n"
                "    state: StagingState\n"
                '    operation: Literal["publish", "cleanup"]\n'
                "    retirement_evidence: "
                "DescriptorRetirementEvidence | None\n"
                "\n"
                "PublicationError:\n"
                "    state: PublicationState\n"
                "    evidence: PublicationRecoveryEvidence | None\n"
                "    destination: PortableRelativePath | None\n"
                "    retirement_evidence: "
                "DescriptorRetirementEvidence | None\n"
                "\n"
                "StagingAdmissionError:\n"
                "    staging: PortableRelativePath\n"
                "    entry_may_remain: bool\n"
                "\n"
                "StagingCleanupError:\n"
                "    state: StagingCleanupState\n"
                "    evidence: StagingCleanupRecoveryEvidence\n"
                "    staging: PortableRelativePath\n"
                "    retirement_evidence: "
                "DescriptorRetirementEvidence | None\n"
                "\n"
                "DescriptorRetirementError:\n"
                "    state: StagingState\n"
                "    operation: Literal[\n"
                '        "write",\n'
                '        "mkdir",\n'
                '        "write_file",\n'
                '        "seal",\n'
                '        "publish",\n'
                '        "cleanup",\n'
                '        "context_exit",\n'
                '        "finalization",\n'
                "    ]\n"
                "    destination: PortableRelativePath\n"
                "    staging: PortableRelativePath\n"
                "    terminal_result: PublicationResult | "
                "StagingCleanupResult | None\n"
                "    retirement_evidence: DescriptorRetirementEvidence"
            ),
            "The frozen publication states remain exactly:": (
                "NOT_COMMITTED\n"
                "COMMITTED_DURABLE\n"
                "COMMITTED_DURABILITY_UNCERTAIN\n"
                "COMMIT_OUTCOME_UNCERTAIN"
            ),
            "the proven publication state:": (
                "COMMITTED_DURABLE\nCOMMITTED_DURABILITY_UNCERTAIN"
            ),
            "The source/destination outcome priority is exhaustive:": (
                "1. post-call destination=original-staged-identity -> commit "
                "proven\n"
                "   a. first post-call source=absent-or-foreign -> source "
                "state is consistent with commit\n"
                "   b. later post-call "
                "source=replaced-or-contradictory-or-uninspectable -> "
                "post-commit namespace-verification failure\n"
                "   c. post-call source=original-staged-identity -> "
                "contradictory single-link evidence and post-commit "
                "namespace-verification failure\n"
                "2. post-call destination-not-original and source=exact and "
                "destination=absent -> NOT_COMMITTED\n"
                "3. post-call destination-not-original and source=exact and "
                "destination=foreign -> collision and NOT_COMMITTED\n"
                "4. post-call destination-not-original and every other "
                "absent, foreign, replaced, contradictory, or uninspectable "
                "combination -> COMMIT_OUTCOME_UNCERTAIN"
            ),
            "The frozen cleanup states are:": (
                "NOT_DISCARDED\n"
                "DISCARDED_DURABLE\n"
                "DISCARDED_DURABILITY_UNCERTAIN\n"
                "DISCARD_OUTCOME_UNCERTAIN"
            ),
            "The claim schema is:": (
                "research_platform.hpc.exclusive_claim.v1"
            ),
        }
        for marker, expected in exact_fenced_contracts.items():
            with self.subTest(marker=marker):
                self.assertEqual(_fenced_block_after(adr, marker), expected)

        self.assertEqual(
            _fenced_block_after(
                adr,
                "Their exact signatures are:",
                language="python",
            ),
            "open_exclusive_staged_file(\n"
            "    trusted_root: TrustedRoot,\n"
            "    *,\n"
            "    destination: PortableRelativePath,\n"
            ") -> AbstractContextManager[StagedFileHandle]\n"
            "\n"
            "open_exclusive_staged_directory(\n"
            "    trusted_root: TrustedRoot,\n"
            "    *,\n"
            "    destination: PortableRelativePath,\n"
            ") -> AbstractContextManager[StagedDirectoryHandle]\n"
            "\n"
            "publish_completed_file(\n"
            "    trusted_root: TrustedRoot,\n"
            "    staging: StagedFileHandle,\n"
            ") -> PublicationResult\n"
            "\n"
            "publish_completed_directory(\n"
            "    trusted_root: TrustedRoot,\n"
            "    staging: StagedDirectoryHandle,\n"
            ") -> PublicationResult\n"
            "\n"
            "cleanup_owned_staging(\n"
            "    trusted_root: TrustedRoot,\n"
            "    staging: StagedFileHandle | StagedDirectoryHandle,\n"
            ") -> StagingCleanupResult",
        )

        self.assertEqual(
            _fenced_block_after(
                adr,
                "The exact public enum ABI is:",
                language="python",
            ),
            "class PublicationState(str, Enum):\n"
            '    NOT_COMMITTED = "not_committed"\n'
            '    COMMITTED_DURABLE = "committed_durable"\n'
            "    COMMITTED_DURABILITY_UNCERTAIN = "
            '"committed_durability_uncertain"\n'
            '    COMMIT_OUTCOME_UNCERTAIN = "commit_outcome_uncertain"\n'
            "\n"
            "class StagingState(str, Enum):\n"
            '    OPEN = "open"\n'
            '    SEALED = "sealed"\n'
            '    NOT_COMMITTED = "not_committed"\n'
            '    PUBLISHED = "published"\n'
            '    DISCARDED = "discarded"\n'
            '    RETIRED = "retired"\n'
            "\n"
            "class StagingCleanupState(str, Enum):\n"
            '    NOT_DISCARDED = "not_discarded"\n'
            '    DISCARDED_DURABLE = "discarded_durable"\n'
            "    DISCARDED_DURABILITY_UNCERTAIN = "
            '"discarded_durability_uncertain"\n'
            '    DISCARD_OUTCOME_UNCERTAIN = "discard_outcome_uncertain"',
        )

        self.assertEqual(
            _fenced_block_after(
                adr,
                "The frozen staging domain is:",
                language="python",
            ),
            'PUBLICATION_STAGING_DOMAIN = '
            'b"research-platform:hpc:publication-staging:v1\\0"',
        )

        incompatible_contracts = (
            "this contract-only change freezes its implementation boundary",
            "implementation and hosted acceptance remain pending",
            "not implemented or exported by this contract-only gate",
            "remain unimplemented and unexported by this contract-only "
            "amendment",
            "implemented by this staged snapshot",
            "exported by this staged H2c1 snapshot",
            "H2c1 no-replace publication, H2c2 exclusive claims, and H2d "
            "receipt envelopes/complete H2 acceptance remain pending",
            "This contract-only change freezes H2c1 but does not implement it",
            "The pending H2c1 publication matrix",
            "H2a and H2b create no remote or workflow capability",
            "the staging reservation remains present after successful rename",
            "exclusive staging creation before its authoritative "
            "descriptor-relative parent enumeration",
            "Every completed publish or cleanup path retires all live "
            "descriptor authority",
            "absence of unexpected or nonempty state",
            "If the staging root still exists, including known partial "
            "expected residue",
            "conflicting namespace evidence is `COMMIT_OUTCOME_UNCERTAIN`",
            "Only missing, replaced, contradictory, or uninspectable "
            "source/destination evidence is `COMMIT_OUTCOME_UNCERTAIN`",
            "Every typed publication failure carries its exact state and "
            "bounded recovery evidence",
            "Acquisition proceeds in this exact order",
            "Release then:",
            "research-platform:hpc:exclusive-claim-directory:v1",
            'claim_directory_name = ".rp-claim-v1-"',
            "NOT_RELEASED",
            "RELEASED_DURABLE",
            "RELEASED_DURABILITY_UNCERTAIN",
            "RELEASE_OUTCOME_UNCERTAIN",
        )
        for statement in incompatible_contracts:
            with self.subTest(incompatible=statement):
                self.assertNotIn(_normalized_text(statement), normalized_adr)
        for deferred_api in (
            "build_claim_record",
            "acquire_exclusive_claim",
            "release_exclusive_claim",
        ):
            with self.subTest(deferred_api=deferred_api):
                self.assertNotIn(deferred_api, adr)

        h2_order = tuple(
            headline_adr.index(f"**H2{suffix}:**")
            for suffix in ("a", "b", "c", "d")
        )
        self.assertEqual(h2_order, tuple(sorted(h2_order)))
        full_gate_order = tuple(
            headline_adr.index(f"**H{gate}:**") for gate in range(13)
        )
        self.assertEqual(full_gate_order, tuple(sorted(full_gate_order)))

        self.assertTrue((SAFETY_ROOT / "publication.py").is_file())
        for reserved_module in ("claims.py", "receipts.py"):
            with self.subTest(reserved_module=reserved_module):
                self.assertFalse((SAFETY_ROOT / reserved_module).exists())


if __name__ == "__main__":
    unittest.main()
