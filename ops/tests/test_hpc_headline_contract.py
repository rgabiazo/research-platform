from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
ADR = REPO_ROOT / "docs" / "decisions" / "ADR-0022-headline-hpc-execution-contract.md"
SAFETY_ADR = REPO_ROOT / "docs" / "decisions" / "ADR-0023-hpc-safety-primitives.md"
DECISION_INDEX = REPO_ROOT / "docs" / "decisions" / "README.md"
ARCHITECTURE = REPO_ROOT / "ARCHITECTURE.md"
ROADMAP = REPO_ROOT / "ROADMAP.md"
LEGACY_BLUEPRINT = REPO_ROOT / "BLUEPRINT.md"
CAPABILITIES = REPO_ROOT / "docs" / "capabilities.md"

CONTRACT_DOCUMENTS = (
    ADR,
    SAFETY_ADR,
    DECISION_INDEX,
    ARCHITECTURE,
    ROADMAP,
    CAPABILITIES,
)
PRIVACY_DOCUMENTS = (*CONTRACT_DOCUMENTS, LEGACY_BLUEPRINT)
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
HEADLINE = (
    "Plan-first SLURM execution, monitoring, cancellation, and verified retrieval "
    "for a deterministic synthetic tabular workload using the same "
    "private-overlay/external-root topology as BYOD, validated end to end on one "
    "documented cluster environment."
)
CURRENT_HPC_CLASSIFICATIONS = {
    "Local HPC setup, target inspection, and plan rendering": "Plan/validation only",
    "SSH doctor and remote-data verification": "Experimental or external-runtime",
    "Local recorded HPC status": "Plan/validation only",
    "Live scheduler status": "Experimental or external-runtime",
    "Local cancellation-request rendering": "Plan/validation only",
    "Explicit HPC transfer, bootstrap, submission, and retrieval": (
        "Experimental or external-runtime"
    ),
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def _prose(path: Path) -> str:
    """Normalize prose while removing Markdown's repeated blockquote marker."""

    text = re.sub(r"(?m)^\s*>\s?", "", _text(path)).replace("**", "")
    return " ".join(text.split())


def _require_terms(test_case: unittest.TestCase, text: str, terms: tuple[str, ...]) -> None:
    folded = text.casefold()
    for term in terms:
        if term.casefold() not in folded:
            test_case.fail(f"Missing contract term: {term}")


def _matrix_statuses() -> dict[str, str]:
    text = _text(CAPABILITIES)
    table = text.split("<!-- capability-matrix:start -->", 1)[1].split(
        "<!-- capability-matrix:end -->", 1
    )[0]
    lines = [line for line in table.splitlines() if line.startswith("|")]
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    statuses: dict[str, str] = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        row = dict(zip(headers, cells, strict=True))
        statuses[row["Surface"].replace("`", "")] = row["Status"]
    return statuses


def _assert_local_links_resolve(test_case: unittest.TestCase, document: Path) -> None:
    for raw_target in LINK_PATTERN.findall(_text(document)):
        target = raw_target.strip().strip("<>")
        path_text = target.split("#", 1)[0]
        if not path_text or "://" in path_text or path_text.startswith("mailto:"):
            continue
        resolved = (document.parent / path_text).resolve()
        resolved.relative_to(REPO_ROOT)
        test_case.assertTrue(
            resolved.is_file(),
            f"Broken link in {document.relative_to(REPO_ROOT)}: {raw_target}",
        )


def _public_markdown_documents() -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-c", "--", "*.md"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    relative_paths = set(completed.stdout.splitlines())
    if ROADMAP.is_file():
        relative_paths.add(ROADMAP.relative_to(REPO_ROOT).as_posix())
    return tuple(
        REPO_ROOT / relative
        for relative in sorted(relative_paths)
        if (REPO_ROOT / relative).is_file()
    )


class HpcHeadlineContractTests(unittest.TestCase):
    def test_adr_exists_is_indexed_and_preserves_current_status(self) -> None:
        self.assertTrue(ADR.is_file())
        index = _text(DECISION_INDEX)
        self.assertIn(ADR.name, index)

        text = _prose(ADR)
        self.assertIn("Accepted", text)
        self.assertRegex(text.casefold(), r"implementation (?:is )?incomplete")
        self.assertRegex(text.casefold(), r"no live[- ]cluster (?:has )?(?:yet )?been validated")
        self.assertIn("plan-first", text.casefold())
        self.assertRegex(text.casefold(), r"experimental(?:/| or )external-runtime")
        self.assertIn("future conditional claim", text.casefold())
        self.assertIn("forbidden", text.casefold())
        self.assertIn("fake-remote", text.casefold())
        self.assertIn("live-cluster acceptance", text.casefold())
        for unsupported_claim in (
            "implementation is complete",
            "live-cluster validation is complete",
            "live-cluster validation has passed",
        ):
            self.assertNotIn(unsupported_claim, text.casefold())

        statuses = _matrix_statuses()
        for surface, expected in CURRENT_HPC_CLASSIFICATIONS.items():
            self.assertEqual(statuses.get(surface), expected, surface)

    def test_narrow_headline_and_explicit_exclusions_are_complete(self) -> None:
        text = _prose(ADR)
        self.assertIn(HEADLINE, text)
        _require_terms(self, text, (
            "universal SLURM compatibility",
            "general Alliance or Nibi compatibility",
            "other provider not actually tested",
            "more than the actually tested environment",
            "arbitrary future workflow support",
            "real-data neuroimaging",
            "raw BIDS-to-analysis execution",
            "FSL",
            "ANTs",
            "SPM",
            "DeepPrep",
            "FEAT",
            "fMRIPost-AROMA",
            "containerized neuroimaging validation",
            "remote ROI or MVPA execution",
            "support for multiple providers",
            "PyPI installation",
            "unattended credential or MFA handling",
            "automatic replacement",
            "destructive recovery",
        ))

        self.assertIn("optional", text.casefold())
        self.assertIn("private neuroimaging integration", text.casefold())
        self.assertIn("cannot substitute", text.casefold())
        self.assertIn("deterministic synthetic", text.casefold())
        self.assertIn("no participant or private study data", text.casefold())

    def test_readiness_and_staging_converge_before_submission(self) -> None:
        text = _text(ADR)
        for state in (
            "planned",
            "runtime-planned",
            "provisioning",
            "runtime-ready",
            "transfer-planned",
            "staging",
            "staged",
            "submitting",
            "submitted",
            "queued",
            "running",
            "scheduler-completed",
            "remote-output-verified",
            "retrieval-staging",
            "retrieved",
        ):
            self.assertIn(state, text, state)
        self.assertRegex(text, r"runtime-ready\s*\+\s*staged")
        self.assertRegex(text, r"runtime-ready\s*\+\s*staged[\s\S]{0,160}->\s*submitting")
        self.assertIn("Runtime artifacts may need staging before readiness", text)
        self.assertRegex(
            text,
            r"Scheduler completion is not scientific(?:-output)? success",
        )

        for state in (
            "cancel-requested",
            "cancel-submitted",
            "scheduler-cancelled",
            "cancel-uncertain",
            "stage-failed",
            "provisioning-failed",
            "runtime-not-ready",
            "submission-failed",
            "scheduler-failed",
            "scheduler-timeout",
            "scheduler-out-of-memory",
            "scheduler-preempted",
            "scheduler-node-failure",
            "scheduler-accounting-pending",
            "remote-execution-failed",
            "remote-receipt-missing",
            "remote-output-invalid",
            "retrieval-failed",
            "retrieval-invalid",
        ):
            self.assertIn(state, text, state)
        self.assertIn("cancellation race", text.casefold())
        self.assertRegex(
            text.casefold(),
            r"does not satisfy\s+cancellation acceptance",
        )

    def test_state_invariants_fail_closed_and_preserve_ownership(self) -> None:
        text = _prose(ADR)
        _require_terms(self, text, (
            "monotonic and atomically persisted",
            "COMPLETED",
            "valid remote execution-success receipt",
            "controlled scientific failure",
            "failure receipt",
            "never a success receipt",
            "Abrupt node loss",
            "must never be inferred as success",
            "transfer command returned zero",
            "fails closed",
            "terminal failure",
            "cancellation request",
            "not cancellation confirmation",
            "scheduler-accounting delay",
            "Local and remote receipts bind the same run",
            "silently merges with, replaces, or overwrites",
            "Exact-match reuse or resume is distinct from replacement",
            "exclusive ownership",
            "created atomically",
            "Cleanup is separately authorized",
        ))

    def test_source_identity_receipts_and_publishable_evidence_are_bounded(self) -> None:
        text = _prose(ADR)
        self.assertIn("originating Git commit", text)
        self.assertIn("canonical source/release payload inventory", text)
        self.assertIn("SHA-256 tree digest", text)
        self.assertIn("history-free public export", text)
        self.assertIn("Commit identity alone cannot prove equivalence", text)

        _require_terms(self, text, (
            "runtime readiness receipts",
            "managed transfer/staging",
            "scheduler submission receipts",
            "scheduler observation and terminal accounting receipts",
            "remote execution success or controlled failure receipts",
            "local retrieval and validation receipts",
        ))
        _require_terms(self, text, (
            "receipt schema version",
            "run ID",
            "operation ID",
            "reviewed plan digest",
            "source commit",
            "canonical source/payload digest",
            "stage inventory and tree digest",
            "runtime identity and package version",
            "target/profile identity",
            "scheduler job ID",
            "raw and normalized scheduler state",
            "scheduler exit code, reason, and timestamps",
            "exact output inventory",
            "byte sizes and SHA-256 digests",
            "previous receipt digests",
            "creation and finalization timestamps",
        ))

        self.assertIn("safe relative paths", text)
        self.assertIn("Private ignored operational receipts", text)
        self.assertIn("sanitized publishable evidence", text.casefold())

    def test_provider_runtime_transfer_and_submission_gates_are_explicit(self) -> None:
        text = _prose(ADR)
        _require_terms(self, text, (
            "genuinely generic SSH/SLURM target template",
            "Alliance/MFA behavior only as an optional provider integration",
            "subprocess-free offline validation",
            "supported Python version",
            "isolated virtual environment",
            "user-site packages disabled",
            "expected `rp` version",
            "coordinated package identity",
            "`pip check`",
            "no unplanned package-index or outbound-network dependency",
            "exact source inventory",
            "regular-file-only policy",
            "symlink, device, socket, FIFO, and unsafe-path rejection",
            "SHA-256 file and tree digests",
            "source-stability checks",
            "unique remote incoming directory",
            "exclusive remote operation claim",
            "same-filesystem atomic no-replace promotion",
            "no unsafe overwrite fallback",
            "matching successful stage and readiness receipts",
            "same-identity resume/recovery",
            "`sbatch --parsable`",
            "exactly one numeric allocation ID",
            "duplicate submission",
            "explicit unresolved state",
        ))

    def test_monitoring_cancellation_execution_and_retrieval_fail_closed(self) -> None:
        text = _prose(ADR)
        _require_terms(self, text, (
            "`squeue` for active state",
            "`sacct`",
            "exact allocation-row selection",
            "accounting delay",
            "idempotent monotonic reconciliation",
            "explicit execution authorization",
            "explicit job-ID confirmation",
            "exactly one intended `scancel` invocation",
            "`cancel-submitted`, not `cancelled`",
            "terminal scheduler accounting can confirm",
            "`cancel-uncertain`",
            "successful execution receipt",
            "controlled-failure receipt",
            "workflow exit success",
            "atomic finalization",
            "scheduler-completed evidence",
            "matching valid remote execution-success receipt",
            "recovery/quarantine retrieval path",
            "unique local sibling staging directory",
            "unique local sibling staging directory, exclusive ownership",
            "atomic no-replace promotion",
            "same-identity interrupted-retrieval resume",
            "no merge, overwrite, replacement",
        ))

    def test_all_implementation_gates_and_live_acceptance_evidence_are_required(self) -> None:
        text = _prose(ADR)
        expected_gate_fragments = {
            "H0": "architecture and evidence contract",
            "H1": "provider-neutral setup",
            "H2": "shared safety primitives",
            "H3": "remote runtime readiness",
            "H4": "transactional upload/staging",
            "H5": "duplicate-safe submission",
            "H6": "terminal reconciliation",
            "H7": "executed cancellation",
            "H8": "remote execution receipts",
            "H9": "transactional retrieval",
            "H10": "fake-remote acceptance",
            "H11": "live-cluster acceptance",
            "H12": "claim promotion",
        }
        for gate, fragment in expected_gate_fragments.items():
            self.assertRegex(text, rf"\b{gate}:\s*[^.]*{re.escape(fragment)}", gate)
        self.assertIn("separate commit", text)
        self.assertRegex(text, r"green CI (?:after each|before the next)")

        _require_terms(self, text, (
            "private target configuration remains untracked",
            "credentials and MFA remain external",
            "source commit and canonical payload digest",
            "runtime-readiness receipt passes",
            "collision at the same destination fails closed",
            "interrupted upload",
            "exactly one successful canary",
            "exactly one numeric job ID",
            "duplicate submission is prevented",
            "active state is observed through `squeue`",
            "terminal completion and `0:0`",
            "deliberate nonzero-exit job",
            "waiting job",
            "exactly one intended `scancel`",
            "cancellation is confirmed",
            "cancellation race",
            "tabular preprocessing, training, and evaluation",
            "complete expected inventory",
            "validates every declared byte size and SHA-256 digest",
            "invalid or incomplete output cannot promote",
            "interrupted retrieval",
            "recovery/quarantine path",
            "local and remote receipt chains match",
            "Publishable evidence is sanitized",
            "cleanup is reviewed and separately authorized",
        ))

    def test_roadmap_keeps_remote_and_fmripost_promotion_conditional(self) -> None:
        text = _prose(ROADMAP)
        self.assertIn(ADR.name, text)
        self.assertIn(SAFETY_ADR.name, text)
        _require_terms(self, text, (
            "conditional milestones",
            "does not change a current capability classification",
            "one deterministic synthetic remote lifecycle",
            "one documented, separately reviewed SLURM environment",
            "fake-remote acceptance",
            "separately authorized live-cluster acceptance using the deterministic synthetic workload",
            "reviewed planning",
            "safe staging",
            "duplicate-safe submission",
            "terminal-state reconciliation",
            "confirmed cancellation",
            "outcome receipts",
            "verified retrieval",
            "collision rejection",
            "explicit no-overwrite publication",
            "separate conditional neuroimaging target",
            "separate accepted tracked decision and reproducible evidence",
            "before any fMRIPost support claim or capability-classification change",
            "fMRIPost remains Experimental or external-runtime",
            "cannot replace fake-remote or live-cluster synthetic acceptance",
            "A roadmap item becomes supported only after",
            "focused tests",
            "checked-in public synthetic example",
            "documented prerequisites, limitations, and scientific boundaries",
            "privacy, scientific, and licensing review",
            "explicit capability-matrix update and release review",
            "exact accepted source commit",
            "canonical source/release payload inventory with a SHA-256 tree digest",
            "coordinated artifact identities",
            "accepted site and profile identity",
            "successful and controlled-failure outcomes",
            "collision rejection",
            "invalid-output rejection",
            "interruption and recovery",
            "verified retrieval",
            "versioned claims and receipts",
            "Full operational receipts remain private",
            "sanitized publishable projection requires privacy and technical review",
            "not sufficient by itself to establish support",
        ))

        self.assertIn("not a supported or released public workflow", _prose(ARCHITECTURE))

    def test_blueprint_is_only_a_compatibility_notice(self) -> None:
        text = _prose(LEGACY_BLUEPRINT)
        _require_terms(self, text, (
            "remains only to preserve historical references",
            "not the authority for current architecture, present support, future commitments, or release requirements",
            "[ARCHITECTURE.md](ARCHITECTURE.md)",
            "[ROADMAP.md](ROADMAP.md)",
            "[docs/capabilities.md](docs/capabilities.md)",
            "[accepted decision records](docs/decisions/README.md)",
        ))
        self.assertNotIn("## Phase", _text(LEGACY_BLUEPRINT))

    def test_changed_links_resolve_and_no_private_evidence_is_embedded(self) -> None:
        public_markdown = _public_markdown_documents()
        self.assertTrue(public_markdown)
        for document in public_markdown:
            _assert_local_links_resolve(self, document)

        combined = "\n".join(_text(document) for document in PRIVACY_DOCUMENTS)
        for forbidden in (
            "/Users/",
            "/Volumes/",
            "/private/",
            "~/.ssh",
            "BEGIN OPENSSH PRIVATE KEY",
            "BEGIN RSA PRIVATE KEY",
            "ssh-rsa ",
            "PASTE_",
            "TODO",
            "TBD",
        ):
            self.assertNotIn(forbidden, combined, forbidden)
        self.assertNotRegex(combined, r"\bsub-[0-9]+\b")
        self.assertNotRegex(combined, r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)\b")
        self.assertNotRegex(combined, r"\b[0-9a-fA-F]{40}\b")
        self.assertNotRegex(combined, r"\b[0-9a-fA-F]{64}\b")


if __name__ == "__main__":
    unittest.main()
