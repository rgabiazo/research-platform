from __future__ import annotations

import argparse
import json
from pathlib import Path

from .events.service import build_events, plan_events, publish_events


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research BIDS helpers for configured staged and published event-building flows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    events_parser = subparsers.add_parser(
        "events",
        help="BIDS helpers for configured staged and published task-event flows.",
    )
    event_subparsers = events_parser.add_subparsers(dest="events_command", required=True)

    for name in ("plan", "build"):
        command = event_subparsers.add_parser(
            name,
            help=f"{name.title()} configured task events into staged outputs.",
        )
        command.add_argument("--spec", required=True, help="Base event spec path.")
        command.add_argument(
            "--source",
            required=True,
            help="Raw source table path. Required columns are defined by the selected event specification.",
        )
        command.add_argument("--artifact-root", required=True, help="Artifact output root for staged outputs and manifests.")
        command.add_argument(
            "--dataset-root",
            help="Optional BIDS dataset root for BOLD anchor resolution and inherited entities.",
        )
        command.add_argument(
            "--session",
            help="Optional session override used during source resolution and anchor matching.",
        )
        command.add_argument(
            "--backend",
            choices=("polars", "pandas"),
            default="polars",
            help=(
                "Tabular backend for numeric parsing and writing. "
                "Use pandas for pandas-compatible checked fixture output; "
                "polars remains the default."
            ),
        )
        if name == "build":
            command.add_argument("--write-sidecars", action="store_true", help="Stage BIDS events sidecar JSON files.")
            command.add_argument(
                "--copy-stimuli",
                action="store_true",
                help="Stage stimuli files and rewrite stim_file values to dataset-relative stimuli/... paths.",
            )

    publish = event_subparsers.add_parser("publish", help="Publish staged outputs from manifest.")
    publish.add_argument("--dataset-root", required=True, help="Dataset root to publish into.")
    publish.add_argument("--manifest", required=True, help="Build manifest path.")
    publish.add_argument("--overwrite", action="store_true", help="Allow publish to overwrite existing dataset files.")

    return parser


def _run_plan(args: argparse.Namespace) -> int:
    manifest = plan_events(
        spec_path=args.spec,
        source_path=args.source,
        artifact_root=args.artifact_root,
        backend=args.backend,
        dataset_root=args.dataset_root,
        session=args.session,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _run_build(args: argparse.Namespace) -> int:
    manifest_path = build_events(
        spec_path=args.spec,
        source_path=args.source,
        artifact_root=args.artifact_root,
        backend=args.backend,
        dataset_root=args.dataset_root,
        session=args.session,
        write_sidecars=args.write_sidecars,
        copy_stimuli=args.copy_stimuli,
    )
    print(manifest_path)
    return 0


def _run_publish(args: argparse.Namespace) -> int:
    publish_events(dataset_root=args.dataset_root, manifest_path=args.manifest, overwrite=args.overwrite)
    print(Path(args.dataset_root))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    commands = {
        ("events", "plan"): _run_plan,
        ("events", "build"): _run_build,
        ("events", "publish"): _run_publish,
    }
    return commands[(args.command, args.events_command)](args)


if __name__ == "__main__":
    raise SystemExit(main())
