#!/usr/bin/env python3
"""Run the frozen Attempt04 paired cohort using the locked Attempt01 protocol.

This thin entry point reuses the prior tested prompt construction, parser,
generation parameters, and raw-response logging. It redirects all outputs into
the new Attempt04 directory and never changes or overwrites prior attempts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "outputs/qwen32_router_generalization_20261118_attempt04"
SERVICE_ATTEMPT = os.environ.get("EA_ATPO_SERVICE_ATTEMPT", "attempt01")
SERVICE_DIR = RUN / "services" / SERVICE_ATTEMPT


def configure():
    sys.path.insert(0, str(HERE))
    import run_qwen32_defer_validation as protocol  # noqa: E402

    protocol.OUT = RUN
    protocol.SELECTION = RUN / "case_selection.json"
    protocol.BASE_URL = "http://127.0.0.1:30257/v1"
    protocol.EXPERT_URL = "http://127.0.0.1:30256/v1"
    protocol.BASE_ALIAS = "qwen3-8b-generalization-base"
    protocol.EXPERT_ALIAS = "qwen3-32b-awq-generalization-expert"
    original_write_manifest = protocol.write_manifest

    def write_attempt04_manifest(cases, meta, outdir, smoke):
        original_write_manifest(cases, meta, outdir, smoke)
        path = outdir / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["experiment"] = "Attempt04 fresh, history-excluded paired Qwen3-8B / Qwen3-32B-AWQ Router generalization cohort"
        manifest["service_manifests"] = {
            "base": str(SERVICE_DIR / "base/service_manifest.json"),
            "expert": str(SERVICE_DIR / "expert/service_manifest.json"),
        }
        manifest["service_attempt"] = SERVICE_ATTEMPT
        manifest["attempt02_superseded_before_inference"] = True
        manifest["attempt03_stopped_no_model_inference"] = True
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    protocol.write_manifest = write_attempt04_manifest
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run the previously isolated 20-case protocol smoke only")
    parser.add_argument("--resume", action="store_true", help="resume an already-started Attempt04 output file set")
    args = parser.parse_args()
    protocol = configure()
    outdir = RUN / ("smoke_20" if args.smoke else "paired_rollouts")
    existing = [outdir / name for name in ("base_answer_trajectories.jsonl", "qwen32_defer_trajectories.jsonl")]
    if any(path.exists() for path in existing) and not args.resume:
        raise SystemExit("Attempt04 rollout files already exist; audit them and pass --resume to continue.")
    if not args.smoke and not (RUN / "case_selection.json").is_file():
        raise SystemExit("Frozen Attempt04 case_selection.json is missing; no inference started.")
    asyncio.run(protocol.main_async(argparse.Namespace(smoke=args.smoke, resume=args.resume)))


if __name__ == "__main__":
    main()
