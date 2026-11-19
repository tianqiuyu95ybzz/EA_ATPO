#!/usr/bin/env python3
"""Refuse model loading on excluded, busy, or under-resourced GPUs."""
from __future__ import annotations

import argparse
import json
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=150000)
    parser.add_argument("--max-util", type=int, default=20)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.gpu in {1, 7}:
        raise SystemExit(f"GPU {args.gpu} is excluded by the standing project instruction")
    line = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    cols = [x.strip() for x in line.split(",")]
    row = {
        "gpu": int(cols[0]), "name": cols[1], "uuid": cols[2],
        "total_mib": int(cols[3]), "used_mib": int(cols[4]),
        "free_mib": int(cols[5]), "utilization_pct": int(cols[6]),
    }
    raw_pids = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    pids = [x.strip() for x in raw_pids.splitlines() if x.strip() and x.strip().upper() != "N/A"]
    report = {
        "status": "passed" if row["free_mib"] >= args.min_free_mib and row["utilization_pct"] <= args.max_util and not pids else "blocked",
        **row, "minimum_free_mib": args.min_free_mib,
        "maximum_utilization_pct": args.max_util,
        "preexisting_compute_pids": pids,
        "excludes_gpus": [1, 7],
    }
    print(json.dumps(report, indent=2))
    if args.manifest:
        from pathlib import Path
        path = Path(args.manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
