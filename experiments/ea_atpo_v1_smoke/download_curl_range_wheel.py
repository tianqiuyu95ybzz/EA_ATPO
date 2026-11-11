#!/usr/bin/env python3
"""Resumable, checksum-verified range downloader using the configured curl proxy."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    state_path = output.with_name(output.name + ".curl-ranges.json")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o664)
    if os.fstat(fd).st_size != args.size:
        os.ftruncate(fd, args.size)

    identity = {"url": args.url, "size": args.size, "sha256": args.sha256,
                "chunk_size": args.chunk_size}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if any(state.get(k) != v for k, v in identity.items()):
            raise SystemExit("Existing range state belongs to a different artifact/config")
    else:
        state = {**identity, "completed_chunks": []}
    done = {int(i) for i in state.get("completed_chunks", [])}
    chunks = (args.size + args.chunk_size - 1) // args.chunk_size
    lock = threading.Lock()

    def save_state() -> None:
        state["completed_chunks"] = sorted(done)
        temp = state_path.with_suffix(state_path.suffix + ".tmp")
        temp.write_text(json.dumps(state, sort_keys=True) + "\n")
        os.replace(temp, state_path)

    def fetch(index: int) -> int:
        start = index * args.chunk_size
        end = min(args.size, start + args.chunk_size) - 1
        expected = end - start + 1
        chunk_file = part.with_name(f"{part.name}.range-{index:04d}")
        for attempt in range(1, args.retries + 1):
            cmd = [
                "curl", "--silent", "--show-error", "--fail", "--location",
                "--retry", str(args.retries), "--retry-all-errors", "--retry-delay", "2",
                "--max-time", "900", "-H", f"Range: bytes={start}-{end}",
                "-D", "-", "-o", str(chunk_file), "-w", "\\n__RANGE_RESULT__%{http_code}\\t%{size_download}",
                args.url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            marker = re.search(r"__RANGE_RESULT__(\d{3})\t(\d+)", result.stdout)
            headers = result.stdout.split("__RANGE_RESULT__", 1)[0]
            content_ranges = re.findall(r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", headers)
            size = chunk_file.stat().st_size if chunk_file.exists() else -1
            valid = (
                result.returncode == 0 and marker is not None
                and marker.group(1) == "206" and int(marker.group(2)) == expected
                and size == expected and content_ranges
                and tuple(map(int, content_ranges[-1])) == (start, end, args.size)
            )
            if valid:
                with chunk_file.open("rb") as stream:
                    data = stream.read()
                if len(data) != expected:
                    raise RuntimeError(f"range {index}: short local chunk")
                written = os.pwrite(fd, data, start)
                if written != expected:
                    raise RuntimeError(f"range {index}: short pwrite {written}/{expected}")
                chunk_file.unlink(missing_ok=True)
                os.fsync(fd)
                with lock:
                    done.add(index)
                    save_state()
                    print(f"completed range {len(done)}/{chunks}", flush=True)
                return index
            detail = (result.stderr or "").strip().replace("\n", " ")[:240]
            chunk_file.unlink(missing_ok=True)
            if attempt == args.retries:
                raise RuntimeError(f"range {index} failed after retries (curl={result.returncode}; {detail})")
        raise AssertionError("unreachable")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, i) for i in range(chunks) if i not in done]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        os.fsync(fd)
        digest = hashlib.sha256()
        with part.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual.lower() != args.sha256.lower():
            raise SystemExit(f"SHA256 mismatch: expected {args.sha256}, got {actual}; partial retained")
        os.close(fd)
        part.replace(output)
        state_path.unlink(missing_ok=True)
        print(f"verified {output} sha256={actual}", flush=True)
    except BaseException:
        os.close(fd)
        raise


if __name__ == "__main__":
    main()
