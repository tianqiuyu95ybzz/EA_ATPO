#!/usr/bin/env python3
"""Resumable SHA-256 checked downloader for official Qwen3-32B-AWQ files.

The default source is Qwen's official ModelScope organization because the
Hugging Face TLS endpoint is unavailable from some environments. The pinned
ModelScope revision contains byte-identical weight shards: every shard's size
and SHA-256 is checked against both the Hugging Face LFS manifest and the
ModelScope revision API. Byte ranges resume from saved prefixes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests


REPO_ID = "Qwen/Qwen3-32B-AWQ"
REVISION = "0499c3ac83fdef8810b907a23894ba91e95eddd8"
MODELSCOPE_REVISION = "c14568581d436715d61d40f471dee3cac433d1d9"
MODELSCOPE_FILE_COMMIT = "e70d01d7985077a16f9a7de5746ed87f90d665ff"
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ"
THREAD_LOCAL = threading.local()
PRINT_LOCK = threading.Lock()

EXPECTED_SHARDS = [
    {"name": "model-00001-of-00004.safetensors", "size": 4966384688, "sha256": "ab7988215e3b087c884e0c14ce1994a64aac31dc9a37a4bcf29f5b45f3506e2e"},
    {"name": "model-00002-of-00004.safetensors", "size": 4998723112, "sha256": "46faff349f1dd18ffe76993edd3d9fd32941d042331dc590525409bbe3d37edf"},
    {"name": "model-00003-of-00004.safetensors", "size": 4949671152, "sha256": "6e9f763d8c31ad19f1cd67b0b599a38edffd76a4465a22ff5f7fcac46d820e63"},
    {"name": "model-00004-of-00004.safetensors", "size": 4410702792, "sha256": "87f180889fead12b00b1bdf4dca3a3a14e4510e1e200bf56e31898e120b993d8"},
]

EXPECTED_AUX_FILES = [
    {"name": "config.json", "size": 907, "sha256": "0e0c09d430b12e84773ef83ca83dba7c9174caf5a94bb940cd895b82df6c6528"},
    {"name": "generation_config.json", "size": 239, "sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"},
    {"name": "merges.txt", "size": 1671853, "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"},
    {"name": "model.safetensors.index.json", "size": 131598, "sha256": "7d0427c7d238f401b5d38ec08dca07cc8a921aa69d73769b2b7983520de504a1"},
    {"name": "tokenizer.json", "size": 11422654, "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"},
    {"name": "tokenizer_config.json", "size": 9732, "sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"},
    {"name": "vocab.json", "size": 2776833, "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"},
]


def session() -> requests.Session:
    current = getattr(THREAD_LOCAL, "session", None)
    if current is None:
        current = requests.Session()
        current.headers.update({"User-Agent": "EA-ATPO-Range-Recovery/1.0"})
        THREAD_LOCAL.session = current
    return current


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_url(source: str, name: str) -> str:
    if source == "modelscope":
        return f"https://www.modelscope.cn/models/{REPO_ID}/resolve/{MODELSCOPE_REVISION}/{name}"
    return f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{name}"


def fetch_modelscope_manifest() -> dict[str, dict[str, object]]:
    response = requests.get(
        f"https://www.modelscope.cn/api/v1/models/{REPO_ID}/repo/files",
        params={"Revision": MODELSCOPE_REVISION, "Recursive": "true"},
        timeout=(15, 45),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("Code") != 200:
        raise RuntimeError(f"ModelScope manifest API returned Code={payload.get('Code')}")
    files = {str(item["Name"]): item for item in payload["Data"].get("Files", [])}
    expected = EXPECTED_SHARDS + EXPECTED_AUX_FILES
    for item in expected:
        actual = files.get(item["name"])
        if actual is None:
            raise RuntimeError(f"pinned ModelScope revision is missing {item['name']}")
        if int(actual.get("Size", -1)) != item["size"] or str(actual.get("Sha256", "")).lower() != item["sha256"]:
            raise RuntimeError(f"ModelScope metadata mismatch for {item['name']}: {actual.get('Size')} / {actual.get('Sha256')}")
    return files


def fetch_hf_manifest(model_dir: Path) -> dict[str, dict[str, object]]:
    tree_path = model_dir / ".cache/huggingface/trees" / f"{REVISION}.json"
    if not tree_path.is_file():
        # The lock below contains the official LFS SHA-256 values collected
        # from this pinned Hugging Face revision. This allows clean installs.
        return {}
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    return payload.get("files", {})


def read_file_manifest(model_dir: Path, source: str, source_files: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    if source == "modelscope":
        # fetch_modelscope_manifest already compared each value with the
        # pinned Hugging Face LFS hashes above.
        return list(EXPECTED_SHARDS)
    hf_files = fetch_hf_manifest(model_dir)
    result = []
    for expected in EXPECTED_SHARDS:
        entry = hf_files.get(expected["name"])
        if entry:
            actual = {"size": int(entry.get("lfs_size", entry.get("size", -1))),
                      "sha256": str(entry.get("lfs_sha256", "")).lower()}
            if actual["size"] != expected["size"] or actual["sha256"] != expected["sha256"]:
                raise RuntimeError(f"pinned Hugging Face tree mismatch for {expected['name']}: {actual}")
        result.append(dict(expected))
    return result


def download_auxiliary_files(model_dir: Path, source: str, source_files: dict[str, dict[str, object]], read_timeout_s: int) -> None:
    for entry in EXPECTED_AUX_FILES:
        name = str(entry["name"])
        destination = model_dir / name
        if destination.is_file() and destination.stat().st_size == entry["size"] and sha256(destination) == entry["sha256"]:
            print(f"verified existing metadata {name}", flush=True)
            continue
        url = source_url(source, name)
        temp = destination.with_suffix(destination.suffix + f".download.tmp.{os.getpid()}")
        temp.parent.mkdir(parents=True, exist_ok=True)
        try:
            digest = hashlib.sha256()
            count = 0
            with session().get(url, stream=True, timeout=(30, read_timeout_s), allow_redirects=True) as response:
                response.raise_for_status()
                with temp.open("wb") as output:
                    for block in response.iter_content(chunk_size=1 << 20):
                        if block:
                            output.write(block)
                            digest.update(block)
                            count += len(block)
            actual_sha = digest.hexdigest()
            if count != entry["size"] or actual_sha != entry["sha256"]:
                raise RuntimeError(f"{name} integrity mismatch: size={count}/{entry['size']}, sha256={actual_sha}")
            temp.replace(destination)
            print(f"verified metadata {name} sha256={entry['sha256']}", flush=True)
        except Exception:
            temp.unlink(missing_ok=True)
            raise


def fetch_range(args) -> tuple[str, int, int]:
    url, name, start, end, total, path, retries, read_timeout_s = args
    expected = end - start + 1
    path = Path(path)
    if path.exists() and path.stat().st_size == expected:
        return name, start, end
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")

    # Reuse the longest saved prefix from a terminated downloader. A live
    # writer is never taken over; temp-file suffixes contain their owner PID.
    if not temp_path.exists():
        candidates = sorted(
            path.parent.glob(path.name + ".tmp.*"),
            key=lambda candidate: candidate.stat().st_size if candidate.exists() else 0,
            reverse=True,
        )
        for candidate in candidates:
            try:
                owner_pid = int(candidate.name.rsplit(".tmp.", 1)[1])
            except (ValueError, IndexError):
                continue
            try:
                os.kill(owner_pid, 0)
                owner_active = True
            except ProcessLookupError:
                owner_active = False
            except PermissionError:
                owner_active = True
            if owner_active:
                continue
            try:
                saved_size = candidate.stat().st_size
            except FileNotFoundError:
                continue
            if 0 < saved_size <= expected:
                candidate.replace(temp_path)
                print(f"resuming {name} bytes {start}-{end} from saved prefix of {saved_size} bytes", flush=True)
                break

    saved_size = temp_path.stat().st_size if temp_path.exists() else 0
    if saved_size > expected:
        temp_path.unlink(missing_ok=True)
        saved_size = 0
    if saved_size == expected:
        temp_path.replace(path)
        return name, start, end

    last_error = None
    for attempt in range(retries):
        try:
            resume_start = start + saved_size
            expected_remaining = end - resume_start + 1
            headers = {"Range": f"bytes={resume_start}-{end}"}
            with session().get(url, headers=headers, stream=True, timeout=(30, read_timeout_s), allow_redirects=True) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code}; server did not honor byte range")
                content_range = response.headers.get("Content-Range", "")
                expected_header = f"bytes {resume_start}-{end}/{total}"
                if content_range != expected_header:
                    raise RuntimeError(f"unexpected Content-Range {content_range!r}; expected {expected_header!r}")
                received = 0
                with temp_path.open("ab" if saved_size else "wb") as output:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            output.write(chunk)
                            received += len(chunk)
                saved_size += received
                if received != expected_remaining or saved_size != expected:
                    raise RuntimeError(f"short resumed range {received}/{expected_remaining} bytes; saved prefix total {saved_size}/{expected}")
            temp_path.replace(path)
            return name, start, end
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            # Leave the valid prefix in place. The next retry (or process)
            # resumes exactly where this response stopped.
            try:
                saved_size = temp_path.stat().st_size
            except FileNotFoundError:
                saved_size = 0
            if saved_size > expected:
                temp_path.unlink(missing_ok=True)
                saved_size = 0
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"{name} bytes {start}-{end} failed after {retries} tries: {last_error}")


def assemble(model_dir: Path, entry: dict[str, object], source: str, chunk_size: int, workers: int, read_timeout_s: int) -> None:
    name = str(entry["name"])
    size = int(entry["size"])
    expected_sha = str(entry["sha256"])
    destination = model_dir / name
    if destination.exists():
        actual_size = destination.stat().st_size
        actual = sha256(destination) if actual_size == size else "size-mismatch"
        if actual_size == size and actual == expected_sha:
            print(f"verified existing {name} ({size} bytes)", flush=True)
            return
        print(f"discarding unverified existing {name}: size={actual_size}/{size}, sha256={actual}", flush=True)
        destination.unlink()

    part_dir = model_dir / ".range_parts" / name
    part_dir.mkdir(parents=True, exist_ok=True)
    url = source_url(source, name)
    ranges = []
    for start in range(0, size, chunk_size):
        end = min(size - 1, start + chunk_size - 1)
        part_path = part_dir / f"{start:012d}-{end:012d}.part"
        ranges.append((url, name, start, end, size, str(part_path), 6, read_timeout_s))

    pending_ranges = []
    completed_before = 0
    for task in ranges:
        part = Path(task[5])
        if part.exists() and part.stat().st_size == task[3] - task[2] + 1:
            completed_before += 1
        else:
            pending_ranges.append(task)
    print(f"{name}: {completed_before}/{len(ranges)} ranges already complete; downloading with resumable Range requests", flush=True)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_range, task) for task in pending_ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 8 == 0 or completed == len(pending_ranges):
                with PRINT_LOCK:
                    print(f"{name}: {completed_before + completed}/{len(ranges)} ranges", flush=True)

    assembling = destination.with_suffix(destination.suffix + ".ranged-assembling")
    digest = hashlib.sha256()
    total_written = 0
    with assembling.open("wb") as output:
        for task in ranges:
            part = Path(task[5])
            if part.stat().st_size != task[3] - task[2] + 1:
                raise RuntimeError(f"bad range size in {part}")
            with part.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    output.write(block)
                    digest.update(block)
                    total_written += len(block)
        output.flush()
        os.fsync(output.fileno())
    actual_sha = digest.hexdigest()
    if total_written != size or actual_sha != expected_sha:
        assembling.unlink(missing_ok=True)
        # A final Hub LFS mismatch means at least one individually complete
        # range was bad. Discard this shard's ranges so the next resumable run
        # fetches them again rather than repeating the same bad assembly.
        for part in part_dir.glob("*.part"):
            part.unlink(missing_ok=True)
        raise RuntimeError(f"assembled {name} failed integrity check: size={total_written}/{size}, sha256={actual_sha}, expected={expected_sha}")
    assembling.replace(destination)
    for part in part_dir.glob("*.part"):
        part.unlink()
    try:
        part_dir.rmdir()
    except OSError:
        pass
    print(f"verified {destination} sha256={expected_sha}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--read-timeout-s", type=int, default=90)
    parser.add_argument("--source", choices=("huggingface", "modelscope"),
                        default=os.environ.get("EA_ATPO_MODEL_SOURCE", "modelscope"))
    args = parser.parse_args()
    workers = max(1, int(args.max_workers))
    if args.chunk_mib < 8 or args.chunk_mib > 128:
        raise ValueError("--chunk-mib must be between 8 and 128")
    source_files = fetch_modelscope_manifest() if args.source == "modelscope" else {}
    entries = read_file_manifest(args.model_dir, args.source, source_files)
    if not entries:
        raise RuntimeError("pinned Hub tree has no safetensors entries")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    download_auxiliary_files(args.model_dir, args.source, source_files, args.read_timeout_s)
    source_lock = {
        "repo_id": REPO_ID,
        "huggingface_revision": REVISION,
        "selected_source": args.source,
        "modelscope_revision": MODELSCOPE_REVISION if args.source == "modelscope" else None,
        "modelscope_weight_upload_commit": MODELSCOPE_FILE_COMMIT if args.source == "modelscope" else None,
        "weight_files": EXPECTED_SHARDS,
        "auxiliary_files": EXPECTED_AUX_FILES,
        "hash_relation": "all ModelScope file SHA-256 values were checked against the pinned Hugging Face LFS SHA-256 values or verified Hugging Face snapshot files",
    }
    (args.model_dir / "model_source_lock.json").write_text(json.dumps(source_lock, indent=2) + "\n", encoding="utf-8")
    status_path = args.model_dir / "range_download_status.json"
    completed = []
    status = {
        "status": "running", "repo_id": REPO_ID, "revision": REVISION,
        "method": "HTTP Range with per-file SHA-256 verification",
        "source": args.source,
        "source_revision": MODELSCOPE_REVISION if args.source == "modelscope" else REVISION,
        "huggingface_revision": REVISION,
        "modelscope_weight_upload_commit": MODELSCOPE_FILE_COMMIT if args.source == "modelscope" else None,
        "chunk_mib": args.chunk_mib, "max_workers": workers,
        "read_timeout_s": args.read_timeout_s, "range_retries": 6,
        "expected_shards": entries, "completed_shards": completed,
        "started_at_epoch": time.time(),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    try:
        for entry in entries:
            assemble(args.model_dir, entry, args.source, args.chunk_mib * (1 << 20), workers, args.read_timeout_s)
            completed.append(str(entry["name"]))
            status["completed_shards"] = list(completed)
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        status["status"] = "interrupted_or_failed_resumable"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["completed_shards"] = list(completed)
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise
    # Old HF CLI fragments are safe to remove only after every official shard
    # has passed exact size and SHA-256 checks.
    fragment_dir = args.model_dir / ".cache/huggingface/download"
    if fragment_dir.exists():
        for fragment in fragment_dir.glob("*.incomplete"):
            fragment.unlink()
    status["status"] = "all_shards_verified"
    status["completed_at_epoch"] = time.time()
    status["completed_shards"] = list(completed)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"all {len(entries)} AWQ weight shards verified from {args.source} at source revision {status['source_revision']}; HF reference revision {REVISION}", flush=True)


if __name__ == "__main__":
    main()
