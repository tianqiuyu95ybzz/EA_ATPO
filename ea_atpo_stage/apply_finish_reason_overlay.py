#!/usr/bin/env python3
"""Apply the EA-only SGLang finish-reason telemetry overlay to the pinned ATPO checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one patch anchor in {path}; found {text.count(old)}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: apply_finish_reason_overlay.py <ATPO-official/codes>", file=sys.stderr)
        return 2
    code = Path(sys.argv[1]).resolve()
    async_server = code / "verl/workers/rollout/async_server.py"
    sglang = code / "verl/workers/rollout/sglang_rollout/sglang_rollout.py"
    changed = replace_once(
        async_server,
        '    """logprobs of response token ids"""\n',
        '    """logprobs of response token ids"""\n'
        '    finish_reason: Optional[str] = None\n'
        '    """engine completion reason, when the backend exposes it"""\n',
    )
    changed |= replace_once(
        sglang,
        '        return TokenOutput(token_ids=token_ids, log_probs=log_probs)\n',
        '        finish_reason = output.get("meta_info", {}).get("finish_reason", {}).get("type")\n'
        '        return TokenOutput(token_ids=token_ids, log_probs=log_probs, finish_reason=finish_reason)\n',
    )
    print(f"EA SGLang finish_reason telemetry overlay: {'applied' if changed else 'already present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
