# Local–remote audit

Starting branch `codex/ea-atpo-gold-oracle-audit-20261121`, HEAD `308131078e9208622b7ceef137dcfbc2ad81b304`; tracked and non-ignored status clean. Previous Pilot data and outputs remain intact. This directory is the sole scope of the new commit; raw dataset, model weights, per-case outputs, invalid outputs and patient trajectories stay under ignored private paths. The model paths in `MODEL_SHA256SUMS` are read-only references, not committed weights.

Before commit, run `git diff --cached --check`, inspect the staged file list and sizes, and verify `sha256sum -c SHA256SUMS` from this directory. After push, run `git fetch origin codex/ea-atpo-gold-oracle-audit-20261121`, compare local HEAD and remote-tracking branch, inspect `git status --short --branch --untracked-files=all` and `git log origin/codex/ea-atpo-gold-oracle-audit-20261121..HEAD`. The actual resulting commit and verification are reported in the delivery message to avoid a self-referential commit hash.
