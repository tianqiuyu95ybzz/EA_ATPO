# EA-ATPO research repository

This repository contains the research additions, experiment protocols, and
implementation notes for the EA-ATPO project.

The official ATPO implementation is an external dependency:

- Upstream: <https://github.com/Quark-Medical/ATPO>
- Local checkout used during experiments: `ATPO-official/`

Large/private datasets, model weights, checkpoints, logs, generated outputs,
and connection credentials are intentionally excluded from Git. Reproduce
the local environment by checking out the upstream ATPO repository and
following the dated experiment plans in this repository.

For the baseline smoke, use the pinned upstream snapshot and leave it unmodified:

```bash
git clone https://github.com/Quark-Medical/ATPO.git ATPO-official
git -C ATPO-official checkout 25af83688c06917f94245a4db06be7d0924c090c
```

The smoke runner reads code and source JSONL files from this ignored checkout.

