# Version policy

V1 denotes the original ATPO implementation and historical results; V2 denotes the ASK/ANSWER/DEFER protocol; V3 denotes the Expert and cost-sensitive Router research. `3.0.0-dev` is an unfinished code and documentation state, not a trained model or a validated result.

Version experiment configurations separately from model weights. A formal run must record its immutable experiment ID, Git commit, model and tokenizer revisions, configuration and data-manifest hashes, seed, actual GPU/environment, commands, per-epoch checkpoint hashes, results and report. Never modify a published run configuration in place; create a new version and preserve the previous one. Large model weights, adapters and patient/Gold data stay on PFS; only safe manifests and hashes belong in Git.

No data or outputs are moved by this policy. Existing paths and scripts remain valid. Negative and failed results are retained.
