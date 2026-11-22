# Versioning and artifact rules

## Research stages

- `V1`, `V2`, and `V3` refer to research-method stages. They are not semantic
  software-version tags.
- Keep each experiment under a dated, stage-specific directory. Do not reuse an
  output directory for a rerun.
- Every run should record the source commit, model repository and immutable
  revision, tokenizer/template, data manifest, split, decoding settings, seed,
  costs, environment, and output hashes.
- Preserve failed, smoke, and development runs. Mark their evidence class
  instead of overwriting them with later results.

## Branches and commits

- Use a dated branch for migrations and experiments, for example
  `codex/ea-atpo-v3-migration-20261122`.
- Do not force-push or rewrite shared history. Fetch and compare the pushed SHA
  after each publication.
- Never commit from another worktree's dirty state. Copy only individually
  reviewed, non-sensitive artifacts into the current branch.

## Data and model assets

- GitHub contains source code, configurations, documentation, and reviewed
  non-sensitive aggregates.
- Patient dialogue, source/case mapping, free-text predictions, Gold, review
  forms, private embeddings, and case-level outcomes remain in access-controlled
  PFS storage.
- Foundation-model weights, SFT adapters, Actor/Critic checkpoints, and
  unapproved Router checkpoints stay out of the public repository. Track their
  location and provenance in the restricted PFS inventory.
- A model name is not a model identity: pin the model revision and state whether
  the run used the original base, SFT adapter, or post-ATPO checkpoint.

## Evidence labels

Use `engineering`, `exploratory_dev`, `confirmatory`, or `not_run` in reports.
Do not call a Dev set a test set. Keep N/A values distinct from zero.
