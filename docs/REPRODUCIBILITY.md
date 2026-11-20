# Reproducibility

The source audit and review gates are documented in `experiments/ea_atpo_v3/expert_sft/reports/` and the follow-up audit reports. The reproducible scripts, file hashes and PFS artifact inventory are under `experiments/ea_atpo_v3/expert_sft/`; the follow-up independently rechecks the source files and split hashes. Never overwrite a run directory. Validate the allowlist, GPU memory and other processes immediately before each load.

Before reporting model performance, preserve raw outputs privately, freeze model and validation-selected settings, run paired case-level evaluation, and bootstrap by case. Smoke or static tests do not establish medical performance.
