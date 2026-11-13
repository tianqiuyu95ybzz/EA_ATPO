# EA-ATPO V1 real smoke attempt 2

- GPUs: physical 4 and 6, visible as a two-device distributed group. GPUs 1 and 7 were not used.
- The Qwen3-8B Actor/Critic and SGLang TP=2 service initialized successfully; no OOM occurred.
- Two actual model-generated episodes were recorded. Action events: ANSWER 2, ASK 2, CONSULT 2, DEFER 0. Both episodes terminated with an answer; 0/2 were correct. This is an engineering smoke, not an accuracy estimate.
- One PPO update completed and Actor/Critic model plus optimizer checkpoint files were written under `checkpoints/global_step_1/`.
- The process then exited during post-update logging because the EA reward adapter lacked legacy `is_valid` and related metric fields required by the unchanged ATPO metric function.
- The EA reward adapter now derives validity and correctness from completed episode traces and supplies the legacy non-EA counters as zero. A focused compatibility test passes.
- The raw log, traces, rollout dump, partial checkpoint and full run manifest are retained in this directory.
