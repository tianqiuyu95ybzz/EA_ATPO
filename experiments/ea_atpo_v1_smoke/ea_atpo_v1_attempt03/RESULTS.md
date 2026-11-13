# EA-ATPO V1 real smoke attempt 3

- GPUs: physical 4 and 6, configured as a two-device distributed group. GPUs 1 and 7 were not used.
- Qwen3-8B Actor/Critic and the SGLang TP=2 server initialized without OOM. Two model-generated episodes completed, followed by one PPO update and an Actor/Critic checkpoint.
- Episode/action summary: 2 episodes; action events {'ASK': 2, 'CONSULT': 2, 'ANSWER': 2}; 2 final answers; 0 correct episodes. This tiny run is an engineering smoke, not a performance estimate.
- Post-update accuracy logging still failed because ATPO removes per-sample `statistics` in `AgentLoopWorker.generate_sequences` before calling the reward manager.
- The next fix wraps `_postprocess` in the EA entrypoint to copy required metric fields before that removal. Focused tests for the bridge and metric contract pass.
- Manifest, trace, rollout, log, and partial checkpoint are retained here.
