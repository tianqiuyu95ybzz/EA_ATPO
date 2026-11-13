# EA-ATPO V1 real smoke attempt 4

- GPUs: physical 4 and 6; GPUs 1 and 7 were not used.
- Qwen3-8B Actor/Critic and SGLang TP=2 initialized without OOM. Two actual model-generated episodes completed; one PPO update wrote an Actor/Critic checkpoint.
- Action event counts: {'ASK': 2, 'CONSULT': 2, 'ANSWER': 2}; final answers: 2/2; correct: 0/2. These two cases are an engineering smoke only.
- Post-update logging again failed because the metric bridge had been installed only in the driver process, not the independent Ray AgentLoopWorker process.
- The current fix installs the bridge when the EA loop module is imported inside that worker. The 22 focused protocol/adapter tests pass; attempt 5 will verify the worker-process hook.
- Full manifest, trace, rollout, log, and partial checkpoint are retained here.
