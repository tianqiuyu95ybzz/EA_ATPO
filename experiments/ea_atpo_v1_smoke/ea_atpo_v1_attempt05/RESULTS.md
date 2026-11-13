# EA-ATPO V1 real smoke attempt 5

- GPUs: physical 4 and 6; GPUs 1 and 7 were not used.
- Qwen3-8B Actor/Critic and SGLang TP=2 initialized without OOM. Two actual model episodes ran; one PPO update saved an Actor/Critic checkpoint.
- Action counts: {'ASK': 2, 'CONSULT': 2, 'ANSWER': 2}; final answers 2/2; correct 0/2. Tiny engineering sample only.
- The run stopped after the update because ATPO's legacy logger still lacked `is_valid`. The prior runtime bridge was ineffective and has been removed.
- The current fix derives validity/correctness in `EARewardManager` from completed response tokens and `reward_model.ground_truth`, then returns required post-hoc metric fields. Focused metric and protocol tests pass.
- Full manifest, traces, rollout dump, log, and partial checkpoint are retained here.
