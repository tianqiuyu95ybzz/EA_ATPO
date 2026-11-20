# EA-ATPO V3 follow-up execution
1. Original state: baseline 80834ad4890a4200549238a4fc28a970c8eea9f6 on the previously pushed V3 branch; this independent branch starts there.
2–3. Data and Gold: see DATA_AUDIT.md. 291 pending; historical ATPO labels have source provenance but no completed human approval. Private gold_audit.csv is on PFS.
4–5. Expert A/B: both_medical implementation not found in tracked current source, related worktrees or git commit search. User now requests identical independent medical SFT; earlier Base-aware A configuration is not silently substituted. Formal training did not start. Previous 10-step AWQ smoke passed; separate loss integration had an interrupted attempt. Expert B's independent weight copy exists on PFS.
6–11. Paired evaluation, confidence intervals, Router and costs: not run; there are no trained experts or reviewed Gold. Historical results are not this run's results.
12. RL: not started; conditions unmet.
13–18. Local/remote audit: see GITHUB_LOCAL_REMOTE_AUDIT.md. All local branch HEAD commits were reachable from remote refs at audit time. No other worktree changed and no files were cleaned. Historical tracked trajectory Gold/dialogue is a protected-data risk; history was not rewritten. Final remote SHA must be verified after this report is committed.
19. PFS: previous V3 worktree stores private queue/Base outputs under experiments/ea_atpo_v3/expert_sft/data/private, independent B model under models/expert_b_awq_init and smoke adapter under outputs/awq_smoke_20261120_attempt01. Its full file inventory is data/pfs_artifact_inventory_v2.json.
20. Outstanding: human Gold approval, definition/implementation of both_medical, safe GPU availability, formal SFT, independent evaluation, and Router admission. No unsupported claims are made.
