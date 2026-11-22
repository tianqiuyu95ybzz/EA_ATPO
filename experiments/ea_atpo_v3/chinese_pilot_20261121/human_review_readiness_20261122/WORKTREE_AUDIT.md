# 分支与本地资产审计（2026-11-22）

只读核查 15 个相关 worktree：检查时各工作分支本地 HEAD 与 `git ls-remote --heads origin` 的对应远端 SHA 一致。当前 C worktree 的新增文件是本轮待提交工作；忽略目录不在普通 Git status 中，远端分支一致不证明原始数据或权重备份。

主 `EA_ATPO_V1` 工作区有 1 个已跟踪修改和 1,937 个未跟踪路径；其中 1,883 个位于 V2 `parallel_cache`，另有恢复目录、状态日志、含病例/标签的 CSV 及一个本地 `policy_lock.json`。repo-management worktree 有 1 个未跟踪审计脚本。其余已检查 worktree 在当时无非忽略工作区改动。这些文件未逐条完成敏感性、归属和远端恢复能力验证；部分可能是重要成果，不能声称已推送，也不应在没有归属与敏感性核对时自动上传。既有报告与 `router_metrics.csv` 等已跟踪，但不能由此推定本地未跟踪 policy lock 或其所有依赖已备份。未修改、移动、提交或清理其他 worktree。

本轮仅提交本目录的非敏感方法、状态和脚本；所有评审原文、病例映射、评分表和模型权重保持本地隔离。最终当前分支推送后另做 fetch 与 SHA 核对。
