# 实验分支与遗留工作区核对

本轮独立 worktree 从历史 Oracle 分支 `93c8b75a497b6a880953e39d0bee5ba134eeb557` 创建，未合并或编辑其他 worktree。`git ls-remote --heads origin` 显示 V1、V2、V3 Expert SFT、Router Training、Oracle 和上一轮公开数据调查等既有分支的远端 SHA 均与本地对应分支 HEAD 一致，上一轮公开数据调查分支为 `da0167fdfe73c71816af40184bd35294746ee2ad`。本次仅确认分支 HEAD，不等于确认所有非 Git 大文件已备份。

只读 `git status --porcelain --untracked-files=all` 发现：主 `EA_ATPO_V1` 工作区已有 1,937 个未跟踪路径（均在 `experiments` 下）和 1 个已跟踪修改（在 `ea_atpo_stage` 下）；`EA_ATPO_V1_repo_management_20261120` 有 1 个未跟踪 `scripts` 路径；其他既有 worktree 在检查时无 Git 状态项。本轮没有修改、提交、推送或清理这些工作区文件。未逐个读取其内容，不能断言它们已备份、无敏感数据或是本轮遗漏；应由对应工作区负责人单独核查。忽略目录及 PFS checkpoint 不会出现在上述计数中。

本实验私有作者源 ZIP、逐动作带答案映射保留在独立 worktree 的 `experiments/ea_atpo_v3/gold_oracle_audit_20261121/private/`。仅汇总、非 Gold 的候选 ID、代码及 checksum 可公开；不上传原题、答案、患者数据或模型文件。
