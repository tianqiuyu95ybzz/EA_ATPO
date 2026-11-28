# DDXPlus 官方 BASD 单病例验收记录

**状态为 BLOCKED，验收未执行。** 在 amax 的可访问范围内没有定位到已登记的 DDXPlus 英文 v2 受控备份，按执行要求停在恢复和字节校验之前。没有另行下载数据、替换版本、复制数据或运行环境。

## 前置检查

- 服务器检出 `codex/ea-atpo-v3-migration-20261122`，HEAD 为 `f3fb006ebcfc464911d43520def073cbb95a0eda`，无未提交或未跟踪文件。最新审计分支经远端核验为 `905bb1cc22234046c10f80ac23ddc2fbbf0ac1ff`。在远端分支名和已发布审计树中没有定位到同名 BASD 验收交付。
- 上一轮报告指向历史实验相对目录 `experiments/ea_atpo_v3/gold_dataset_admission_20261121/private/ddx_data/` 和 `SHA256_MANIFEST.json`。它们登记了文件来源、历史大小和 SHA-256。Git 仓库内有历史 DDXPlus 运行与核验脚本，未发现可恢复该备份的专用脚本或已完成的 BASD 验收脚本。原 DDXPlus `test_cpu.py` 验证项目原型，官方 BASD 的结果尚待现场执行。
- 旧 `/pfs` 根在 amax 不存在，`EA_ATPO_OLD_WORKSPACE`、`EA_ATPO_WORKSPACE_COPY` 和 `EA_ATPO_PRIVATE_AUDIT` 未配置。项目工作树、用户两个数据根、ATPO 目录、账号目录及缓存的定向名称检索均完成，未找到这三个原件或可识别的备份。更宽的递归扫描到达 40 秒上限，其他位置仍为 **UNKNOWN**。现有证据确认的是登记指纹，不能确认有可访问的备份副本。
- 复核 Figshare v2 的元数据，三个官方文件大小均与历史登记值一致。官方 MD5 和历史 SHA-256 已分别写入清单，未生成任何现场文件摘要。
- 官方 BASD 源码固定在 `69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66`，核实了 `reset()`、`initialize_state(indices=[i])` 和 `step(s, a_p, done)` 的源码定义。依赖记录来自包元数据，尚未做运行兼容性检查。

[上次 DDXPlus 审计](../three_action_environment_audit_20261128/DDXPLUS_AUDIT.md) · [历史数据路径与复现说明](../ddx_phase3_20261121/README.md) · [历史指纹](../gold_dataset_admission_20261121/SHA256_MANIFEST.json) · [PFS 资产索引](../../../PFS_ARTIFACT_INDEX.md) · [官方 v2 元数据](https://api.figshare.com/v2/articles/22687585/versions/2)

## 验收状态

| 验收项 | 当前结果 |
| --- | --- |
| 三文件恢复 | 阻塞，0/3 |
| SHA-256 核验 | 未执行。登记值和官方 MD5 已记录 |
| BASD 成功初始化 | 未执行 |
| 两次公开 Reset 一致 | 未执行 |
| 单次 ASK 只新增目标证据 | 未执行 |
| 隐藏证据与 Gold 序列化隔离 | 未执行。已记录静态字段契约 |
| Standard Oracle 独立读取 Gold | 未执行 |
| 运行复现 | 保留源码和依赖检查，未选择病例或固化运行配置 |

正式小规模三动作同状态试验暂不准入。需要先获得三个已登记文件的受控可访问路径或恢复原授权挂载，再从验收入口执行。因备份门槛未通过，本次没有新建 `ddxplus_basd_smoke_test.py`，也没有运行 Qwen3-8B、Router、ATPO 树搜索、PPO、Q 数据生产或 IMCS-21。ReMeDi 数据及 Test 均未被访问。

[恢复指纹清单](restore_manifest.json) · [公开状态和评价端边界](STATE_ISOLATION.md) · [环境记录](environment.txt)
