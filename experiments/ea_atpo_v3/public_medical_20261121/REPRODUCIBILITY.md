# 复现与隔离

在独立 worktree 下使用 Python 3 和 requests；无需 GPU。以下命令创建新的 `private`，若目录存在则拒绝覆盖：

```sh
python3 experiments/ea_atpo_v3/public_medical_20261121/scripts/download_samples.py --out experiments/ea_atpo_v3/public_medical_20261121/private
python3 experiments/ea_atpo_v3/public_medical_20261121/scripts/complete_downloads.py
python3 experiments/ea_atpo_v3/public_medical_20261121/scripts/download_meddialog_sample.py
python3 experiments/ea_atpo_v3/public_medical_20261121/scripts/audit_samples.py
```

数据平台与网络可能变化；DISC/Aquila API 为非 revision 固定快照，精确重现以本轮 private 样本和 download_manifest 的 SHA 为准。Drive MedDialog 下载仅保留前200样本，不是全量版本校验。MedDG 的 pickle 只通过拒绝全部全局对象的受限加载器读取，不加载不可信类。复现前先保护现有 private 目录，切勿覆盖；后续脚本会拒绝覆盖样本，但不是断点续传工具。

转换示例需自行设置只存在于 `private` 下的新输出路径：

```sh
python3 experiments/ea_atpo_v3/public_medical_20261121/scripts/convert_sft.py --dataset remedi --input experiments/ea_atpo_v3/public_medical_20261121/private/remedi_base.json --output experiments/ea_atpo_v3/public_medical_20261121/private/example.jsonl --max-dialogues 20
```

此转换器仅为格式研究，目标为每个既有 doctor 回答，输入严格截止该回答之前；不得当作经过验证的独立评测 Gold。哈希分组只在单源完整对话内，不能保证同一患者的跨对话、跨库隔离或近重复隔离。需要在授权核验、真实患者分组、泄漏核查、Qwen 模板和输出质量验证之后才能用于正式训练。

生成格式仅供审计，不提供训练 loss mask；同一前缀中历史医生回复需要屏蔽 loss。同时真实数据存在连续同角色轮次，需要先验证 Qwen 实际 chat template 并设计保留语义的规范化规则。未完成这些步骤前禁止将转换文件视为可训练数据。
