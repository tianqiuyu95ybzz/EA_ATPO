# CPU Oracle 冒烟

仅 DDXPlus 通过**原型级候选**（不是正式实验准入）；取官方 validate 文件前 50 条、无性能筛选。50/50 官方 PATHOLOGY 可精确匹配官方 49 类之一；每例验证初始化只含首证据、一个新隐藏证据及一次未知值、重复询问零费用、视图不含病种或差分、Oracle A/B 在模拟 Base 正确与错误时的边界；均 PASS。完整可重现统计见 `cpu_smoke.json` 和 `ddx_cpu_smoke.py`。症状及鉴别诊断、病例原文和测试标签不提交。此处构造的 Base 标签仅用于边界测试，不是模型预测、合成 Gold 或效果评估。

IMCS/DX 因标签来源或许可尚未满足准入，未做候选级 Oracle smoke；ReMeDi/MedDG 没有可验证目标，也未重复运行。无模型调用、训练或 GPU。请从已获取的官方文件按脚本 `--zip --evidences --conditions` 参数复现；输入位于 ignored private，远端保存代码和文件 hashes，需从官方重新获取原始文件才能复现。
