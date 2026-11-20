# 来源与权利核查

许可证与上游病患内容、隐私同意、商业训练、模型发布和再分发权利不是同一问题。下列状态为调查结果，不构成法律结论；未明确覆盖者标为待核验。

| 数据 | 官方信息 | 内容来源/限制 | 当前判断 |
|---|---|---|---|
| [ReMeDi](https://github.com/yanguojun123/Medical-Dialogue) | 仓库标示 MIT | [论文](https://arxiv.org/abs/2109.00430)描述春雨医生在线对话；仓库许可对原问诊文本、患者同意及商业训练的权利链未核实 | 下载可行；数据商业使用与再分发待确认 |
| [MedDG](https://github.com/lwgkzl/MedDG) | 仓库 GPL-3.0 | [论文](https://arxiv.org/abs/2010.07497)描述线上医疗咨询数据；仓库代码许可是否完整覆盖 Drive 文件及原始网页内容待确认 | 下载可行；数据和权利链待确认 |
| [DISC-Med-SFT](https://huggingface.co/datasets/Flmc/DISC-Med-SFT) | 数据卡 Apache-2.0 | [模型文档](https://huggingface.co/Flmc/DISC-MedLLM)描述 MedDialog 重构、知识图谱和其他来源；各上游条款及生成方式需逐来源核对 | 卡片许可不能单独证明全部第三方内容可商用 |
| [AquilaMed-Instruct](https://huggingface.co/datasets/BAAI/AquilaMed-Instruct) | 数据卡 Apache-2.0 | 卡片列出六类上游，包括 MedDialog、Huatuo26M、ChatDoctor；问答、考试、对话混合 | 各来源权利链及比例待核验 |
| [MedDialog](https://huggingface.co/datasets/UCSD26/medical_dialog) | 数据卡 license unknown | 医患平台数据；官方脚本 train 链接可读，指定 GitHub 链接 404；网页版权、隐私授权与商业使用不明 | 不假设商业训练或再分发获授权 |

所有实际样本文本保存在 PFS private，仓库只提交摘要、源和摘要值。原始内容可能含敏感信息，不能由公开下载推断可任意再发布。数据转换 smoke 不构成版权或临床认证。
