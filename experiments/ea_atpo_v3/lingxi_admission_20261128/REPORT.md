# EA-ATPO V3 中文数据准入与诊断配对 Pilot

日期为 2026-11-28。执行节点为 amax，项目源提交为 `f3fb006ebcfc464911d43520def073cbb95a0eda`。此前的 Blackwell 恢复记录见 `b096a1093ba713ff0f2ffee61f2a715f84eed62d`。

## 数据准入

| 资源 | 实查结论 |
| --- | --- |
| AI Hospital / MVME | 固定源码 `870fc38c1daffa549c20332d273a8ad71b9d9fe1`。患者模拟器读取病历中的现病史等信息。评价器从病历取诊断结果，缺省时回退至初步诊断，同时使用检查和诊断依据。缺少初始状态下可独立使用的专家回答证据。作者鼓励科研使用，原始病例来自第三方。此次只审代码与发布说明。 |
| RJUA-SPs | 发布者声明 45 个泌尿专科虚拟标准化患者病例，专家编写，数据 CC BY-NC-SA 4.0。完整下载入口此次未验证可用。 |
| LingxiDiag-16K | 作者发布的中文合成精神科对话，数据 CC BY-NC 4.0。具备原对话、病例标识和四分类基准标签。准入限于非商业探索性合成诊断分类，分别评估 ANSWER 和 ASK Replay。状态匹配的 DEFER 专家答案仍待落实。 |

[MVME 论文](https://aclanthology.org/2025.coling-main.680/) · [MVME 源码](https://github.com/LibertFan/AI_Hospital/tree/870fc38c1daffa549c20332d273a8ad71b9d9fe1) · [RJUA-SPs](https://github.com/AQ-MedAI/SPs_benchmark) · [LingxiDiagBench 论文](https://arxiv.org/abs/2602.09379)

LingxiDiag 来源为 Shihao Xu 等发布的数据集，固定 revision `653baa2e5a5a08ba9ed2b1a83f9b443f07c5f8cd)。本 Pilot 从对话提取开头的患者首轮、原医生首个带问号的提问及立即跟随的患者回答，按已有标签评分。案例对话和派生样本仅留在受控目录。该版本发布了 train 和 validation 文件。
[固定数据版本](https://huggingface.co/datasets/XuShihao6715/LingxiDiag-16K/tree/653baa2e5a5a08ba9ed2b1a83f9b443f07c5f8cd) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## 数据检查与结果

- Train 为 14,000 行、13,997 个唯一病例标识。Validation 为 1,000 行、1,000 个唯一病例标识。跨 split 共享标识和去空白后完全相同的完整对话数均为 0。未测近重复或模型预训练污染。
- 847 个 validation 病例通过四分类标签、回放结构和精确重复筛选。按病例标识加固定 salt 的 SHA-256 排序，每类选 2 例。测试集和项目旧 Frozen Holdout 均未读取。
- Base 为原始 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`。GPU 0 使用 BF16、batch 1、seed 0、关闭 thinking、贪心解码和最多 4 个新 token。只接收 A/B/C/D 单字母输出。16 次调用均解析成功，PyTorch 峰值分配 15,674 MiB。V2 训练框架的 19 个 CPU 测试通过。

| 策略 | 标签匹配 | 说明 |
| --- | ---: | --- |
| Always ANSWER | 0/8 | 仅见对话首条患者发言 |
| ASK Replay 一次后 ANSWER | 2/8 | 追加原医生问题及原患者反馈，2 例由错变对，6 例仍错 |
| DEFER | N/A | 缺少与初始状态匹配的专家输出与评分依据 |

样本按类别平衡，规模仅够检查处理、推理及标签评分通路。ASK Replay 有 6/8 次预测为混合类别，结果不足以估计追问收益、自主提问能力或效用。最终基准标签只参与离线评分。没有正式三动作数据、开放式医学质量结果或 Router 训练结果。

## 复现

将 `lingxi_pilot.py` 复制到受控目录，把对应 revision 两个 Parquet 文件放到脚本旁的 `lingxi/`。下载来源可用原 Hugging Face 或能提供相同字节的镜像。复现前核验：

```text
5e2a90354c34797c30a353ef68a9b9ca14c32128fea6e5c2c733143ce99c0360  train-00000-of-00001.parquet
a9ec0ad1ea6a9fa872b7aff46578ebb3ea5b5c2bd87cec133a74a5898a20bfad  validation-00000-of-00001.parquet
```

```bash
CUDA_VISIBLE_DEVICES='' python lingxi_pilot.py prepare
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  timeout 150 python -u lingxi_pilot.py infer "$MODEL_DIR"
```

运行环境与 GPU 见上一次 Blackwell 恢复报告。准备阶段汇总 SHA-256 为 `8c422a66808f637db920bd3b816a3f61c7e1abd8ef884bac1cabadb367ca74d7`。原始推理日志 SHA-256 为 `6b2a70a104b7b7b966b2c01042c9970ea8d34e2d2ba017d694b27d9ac783d0f0`。数据、原始输出和日志保留在受控服务器目录，GitHub 只发布代码和汇总。
