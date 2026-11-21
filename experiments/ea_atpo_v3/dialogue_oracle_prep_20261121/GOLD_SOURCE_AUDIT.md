# Gold 来源审计

[ReMeDi 作者仓库](https://github.com/yanguojun123/Medical-Dialogue)及[论文](https://arxiv.org/abs/2109.00430)说明 base 有 1,557 段医患对话及细粒度 intent/action/slot/value 人工标注。原文件含 dialogue、按序 role、sentence、turn、actions 等；可区分患者与医生，含多轮。医生发言是原始对话响应，部分可能包含诊断，但不是由本次审核确认的正确最终诊断；语义标注不是对任意 Base 答案的医学正确性标签。没有发现可支撑本实验的逐状态验证 Gold 或 Base 正确性。原文件未提供患者 ID；处理后的官方 train/dev/test 与原对话的映射尚未建立。

[MedDG 作者仓库](https://github.com/lwgkzl/MedDG)及[论文](https://arxiv.org/abs/2010.07497)说明任务为医生回复生成和实体预测，标注疾病、症状、属性、检查、药物等实体。train.pk 按序包含 Patients/Doctor、Sentence 和实体字段，未见患者或官方对话 ID。医生回复可作为生成任务的观察参考；疾病实体是提及或标注的实体，并非已核实最终诊断或对任意回答的正确性判定。不能将任一医生句或实体当作已验证 Oracle Gold。严格按当前可证明标准，两者 Gold 和 Oracle 可用率均为 0%，并非声称医生回复必然错误。

两源有真实原始患者后续发言，但只能在原医生发言之后按原序观察；不可将其设为新问题的通用答案。ReMeDi 作者仓库声明 MIT、MedDG 仓库为 GPL-3.0；上游问诊内容、个人信息、训练与再分发权利链仍需确认。仓库不上传数据、参考或患者内容。详见上一轮 `DATA_LICENSE_AND_PROVENANCE.md`。
