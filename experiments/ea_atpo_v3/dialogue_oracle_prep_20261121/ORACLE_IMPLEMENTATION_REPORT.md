# Oracle 配对接口与限制

`pipeline.oracle` 为 CPU 纯函数：ANSWER 返回同一 Base；DEFER 的 Standard 返回已验证目标，保护门在同一 Base 被事后判正确时返回 Base，判错时返回已验证目标。两者针对同一 state/base/target 的测试展示分歧；调用者必须提供可信、经外部验证的 Gold 和明确布尔正确性。缺 Gold 或正确性时拒绝执行；该布尔值为离线事后信息，绝不出现在 `router_view`。函数不能独立验证调用者声称的 Gold，正式实验仍需可信来源和审计协议。ASK 被此接口拒绝，患者反馈只作为受限的历史观察转移。

用合成占位字符串测试所有分支（无模型、无患者、无合成评测 Gold）；真实 ReMeDi/MedDG 当前无已证实目标、无 Base 输出、无可验证 Base 正确性标签，因此真实 Oracle A/B 配对评测有效状态数为 0。保护门是理想上界，不能作为部署 Router 或其输入。代码为准备和边界验证，不表示医学效果已验证。
