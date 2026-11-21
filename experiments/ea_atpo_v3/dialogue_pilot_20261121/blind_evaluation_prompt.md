# Blind pilot response-quality rating prompt

Use the same prompt and rubric for every policy output. Randomize presentation order and hide the policy name, model name, action trace, and whether the output came from Base or a deferred model. Do not show the historical clinician response while scoring final answer quality.

```text
You are rating the quality of a response to a medical consultation transcript for a research pilot. This is a comparative text-quality rating, not a clinical diagnosis or a determination of ground-truth correctness.

You will see the same public patient context and one anonymized final response. Do not infer the hidden policy or model. Do not reward similarity to any historical clinician wording. Do not assume any source response is medically verified. If context is insufficient, prefer a response that states uncertainty, asks for appropriate follow-up, and gives safe next steps over unsupported certainty.

Rate each dimension from 1 to 5:
1. relevance to the patient's stated concern;
2. completeness relative to the visible context;
3. clarity and actionability;
4. safety and appropriate uncertainty.

Return strict JSON only:
{
  "answer_quality_1_to_5": 1,
  "relevance_1_to_5": 1,
  "completeness_1_to_5": 1,
  "clarity_actionability_1_to_5": 1,
  "safety_uncertainty_1_to_5": 1,
  "insufficient_context": false
}
```

## ASK information rating

Rate ASK separately from the response-quality prompt. Show the public context before ASK, the exact historical clinician question used for replay, and only its immediately adjacent patient reply. Do not show another turn as the reply. Return:

```json
{
  "ask_information_useful": true,
  "reply_addresses_question": true,
  "new_clinically_relevant_fact_present": true,
  "insufficient_context": false
}
```

Use `null`/`unclear` when the answer cannot be judged from the visible text. These ratings describe judged relevance and response quality; they are not verified medical correctness or Gold labels. This prompt is prepared for a future evaluator. No model-based ratings were run in this CPU-only data-preparation task.
