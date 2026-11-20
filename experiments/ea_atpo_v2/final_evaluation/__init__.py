"""Final case-level evaluation tools for EA-ATPO V2."""

ACTION_ORDER = ("ASK", "ANSWER", "DEFER")
LEARNED_POLICIES = ("q_guided_distillation", "relative_advantage", "learned_router")
BASELINE_POLICIES = ("always_answer", "always_ask", "always_defer")
DEPLOYABLE_POLICIES = BASELINE_POLICIES + LEARNED_POLICIES
ALL_POLICIES = DEPLOYABLE_POLICIES + ("oracle_upper_bound",)

DEFAULT_COSTS = {"ASK": 0.01, "ANSWER": 0.0, "DEFER": 0.05}
