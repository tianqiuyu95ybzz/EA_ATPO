"""EA-ATPO entrypoint that adds only an EA reward-accounting adapter."""

from recipe.atpo import mt_main as _atpo_main
from recipe.atpo.agent_loop import AgentLoopWorker

from .ea_reward_manager import EARewardManager
from .ea_metrics import patch_agent_loop_worker


_load_atpo_reward_manager = _atpo_main.load_reward_manager
patch_agent_loop_worker(AgentLoopWorker)


def _load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    if config.reward_model.get("reward_manager", "naive") == "ea_atpo":
        return EARewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=_atpo_main.get_custom_reward_fn(config),
            reward_fn_key=config.data.reward_fn_key,
            reward_coefficients=config.get("reward_coefficients", {}),
            **reward_kwargs,
        )
    return _load_atpo_reward_manager(config, tokenizer, num_examine, **reward_kwargs)


# mt_main.main resolves this global when it constructs the trainer.
_atpo_main.load_reward_manager = _load_reward_manager


if __name__ == "__main__":
    _atpo_main.main()
