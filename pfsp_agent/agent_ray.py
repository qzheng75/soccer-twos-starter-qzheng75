"""
PFSP-trained agent for soccer-twos competition.

Loads the "default" policy from the PFSP checkpoint and uses it for inference.
"""

import os
from typing import Dict

import gym
import numpy as np
import ray
import soccer_twos
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface

ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results",
    "PPO_PFSP",
    "PPO_Soccer_5383d_00000_0_2026-03-19_12-11-46",
    "checkpoint_000883",
    "checkpoint-883",
)
POLICY_NAME = "default"

POOL_SIZE = 5  # must match train_pfsp.py


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """A RLLib wrapper so our env can inherit from MultiAgentEnv."""

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


def create_rllib_env(env_config: dict = {}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return RLLibWrapper(env)


class PFSPAgent(AgentInterface):
    """
    Agent that uses the trained PFSP (Prioritized Fictitious Self-Play) policy.

    The "default" policy from the checkpoint is used for all team members.
    """

    def __init__(self, env: gym.Env):
        super().__init__()

        ray.init(ignore_reinit_error=True)
        tune.registry.register_env("Soccer", create_rllib_env)

        obs_space = env.observation_space
        act_space = env.action_space

        policies = {
            "default": (None, obs_space, act_space, {}),
            "opponent_active": (None, obs_space, act_space, {}),
        }
        for i in range(POOL_SIZE):
            policies[f"opponent_{i}"] = (None, obs_space, act_space, {})

        config = {
            "framework": "torch",
            "num_workers": 0,
            "num_gpus": 0,
            "env": "Soccer",
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": tune.function(lambda agent_id: "default"),
                "policies_to_train": [],
            },
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },
            "explore": False,
        }

        self.trainer = PPOTrainer(config=config)
        self.trainer.restore(os.path.normpath(CHECKPOINT_PATH))

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Compute actions for all team members using the trained PFSP policy.

        Args:
            observation: dict mapping team member id -> observation array (336,)
        Returns:
            dict mapping team member id -> action array (MultiDiscrete [3,3,3])
        """
        actions = {}
        for player_id, obs in observation.items():
            actions[player_id] = self.trainer.compute_action(
                obs, policy_id=POLICY_NAME, explore=False
            )
        return actions
