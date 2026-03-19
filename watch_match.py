"""
Watch a match using a trained PFSP policy.

Usage:
    python watch_match.py
    python watch_match.py --checkpoint path/to/checkpoint-300
    python watch_match.py --checkpoint path/to/checkpoint-300 --opponent random
    python watch_match.py --episodes 3

Team 0 (blue,  agents 0 & 1): trained "default" policy from the checkpoint
Team 1 (orange, agents 2 & 3): same trained policy by default, or random if --opponent random
"""

import argparse
import os

import numpy as np
import ray
import soccer_twos
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from utils import create_rllib_env

# ---------------------------------------------------------------------------
# Default checkpoint — latest from train_pfsp.py run
# ---------------------------------------------------------------------------
_TRIAL_DIR = (
    "./ray_results/PPO_PFSP/"
    "PPO_Soccer_5383d_00000_0_2026-03-19_12-11-46"
)
DEFAULT_CHECKPOINT = os.path.join(_TRIAL_DIR, "checkpoint_000300", "checkpoint-300")


POOL_SIZE = 5  # must match train_pfsp.py


def build_trainer(checkpoint_path, obs_space, act_space):
    """Restore a PPOTrainer from checkpoint with the same config used in train_pfsp.py."""
    # Must include all policies that were saved in the checkpoint.
    policies = {
        "default": (None, obs_space, act_space, {}),
        "opponent_active": (None, obs_space, act_space, {}),
    }
    for i in range(POOL_SIZE):
        policies[f"opponent_{i}"] = (None, obs_space, act_space, {})

    config = {
        "framework": "torch",
        "num_workers": 0,       # no rollout workers needed for inference
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
    trainer = PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    return trainer


def run_match(checkpoint_path, opponent="trained", num_episodes=1):
    ray.init(ignore_reinit_error=True)
    tune.registry.register_env("Soccer", create_rllib_env)

    # Build a headless temp env just to get spaces
    temp_env = create_rllib_env({})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    trainer = build_trainer(checkpoint_path, obs_space, act_space)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Team 0 (agents 0,1): trained policy")
    print(f"Team 1 (agents 2,3): {'trained policy (self-play)' if opponent == 'trained' else 'random'}")

    for ep in range(num_episodes):
        # watch=True uses the audience-friendly binary at 1x speed with graphics
        env = soccer_twos.make(worker_id=1)
        obs = env.reset()
        done = {"__all__": False}
        total_reward = {i: 0.0 for i in range(4)}
        step = 0

        while not done["__all__"]:
            actions = {}
            for agent_id, agent_obs in obs.items():
                if agent_id in (0, 1) or opponent == "trained":
                    actions[agent_id] = trainer.compute_action(
                        agent_obs, policy_id="default", explore=False
                    )
                else:
                    actions[agent_id] = act_space.sample()

            obs, reward, done, _ = env.step(actions)
            for agent_id, r in reward.items():
                total_reward[agent_id] += r
            step += 1

        env.close()

        team0 = total_reward[0] + total_reward[1]
        team1 = total_reward[2] + total_reward[3]
        print(
            f"Episode {ep + 1}/{num_episodes} finished after {step} steps — "
            f"Team0 reward: {team0:.2f}, Team1 reward: {team1:.2f}"
        )

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the RLLib checkpoint file (e.g. checkpoint_000300/checkpoint-300)",
    )
    parser.add_argument(
        "--opponent",
        choices=["trained", "random"],
        default="trained",
        help="What policy team 1 uses (default: trained, i.e. self-play)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to watch",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        print("Pass a valid path with --checkpoint")
        raise SystemExit(1)

    run_match(args.checkpoint, opponent=args.opponent, num_episodes=args.episodes)
