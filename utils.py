from random import uniform as randfloat

import numpy as np
import gym
from ray.rllib import MultiAgentEnv
import soccer_twos

# ---------------------------------------------------------------------------
# Field geometry (soccer-twos Unity environment, z-axis is the attack direction)
#
# Team 0 (agents 0, 1) attacks toward +z  →  target goal at +z
# Team 1 (agents 2, 3) attacks toward -z  →  target goal at -z
#
# Verify these by printing info["player_info"]["position"] early in training.
# ---------------------------------------------------------------------------
_GOAL_TEAM0_ATTACKS = np.array([0.0,  24.0])   # goal team-0 scores into (+z side)
_GOAL_TEAM1_ATTACKS = np.array([0.0, -24.0])   # goal team-1 scores into (-z side)

# Scale factors — keep dense rewards small relative to the sparse ±1 goal reward
_W_CHASE    = 0.001   # penalty per unit of distance between player and ball
_W_ATTACK   = 0.001   # penalty per unit of distance between ball and target goal
_W_VELOCITY = 0.002   # reward per unit of ball speed projected toward target goal


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.

    Adds dense intermediate rewards on top of the sparse ±1 goal reward:
      1. Ball proximity   — reward agent for staying close to the ball
      2. Ball advancement — reward ball being close to the opponent's goal
      3. Ball velocity    — reward ball moving toward the opponent's goal
    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self._shape_reward(reward, info)
        return obs, reward, done, info

    def _shape_reward(self, reward, info):
        shaped = dict(reward)

        for agent_id, base_reward in reward.items():
            agent_info = info.get(agent_id)
            if not agent_info:
                # info not available for this agent (can happen at episode start)
                continue

            try:
                player_pos = np.array(agent_info["player_info"]["position"])
                ball_pos   = np.array(agent_info["ball_info"]["position"])
                ball_vel   = np.array(agent_info["ball_info"]["velocity"])
            except (KeyError, TypeError):
                continue

            # Which goal does this agent attack?
            target_goal = (
                _GOAL_TEAM0_ATTACKS if agent_id in (0, 1) else _GOAL_TEAM1_ATTACKS
            )

            # 1. Ball proximity: encourage agent to stay close to the ball
            dist_to_ball = np.linalg.norm(player_pos - ball_pos)
            r_chase = -_W_CHASE * dist_to_ball

            # 2. Ball advancement: encourage ball to be near the opponent's goal
            dist_ball_to_goal = np.linalg.norm(ball_pos - target_goal)
            r_attack = -_W_ATTACK * dist_ball_to_goal

            # 3. Ball velocity: encourage ball moving toward the opponent's goal
            goal_dir = target_goal - ball_pos
            goal_dir_norm = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
            r_velocity = _W_VELOCITY * float(np.dot(ball_vel, goal_dir_norm))

            shaped[agent_id] = base_reward + r_chase + r_attack + r_velocity

        return shaped


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
