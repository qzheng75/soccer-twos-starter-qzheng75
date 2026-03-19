"""
Prioritized Fictitious Self-Play (PFSP) for Soccer-Twos.

Reference: Vinyals et al. (2019) "Grandmaster level in StarCraft II using multi-agent
reinforcement learning" (AlphaStar). Section on PFSP curriculum.

Core idea:
  - Maintain a circular pool of N past policy snapshots.
  - Sample opponents with probability proportional to f(win_rate),
    where f(p) = (1 - p)^alpha.  This prioritizes opponents we rarely beat
    (hard but learnable) over opponents we always beat or always lose to.
  - Win rates are tracked per opponent via exponential moving average (EMA)
    using episode_reward_mean from each training iteration.
  - Every SNAPSHOT_EVERY training iterations the current policy is frozen
    into the next pool slot (overwriting the oldest when the pool is full).

RLLib 1.4.0 compatibility notes:
  - policy_mapping_fn is called as fn(agent_id) with no episode arg.
    Therefore we use a single fixed "opponent_active" slot for agents 2 & 3,
    and swap its weights each iteration via on_train_result on the driver.
  - All PFSP logic (sampling, win-rate updates, snapshots) runs on the driver
    in on_train_result — workers need no access to shared state.
"""

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import create_rllib_env

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
POOL_SIZE = 5          # max number of frozen opponent snapshots
SNAPSHOT_EVERY = 30    # freeze a new snapshot every N training iterations
WIN_RATE_EMA = 0.1     # EMA coefficient for win-rate updates (higher = faster)
PFSP_ALPHA = 2.0       # weight = (1 - win_rate)^alpha; higher = harder focus
NUM_ENVS_PER_WORKER = 3


# ---------------------------------------------------------------------------
# Shared PFSP state — Ray remote actor on the driver, no worker access needed
# ---------------------------------------------------------------------------
@ray.remote
class PFSPStore:
    """
    Holds all PFSP bookkeeping.  Only called from the driver (on_train_result).
    """

    def __init__(self, pool_size: int, pfsp_alpha: float):
        self.pool_size = pool_size
        self.pfsp_alpha = pfsp_alpha
        self.active_pool = []    # ordered list of slot IDs currently filled
        self.win_rates = {}      # slot_id -> EMA win rate in [0, 1]
        self.slot_ptr = 0        # monotonically increasing; slot = ptr % pool_size
        self.active_slot = None  # slot whose weights are in "opponent_active"

    def add_snapshot(self) -> str:
        """Allocate next circular slot. Returns slot_id for weight copying."""
        slot_id = f"opponent_{self.slot_ptr % self.pool_size}"
        if slot_id not in self.active_pool:
            self.active_pool.append(slot_id)
        self.win_rates[slot_id] = 0.5   # assume 50% win rate for fresh snapshot
        self.slot_ptr += 1
        return slot_id

    def update_active_win_rate(self, win_rate_est: float, ema_alpha: float):
        """EMA update for the currently active opponent slot."""
        if self.active_slot and self.active_slot in self.win_rates:
            old = self.win_rates[self.active_slot]
            self.win_rates[self.active_slot] = (
                (1.0 - ema_alpha) * old + ema_alpha * win_rate_est
            )

    def pfsp_sample(self) -> str:
        """
        PFSP-weighted sample from the active pool.
        Sets self.active_slot and returns the chosen slot_id.
        Returns None if the pool is still empty.
        """
        if not self.active_pool:
            return None
        weights = np.array([self._weight(self.win_rates[p]) for p in self.active_pool])
        weights /= weights.sum()
        chosen = str(np.random.choice(self.active_pool, p=weights))
        self.active_slot = chosen
        return chosen

    def get_state(self) -> dict:
        return {
            "active_pool": list(self.active_pool),
            "win_rates": {
                k: round(v, 3)
                for k, v in self.win_rates.items()
                if k in self.active_pool
            },
            "active_slot": self.active_slot,
        }

    def _weight(self, win_rate: float) -> float:
        return max((1.0 - win_rate) ** self.pfsp_alpha, 1e-4)


# ---------------------------------------------------------------------------
# Policy mapping
# ---------------------------------------------------------------------------
def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    RLLib 1.4.0 calls this as fn(agent_id) — no episode argument.

    Team 0 (agents 0, 1): always "default" (the policy being trained).
    Team 1 (agents 2, 3): always "opponent_active" (weights swapped each iter).
    """
    if agent_id in (0, 1):
        return "default"
    return "opponent_active"


# ---------------------------------------------------------------------------
# Callbacks — all PFSP logic runs on driver in on_train_result
# ---------------------------------------------------------------------------
def make_callback_class(store_handle):
    """
    Factory returning a DefaultCallbacks subclass with the store handle
    baked in.  RLLib 1.4.0 requires a class (not instance) in the config.
    """

    class PFSPCallback(DefaultCallbacks):
        _store = store_handle
        _train_iter = 0

        def on_train_result(self, **info):
            """
            Runs on driver after each training iteration.

            Each iteration:
              1. Update win rate for the currently active opponent using
                 episode_reward_mean normalized to [0, 1].
              2. Optionally freeze current "default" weights as a new snapshot.
              3. PFSP-sample next opponent and copy its weights to "opponent_active".
            """
            PFSPCallback._train_iter += 1
            trainer = info["trainer"]
            result = info["result"]

            # --- 1. Update win rate ---
            # episode_reward_mean is per-agent mean; in soccer-twos zero-sum,
            # team 0 reward per game is in {-1, 0, +1} so the mean is in [-1, 1].
            # Map to [0, 1] as a win-rate estimate.
            mean_reward = result.get("episode_reward_mean", 0.0)
            win_rate_est = (mean_reward + 1.0) / 2.0
            self._store.update_active_win_rate.remote(win_rate_est, WIN_RATE_EMA)

            # --- 2. Snapshot current policy ---
            should_snap = (
                PFSPCallback._train_iter == 1           # always snapshot immediately
                or PFSPCallback._train_iter % SNAPSHOT_EVERY == 0
            )
            if should_snap:
                slot_id = ray.get(self._store.add_snapshot.remote())
                default_weights = trainer.get_weights(["default"])["default"]
                trainer.set_weights({slot_id: default_weights})

            # --- 3. PFSP-sample and activate next opponent ---
            chosen = ray.get(self._store.pfsp_sample.remote())
            if chosen:
                chosen_weights = trainer.get_weights([chosen])[chosen]
                trainer.set_weights({"opponent_active": chosen_weights})

            state = ray.get(self._store.get_state.remote())
            print(
                f"\n[PFSP iter={PFSPCallback._train_iter}] "
                f"active={state['active_slot']}  "
                f"win_rates={state['win_rates']}\n"
            )

    return PFSPCallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ray.init()

    store = PFSPStore.remote(POOL_SIZE, PFSP_ALPHA)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    # "opponent_active" is the live opponent slot (weights swapped each iter).
    # "opponent_0" .. "opponent_N" are the frozen snapshot pool.
    policies = {
        "default": (None, obs_space, act_space, {}),
        "opponent_active": (None, obs_space, act_space, {}),
    }
    for i in range(POOL_SIZE):
        policies[f"opponent_{i}"] = (None, obs_space, act_space, {})

    analysis = tune.run(
        "PPO",
        name="PPO_PFSP",
        config={
            # system
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # PFSP callbacks
            "callbacks": make_callback_class(store),
            # multiagent
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            # model
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 15_000_000, "time_total_s": 20000},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
