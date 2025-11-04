import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import defaultdict, deque
import dill


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    import gym
    import gymnasium.spaces as gymn_spaces

    if isinstance(space, (gym.spaces.Box, gymn_spaces.Box)):
        return repeated_box(space, n)
    elif isinstance(space, (gym.spaces.Dict, gymn_spaces.Dict)):
        result_space = gymn_spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}!!!")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    """Aggrega una lista di dati numerici o booleani gestendo array eterogenei."""
    if len(data) == 0:
        return 0

    # normalizza tutti gli elementi
    flat = []
    for d in data:
        if d is None:
            continue
        if hasattr(d, "detach"):
            d = d.detach().cpu().numpy()
        d = np.asarray(d)
        if d.ndim == 0:
            d = d.reshape(1)
        flat.append(d)

    if len(flat) == 0:
        return 0

    # concatenazione sicura
    try:
        arr = np.concatenate(flat).ravel()
    except Exception:
        arr = np.array(flat, dtype=object)

    # riduzione secondo il metodo
    if method == "max":
        return np.max(arr)
    elif method == "min":
        return np.min(arr)
    elif method == "mean":
        return np.mean(arr.astype(np.float32))
    elif method == "sum":
        return np.sum(arr)
    else:
        raise NotImplementedError(f"Unknown aggregation method: {method}")


def stack_last_n_obs(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros(
        (n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
    )
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_obs_steps,
        n_action_steps,
        max_episode_steps=None,
        reward_agg_method="max",
    ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method

        self.obs = deque(maxlen=n_obs_steps + 1)
        self.reward = []
        self.done = []
        self.info = defaultdict(lambda: deque(maxlen=n_obs_steps + 1))

    def reset(self, **kwargs):
        """Resetta l'ambiente (Gymnasium style)."""
        obs, info = super().reset(**kwargs)

        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        self.reward = []
        self.done = []
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))

        obs = self._get_obs(self.n_obs_steps)
        return obs, info

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            # interrompi se già finito
            if len(self.done) > 0 and bool(self.done[-1]):
                break

            observation, reward, terminated, truncated, info = super().step(act)

            # normalizza osservazione
            if isinstance(observation, (list, tuple)):
                observation = np.array(observation, dtype=np.float32)
            elif isinstance(observation, dict):
                from mani_skill.utils import common as ms_common
                observation = ms_common.flatten_state_dict(observation)
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)

            done = bool(terminated or truncated)

            self.obs.append(observation)
            self.reward.append(reward)

            # limite temporale
            if (
                self.max_episode_steps is not None
                and len(self.reward) >= self.max_episode_steps
            ):
                truncated = True
                terminated = False
                done = True

            self.done.append(done)
            self._add_info(info)

        # aggregazione n-step
        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)

        # episodi finiti
        any_done = bool(aggregate(self.done, "max"))

        limit_trunc = (
            self.max_episode_steps is not None
            and len(self.reward) >= self.max_episode_steps
        )
        terminated = bool(any_done and not limit_trunc)
        truncated = bool(limit_trunc)

        info = dict_take_last_n(self.info, self.n_obs_steps)

        return observation, reward, terminated, truncated, info

    def _get_obs(self, n_steps=1):
        """Restituisce stack delle ultime osservazioni."""
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs], n_steps
                )
            return result
        else:
            raise RuntimeError("Unsupported space type")

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def get_rewards(self):
        return self.reward

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)

    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
