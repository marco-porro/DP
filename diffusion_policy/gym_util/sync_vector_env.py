import numpy as np
from copy import deepcopy

# ✅ cambiato da gym → gymnasium
import gymnasium as gym
from gymnasium import logger
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.vector.utils import concatenate, create_empty_array
from gymnasium.spaces import Space

__all__ = ["SyncVectorEnv"]


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.
    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.
    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.
    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.
    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super(SyncVectorEnv, self).__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_observation_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        # self._rewards = [0] * self.num_envs
        # self._dones = [False] * self.num_envs
        self._actions = None

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        # ✅ Gymnasium: non ha env.seed(), si usa env.reset(seed=...)
        for env, seed in zip(self.envs, seeds):
            env.reset(seed=seed)


    def reset_wait(self, seed=None, options=None):
        self._dones[:] = False
        observations = []
        infos = []
        for env in self.envs:
            observation, info = env.reset(seed=seed, options=options)
            observations.append(observation)
            infos.append(info)

        # --- MANUAL MERGE: handle lists, tuples, dicts, etc. safely ---
        first_obs = observations[0]

        # force manual path if the observations are non-array types
        if isinstance(first_obs, (list, tuple, dict)):
            if isinstance(first_obs, (list, tuple)):
                # list/tuple of arrays per env -> list of (num_envs, ...)
                self.observations = [
                    np.stack([o[i] for o in observations]) for i in range(len(first_obs))
                ]
            elif isinstance(first_obs, dict):
                # dict of arrays per env -> dict of (num_envs, ...)
                self.observations = {
                    k: np.stack([o[k] for o in observations]) for k in first_obs.keys()
                }
        else:
            # fallback to Gymnasium behaviour
            try:
                self.observations = concatenate(
                    observations, self.observations, self.single_observation_space
                )
            except Exception:
                # final fallback if concatenate still fails
                self.observations = np.stack(observations)

        return (deepcopy(self.observations) if self.copy else self.observations), infos


    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            # ✅ Gymnasium step() -> (obs, reward, terminated, truncated, info)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self._rewards[i] = reward
            self._dones[i] = done
            observations.append(observation)
            infos.append(info)

        # ✅ Combine observations safely (same logic as reset_wait)
        first_obs = observations[0]
        # ensure elements are arrays
        observations = [np.asarray(o) for o in observations]

        need_manual = (not isinstance(self.single_observation_space, Space)) \
                      or isinstance(first_obs, (list, tuple, dict))

        if need_manual:
            if isinstance(first_obs, (list, tuple)):
                self.observations = [
                    np.stack([o[i] for o in observations]) for i in range(len(first_obs))
                ]
            elif isinstance(first_obs, dict):
                self.observations = {
                    k: np.stack([o[k] for o in observations]) for k in first_obs.keys()
                }
            else:
                self.observations = np.stack(observations)
        else:
            try:
                self.observations = concatenate(
                    observations, self.observations, self.single_observation_space
                )
            except Exception:
                # 🔧 robust fallback
                self.observations = np.stack(observations)

        terminateds = np.array(self._dones, dtype=np.bool_)
        truncateds = np.zeros_like(terminateds, dtype=np.bool_)  # placeholder

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            terminateds,
            truncateds,
            infos,
        )


    def close_extras(self, **kwargs):
        [env.close() for env in self.envs]

    def _check_observation_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                break
        else:
            return True
        raise RuntimeError(
            "Some environments have an observation space "
            "different from `{0}`. In order to batch observations, the "
            "observation spaces from all environments must be "
            "equal.".format(self.single_observation_space)
        )
    
    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def call_each(self, name: str, 
            args_list: list=None, 
            kwargs_list: list=None):
        n_envs = len(self.envs)
        if args_list is None:
            args_list = [[]] * n_envs
        assert len(args_list) == n_envs

        if kwargs_list is None:
            kwargs_list = [dict()] * n_envs
        assert len(kwargs_list) == n_envs

        results = []
        for i, env in enumerate(self.envs):
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args_list[i], **kwargs_list[i]))
            else:
                results.append(function)

        return tuple(results)


    def render(self, *args, **kwargs):
        return self.call('render', *args, **kwargs)
    
    def set_attr(self, name: str, values):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)
