import gymnasium as gym
import numpy as np
import torch

class ManiSkillLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        obs_keys=None,
        init_state=None,
        max_steps=400,
    ):
        super().__init__()
        self.env = env
        self.obs_keys = obs_keys or []   # non usato direttamente, ma lasciato per compatibilità
        self.init_state = init_state
        self.max_steps = max_steps
        self._seed = None
        self._step_count = 0
        self.num_envs = getattr(env, "num_envs", 1)

        # Gymnasium space setup
        if hasattr(env, "action_space"):
            self.action_space = env.action_space
        else:
            raise AttributeError("Env must define action_space")

        # Determina observation_space (flattened state)
        obs, _ = env.reset()
        obs_flat = self._flatten_obs(obs)
        low = np.full_like(obs_flat, fill_value=-np.inf)
        high = np.full_like(obs_flat, fill_value=np.inf)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    # ----------------------------------------------------------------------
    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]

    # ----------------------------------------------------------------------
    def _flatten_obs(self, obs):
        """
        Converte i dizionari ManiSkill (torch tensors o numpy) in un singolo vettore np.float32.
        """
        # ManiSkill usa torch.Tensor batched -> scegli env[0]
        if isinstance(obs, dict):
            # ManiSkill 'state' mode -> dict di torch.tensor
            if "agent" in obs and isinstance(obs["agent"], dict):
                values = []
                for v in obs["agent"].values():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                    v = np.asarray(v, dtype=np.float32)
                    # batch dimension
                    if v.ndim > 1:
                        v = v[0]
                    values.append(v.reshape(-1))
                flat = np.concatenate(values, axis=0)
                return flat
            else:
                # già flattened o altro dict
                all_vals = []
                for v in obs.values():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                    v = np.asarray(v, dtype=np.float32)
                    if v.ndim > 1:
                        v = v[0]
                    all_vals.append(v.reshape(-1))
                return np.concatenate(all_vals, axis=0)
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
            if obs.ndim > 1:
                obs = obs[0]
            return obs.astype(np.float32)
        else:
            return np.asarray(obs, dtype=np.float32).reshape(-1)

    # ----------------------------------------------------------------------
    def reset(self, **kwargs):
        """
        Reset compatibile con Gymnasium, gestisce init_state e seed.
        """
        if self.init_state is not None:
            self.env.set_state_dict(self.init_state)

        # evita seed duplicato
        if "seed" in kwargs:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, info = self.env.reset(seed=self._seed, **kwargs)

        self._seed = None
        self._step_count = 0
        obs_flat = self._flatten_obs(obs)
        return obs_flat, info

    # ----------------------------------------------------------------------
    def step(self, action):
        """
        Step compatibile con Gymnasium, converte osservazioni in float32 flattened.
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_flat = self._flatten_obs(obs)
        self._step_count += 1

        # Se il wrapper interno non restituisce truncated, creiamolo noi
        if truncated is None:
            truncated = self._step_count >= self.max_steps

        # Calcolo done solo interno (non nel return)
        done = terminated or truncated

        return obs_flat, float(reward), terminated, truncated, info

    # ----------------------------------------------------------------------
    def render(self, **kwargs):
        """
        Usa l'API Gymnasium per render. Converte torch -> numpy uint8.
        """
        # In Gymnasium, il render_mode è definito al momento della creazione dell'env
        try:
            frame = self.env.render(**kwargs)
        except TypeError:
            # Fallback: alcuni wrapper non accettano kwargs
            frame = self.env.render()

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            fmin, fmax = float(frame.min()), float(frame.max())
            if fmax <= 1.0:
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)
        return frame

    # ----------------------------------------------------------------------
    def get_observation(self):
        """
        Restituisce l'osservazione corrente flattenata.
        """
        obs = self.env.unwrapped.get_obs() if hasattr(self.env.unwrapped, "get_obs") else None
        if obs is None:
            obs, _ = self.env.reset()
        return self._flatten_obs(obs)
