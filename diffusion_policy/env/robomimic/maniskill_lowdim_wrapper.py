from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

# ManiSkill3 envs
import mani_skill.envs

class ManiSkillLowdimWrapper(gym.Env):
    """
    Wrapper per ambienti ManiSkill (Gymnasium >= 0.29)
    che converte le osservazioni in un vettore low-dim concatenato,
    compatibile con i dataset e le policy di Diffusion Policy.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        obs_keys: Optional[List[str]] = None,
        control_mode: str = "pd_joint_delta_pos",
        reward_mode: str = "dense",
        obs_mode: str = "state_dict",
        num_envs: int = 1,
        render_hw: Tuple[int, int] = (256, 256),
        render_camera_name: str = "base_camera",
        init_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Args:
            env_id: nome dell'ambiente ManiSkill (es. "PickCube-v1")
            obs_keys: lista di chiavi del dict di osservazioni da concatenare
            control_mode: controllore (es. "pd_joint_delta_pos")
            reward_mode: tipo di reward ("dense" o "sparse")
            obs_mode: modalità di osservazione ("state_dict", "rgbd", ecc.)
            num_envs: numero di env in parallelo
            render_hw: risoluzione per il render RGB
            render_camera_name: nome della camera (es. "base_camera")
            init_state: stato opzionale da cui ripartire
        """
        super().__init__()

        # Crea l'ambiente ManiSkill (Gymnasium API)
        self.env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode=reward_mode,
            num_envs=num_envs,
            **kwargs
        )

        self.obs_keys = (
            obs_keys
            or [
                "robot0_eef_pos",
                "robot0_eef_rot",
                "robot0_gripper_qpos",
                "object",
            ]
        )
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.init_state = init_state

        # gestione seed / cache stati
        self.seed_state_map: Dict[int, Any] = {}
        self._seed: Optional[int] = None

        # Setup spazi Gymnasium
        act_space = self.env.action_space
        self.action_space = Box(
            low=act_space.low, high=act_space.high, shape=act_space.shape, dtype=np.float32
        )

        obs_example, _ = self.env.reset(seed=0)
        obs_vec = self._flatten_observation(obs_example)
        low = np.full_like(obs_vec, -np.inf, dtype=np.float32)
        high = np.full_like(obs_vec, np.inf, dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    # ----------------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------------

    def _flatten_observation(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatena le chiavi specificate in un unico vettore float32."""
        parts = []
        for k in self.obs_keys:
            val = raw_obs.get(k, None)
            if val is None:
                continue
            val = np.asarray(val, dtype=np.float32).reshape(-1)
            parts.append(val)
        if not parts:
            raise KeyError(f"Nessuna chiave valida trovata in obs: {self.obs_keys}")
        return np.concatenate(parts, axis=0).astype(np.float32)

    # ----------------------------------------------------------------------
    # GYMNASIUM API
    # ----------------------------------------------------------------------

    def seed(self, seed: Optional[int] = None):
        """Imposta il seed randomico e lo memorizza localmente."""
        np.random.seed(seed=seed)
        self._seed = seed
        return [seed]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Resetta l’ambiente secondo Gymnasium API."""
        if seed is not None:
            self._seed = seed

        if self.init_state is not None:
            # Reset deterministico a uno stato noto
            obs, info = self.env.reset(seed=self._seed, options={"state_dict": self.init_state})
        elif self._seed is not None:
            # Reset per seed specifico con caching
            if self._seed in self.seed_state_map:
                obs, info = self.env.reset(
                    seed=self._seed, options={"state_dict": self.seed_state_map[self._seed]}
                )
            else:
                obs, info = self.env.reset(seed=self._seed)
                if "env_states" in info:
                    self.seed_state_map[self._seed] = info["env_states"]
        else:
            # Reset casuale standard
            obs, info = self.env.reset()

        obs_vec = self._flatten_observation(obs)
        return obs_vec, info

    def step(self, action: np.ndarray):
        """
        Esegue uno step nell’ambiente.
        ManiSkill (Gymnasium >=0.29) restituisce:
        obs, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_vec = self._flatten_observation(obs)
        done = bool(np.any(terminated) or np.any(truncated))
        return obs_vec, float(np.mean(reward)), done, info

    def render(self, mode: str = "rgb_array"):
        """Renderizza l'ambiente in RGB."""
        h, w = self.render_hw
        frame = self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )
        if isinstance(frame, list):
            frame = frame[0]
        return np.asarray(frame)

    def close(self):
        """Chiude l'ambiente ManiSkill."""
        try:
            self.env.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Test rapido (solo CPU, state-based)
# ----------------------------------------------------------------------

def test():
    import matplotlib.pyplot as plt

    wrapper = ManiSkillLowdimWrapper(
        env_id="PickCube-v1",
        obs_keys=["agent", "cube", "goal_site", "panda"],
        control_mode="pd_joint_delta_pos",
        obs_mode="state_dict",
        reward_mode="dense",
    )

    obs, info = wrapper.reset(seed=0)
    print("Initial obs shape:", obs.shape)

    done = False
    total_rew = 0
    while not done:
        act = wrapper.action_space.sample()
        obs, rew, done, info = wrapper.step(act)
        total_rew += rew

    print("Episode reward:", total_rew)
    img = wrapper.render()
    plt.imshow(img)
    plt.title("ManiSkill Render")
    plt.show()


if __name__ == "__main__":
    test()
