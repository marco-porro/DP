from typing import Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import common as ms_common  # for flatten_state_dict


def _to_flat_obs(x: Union[np.ndarray, dict]) -> np.ndarray:
    if isinstance(x, dict):
        x = ms_common.flatten_state_dict(x)
    return np.asarray(x, dtype=np.float32).reshape(-1)


class ManiSkillLowdimWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        obs_mode: str = "state",
        control_mode: str = "pd_joint_delta_pos",
        reward_mode: str = "dense",
        render_mode: str = "rgb_array",
        record_dir: Optional[str] = None,
        record_fps: int = 30,
        save_traj: bool = False,
        save_video: bool = True,
        max_steps: int = 400,
        **env_kwargs,
    ):
        import mani_skill.envs

        # Crea l’ambiente ManiSkill
        env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode=reward_mode,
            render_mode=render_mode,
            **env_kwargs,
        )

        # ✅ Forza la durata massima dell’episodio
        # ManiSkill 3 non sempre espone _max_episode_steps, quindi proviamo tutte le varianti
        for attr in ["max_episode_steps", "_max_episode_steps"]:
            try:
                if hasattr(env, attr):
                    setattr(env, attr, max_steps)
                if hasattr(env.unwrapped, attr):
                    setattr(env.unwrapped, attr, max_steps)
            except Exception:
                pass

        # Conversione a Gym compatibile (CPU)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

        # Aggiungi registrazione video (senza troncamento forzato)
        if record_dir is not None:
            env = RecordEpisode(
                env,
                output_dir=record_dir,
                save_trajectory=save_traj,
                trajectory_name="trajectory",
                save_video=save_video,
                video_fps=record_fps,
                max_steps_per_video=None,  # ✅ evita trunc
            )

        self.env = env

        # Spazi osservazioni / azioni
        obs, _ = self.env.reset(seed=0)
        obs = _to_flat_obs(obs)
        self.observation_space = Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

        low = np.asarray(self.env.action_space.low, np.float32).reshape(-1)
        high = np.asarray(self.env.action_space.high, np.float32).reshape(-1)
        self.action_space = Box(low, high, dtype=np.float32)

        self.max_steps = max_steps
        self.record_fps = record_fps

    # ---------------------------- GYM API ----------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return _to_flat_obs(obs), info

    def step(self, action):
        action = np.asarray(action, np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return _to_flat_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self, **kwargs):
        frame = self.env.render(**kwargs)
        try:
            import torch
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
        except Exception:
            pass
        frame = np.asarray(frame)
        if frame.ndim == 4 and frame.shape[-1] in (1, 3, 4):
            frame = frame[0]
        if frame.dtype != np.uint8:
            fmin, fmax = float(frame.min()), float(frame.max())
            if fmax <= 1.0:
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)
        return frame

    def seed(self, seed: Optional[int] = None):
        self.env.reset(seed=seed)
        return [seed]

    def close(self):
        self.env.close()
