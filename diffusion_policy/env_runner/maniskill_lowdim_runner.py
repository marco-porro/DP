import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources")

import os
import math
import dill
import wandb
import tqdm
import torch
import pathlib
import collections
import numpy as np
import wandb.sdk.data_types.video as wv

from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.maniskill.maniskill_lowdim_wrapper import ManiSkillLowdimWrapper


class ManiSkillLowdimRunner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir,
        env_id="PickCube-v1",
        n_train=1,
        n_train_vis=1,
        train_start_seed=0,
        n_test=1,
        n_test_vis=1,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        n_latency_steps=0,
        render_mode="rgb_array",
        fps=30,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        control_mode="pd_joint_delta_pos",
        reward_mode="dense",
        obs_mode="state",
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        # ----------------------------------------------------------------------
        # Funzione di creazione ambiente (usa solo ManiSkillLowdimWrapper)
        # ----------------------------------------------------------------------
        def env_fn():
            env = MultiStepWrapper(
                ManiSkillLowdimWrapper(
                    env_id=env_id,
                    control_mode=control_mode,
                    reward_mode=reward_mode,
                    render_mode=render_mode,
                    obs_mode=obs_mode,
                    record_dir=os.path.join(output_dir, "media"),
                    record_fps=fps,
                    save_traj=False,
                    save_video=True,
                    max_steps=max_steps,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps * env_n_action_steps,
            )
            return env

        env_fns = [env_fn] * n_envs
        env_seeds, env_prefixs, env_init_fn_dills = [], [], []

        # ----------------------------------------------------------------------
        # TRAIN ENVIRONMENTS
        # ----------------------------------------------------------------------
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                env.seed(seed)
            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # ----------------------------------------------------------------------
        # TEST ENVIRONMENTS
        # ----------------------------------------------------------------------
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                env.seed(seed)
            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # ----------------------------------------------------------------------
        # Vectorized environment
        # ----------------------------------------------------------------------
        self.env = SyncVectorEnv(env_fns)
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.env_id = env_id

    # ----------------------------------------------------------------------
    # ROLLOUT
    # ----------------------------------------------------------------------
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_rewards = [None] * n_inits
        all_video_paths = [None] * n_inits
        log_data = dict()

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start

            init_fns = self.env_init_fn_dills[this_global_slice]
            env.call_each("run_dill_function", args_list=[(x,) for x in init_fns])

            obs, _ = env.reset()
            past_action = None
            policy.reset()

            max_reward_vec = -np.inf * np.ones(this_n_active_envs, dtype=np.float32)

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Rollout {self.env_id} {chunk_idx+1}/{n_chunks}",
                leave=True,
                mininterval=self.tqdm_interval_sec,
                smoothing=0.1,
            )

            done = False
            while not done:
                np_obs_dict = {"obs": obs[:, : self.n_obs_steps].astype(np.float32)}
                if self.past_action and (past_action is not None):
                    np_obs_dict["past_action"] = past_action[:, -(self.n_obs_steps - 1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device, dtype=dtype))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
                action = np_action_dict["action"][:, self.n_latency_steps:]

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action detected.")

                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, terminated, truncated, info = env.step(env_action)
                done = np.logical_or(terminated, truncated)
                done = np.all(done)

                max_reward_vec = np.maximum(max_reward_vec, reward.astype(np.float32))
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            all_rewards[this_global_slice] = max_reward_vec.copy()

        # ----------------------------------------------------------------------
        # Logging finale su W&B
        # ----------------------------------------------------------------------
        media_dir = pathlib.Path(self.output_dir) / "media"
        videos = sorted(media_dir.glob("*.mp4"))

        max_rewards = collections.defaultdict(list)
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

        for idx, video_path in enumerate(videos[:n_inits]):
            prefix = self.env_prefixs[idx]
            seed = self.env_seeds[idx]
            sim_video = wandb.Video(str(video_path), format="mp4")
            log_data[prefix + f"sim_video_{seed}"] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            log_data[name] = np.mean(value)

        return log_data

    def undo_transform_action(self, action):
        if self.rotation_transformer is None:
            return action
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)
        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)
        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction
