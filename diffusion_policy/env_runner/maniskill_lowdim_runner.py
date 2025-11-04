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
import numpy as np
import glob
import time
import wandb.sdk.data_types.video as wv
import collections

from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.maniskill.maniskill_lowdim_wrapper import ManiSkillLowdimWrapper

from mani_skill.utils.wrappers import RecordEpisode


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

        # numero env == train + test (uno per ciascun rollout in parallelo)
        if n_envs is None:
            n_envs = n_train + n_test

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        # ----------------------------------------------------------------------
        # Costruttore di env con prefisso distinto ('train' | 'test')
        # ----------------------------------------------------------------------
        def make_env_fn(prefix: str):
            def env_fn():
                import gymnasium as gym
                import mani_skill.envs

                base_env = gym.make(
                    env_id,
                    obs_mode=obs_mode,
                    control_mode=control_mode,
                    reward_mode=reward_mode,
                    render_mode=render_mode,
                )

                # forza max steps
                for attr in ["max_episode_steps", "_max_episode_steps"]:
                    try:
                        if hasattr(base_env, attr):
                            setattr(base_env, attr, max_steps)
                        if hasattr(base_env.unwrapped, attr):
                            setattr(base_env.unwrapped, attr, max_steps)
                    except Exception:
                        pass

                wrapped_env = ManiSkillLowdimWrapper(base_env, max_steps=max_steps)

                # output_dir separato per train / test
                rec_dir = os.path.join(output_dir, "media", prefix)
                os.makedirs(rec_dir, exist_ok=True)
                wrapped_env_rec = RecordEpisode(
                    wrapped_env,
                    output_dir=rec_dir,
                    max_steps_per_video=max_steps,
                    save_video=True,
                    save_trajectory=False,
                    video_fps=fps,
                )

                env = MultiStepWrapper(
                    wrapped_env_rec,
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps,
                )
                return env

            return env_fn

        # ----------------------------------------------------------------------
        # Costruisci env_fns con prefisso coerente all'indice: prima train poi test
        # ----------------------------------------------------------------------
        env_fns = []
        env_fns.extend([make_env_fn("train")] * n_train)
        env_fns.extend([make_env_fn("test")] * n_test)

        env_seeds, env_prefixs, env_init_fn_dills = [], [], []

        # ----------------------------------------------------------------------
        # TRAIN initializations
        # ----------------------------------------------------------------------
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = (i < n_train_vis)

            def init_fn(env, seed=seed, enable_render=enable_render):
                # env: MultiStepWrapper -> env.env: RecordEpisode -> env.env.env: ManiSkillLowdimWrapper
                assert isinstance(env.env.env, ManiSkillLowdimWrapper)
                env.env.env.init_state = None
                # Gymnasium: il seed effettivo va passato via reset
                env.reset(seed=seed)
                # RecordEpisode salva comunque; enable_render lo usiamo solo per decidere il logging più avanti

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # ----------------------------------------------------------------------
        # TEST initializations
        # ----------------------------------------------------------------------
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = (i < n_test_vis)

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env.env, ManiSkillLowdimWrapper)
                env.env.env.init_state = None
                env.reset(seed=seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # ----------------------------------------------------------------------
        env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.env_id = env_id
        self.n_train = n_train
        self.n_test = n_test
        self.n_train_vis = n_train_vis
        self.n_test_vis = n_test_vis

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

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # set per evitare di riusare lo stesso file quando si fa glob
        used_video_paths = set()

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])
            obs, _ = env.reset()
            past_action = None
            policy.reset()

            chunk_start_time = time.time()
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Rollout {self.env_id} Lowdim {chunk_idx+1}/{n_chunks}",
                leave=True,
                mininterval=self.tqdm_interval_sec,
                smoothing=0.1,
            )

            done = False
            total_rewards = np.zeros(n_envs, dtype=np.float32)

            while not done:
                np_obs_dict = {"obs": obs[:, : self.n_obs_steps].astype(np.float32)}
                if self.past_action and (past_action is not None):
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.n_obs_steps - 1):
                    ].astype(np.float32)

                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device, dtype=dtype),
                )

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

                action = np_action_dict["action"][:, self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action detected.")

                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, terminated, truncated, info = env.step(env_action)

                if hasattr(reward, "detach"):
                    reward = reward.detach().cpu().numpy()
                reward = np.asarray(reward, dtype=np.float32).reshape(-1)
                total_rewards[: len(reward)] += reward

                if hasattr(terminated, "detach"):
                    terminated = terminated.detach().cpu().numpy()
                if hasattr(truncated, "detach"):
                    truncated = truncated.detach().cpu().numpy()
                terminated = np.asarray(terminated, dtype=bool).reshape(-1)
                truncated = np.asarray(truncated, dtype=bool).reshape(-1)
                done_vec = np.logical_or(terminated, truncated)
                done = bool(np.all(done_vec))

                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            # se RecordEpisode espone flush_video, forwarder di gym.Wrapper la raggiunge
            try:
                env.call_each("flush_video")
            except Exception:
                pass

            # ---- raccolta video per prefisso coerente (train/test) ----
            for local_idx in range(this_n_active_envs):
                global_idx = start + local_idx
                prefix = self.env_prefixs[global_idx].rstrip("/")  # 'train' o 'test'
                media_dir = os.path.join(self.output_dir, "media", prefix)

                # prendiamo i file nuovi di questo chunk e non ancora usati
                candidates = sorted(
                    [
                        p for p in glob.glob(os.path.join(media_dir, "*.mp4"))
                        if os.path.getmtime(p) >= chunk_start_time and p not in used_video_paths
                    ]
                )
                if len(candidates) > 0:
                    chosen = candidates[0]
                    used_video_paths.add(chosen)
                    all_video_paths[global_idx] = chosen

            # rewards (una per env attivo del chunk)
            for j in range(this_n_active_envs):
                all_rewards[start + j] = float(total_rewards[j])

        # ----------------------------------------------------------------------
        # Log stile DP
        # ----------------------------------------------------------------------
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]  # 'train/' o 'test/'
            reward_sum = all_rewards[i] if all_rewards[i] is not None else 0.0
            max_rewards[prefix].append(reward_sum)
            log_data[prefix + f"sim_max_reward_{seed}"] = reward_sum

            video_path = all_video_paths[i]
            if video_path is not None and os.path.exists(video_path):
                # Mostra solo fino a *_vis per evitare spam
                is_train = prefix == "train/"
                idx_in_split = seed - (self.env_seeds[0] if is_train else self.env_seeds[self.n_train])
                show_vis = (idx_in_split < (self.n_train_vis if is_train else self.n_test_vis))
                if show_vis:
                    sim_video = wandb.Video(video_path, format="mp4")
                    log_data[prefix + f"sim_video_{seed}"] = sim_video

        for prefix, values in max_rewards.items():
            mean_score = float(np.mean(values)) if len(values) > 0 else 0.0
            log_data[prefix + "mean_score"] = mean_score

        return log_data

    # ----------------------------------------------------------------------
    def undo_transform_action(self, action):
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
