from typing import Dict
import torch
import numpy as np
import copy
from threadpoolctl import threadpool_limits
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
    array_to_stats
)


class ManiSkillReplayImageDataset(BaseImageDataset):
    """
    Dataset per Diffusion Policy che legge un ReplayBuffer già convertito in formato Zarr
    (contenente osservazioni RGB-D e azioni ManiSkill).
    Nessuna conversione HDF5 viene fatta qui — il .zarr deve già esistere.
    """
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None):
        super().__init__()

        # carica direttamente il ReplayBuffer da .zarr
        replay_buffer = ReplayBuffer.copy_from_path(dataset_path)

        # parse obs chiavi dal meta
        rgb_keys, lowdim_keys = [], []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            t = attr.get("type", "low_dim")
            if t == "rgb":
                rgb_keys.append(key)
            elif t == "low_dim":
                lowdim_keys.append(key)

        # maschere train/val
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # costruisci sampler
        key_first_k = {}
        if n_obs_steps is not None:
            for k in rgb_keys + lowdim_keys:
                key_first_k[k] = n_obs_steps

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.max_train_episodes = max_train_episodes

    # --- validation dataset ---
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    # --- normalizer automatico ---
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # normalizzazione azioni
        stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = get_range_normalizer_from_stat(stat)

        # normalizzazione osservazioni lowdim
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        # normalizzazione immagini
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    # --- campionamento ---
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # (T,H,W,C) → (T,C,H,W), normalizza in [0,1]
            obs_dict[key] = np.moveaxis(
                data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32))
        }
        return torch_data
