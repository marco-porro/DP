from typing import Dict, List, Optional
import os
import copy
import torch
import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.common.normalize_util import (
    array_to_stats, get_identity_normalizer_from_stat
)

# ======================================================================
# Utility: crea un normalizzatore lineare semplice
# ======================================================================

def normalizer_from_stat(stat: Dict[str, np.ndarray]) -> SingleFieldLinearNormalizer:
    """Crea un normalizzatore lineare simmetrico da statistiche."""
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1.0 / max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )

# ======================================================================
# Dataset principale
# ======================================================================

class ManiSkillReplayLowdimDataset(BaseLowdimDataset):
    """
    Dataset compatibile con Diffusion Policy per dati ManiSkill convertiti in Zarr.
    Si aspetta una struttura:
        data/
            obs (N, Dobs)
            action (N, Da)
        meta/
            episode_ends (E,)
    """

    def __init__(self,
                 dataset_path: str,
                 horizon: int = 1,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 val_ratio: float = 0.0,
                 seed: int = 42,
                 max_train_episodes: Optional[int] = None,
                 normalize_actions: bool = True,
                 normalize_obs: bool = True,
                 use_symmetric_normalizer: bool = True):
        """
        Args:
            dataset_path: path alla directory .zarr
            horizon: lunghezza sequenza osservazioni/azioni
            pad_before/pad_after: padding per i campioni ai bordi
            val_ratio: proporzione episodi di validazione
            seed: per split casuale
            max_train_episodes: opzionale, limita #episodi per training
            normalize_actions: se True, normalizza azioni
            normalize_obs: se True, normalizza osservazioni
            use_symmetric_normalizer: se True, usa normalizzatore [-1,1]
        """
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.normalize_actions = normalize_actions
        self.normalize_obs = normalize_obs
        self.use_symmetric_normalizer = use_symmetric_normalizer

        # ------------------------------------------------------------------
        # 1. Carica il dataset Zarr
        # ------------------------------------------------------------------
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Zarr dataset not found: {dataset_path}")

        print(f"📂 Loading ManiSkill dataset from: {dataset_path}")
        z = zarr.open(dataset_path, mode='r')
        z_data = z["data"]
        z_meta = z["meta"]

        obs = np.asarray(z_data["obs"])
        actions = np.asarray(z_data["action"])
        episode_ends = np.asarray(z_meta["episode_ends"])

        # ------------------------------------------------------------------
        # 2. Costruisci ReplayBuffer
        # ------------------------------------------------------------------
        replay_buffer = ReplayBuffer.create_empty_numpy()

        start = 0
        for ep_idx, end in enumerate(tqdm(episode_ends, desc="Building ReplayBuffer from Zarr")):
            ep_obs = obs[start:end]
            ep_act = actions[start:end]
            episode = {"obs": ep_obs, "action": ep_act}
            replay_buffer.add_episode(episode)
            start = end

        self.replay_buffer = replay_buffer

        # ------------------------------------------------------------------
        # 3. Split train/val
        # ------------------------------------------------------------------
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.train_mask = train_mask
        self.val_mask = val_mask

        # ------------------------------------------------------------------
        # 4. SequenceSampler per training
        # ------------------------------------------------------------------
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.sampler = sampler
        print(f"✅ Loaded {replay_buffer.n_episodes} episodes | "
              f"Train={train_mask.sum()} | Val={val_mask.sum()}")

    # ------------------------------------------------------------------
    # Validation dataset
    # ------------------------------------------------------------------
    def get_validation_dataset(self):
        val_copy = copy.copy(self)
        val_copy.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        return val_copy

    # ------------------------------------------------------------------
    # Normalizer construction
    # ------------------------------------------------------------------
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Normalizer per azioni
        act_stat = array_to_stats(self.replay_buffer["action"])
        if self.normalize_actions:
            if self.use_symmetric_normalizer:
                this_norm = normalizer_from_stat(act_stat)
            else:
                this_norm = get_identity_normalizer_from_stat(act_stat)
        else:
            this_norm = get_identity_normalizer_from_stat(act_stat)
        normalizer["action"] = this_norm

        # Normalizer per osservazioni
        obs_stat = array_to_stats(self.replay_buffer["obs"])
        if self.normalize_obs:
            if self.use_symmetric_normalizer:
                this_norm = normalizer_from_stat(obs_stat)
            else:
                this_norm = get_identity_normalizer_from_stat(obs_stat)
        else:
            this_norm = get_identity_normalizer_from_stat(obs_stat)
        normalizer["obs"] = this_norm

        return normalizer

    # ------------------------------------------------------------------
    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.sampler)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
