if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.maniskill_lowdim_policy import ManiSkillLowdimPolicy
from diffusion_policy.dataset.maniskill_replay_lowdim_dataset import ManiSkillReplayLowdimDataset
from diffusion_policy.env_runner.maniskill_lowdim_runner import ManiSkillLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainManiSkillLowdimWorkspace(BaseWorkspace):
    """
    Workspace per training di policy lowdim su ambienti ManiSkill.
    Adattato da TrainRobomimicLowdimWorkspace.
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # =========== Seed ===========
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # =========== Policy ===========
        self.model: ManiSkillLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        # Stato di training
        self.global_step = 0
        self.epoch = 0

    # ============================================================
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # =========== Resume training ===========
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        # =========== Dataset ===========
        dataset: ManiSkillReplayLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, ManiSkillReplayLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # normalizer → model
        self.model.set_normalizer(normalizer)

        # =========== Environment Runner ===========
        env_runner: ManiSkillLowdimRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir
        )
        assert isinstance(env_runner, ManiSkillLowdimRunner)

        # =========== Logging ===========
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})

        # =========== Checkpoint Manager ===========
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # =========== Device ===========
        device = torch.device(cfg.training.device)
        self.model.to(device)

        # Debug mode ridotto
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        # =========== Training Loop ===========
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                # ---------- Training ----------
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        info = self.model.train_on_batch(batch, epoch=self.epoch)

                        loss_value = info["loss"]
                        tepoch.set_postfix(loss=loss_value, refresh=False)
                        train_losses.append(loss_value)
                        step_log = {
                            "train_loss": loss_value,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                        }

                        is_last_batch = (batch_idx == len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and (
                            batch_idx >= cfg.training.max_train_steps - 1
                        ):
                            break

                # ---------- Fine epoca ----------
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ---------- Evaluation ----------
                self.model.eval()

                # rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model)
                    step_log.update(runner_log)

                # validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                info = self.model.train_on_batch(
                                    batch, epoch=self.epoch, validate=True
                                )
                                val_losses.append(info["loss"])
                                if (cfg.training.max_val_steps is not None) and (
                                    batch_idx >= cfg.training.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = np.mean(val_losses)
                            step_log["val_loss"] = val_loss

                # ---------- Checkpoint ----------
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # ---------- Log Epoch ----------
                self.model.train()
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


# ============================================================
# Hydra entry point
# ============================================================
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainManiSkillLowdimWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
