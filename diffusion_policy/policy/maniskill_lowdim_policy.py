from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class ManiSkillLowdimPolicy(BaseLowdimPolicy):
    """
    Policy per ambienti ManiSkill low-dim, compatibile con BaseLowdimPolicy.
    Versione adattata da RobomimicLowdimPolicy, ma indipendente da robomimic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        device: str = "cpu",
    ):
        """
        Args:
            obs_dim: dimensione vettore osservazione
            action_dim: dimensione vettore azione
            hidden_dim: dimensione layer nascosti
            n_layers: numero layer MLP
            device: cpu o cuda
        """
        super().__init__()

        self.obs_key = "obs"
        self.normalizer = LinearNormalizer()
        self.device = torch.device(device)

        # MLP semplice per mapping obs -> action
        layers = []
        input_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

        self.net.to(self.device, dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

    # ----------------------------------------------------------------------
    # Device management
    # ----------------------------------------------------------------------
    def to(self, *args, **kwargs):
        """Trasferisce il modello e il normalizzatore su device/dtype."""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.device = device
        self.net.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Inference step:
        - riceve obs_dict con chiave 'obs' -> Tensor (B,To,Do)
        - produce action Tensor (B,Ta,Da)
        """
        obs = obs_dict["obs"]
        obs = self.normalizer["obs"].normalize(obs)

        # prendiamo l’ultimo step di osservazione
        current_obs = obs[:, -1, :]  # (B, Do)
        action = self.net(current_obs)  # (B, Da)
        action = self.normalizer["action"].unnormalize(action)

        # formato standard DP: (B, 1, Da)
        return {"action": action[:, None, :]}

    def reset(self):
        """Resetta eventuale stato interno (policy stateless)."""
        pass

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Carica parametri di normalizzazione (bias/scale)."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def train_on_batch(self, batch, epoch: int = 0, validate: bool = False):
        """
        Esegue un passo di training supervisionato su un batch:
        batch: dict {'obs': (B,To,Do), 'action': (B,Ta,Da)}
        """
        self.net.train(not validate)
        nbatch = self.normalizer.normalize(batch)

        obs = nbatch["obs"][:, -1, :]  # (B, Do)
        target_action = nbatch["action"][:, 0, :]  # (B, Da)
        pred_action = self.net(obs)

        loss = nn.functional.mse_loss(pred_action, target_action)
        if not validate:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item()}

    def get_optimizer(self):
        """Restituisce l’ottimizzatore principale (per training DP)."""
        return self.optimizer
