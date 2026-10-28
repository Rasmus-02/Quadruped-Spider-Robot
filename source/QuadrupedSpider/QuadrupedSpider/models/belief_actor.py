import torch
import torch.nn as nn


class BeliefEncoder(nn.Module):
    def __init__(self, step_dim: int, history_len: int = 20, latent_dim: int = 64):
        super().__init__()
        # 1D Conv across time: Input shape [Batch, Channels, Time]
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=step_dim, out_channels=64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten(),
        )
        
        # Calculate flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, step_dim, history_len)
            flat_dim = self.conv(dummy).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        # obs_history comes in as [Batch, Time, Channels]. Conv1d expects [Batch, Channels, Time]
        x = obs_history.transpose(1, 2)
        x = self.conv(x)
        return self.fc(x)


class BeliefDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, priv_dim: int = 121):
        super().__init__()
        # Reconstructs the clean privileged height scan (e.g., 11x11 = 121 points)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, priv_dim)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class StudentActorWithBelief(nn.Module):
    def __init__(self, prop_dim: int, extero_dim: int, priv_dim: int, action_dim: int = 12, history_len: int = 20):
        super().__init__()
        step_dim = prop_dim + extero_dim
        self.encoder = BeliefEncoder(step_dim=step_dim, history_len=history_len, latent_dim=64)
        self.decoder = BeliefDecoder(latent_dim=64, priv_dim=priv_dim)
        
        # Action head: conditioned on the latent belief and the current immediate proprioception
        self.actor_mlp = nn.Sequential(
            nn.Linear(64 + prop_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs_history: torch.Tensor, current_prop: torch.Tensor):
        latent = self.encoder(obs_history)
        action = self.actor_mlp(torch.cat([latent, current_prop], dim=-1))
        return action, latent