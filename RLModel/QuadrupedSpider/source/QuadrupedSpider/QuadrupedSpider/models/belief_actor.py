import torch
import torch.nn as nn

class StudentActorCritic(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actions: int,
        prop_dim: int = 45,
        extero_dim: int = 121,
        priv_dim: int = 124,
        latent_dim: int = 64,
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 2,
        student_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        **kwargs,
    ):
        super().__init__()

        self.prop_dim = prop_dim
        self.extero_dim = extero_dim
        self.priv_dim = priv_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.hidden = None
        self.reconstruction = None

        act_cls = nn.ELU if activation.lower() == "elu" else nn.ReLU

        # 1. Proprioceptive temporal encoder (feeds into GRU)
        self.proprio_net = nn.Sequential(
            nn.Linear(prop_dim, 128),
            act_cls(),
            nn.Linear(128, 128),
            act_cls(),
        )

        # Recurrent belief GRU (ONLY processes proprioception)
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
        )

        # 2. Exteroceptive spatial encoder (Independent MLP)
        self.extero_net = nn.Sequential(
            nn.Linear(extero_dim, 256),
            act_cls(),
            nn.Linear(256, 128),
            act_cls(),
        )

        # Attention Gate: Uses GRU belief to scale/mute exteroception
        self.gate = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 128),
            nn.Sigmoid()
        )

        # 3. Action and reconstruction heads (Input: GRU + Gated Exteroception)
        self.action_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim + 128, 128),
            act_cls(),
            nn.Linear(128, num_actions),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim + 128, 128),
            act_cls(),
            nn.Linear(128, priv_dim),
        )

        self.register_buffer("action_std", torch.zeros(num_actions))

    def _extract_inputs(self, obs):
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            proprio = obs["proprioception"]
            extero = obs["exteroception"]
        else:
            proprio = obs[:, :self.prop_dim]
            extero = obs[:, self.prop_dim:self.prop_dim + self.extero_dim]
        return proprio, extero

    def act(self, obs, **kwargs):
        proprio, extero = self._extract_inputs(obs)

        # Process proprioception through GRU
        p_feat = self.proprio_net(proprio).unsqueeze(1)
        gru_out, self.hidden = self.rnn(p_feat, self.hidden)
        belief_state = gru_out.squeeze(1)

        # Process exteroception and apply attention gate
        e_feat = self.extero_net(extero)
        alpha = self.gate(belief_state)
        gated_extero = alpha * e_feat

        # Combine internal belief with gated spatial data
        latent = torch.cat([belief_state, gated_extero], dim=-1)

        actions = self.action_head(latent)
        self.reconstruction = self.reconstruction_head(latent)
        return actions

    def act_inference(self, obs_dict, **kwargs):
        prop, ext = self._extract_inputs(obs_dict)

        p_feat = self.proprio_net(prop).unsqueeze(1)
        out, self.hidden = self.rnn(p_feat, self.hidden)
        belief_state = out.squeeze(1)

        e_feat = self.extero_net(ext)
        alpha = self.gate(belief_state)
        gated_extero = alpha * e_feat

        latent = torch.cat([belief_state, gated_extero], dim=-1)

        self.reconstruction = self.reconstruction_head(latent)
        return self.action_head(latent)

    def reset(self, dones=None, hidden_states=None):
        if dones is None:
            self.hidden = None if hidden_states is None else hidden_states.detach().clone()
        elif self.hidden is not None:
            keep = (~dones.reshape(1, -1, 1).bool()).to(self.hidden.dtype)
            self.hidden = self.hidden * keep

    def get_hidden_states(self):
        return None if self.hidden is None else self.hidden.detach().clone()

    def detach_hidden_states(self):
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

    def evaluate(self, obs, **kwargs):
        return self.act_inference(obs, **kwargs)