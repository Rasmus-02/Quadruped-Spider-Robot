import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization


class PerFootExteroEncoder(nn.Module):
    """Exteroceptive Encoder (g_e): encodes each 52-point foot scan via [80, 60] -> 24 and concatenates to 96."""

    def __init__(self, num_feet: int = 4, points_per_foot: int = 52, foot_latent_dim: int = 24, act_cls=nn.LeakyReLU):
        super().__init__()
        self.num_feet = num_feet
        self.points_per_foot = points_per_foot
        self.foot_latent_dim = foot_latent_dim
        self.out_dim = num_feet * foot_latent_dim

        self.net = nn.Sequential(
            nn.Linear(points_per_foot, 80),
            act_cls(),
            nn.Linear(80, 60),
            act_cls(),
            nn.Linear(60, foot_latent_dim),
            act_cls(),
        )

    def forward(self, extero: torch.Tensor) -> torch.Tensor:
        batch_shape = extero.shape[:-1]
        per_foot = extero.reshape(*batch_shape, self.num_feet, self.points_per_foot)
        encoded = self.net(per_foot)
        return encoded.reshape(*batch_shape, self.out_dim)


class TeacherActorCritic(nn.Module):
    """Privileged teacher policy with per-foot exteroceptive encoder (g_e) and privileged encoder (g_p)."""

    num_feet = 4
    foot_latent_dim = 24
    extero_latent_dim = 96
    priv_latent_dim = 24
    belief_dim = 120
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dims=(256, 160, 128),
        critic_hidden_dims=(256, 160, 128),
        activation: str = "leaky_relu",
        init_noise_std: float = 0.8,
        **kwargs,
    ):
        super().__init__()
        self.obs_groups = obs_groups
        self.num_actions = num_actions

        self.prop_dim = int(obs["proprioception"].shape[-1])
        self.extero_dim = int(obs["exteroception"].shape[-1])
        self.priv_dim = int(obs["privileged"].shape[-1])
        self.points_per_foot = self.extero_dim // self.num_feet

        act_cls = nn.LeakyReLU if activation.lower() == "leaky_relu" else nn.ELU

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.prop_normalizer = EmpiricalNormalization(self.prop_dim)
            self.priv_normalizer = EmpiricalNormalization(self.priv_dim)
        else:
            self.prop_normalizer = nn.Identity()
            self.priv_normalizer = nn.Identity()

        # g_e ([80, 60] -> 24 per foot = 96) & g_p ([64, 32] -> 24)
        self.extero_encoder = PerFootExteroEncoder(
            num_feet=self.num_feet,
            points_per_foot=self.points_per_foot,
            foot_latent_dim=self.foot_latent_dim,
            act_cls=act_cls,
        )
        self.priv_encoder = nn.Sequential(
            nn.Linear(self.priv_dim, 64),
            act_cls(),
            nn.Linear(64, 32),
            act_cls(),
            nn.Linear(32, self.priv_latent_dim),
            act_cls(),
        )

        # Main MLP [256, 160, 128]
        actor_layers = []
        in_dim = self.prop_dim + self.belief_dim
        for h_dim in actor_hidden_dims:
            actor_layers.extend((nn.Linear(in_dim, h_dim), act_cls()))
            in_dim = h_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.action_head = nn.Sequential(*actor_layers)

        self.critic_extero_encoder = PerFootExteroEncoder(
            num_feet=self.num_feet,
            points_per_foot=self.points_per_foot,
            foot_latent_dim=self.foot_latent_dim,
            act_cls=act_cls,
        )
        self.critic_priv_encoder = nn.Sequential(
            nn.Linear(self.priv_dim, 64),
            act_cls(),
            nn.Linear(64, 32),
            act_cls(),
            nn.Linear(32, self.priv_latent_dim),
            act_cls(),
        )
        critic_layers = []
        c_in_dim = self.prop_dim + self.belief_dim
        for h_dim in critic_hidden_dims:
            critic_layers.extend((nn.Linear(c_in_dim, h_dim), act_cls()))
            c_in_dim = h_dim
        critic_layers.append(nn.Linear(c_in_dim, 1))
        self.critic_mlp = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        torch.distributions.Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def _split_obs(self, obs):
        if hasattr(obs, "keys"):
            return obs["proprioception"], obs["exteroception"], obs["privileged"]
        prop = obs[..., : self.prop_dim]
        extero = obs[..., self.prop_dim : self.prop_dim + self.extero_dim]
        priv = obs[..., self.prop_dim + self.extero_dim : self.prop_dim + self.extero_dim + self.priv_dim]
        return prop, extero, priv

    def encode_actor_inputs(self, obs):
        prop, extero, priv = self._split_obs(obs)
        prop_norm = self.prop_normalizer(prop)
        priv_norm = self.priv_normalizer(priv)
        l_e = self.extero_encoder(extero)
        l_p = self.priv_encoder(priv_norm)
        latent = torch.cat((l_e, l_p), dim=-1)
        return prop_norm, latent

    def act_inference(self, obs):
        prop_norm, latent = self.encode_actor_inputs(obs)
        return self.action_head(torch.cat((prop_norm, latent), dim=-1))

    def act(self, obs, **kwargs):
        mean = self.act_inference(obs)
        std = self.std.clamp(min=1e-3, max=2.0).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def evaluate(self, critic_obs, **kwargs):
        prop, extero, priv = self._split_obs(critic_obs)
        prop_norm = self.prop_normalizer(prop)
        priv_norm = self.priv_normalizer(priv)
        l_e = self.critic_extero_encoder(extero)
        l_p = self.critic_priv_encoder(priv_norm)
        return self.critic_mlp(torch.cat((prop_norm, l_e, l_p), dim=-1))

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            prop, _, priv = self._split_obs(obs)
            self.prop_normalizer.update(prop)
            self.priv_normalizer.update(priv)


class StudentActorCritic(nn.Module):
    """Gated Recurrent Belief State Student Policy."""

    num_feet = 4
    foot_latent_dim = 24
    extero_latent_dim = 96
    priv_latent_dim = 24
    belief_dim = 120

    def __init__(
        self,
        num_actions: int,
        obs: dict | None = None,
        prop_dim: int | None = None,
        extero_dim: int | None = None,
        priv_dim: int | None = None,
        rnn_hidden_dim: int = 50,
        rnn_num_layers: int = 2,
        student_hidden_dims=(256, 160, 128),
        activation: str = "leaky_relu",
        **kwargs,
    ):
        super().__init__()

        if obs is not None and hasattr(obs, "keys"):
            if prop_dim is None and "proprioception" in obs:
                prop_dim = int(obs["proprioception"].shape[-1])
            if extero_dim is None and "exteroception" in obs:
                extero_dim = int(obs["exteroception"].shape[-1])
            if priv_dim is None and "privileged" in obs:
                priv_dim = int(obs["privileged"].shape[-1])

        if prop_dim is None or extero_dim is None:
            raise ValueError("StudentActorCritic requires `obs` or explicit `prop_dim` and `extero_dim`.")
        if priv_dim is None:
            priv_dim = 3

        self.prop_dim = int(prop_dim)
        self.extero_dim = int(extero_dim)
        self.priv_dim = int(priv_dim)
        self.points_per_foot = self.extero_dim // self.num_feet
        self.map_cells = self.extero_dim
        self.hidden = None
        self.reconstruction = None
        self.priv_reconstruction = None

        act_cls = nn.LeakyReLU if activation.lower() == "leaky_relu" else nn.ELU

        self.prop_normalizer = EmpiricalNormalization(self.prop_dim)

        # Shared per-foot exteroceptive encoder g_e (reused from Teacher)
        self.extero_encoder = PerFootExteroEncoder(
            num_feet=self.num_feet,
            points_per_foot=self.points_per_foot,
            foot_latent_dim=self.foot_latent_dim,
            act_cls=act_cls,
        )

        # 2-layer stacked GRU with 50 hidden units (or rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=self.prop_dim + self.extero_latent_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=False,
        )

        # g_a ([64, 64] -> 96) with sigmoid gate
        self.encoder_gate = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 64),
            act_cls(),
            nn.Linear(64, 64),
            act_cls(),
            nn.Linear(64, self.extero_latent_dim),
            nn.Sigmoid(),
        )

        # g_b ([64, 64] -> 120) belief state encoder
        self.belief_proj = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 64),
            act_cls(),
            nn.Linear(64, 64),
            act_cls(),
            nn.Linear(64, self.belief_dim),
        )

        # Main Action MLP [256, 160, 128] (reused from Teacher)
        actor_layers = []
        in_dim = self.prop_dim + self.belief_dim
        for h_dim in student_hidden_dims:
            actor_layers.extend((nn.Linear(in_dim, h_dim), act_cls()))
            in_dim = h_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.action_head = nn.Sequential(*actor_layers)

        # Belief Decoder with per-sample gate + exteroceptive decoder + privileged decoder
        self.decoder_gate = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 64),
            act_cls(),
            nn.Linear(64, 64),
            act_cls(),
            nn.Linear(64, self.extero_dim),
            nn.Sigmoid(),
        )
        self.extero_decoder = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 128),
            act_cls(),
            nn.Linear(128, 128),
            act_cls(),
            nn.Linear(128, self.extero_dim),
        )
        self.priv_decoder = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 64),
            act_cls(),
            nn.Linear(64, 32),
            act_cls(),
            nn.Linear(32, self.priv_dim),
        )
        self.register_buffer("action_std", torch.zeros(num_actions))

    def load_from_teacher(self, teacher: TeacherActorCritic):
        """Initialize normalizer, exteroceptive encoder g_e, and action MLP from trained Teacher."""
        self.prop_normalizer.load_state_dict(teacher.prop_normalizer.state_dict())
        self.prop_normalizer.eval()
        for p in self.prop_normalizer.parameters():
            p.requires_grad = False
        self.extero_encoder.load_state_dict(teacher.extero_encoder.state_dict())
        self.action_head.load_state_dict(teacher.action_head.state_dict())

    def _extract_inputs(self, obs):
        if hasattr(obs, "keys"):
            proprio = obs["proprioception"]
            extero = obs["exteroception"]
        else:
            proprio = obs[..., : self.prop_dim]
            extero = obs[..., self.prop_dim : self.prop_dim + self.extero_dim]
        return proprio, extero

    def _compute_belief_and_decode(
        self,
        prop_norm: torch.Tensor,
        extero: torch.Tensor,
        l_e: torch.Tensor,
        b_prime: torch.Tensor,
    ):
        # alpha = sigmoid(g_a(b'_t)), b_t = g_b(b'_t) + pad_24(l_t^e * alpha)
        alpha = self.encoder_gate(b_prime)
        gated_extero = l_e * alpha
        zero_pad = torch.zeros(
            (*gated_extero.shape[:-1], self.priv_latent_dim),
            device=gated_extero.device,
            dtype=gated_extero.dtype,
        )
        b_t = self.belief_proj(b_prime) + torch.cat((gated_extero, zero_pad), dim=-1)

        # Belief Decoder (gated skip connection + learned exteroceptive decoder)
        valid_sample = (extero < 0.98).to(extero.dtype)
        dec_alpha = self.decoder_gate(b_prime) * valid_sample
        est_extero = (dec_alpha * extero + self.extero_decoder(b_prime)).clamp(-1.0, 1.0)
        est_priv = self.priv_decoder(b_prime)

        actions = self.action_head(torch.cat((prop_norm, b_t), dim=-1))
        return actions, est_extero, est_priv

    def _forward(self, obs):
        proprio, extero = self._extract_inputs(obs)
        prop_norm = self.prop_normalizer(proprio)
        l_e = self.extero_encoder(extero)
        rnn_in = torch.cat((prop_norm, l_e), dim=-1).unsqueeze(0)
        b_prime, self.hidden = self.rnn(rnn_in, self.hidden)
        b_prime = b_prime.squeeze(0)

        actions, est_extero, est_priv = self._compute_belief_and_decode(prop_norm, extero, l_e, b_prime)
        self.reconstruction = est_extero
        self.priv_reconstruction = est_priv
        return actions

    def forward_sequence(self, proprio_seq: torch.Tensor, extero_seq: torch.Tensor, dones_seq: torch.Tensor):
        seq_len = proprio_seq.shape[0]
        prop_norm = self.prop_normalizer(proprio_seq)
        l_e = self.extero_encoder(extero_seq)
        rnn_in = torch.cat((prop_norm, l_e), dim=-1)

        if not dones_seq.any():
            b_prime, self.hidden = self.rnn(rnn_in, self.hidden)
        else:
            outputs = []
            for t in range(seq_len):
                out_t, self.hidden = self.rnn(rnn_in[t : t + 1], self.hidden)
                outputs.append(out_t)
                if dones_seq[t].any():
                    keep = (~dones_seq[t].reshape(1, -1, 1)).to(self.hidden.dtype)
                    self.hidden = self.hidden * keep
            b_prime = torch.cat(outputs, dim=0)

        actions, est_extero, est_priv = self._compute_belief_and_decode(prop_norm, extero_seq, l_e, b_prime)
        self.reconstruction = est_extero[-1]
        self.priv_reconstruction = est_priv[-1]
        return actions, est_extero, est_priv

    def act(self, obs):
        return self._forward(obs)

    def act_inference(self, obs_dict):
        return self._forward(obs_dict)

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

    def evaluate(self, obs):
        return self.act_inference(obs)