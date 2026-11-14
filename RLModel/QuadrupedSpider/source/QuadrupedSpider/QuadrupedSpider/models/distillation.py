import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import Distillation
from rsl_rl.runners import OnPolicyRunner


class DistillationRolloutBuffer:
    """Rollout buffer storing student inputs, teacher actions, and reconstruction targets[cite: 47]."""

    def __init__(self, num_envs: int, num_steps: int, obs_example, num_actions: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0
        self.initial_hidden = None

        keep_groups = ("proprioception", "exteroception", "teacher_exteroception", "privileged")
        self.observations = {
            group: torch.zeros(
                (num_steps, num_envs, *obs_example[group].shape[1:]),
                dtype=torch.float32,
                device=device,
            )
            for group in keep_groups
            if group in obs_example
        }
        self.teacher_actions = torch.zeros(
            (num_steps, num_envs, num_actions), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)

    def set_initial_hidden(self, hidden):
        if hidden is None:
            self.initial_hidden = None
            return
        with torch.inference_mode(False):
            self.initial_hidden = hidden.detach().clone().to(self.device)

    def add(self, obs, teacher_actions, dones):
        for group in self.observations:
            self.observations[group][self.step].copy_(obs[group])
        self.teacher_actions[self.step].copy_(teacher_actions)
        self.dones[self.step].copy_(dones.bool())
        self.step += 1

    def clear(self):
        self.step = 0
        self.initial_hidden = None


class NativeBeliefDistillation(Distillation):
    """Dual-loss belief state distillation algorithm[cite: 36, 41, 47]."""

    def __init__(
        self,
        student_policy,
        teacher_policy,
        lambda_bc=1.0,
        lambda_rec=0.5,
        sequence_length=10,
        **kwargs,
    ):
        super().__init__(student_policy, **kwargs)
        self.student = self.policy
        self.teacher = teacher_policy

        self.rnd = False
        self.empirical_normalization = False

        self.teacher.eval()
        self.teacher.requires_grad_(False)

        if hasattr(self.student, "load_from_teacher"):
            self.student.load_from_teacher(self.teacher)

        self.lambda_bc = lambda_bc
        self.lambda_rec = lambda_rec
        self.sequence_length = max(1, int(sequence_length))
        self.custom_storage = None
        self.last_obs = None
        self.last_teacher_actions = None

    def init_storage(self, num_envs, num_steps, obs, num_actions=12):
        self.custom_storage = DistillationRolloutBuffer(
            num_envs=num_envs,
            num_steps=num_steps,
            obs_example=obs,
            num_actions=num_actions,
            device=self.device,
        )

    def _build_teacher_obs(self, obs):
        return {
            "proprioception": obs["proprioception"],
            "exteroception": obs.get("teacher_exteroception", obs["exteroception"]),
            "privileged": obs["privileged"],
        }

    def act(self, obs):
        self.last_obs = obs
        if self.custom_storage is not None and self.custom_storage.step == 0:
            self.custom_storage.set_initial_hidden(self.student.get_hidden_states())

        actions = self.student.act(obs)
        self.transition.actions = actions.detach()

        with torch.no_grad():
            teacher_obs = self._build_teacher_obs(obs)
            teacher_actions = self.teacher.act_inference(teacher_obs)

        self.last_teacher_actions = teacher_actions.detach()
        self.transition.privileged_actions = self.last_teacher_actions
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self.custom_storage is not None and self.last_obs is not None:
            self.custom_storage.add(self.last_obs, self.last_teacher_actions, dones)

        if hasattr(self.student, "reset"):
            self.student.reset(dones)

    def compute_returns(self, last_critic_obs):
        pass

    def update(self):
        storage = self.custom_storage
        if storage is None:
            raise RuntimeError("Rollout storage has not been initialized.")

        totals = {"behavior": 0.0, "belief_map": 0.0}
        count = 0

        proprio_all = storage.observations["proprioception"][: storage.step].clone()
        extero_all = storage.observations["exteroception"][: storage.step].clone()
        target_extero_key = "teacher_exteroception" if "teacher_exteroception" in storage.observations else "exteroception"
        target_extero_all = storage.observations[target_extero_key][: storage.step].clone()
        target_priv_all = storage.observations["privileged"][: storage.step].clone()
        teacher_act_all = storage.teacher_actions[: storage.step].clone()
        dones_all = storage.dones[: storage.step].clone()

        for _ in range(self.num_learning_epochs):
            if hasattr(self.student, "reset"):
                self.student.reset(hidden_states=storage.initial_hidden)

            for seq_start in range(0, storage.step, self.sequence_length):
                seq_end = min(seq_start + self.sequence_length, storage.step)

                student_actions, est_extero, est_priv = self.student.forward_sequence(
                    proprio_all[seq_start:seq_end],
                    extero_all[seq_start:seq_end],
                    dones_all[seq_start:seq_end],
                )

                target_seq = target_extero_all[seq_start:seq_end]
                noisy_seq = extero_all[seq_start:seq_end]

                # Weight cells where true terrain differs from raw sensor (invisible obstacles, step edges, dropouts) 5x higher
                discrepancy_mask = (target_seq - noisy_seq).abs() > 0.03
                cell_weights = torch.where(discrepancy_mask, 5.0, 1.0)
                sq_err = (est_extero - target_seq).square()
                map_rec_loss = (sq_err * cell_weights).mean()

                behavior_loss = F.mse_loss(student_actions, teacher_act_all[seq_start:seq_end])
                priv_rec_loss = F.mse_loss(est_priv, target_priv_all[seq_start:seq_end])
                rec_loss = map_rec_loss + 0.25 * priv_rec_loss

                loss = self.lambda_bc * behavior_loss + self.lambda_rec * rec_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if hasattr(self.student, "detach_hidden_states"):
                    self.student.detach_hidden_states()

                totals["behavior"] += behavior_loss.item()
                totals["belief_map"] += map_rec_loss.item()
                count += 1

        storage.clear()

        mean_behavior = totals["behavior"] / max(1, count)
        mean_belief_map = totals["belief_map"] / max(1, count)

        return {
            "surrogate": mean_behavior,
            "value_loss": mean_belief_map,
            "behavior_loss": mean_behavior,
            "belief_map_loss": mean_belief_map,
            "reconstruction_loss": mean_belief_map,
        }


class NativeDistillationRunner(OnPolicyRunner):
    """Custom runner that accepts a pre-loaded TeacherActorCritic and trains StudentActorCritic[cite: 47]."""

    def __init__(self, env, train_cfg, teacher_policy, log_dir=None, device="cpu"):
        self.teacher_policy = teacher_policy
        super().__init__(env, train_cfg, log_dir, device)

    def _construct_algorithm(self, obs):
        from .belief_actor import StudentActorCritic

        student_cfg = self.policy_cfg.copy()
        student_cfg.pop("class_name", None)
        self.student = StudentActorCritic(
            num_actions=self.env.num_actions,
            obs=obs,
            **student_cfg,
        ).to(self.device)

        alg_cfg = self.alg_cfg.copy()
        alg_cfg.pop("class_name", None)
        alg = NativeBeliefDistillation(
            student_policy=self.student,
            teacher_policy=self.teacher_policy,
            device=self.device,
            **alg_cfg,
        )

        alg.init_storage(
            num_envs=self.env.num_envs,
            num_steps=self.num_steps_per_env,
            obs=obs,
            num_actions=self.env.num_actions,
        )
        return alg