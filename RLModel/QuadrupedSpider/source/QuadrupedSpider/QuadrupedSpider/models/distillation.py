import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import Distillation
from rsl_rl.runners import OnPolicyRunner


class DistillationRolloutBuffer:
    """Lightweight rollout buffer designed specifically for grouped observation dictionaries."""

    def __init__(self, num_envs: int, num_steps: int, obs_example, device: str = "cpu"):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0

        # Preallocate storage for each observation group
        self.observations = {
            group: torch.zeros(
                (num_steps, num_envs, tensor.shape[-1]),
                dtype=torch.float32,
                device=device,
            )
            for group, tensor in obs_example.items()
        }
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)

    def add(self, obs, dones):
        for group, tensor in obs.items():
            self.observations[group][self.step].copy_(tensor)
        self.dones[self.step].copy_(dones)
        self.step += 1

    def clear(self):
        self.step = 0

    def get_step(self, step_idx: int):
        obs_slice = {
            group: self.observations[group][step_idx]
            for group in self.observations
        }
        return obs_slice, self.dones[step_idx]


class NativeBeliefDistillation(Distillation):
    """Custom belief distillation algorithm for training the student policy using a frozen teacher."""

    def __init__(
        self,
        student_policy,
        teacher_policy,
        lambda_bc=1.0,
        lambda_rec=0.5,
        prop_dim=45,
        **kwargs,
    ):
        super().__init__(student_policy, **kwargs)
        self.student = self.policy
        self.teacher = teacher_policy
        self.prop_dim = prop_dim

        # Attributes expected by OnPolicyRunner
        self.rnd = False
        self.empirical_normalization = False

        # Freeze teacher weights and set to evaluation mode
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        self.lambda_bc = lambda_bc
        self.lambda_rec = lambda_rec
        self.custom_storage = None
        self.last_obs = None

    def init_storage(self, num_envs, num_steps, obs, *args, **kwargs):
        """Initialize custom rollout buffer for grouped observations."""
        self.custom_storage = DistillationRolloutBuffer(
            num_envs=num_envs,
            num_steps=num_steps,
            obs_example=obs,
            device=self.device,
        )

    def act(self, obs):
        """Query student actions and generate teacher target actions during rollouts."""
        self.last_obs = obs

        # 1. Query student actions
        actions = self.student.act(obs)
        self.transition.actions = actions.detach()

        # 2. Query teacher actions using privileged data
        with torch.no_grad():
            teacher_actions = self.teacher.act_inference(obs)

        self.transition.privileged_actions = teacher_actions.detach()
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        """Record rollout step into custom storage and reset terminated environments."""
        if self.custom_storage is not None and self.last_obs is not None:
            self.custom_storage.add(self.last_obs, dones)

        # Reset student recurrent hidden states for environments that have terminated
        if hasattr(self.student, "reset"):
            self.student.reset(dones)

    def compute_returns(self, last_critic_obs):
        """Bypass advantage estimation. Distillation does not utilize value functions or returns."""
        pass

    def update(self):
        """Execute distillation learning epochs combining behavior cloning and state reconstruction losses."""
        totals = {"behavior": 0.0, "reconstruction": 0.0}
        count = 0

        for _ in range(self.num_learning_epochs):
            if hasattr(self.student, "reset"):
                self.student.reset()

            # Iterate sequentially over collected rollout transitions to preserve GRU continuity
            for step_idx in range(self.custom_storage.step):
                obs, dones = self.custom_storage.get_step(step_idx)
                privileged_targets = obs["privileged"]

                # 1. Generate teacher actions without gradients
                with torch.no_grad():
                    teacher_actions = self.teacher.act_inference(obs)

                # 2. Forward pass student model
                student_actions = self.student.act(obs)
                reconstruction = self.student.reconstruction

                # 3. Compute losses (Behavior Cloning + Privileged Reconstruction)
                behavior_loss = F.mse_loss(student_actions, teacher_actions)
                reconstruction_loss = F.mse_loss(reconstruction, privileged_targets)
                loss = (self.lambda_bc * behavior_loss) + (self.lambda_rec * reconstruction_loss)

                # 4. Backpropagation & optimization step
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Detach recurrent state to truncate BPTT to 1 step
                if hasattr(self.student, "detach_hidden_states"):
                    self.student.detach_hidden_states()

                # Reset recurrent state for environments terminated in this step
                if hasattr(self.student, "reset"):
                    self.student.reset(dones)

                totals["behavior"] += behavior_loss.item()
                totals["reconstruction"] += reconstruction_loss.item()
                count += 1

        self.custom_storage.clear()

        mean_behavior = totals["behavior"] / max(1, count)
        mean_reconstruction = totals["reconstruction"] / max(1, count)

        # Keys mapped to satisfy default OnPolicyRunner.log formatting and TensorBoard summaries
        return {
            "surrogate": mean_behavior,
            "value_loss": mean_reconstruction,
            "behavior_loss": mean_behavior,
            "reconstruction_loss": mean_reconstruction,
        }


class NativeDistillationRunner(OnPolicyRunner):
    """Custom runner that accepts a pre-loaded teacher and initializes the native student architecture."""

    def __init__(self, env, train_cfg, teacher_policy, log_dir=None, device="cpu"):
        self.teacher_policy = teacher_policy
        # The base init automatically calls self._construct_algorithm()
        super().__init__(env, train_cfg, log_dir, device)

    def _construct_algorithm(self, obs):
        """Construct the decoupled student policy and native distillation algorithm."""
        from .belief_actor import StudentActorCritic

        # 1. Initialize Student Policy
        student_cfg = self.policy_cfg.copy()
        student_cfg.pop("class_name", None)
        self.student = StudentActorCritic(
            num_actions=self.env.num_actions,
            **student_cfg,
        ).to(self.device)

        # 2. Initialize Algorithm
        alg_cfg = self.alg_cfg.copy()
        alg_cfg.pop("class_name", None)
        alg = NativeBeliefDistillation(
            student_policy=self.student,
            teacher_policy=self.teacher_policy,
            device=self.device,
            **alg_cfg,
        )

        # 3. Initialize Custom Rollout Storage
        alg.init_storage(
            num_envs=self.env.num_envs,
            num_steps=self.num_steps_per_env,
            obs=obs,
        )
        return alg