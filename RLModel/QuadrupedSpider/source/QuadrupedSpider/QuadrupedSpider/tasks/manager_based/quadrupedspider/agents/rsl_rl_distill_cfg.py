from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab.utils import configclass


@configclass
class QuadrupedSpiderDistillRunnerCfg(RslRlBaseRunnerCfg):
    class_name = "DistillationRunner"

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "quadruped_spider_student"
    empirical_normalization = False

    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    teacher_checkpoint = None

    obs_groups = {
        "policy": ["proprioception"],
        "critic": ["proprioception"],
        "proprioception": ["proprioception"],
        "exteroception": ["exteroception"],
        "teacher_exteroception": ["teacher_exteroception"],
        "privileged": ["privileged"],
    }

    @configclass
    class AlgorithmCfg:
        class_name = "BeliefDistillationAlgorithm"
        lambda_bc = 1.0
        lambda_rec = 0.5
        sequence_length = 10
        learning_rate = 5e-4
        num_learning_epochs = 2
        max_grad_norm = 1.0

    @configclass
    class PolicyCfg:
        class_name = "StudentActorCritic"
        rnn_hidden_dim = 50
        rnn_num_layers = 2
        student_hidden_dims = [256, 160, 128]
        activation = "leaky_relu"

    algorithm = AlgorithmCfg()
    policy = PolicyCfg()