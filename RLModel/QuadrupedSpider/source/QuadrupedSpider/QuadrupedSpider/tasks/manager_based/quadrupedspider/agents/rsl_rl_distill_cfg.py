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

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
        "proprioception": ["proprioception"],
        "exteroception": ["exteroception"],
        "privileged": ["privileged"],
    }
    
    @configclass
    class AlgorithmCfg:
        class_name = "BeliefDistillationAlgorithm"
        lambda_bc = 1.0
        lambda_rec = 0.5
        prop_dim = 45
        learning_rate = 5e-4
        num_learning_epochs = 5
        max_grad_norm = 1.0

    @configclass
    class PolicyCfg:
        class_name = "StudentActorCritic"
        prop_dim = 45
        priv_dim = 124
        latent_dim = 64
        rnn_hidden_dim = 256
        rnn_num_layers = 2
        student_hidden_dims = [512, 256, 128]
        teacher_hidden_dims = [512, 256, 128]
        activation = "elu"

    algorithm = AlgorithmCfg()
    policy = PolicyCfg()