from rsl_rl.modules import ActorCritic, StudentTeacher
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

    # Maps directly to the groups defined in StudentObservationsCfg
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher"],
    }

    @configclass
    class AlgorithmCfg:
        class_name = "Distillation"
        learning_rate = 1e-3
        num_learning_epochs = 5
        max_grad_norm = 1.0

    @configclass
    class PolicyCfg:
        class_name = "StudentTeacher"
        # Student network architecture (processes student obs with history)
        student_hidden_dims = [512, 256, 128]
        # Teacher network architecture (matches your trained teacher PPO policy)
        teacher_hidden_dims = [512, 256, 128]
        activation = "elu"
        

    algorithm = AlgorithmCfg()
    policy = PolicyCfg()