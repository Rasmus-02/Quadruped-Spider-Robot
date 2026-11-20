# Validate project structure
python3 zero_agent.py --task Template-Quadrupedspider-v0 --num_envs 16

# Monitor run
tensorboard --logdir /home/rasmus/Robotics/MastersThesis/RLModel/QuadrupedSpider/scripts/rsl_rl/logs/rsl_rl/ --bind_all

## Train
**Teacher**:
python scripts/rsl_rl/train.py --task Isaac-QuadrupedSpider-Teacher-v0 --num_envs 4096 --headless

**Student**:
python scripts/rsl_rl/train.py --task Isaac-QuadrupedSpider-Student-v0 --num_envs 4096 --headless

