# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play an RL agent using Isaac Sim's internal ROS 2 bridge for Xbox teleoperation."""

import argparse
import glob
import os
import sys
import time

# Purge system ROS 2 (3.10) paths from environment so they don't break Python 3.11
sys.path = [p for p in sys.path if not p.startswith("/opt/ros")]
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = ":".join(
        [p for p in os.environ["PYTHONPATH"].split(":") if not p.startswith("/opt/ros")]
    )

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and ROS 2 teleop.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--disable_scan_corruption",
    action="store_true",
    help="Disable student scan noise, dropouts, and blackouts for terrain-visualization diagnosis.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Enable Built-in Isaac Sim ROS 2 Bridge (compiled for internal Python)"""
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

"""Imports that depend on Isaac Sim's internal ROS 2 environment"""
import gymnasium as gym
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
import torch
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as opr

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import QuadrupedSpider.tasks  # noqa: F401
from QuadrupedSpider.models import StudentActorCritic, TeacherActorCritic

from QuadrupedSpider.tasks.manager_based.quadrupedspider.mdp.observations import foot_height_sample_positions

opr.TeacherActorCritic = TeacherActorCritic


class RosTeleopNode(Node):
    """ROS 2 listener node for teleoperation and manual reset commands."""

    def __init__(self):
        super().__init__("isaac_teleop_node")
        self.cmd_vel = [0.0, 0.0, 0.0]
        self.reset_requested = False

        self.twist_sub = self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)
        self.joy_sub = self.create_subscription(Joy, "/joy", self._joy_callback, 10)

    def _cmd_vel_callback(self, msg: Twist):
        max_lin_vel = 1.5
        max_ang_vel = 2.0
        self.cmd_vel = [
            float(msg.linear.x) * max_lin_vel,
            float(msg.linear.y) * max_lin_vel,
            float(msg.angular.z) * max_ang_vel,
        ]

    def _joy_callback(self, msg: Joy):
        if len(msg.buttons) > 0 and msg.buttons[0] == 1:
            self.reset_requested = True

        if len(msg.axes) >= 4:
            lin_x = float(msg.axes[1])
            lin_y = float(msg.axes[0])
            ang_z = float(msg.axes[3])

            deadzone = 0.08
            lin_x = 0.0 if abs(lin_x) < deadzone else lin_x
            lin_y = 0.0 if abs(lin_y) < deadzone else lin_y
            ang_z = 0.0 if abs(ang_z) < deadzone else ang_z

            self.cmd_vel = [lin_x, lin_y, ang_z]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using ROS 2 teleoperation."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.disable_scan_corruption:
        try:
            scan_term = env_cfg.observations.exteroception.noisy_height_grid
        except AttributeError as exc:
            raise ValueError(
                "--disable_scan_corruption requires the student noisy_height_grid observation."
            ) from exc
        scan_term.params.update(noise_std=0.0, dropout_prob=0.0, blackout_prob=0.0)
        env_cfg.observations.exteroception.enable_corruption = False
        print("[INFO] Student scan corruption disabled for this play session.")

    env_cfg.episode_length_s = 1e6
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None
    if getattr(env_cfg, "events", None) is not None and hasattr(
        env_cfg.events, "randomize_hidden_belief_obstacles"
    ):
        env_cfg.events.randomize_hidden_belief_obstacles = None
    for obstacle_name in ("belief_obstacle", "belief_obstacle_05", "belief_obstacle_10", "belief_obstacle_15"):
        if hasattr(env_cfg.scene, obstacle_name):
            setattr(env_cfg.scene, obstacle_name, None)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if os.path.isdir(resume_path):
        checkpoints = glob.glob(os.path.join(resume_path, "**", "model_*.pt"), recursive=True)
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(resume_path, "*.pt"))
        if checkpoints:
            def extract_iter(p):
                base = os.path.basename(p)
                num = "".join(filter(str.isdigit, base))
                return int(num) if num else os.path.getmtime(p)
            resume_path = max(checkpoints, key=extract_iter)
            print(f"[INFO] Auto-resolved latest checkpoint file: {resume_path}")
        else:
            raise FileNotFoundError(f"No .pt checkpoint files found inside directory: {resume_path}")

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
    elif agent_cfg.class_name in ("DistillationRunner", "NativeDistillationRunner"):
        student_cfg = agent_cfg.policy.to_dict() if hasattr(agent_cfg.policy, "to_dict") else dict(agent_cfg.policy)
        student_cfg.pop("class_name", None)

        policy_nn = StudentActorCritic(
            num_actions=env.num_actions,
            obs=env.get_observations(),
            **student_cfg,
        ).to(agent_cfg.device)

        checkpoint = torch.load(resume_path, map_location=agent_cfg.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("student."):
                clean_k = clean_k[len("student."):]
            elif clean_k.startswith("policy."):
                clean_k = clean_k[len("policy."):]
            cleaned_state_dict[clean_k] = v

        policy_nn.load_state_dict(cleaned_state_dict, strict=True)
        policy_nn.eval()
        policy = policy_nn.act_inference
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    belief_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/BeliefTerrain",
        markers={
            "points": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.2)),
            ),
        },
    )
    belief_visualizer = VisualizationMarkers(belief_marker_cfg)

    raw_sensor_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RawSensorTerrain",
        markers={
            "points": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.15)),
            ),
        },
    )
    raw_visualizer = VisualizationMarkers(raw_sensor_marker_cfg)

    if not rclpy.ok():
        rclpy.init()
    ros_node = RosTeleopNode()

    print("\n" + "=" * 65)
    print("[INFO] Isaac Sim ROS 2 Teleop Listener Active:")
    print("  - Listening on /cmd_vel (Twist)")
    print("  - Listening on /joy (Button 0 'A' triggers reset)")
    print("=" * 65 + "\n")

    command_term = None
    if isinstance(env.unwrapped, ManagerBasedRLEnv) and "base_velocity" in env.unwrapped.command_manager.active_terms:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    loop_step = 0
    scan_offset = 0.35

    while simulation_app.is_running():
        start_time = time.time()

        rclpy.spin_once(ros_node, timeout_sec=0.0)

        with torch.inference_mode():
            if ros_node.reset_requested:
                print("[TELEOP] Button A pressed -> Resetting robot!")
                ros_node.reset_requested = False
                all_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
                env.unwrapped.reset(env_ids=all_env_ids)
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset()
                obs = env.get_observations()

            cmd_tensor = torch.tensor(ros_node.cmd_vel, dtype=torch.float32, device=env.unwrapped.device)
            if command_term is not None:
                command_term.command[:] = cmd_tensor.repeat(env.unwrapped.num_envs, 1)

            if hasattr(obs, "keys"):
                if "proprioception" in obs.keys():
                    obs["proprioception"][:, 6:9] = cmd_tensor
                if "policy" in obs.keys():
                    obs["policy"][:, 6:9] = cmd_tensor
            elif isinstance(obs, torch.Tensor):
                obs[:, 6:9] = cmd_tensor

            actions = policy(obs)

            terrain_diagnostic = None
            if "height_scanner" in env.unwrapped.scene.sensors:
                sensor = env.unwrapped.scene.sensors["height_scanner"]
                sensor_pos_z = sensor.data.pos_w[0, 2]

                noisy_scan = (
                    obs["exteroception"][0]
                    if hasattr(obs, "keys") and "exteroception" in obs
                    else None
                )
                sample_positions = foot_height_sample_positions(env.unwrapped)[0].reshape(-1, 3)

                if noisy_scan is not None and sample_positions is not None:
                    valid_scan = (
                        torch.isfinite(noisy_scan)
                        & (noisy_scan < 1.0)
                        & torch.isfinite(sample_positions).all(dim=-1)
                    )
                    if valid_scan.any():
                        world_points_raw = sample_positions[valid_scan].clone()
                        world_points_raw[:, 2] = sensor_pos_z - noisy_scan[valid_scan] - scan_offset
                        raw_visualizer.set_visibility(True)
                        raw_visualizer.visualize(translations=world_points_raw)
                    else:
                        raw_visualizer.set_visibility(False)
                else:
                    raw_visualizer.set_visibility(False)

                reconstruction = getattr(policy_nn, "reconstruction", None)

                if reconstruction is not None:
                    pred_scan = reconstruction[0]
                    if sample_positions is not None and pred_scan.numel() == sample_positions.shape[0]:
                        valid_prediction = (
                            torch.isfinite(pred_scan)
                            & torch.isfinite(sample_positions).all(dim=-1)
                        )
                        if valid_prediction.any():
                            world_points_belief = sample_positions[valid_prediction].clone()
                            world_points_belief[:, 2] = (
                                sensor_pos_z - pred_scan[valid_prediction] - scan_offset
                            )
                            belief_visualizer.set_visibility(True)
                            belief_visualizer.visualize(translations=world_points_belief)
                        else:
                            belief_visualizer.set_visibility(False)
                    else:
                        belief_visualizer.set_visibility(False)

                    target_group = "teacher_exteroception" if (hasattr(obs, "keys") and "teacher_exteroception" in obs) else "privileged"
                    if (
                        noisy_scan is not None
                        and reconstruction.shape[-1] == noisy_scan.numel()
                        and hasattr(obs, "keys")
                        and target_group in obs
                    ):
                        target_scan = obs[target_group][0][-noisy_scan.numel() :]
                        target_z = sensor_pos_z - target_scan - scan_offset
                        predicted_z = sensor_pos_z - pred_scan - scan_offset
                        target_known = torch.isfinite(target_scan) & (target_scan < 1.0)
                        valid_raw_scan = torch.isfinite(noisy_scan) & (noisy_scan < 1.0)
                        comparison_mask = valid_raw_scan & target_known
                        missing_raw_scan = ~valid_raw_scan & target_known
                        prediction_rmse = (
                            torch.sqrt(torch.mean((pred_scan[target_known] - target_scan[target_known]).square()))
                            if target_known.any()
                            else torch.tensor(float("nan"), device=pred_scan.device)
                        )
                        if target_known.any():
                            predicted_z_range = predicted_z[target_known]
                            target_z_range = target_z[target_known]
                            pred_min, pred_max = predicted_z_range.min().item(), predicted_z_range.max().item()
                            target_min, target_max = target_z_range.min().item(), target_z_range.max().item()
                        else:
                            pred_min = pred_max = target_min = target_max = float("nan")
                        if comparison_mask.any():
                            raw_rmse = torch.sqrt(
                                torch.mean(
                                    (noisy_scan[comparison_mask] - target_scan[comparison_mask]).square()
                                )
                            )
                            valid_raw_rmse = raw_rmse.item()
                            valid_prediction_rmse = torch.sqrt(
                                torch.mean(
                                    (pred_scan[comparison_mask] - target_scan[comparison_mask]).square()
                                )
                            ).item()
                        else:
                            valid_raw_rmse = float("nan")
                            valid_prediction_rmse = float("nan")
                        if missing_raw_scan.any():
                            missing_prediction_rmse = torch.sqrt(
                                torch.mean(
                                    (pred_scan[missing_raw_scan] - target_scan[missing_raw_scan]).square()
                                )
                            ).item()
                        else:
                            missing_prediction_rmse = float("nan")
                        dropped_samples = missing_raw_scan.sum().item()
                        terrain_diagnostic = (
                            pred_min,
                            pred_max,
                            target_min,
                            target_max,
                            valid_raw_rmse,
                            valid_prediction_rmse,
                            missing_prediction_rmse,
                            dropped_samples,
                            noisy_scan.numel(),
                            prediction_rmse.item(),
                        )
                else:
                    belief_visualizer.set_visibility(False)

            obs, _, dones, _ = env.step(actions)

            if command_term is not None:
                command_term.command[:] = cmd_tensor.repeat(env.unwrapped.num_envs, 1)

            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

            if loop_step % 60 == 0:
                print(
                    f"[Step {loop_step}] /cmd_vel: [{ros_node.cmd_vel[0]:.2f}, {ros_node.cmd_vel[1]:.2f}, {ros_node.cmd_vel[2]:.2f}] | Action Norm: {torch.norm(actions).item():.3f}"
                )
                if terrain_diagnostic is not None:
                    (
                        pred_min,
                        pred_max,
                        target_min,
                        target_max,
                        valid_raw_rmse,
                        valid_prediction_rmse,
                        missing_prediction_rmse,
                        dropped_samples,
                        total_samples,
                        prediction_rmse,
                    ) = terrain_diagnostic
                    print(
                        "[Terrain] "
                        f"predicted world-z=[{pred_min:.2f}, {pred_max:.2f}] m, "
                        f"contact-corrected target world-z=[{target_min:.2f}, {target_max:.2f}] m, "
                        f"valid raw/prediction RMSE={valid_raw_rmse:.3f}/{valid_prediction_rmse:.3f} m, "
                        f"missing prediction RMSE={missing_prediction_rmse:.3f} m, "
                        f"invalid/dropout samples={dropped_samples}/{total_samples}, "
                        f"reconstruction RMSE={prediction_rmse:.3f} m"
                    )

        loop_step += 1
        if args_cli.video and loop_step == args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    ros_node.destroy_node()
    rclpy.shutdown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()