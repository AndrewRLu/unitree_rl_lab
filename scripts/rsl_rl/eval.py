#can call things like env.command_manager and then get_command or something

#inherit this: https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/envs/manager_based_rl_env.html#ManagerBasedRLEnv


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

##

import itertools

import numpy as np


def main():
    """Eval RSL-RL agent command tracking error."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        # entry_point_key="play_env_cfg_entry_point",
        entry_point_key="eval_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt # environment dt

    eval_duration_s = 5.0 # seconds
    steps_per_command = int(eval_duration_s/dt)

    # command combos
    vx_range = np.linspace(-1.0, 1.0, 5)  # 5 points
    vy_range = np.linspace(-1.0, 1.0, 5)  # 5 points
    wz_range = np.linspace(-1.0, 1.0, 5)  # 5 points

    # Cartesian product
    commands = list(itertools.product(vx_range, vy_range, wz_range))

    results = {}
        
    #-----------------
    # sample & eval
    #-----------------

    for i, (cmd_vx, cmd_vy, cmd_wz) in enumerate(commands):
        obs, _ = env.reset()
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        cmd_term.command[:, 0] = cmd_vx
        cmd_term.command[:, 1] = cmd_vy
        cmd_term.command[:, 2] = cmd_wz

        has_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        step_history = []

        for step in range(steps_per_command):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                has_failed = has_failed | dones

                robot_data = env.unwrapped.scene["robot"].data
                actual_vel = torch.stack([
                    robot_data.root_lin_vel_b[:, 0],  # vx
                    robot_data.root_lin_vel_b[:, 1],  # vy
                    robot_data.root_ang_vel_b[:, 2]   # wz
                ], dim=-1)

                step_history.append(actual_vel)

        valid_mask = ~has_failed 
        
        # Stack step history to shape: (250_steps, num_envs, 3)
        trial_time_series = torch.stack(step_history, dim=0)

        # Extract metrics for successful envs
        results[f"cmd_{i}"] = {
            "target_cmd": (cmd_vx, cmd_vy, cmd_wz),
            "valid_mask": valid_mask.cpu(),  # (num_envs,) boolean tensor
            
            # Unmasked (Full environment data)
            "terrain_levels": env.unwrapped.scene.terrain.terrain_levels.cpu(),  # (num_envs,)
            "terrain_types": env.unwrapped.scene.terrain.terrain_types.cpu(),    # (num_envs,)
            "actual_vel_history": trial_time_series.cpu(),                       # (250, num_envs, 3)
            
            "masked": {
                "terrain_levels": env.unwrapped.scene.terrain.terrain_levels[valid_mask].cpu(),
                "terrain_types": env.unwrapped.scene.terrain.terrain_types[valid_mask].cpu(),
                "actual_vel_history": trial_time_series[:, valid_mask, :].cpu(),  # (250, num_valid_envs, 3)
            },
            
            "success_rate": (valid_mask.sum() / env.num_envs).item()
        }

    torch.save(results, "eval_results.pt")
    print("Sweep complete! Results saved to eval_results.pt")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()