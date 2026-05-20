import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    min_denoise_steps = usr_args.get("min_denoise_steps", 10)
    max_denoise_steps = usr_args.get("max_denoise_steps", 10)
    window_size = usr_args.get("window_size", 5)
    
    # Get robot config from embodiment config (supports multiple robot types)
    left_robot_file = usr_args.get("left_robot_file", None)
    right_robot_file = usr_args.get("right_robot_file", None)
    
    if min_denoise_steps == max_denoise_steps == 10:
        return PI0(train_config_name, model_name, checkpoint_id, pi0_step, 
                   left_robot_file=left_robot_file, right_robot_file=right_robot_file)
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step, (min_denoise_steps, max_denoise_steps),
               left_robot_file=left_robot_file, right_robot_file=right_robot_file, window_size=window_size)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    result = model.get_action()
    actions = result["actions"][:model.pi0_step]
    inference_time = result["policy_timing"]["infer_s"]
    denoise_steps = result.get("num_steps", 10)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================
    return inference_time, denoise_steps

def reset_model(model):
    model.reset_obsrvationwindows()
