# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import cv2
import pyrealsense2 as rs
import numpy as np
import torch
from xarm.wrapper import XArmAPI
import math
import time
import threading
# For loading local openvla
import sys
from pathlib import Path
from typing import Union
sys.path = ["/workspace/openvla"] + sys.path
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)


class OpenVLAConf:
    model_family: str = "openvla"
    # pretrained_checkpoint: Union[str, Path] = "/mnt/data/dreilly1/openvla_trained_models/openvla_trained_xarm7_givenaction"
    # pretrained_checkpoint: Union[str, Path] = "/mnt/data/dreilly1/openvla_trained_models/openvla-7b+XArmToyDataset+b16+lr-0.0005+lora-r32+dropout-0.0--train_on_collected_xarmdata-GripperFix--image_aug--20000_chkpt"
    pretrained_checkpoint: Union[str, Path] = "/mnt/data/dreilly1/openvla_trained_models/openvla_trained_models/openvla-7b+RealWorldTask1_objectbowl+b16+lr-0.0005+lora-r32+dropout-0.0--RealTask1_objectbowl-RemoveZeroActions-run1--image_aug--20000_chkpt"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = False
cap = cv2.VideoCapture(4)
### From xarm_rlds.py
frame_size=(1280, 800)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
current_pos = [300,0,300]
current_rot = [180,0,0]
# gripper_open_pos = 524.9999940395355
gripper_open_pos = 600
# gripper_open_pos = 850
current_gripper_pos = gripper_open_pos
arm = XArmAPI('192.168.1.198')
time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
if not arm.arm._enable_report or arm.arm._stream_type != 'socket':
    arm.arm.get_err_warn_code()
    arm.arm.get_state()
if arm.arm._warn_code != 0:
    arm.arm.clean_warn()
if arm.arm._error_code != 0:
    arm.arm.clean_error()
    arm.arm.motion_enable(enable=True, servo_id=8)
    arm.arm.set_state(0)
if not arm.arm._is_ready:
    arm.arm.motion_enable(enable=True, servo_id=8)
    arm.arm.set_state(state=0)
arm.set_position(*[300, 0, 300, 180, 0, 0], wait=True)
code = arm.set_gripper_mode(0)
code = arm.set_gripper_enable(True)
code = arm.set_gripper_speed(5000)
arm.set_gripper_position(gripper_open_pos, wait=True)


# Debug: Change to velocity
# arm.set_mode(5)
# arm.set_state(0)

prompt = "Pick up the banana and place it in the white bowl"
print(prompt)
breakpoint()
# Load Processor & VLA
# processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     "openvla/openvla-7b",
#     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# ).to("cuda:0")
cfg = OpenVLAConf()
cfg.unnorm_key = "XArmToyDataset"
# Load model
vla = get_model(cfg)
# [OpenVLA] Get Hugging Face processor
processor = None
if cfg.model_family == "openvla":
    processor = get_processor(cfg)
debug = False
frame_idx = -1
while True:
    start_time = time.time()
    ret, frame = cap.read()
    # print(frame.shape)
    frame_idx += 1
    if frame_idx < 60:
        ret, frame = cap.read()
        continue
    ### Copied from xarm_rlds.py
    if ret:
        target_width = 840
        target_height = 630
        h, w, _ = frame.shape
        if h < target_height or w < target_width:
            print("Warning: Frame is smaller than crop size, skipping frame.")
            continue
        # Calculate top-left corner of crop box
        x_start = 50 + (w - target_width) // 2
        y_start = (h - target_height) // 2 + 50
        frame = frame[y_start:y_start + target_height, x_start:x_start + target_width]
        # Optionally draw info
        # frame = cv2.putText(frame, str(action_idx), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        #resize image
        frame = cv2.resize(frame, (640, 480), interpolation= cv2.INTER_LINEAR)
    else:
        print("Warning: Failed to capture frame")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = Image.fromarray(rgb_frame)
    rgb_frame = rgb_frame.resize((256, 256), resample=0)
    rgb_frame = np.array(rgb_frame)

    cv2.imshow("Live view", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
        
    # Predict Action (7-DoF)
    obs = {
        "full_image": rgb_frame
    }
    action = get_action(cfg, vla, obs, prompt, processor=processor)
    print(action)

    if not np.all(np.abs(action[:6]) < 0.001):
        pos = [i * 1000 for i in action[:3]]
        #current_pos[0] -= pos[0]
        current_pos[0] += pos[0]
        #current_pos[1] -= pos[1]
        current_pos[1] += pos[1]
        current_pos[2] += pos[2]
        rot = [i * 180/math.pi for i in action[3:6]]
        #current_rot[0] -= rot[0]
        current_rot[0] += rot[0]
        #current_rot[1] -= rot[1]
        current_rot[1] += rot[1]
        current_rot[2] += rot[2]

        arm.set_position(*[current_pos[0],current_pos[1],current_pos[2],current_rot[0],current_rot[1],current_rot[2]], wait=False)
        # arm.vc_set_cartesian_velocity(action[:6]*1000)
    else:
        # print('skipping movement, action xyz/rot too low')
        pass

    gripper_open = 1 if action[6] > 0.9 else 0
    current_gripper_pos = (gripper_open)*gripper_open_pos

    arm.set_gripper_position(current_gripper_pos, wait=False)
