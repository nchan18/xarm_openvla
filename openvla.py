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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

current_pos = [300,0,300]
current_rot = [180,0,0]
current_gripper_pos = 850

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
arm.set_gripper_position(850, wait=True)

prompt = "pickup the blue cube"

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

frame_idx = -1
while True:
    start_time = time.time()
    ret, frame = cap.read()
    print(frame.shape)
    frame_idx += 1
    
    if frame_idx < 5:
        ret, frame = None, None
        continue

    cv2.imshow("Live view", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# #    frame = frame[125:550, 132:472] # Debug: added by Dominick on Jun 6
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image: Image.Image = Image.fromarray(bgr_frame)

#     # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    pos = [i * 1000 for i in action[:3]]
    current_pos[0] -= pos[0]

    current_pos[1] -= pos[1]
    current_pos[2] += pos[2]
    rot = [i * 180/math.pi for i in action[3:6]]
    current_rot[0] -= rot[0]
    current_rot[1] -= rot[1]
    current_rot[2] += rot[2]
    current_gripper_pos += action[6]*850
    print(pos)
    print(current_pos)
    print(rot)
    print(current_rot)
    print(current_gripper_pos)
    print(f"Current loop time is:{time.time()-start_time}")




    arm.set_position(*[current_pos[0],current_pos[1],current_pos[2],current_rot[0],current_rot[1],current_rot[2]], wait=False)
    # arm.set_position(300,100,200,180,0,0, wait=False)
    arm.set_gripper_position(current_gripper_pos, wait=False)
    # arm.set_gripper_position(100, wait=False)
