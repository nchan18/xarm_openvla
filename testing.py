import sys
from pathlib import Path
from typing import Dict, List, Union
from dataclasses import dataclass, field

import draccus

sys.path.append(".")

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "localhost"
    port: int = 5556

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 60                                         # Max number of timesteps per episode
    control_frequency: float = 5                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)

@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "XArmToyDataset"

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    breakpoint()

if __name__ == "__main__":
    eval_model_in_bridge_env()