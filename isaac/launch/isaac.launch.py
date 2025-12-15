# -*- coding: utf-8 -*-
# Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023 PickNik, LLC. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import re
import os

import carb
import numpy as np
from pathlib import Path

# In older versions of Isaac Sim (prior to 4.0), SimulationApp is imported from
# omni.isaac.kit rather than isaacsim.
try:
    from isaacsim import SimulationApp
except:
    from omni.isaac.kit import SimulationApp
script_dir = os.path.dirname(os.path.realpath(__file__))
isaac_dir = Path(script_dir).parent
MYCOBOT_STAGE_PATH = "/mycobot_320"
TABLE_STAGE_PATH = "/background/Table"
TABLE_USD_PATH = "/Isaac/Environments/Simple_Room/Props/table_low.usd"
MYCOBOT_USD_PATH = os.path.join(isaac_dir, "assets/mycobot_324.usd")
CAMERA_PRIM_PATH = f"{MYCOBOT_STAGE_PATH}/camera_link/realsense_camera"
FRONT_CAMERA_PRIM_PATH = "/front_camera"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
GRAPH_PATH = "/ActionGraph"
REALSENSE_VIEWPORT_NAME = "MyCobot Camera"

CONFIG = {"renderer": "RayTracedLighting", "headless": False}                                                                                                          

simulation_app = SimulationApp(CONFIG)  

from omni.isaac.version import get_version

# Check the major version number of Isaac Sim to see if it's four digits, corresponding
# to Isaac Sim 2023.1.1 or older.  The version numbering scheme changed with the
# Isaac Sim 4.0 release in 2024.
is_legacy_isaacsim = len(get_version()[2]) == 4

# More imports that need to compare after we create the app
from omni.isaac.core import SimulationContext  # noqa E402
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import set_targets
from omni.isaac.core.prims import GeometryPrim, RigidPrim
from omni.isaac.core.utils import (  # noqa E402
    extensions,
    prims,
    rotations,
    stage,
    viewports,
)
from omni.isaac.nucleus import nucleus  # noqa E402
from omni.isaac.core_nodes.scripts.utils import set_target_prims  # noqa E402
from pxr import Gf, UsdGeom, Sdf  # noqa E402
import omni.graph.core as og  # noqa E402
import omni

# enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = nucleus.get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# Preparing stage
viewports.set_camera_view(eye=np.array([2.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

# # Loading the warehouse environment
stage.add_reference_to_stage(
    assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
)


prims.create_prim(
    MYCOBOT_STAGE_PATH,
    "Xform",
    position=np.array([1.8, 0, 0.68383]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    scale=np.array([1.0, 1.0, 1.0]),
    usd_path=MYCOBOT_USD_PATH,
)

prims.create_prim(
    TABLE_STAGE_PATH,
    "Xform",
    position=np.array([2, 0, 0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    usd_path=assets_root_path + TABLE_USD_PATH,
)
robot_camera = Camera(
    prim_path=CAMERA_PRIM_PATH,
    frequency=60,
    resolution=(640, 480),
    orientation=np.array([-0.5, 0.5, 0.5, -0.5]),
)


front_camera = Camera(
    prim_path=FRONT_CAMERA_PRIM_PATH,  
    frequency=60,
    resolution=(640, 480),
    position=np.array([4.0, 0.0, 1.0]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)


# add some objects, spread evenly along the X axis
# with a fixed offset from the robot in the Y and Z
prims.create_prim(
    "/cracker_box",
    "Xform",
    position=np.array([1.8, -0.25, 1.0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
)
prims.create_prim(
    "/sugar_box",
    "Xform",
    position=np.array([1.93, -0.25, 1.0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
)
prims.create_prim(
    "/soup_can",
    "Xform",
    position=np.array([2.1, -0.25, 1.0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
)
prims.create_prim(
    "/mustard_bottle",
    "Xform",
    position=np.array([2.0, 0.15, 1.0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
)

omni.kit.commands.execute('SetRigidBody',
	path=Sdf.Path(TABLE_STAGE_PATH+'/table_low'),
	approximationShape='convexHull',
	kinematic=False)


simulation_app.update()

try:
    ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
    print("Using ROS_DOMAIN_ID: ", ros_domain_id)
except ValueError:
    print("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
    ros_domain_id = 0
except KeyError:
    print("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
    ros_domain_id = 0

# Create an action graph with ROS component nodes
try:
    og_keys_set_values = [
        ("Context.inputs:domain_id", ros_domain_id),
        # Set the /Franka target prim to Articulation Controller node
        ("ArticulationController.inputs:robotPath", MYCOBOT_STAGE_PATH),
        ("PublishJointState.inputs:topicName", "isaac_joint_states"),
        ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
        ("createViewport.inputs:name", REALSENSE_VIEWPORT_NAME),
        ("createViewport.inputs:viewportId", 1),
        ("cameraHelperRgb.inputs:frameId", "sim_camera"),
        ("cameraHelperRgb.inputs:topicName", "rgb"),
        ("cameraHelperRgb.inputs:type", "rgb"),
        ("cameraHelperInfo.inputs:frameId", "sim_camera"),
        ("cameraHelperInfo.inputs:topicName", "camera_info"),
        ("cameraHelperInfo.inputs:type", "camera_info"),
        ("cameraHelperDepth.inputs:frameId", "sim_camera"),
        ("cameraHelperDepth.inputs:topicName", "depth"),
        ("cameraHelperDepth.inputs:type", "depth"),
        ("createViewport2.inputs:name", "front_camera"),
        ("createViewport2.inputs:viewportId", 2),
        ("cameraHelperRgb2.inputs:frameId", "sim_camera2"),
        ("cameraHelperRgb2.inputs:topicName", "/camera1/rgb"),
        ("cameraHelperRgb2.inputs:type", "rgb"),
    ]

    # In older versions of Isaac Sim, the articulation controller node contained a
    # "usePath" checkbox input that should be enabled.
    if is_legacy_isaacsim:
        og_keys_set_values.insert(1, ("ArticulationController.inputs:usePath", True))

    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                (
                    "SubscribeJointState",
                    "omni.isaac.ros2_bridge.ROS2SubscribeJointState",
                ),
                (
                    "ArticulationController",
                    "omni.isaac.core_nodes.IsaacArticulationController",
                ),
                ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                ("OnTick", "omni.graph.action.OnTick"),
                ("Ontick2", "omni.graph.action.OnTick"),
                ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                ("createViewport2", "omni.isaac.core_nodes.IsaacCreateViewport"),
                (
                    "getRenderProduct",
                    "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                ),
                (
                    "getRenderProduct2",
                    "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                ),
                ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                ("setCamera2", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("cameraHelperRgb2", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("cameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("SubscribeJointState.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("SubscribeJointState.outputs:execOut", "PublishClock.inputs:execIn"),
                (
                    "SubscribeJointState.outputs:execOut",
                    "ArticulationController.inputs:execIn",
                ),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishJointState.inputs:timeStamp",
                ),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
                ("SubscribeJointState.outputs:execOut", "createViewport.inputs:execIn"),
                ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                (
                    "getRenderProduct.outputs:renderProductPath",
                    "setCamera.inputs:renderProductPath",
                ),
                ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
                ("Context.outputs:context", "cameraHelperInfo.inputs:context"),
                ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
                (
                    "getRenderProduct.outputs:renderProductPath",
                    "cameraHelperRgb.inputs:renderProductPath",
                ),
                (
                    "getRenderProduct.outputs:renderProductPath",
                    "cameraHelperInfo.inputs:renderProductPath",
                ),
                (
                    "getRenderProduct.outputs:renderProductPath",
                    "cameraHelperDepth.inputs:renderProductPath",
                ),
                ("SubscribeJointState.outputs:execOut", "createViewport2.inputs:execIn"),
                ("createViewport2.outputs:execOut", "getRenderProduct2.inputs:execIn"),
                ("createViewport2.outputs:viewport", "getRenderProduct2.inputs:viewport"),
                ("getRenderProduct2.outputs:execOut", "setCamera2.inputs:execIn"),
                (
                    "getRenderProduct2.outputs:renderProductPath",
                    "setCamera2.inputs:renderProductPath",
                ),
                ("setCamera2.outputs:execOut", "cameraHelperRgb2.inputs:execIn"),
                ("Context.outputs:context", "cameraHelperRgb2.inputs:context"),

                (
                    "getRenderProduct2.outputs:renderProductPath",
                    "cameraHelperRgb2.inputs:renderProductPath",
                ),
            ],
            og.Controller.Keys.SET_VALUES: og_keys_set_values,
        },
    )
except Exception as e:
    print(e)


# Setting the /mycobot target prim to Publish JointState node
set_target_prims(
    primPath="/ActionGraph/PublishJointState", targetPrimPaths=[MYCOBOT_STAGE_PATH]
)

robot_camera_prim = UsdGeom.Camera(
    stage.get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
)
robot_camera_prim.GetHorizontalApertureAttr().Set(20.955)
robot_camera_prim.GetVerticalApertureAttr().Set(15.7)
robot_camera_prim.GetFocalLengthAttr().Set(18.8)
robot_camera_prim.GetFocusDistanceAttr().Set(400)

front_camera_prim = UsdGeom.Camera(
    stage.get_current_stage().GetPrimAtPath(FRONT_CAMERA_PRIM_PATH)
)
front_camera_prim.GetHorizontalApertureAttr().Set(20.955)
front_camera_prim.GetVerticalApertureAttr().Set(15.7)
front_camera_prim.GetFocalLengthAttr().Set(18.8)
front_camera_prim.GetFocusDistanceAttr().Set(400)

set_targets(
    prim=stage.get_current_stage().GetPrimAtPath(GRAPH_PATH + "/setCamera"),
    attribute="inputs:cameraPrim",
    target_prim_paths=[CAMERA_PRIM_PATH],
)
set_targets(
    prim=stage.get_current_stage().GetPrimAtPath(GRAPH_PATH + "/setCamera2"),
    attribute="inputs:cameraPrim",
    target_prim_paths=["/front_camera"],
)

simulation_app.update()

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()

simulation_context.play()
simulation_context.stop()
simulation_context.play()
# Dock the second camera window
# viewport = omni.ui.Workspace.get_window("Viewport")
# rs_viewport = omni.ui.Workspace.get_window(REALSENSE_VIEWPORT_NAME)
# rs_viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT)


while simulation_app.is_running():

    # Run with a fixed step size
    simulation_context.step(render=True)

    # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    )

simulation_context.stop()
simulation_app.close()