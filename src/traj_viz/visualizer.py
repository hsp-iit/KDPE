#!/usr/bin/env python3
from collections import defaultdict
from dataclasses import dataclass
import math
import os
import sys
import threading
import time

 

# from traj_viz.utils.blueprint import build_blueprint
# from traj_viz.utils.blueprint import build_blueprint
from traj_viz.utils.blueprint import build_blueprint
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from uuid import uuid4 
import numpy as np
from pathlib import Path
import rerun as rr


import glob
import h5py
import json
from utils.urdf_logger import URDFLogger
import uuid
from scipy.spatial.transform import Rotation as R




from matplotlib import pyplot as plt
def log_frame(name, pose, rec, scale=0.1, static=False):
    rec.log(f'poses/{name}', rr.Transform3D(translation=pose.pos, mat3x3=pose.ori), static=static)
    rec.log(f'poses/{name}/axes', rr.Arrows3D(origins=np.zeros([3,3]), vectors=np.eye(3) * scale, colors=np.eye(3)), static=static)
    rec.log(f'poses/{name}/point', rr.Points3D([0,0,-0.04], labels=[f'{name} ({pose.grip})'], radii=[0.001, ]), static=static)

    pose_array = np.concatenate([pose.pos, R.from_matrix(pose.ori).as_rotvec(), pose.grip])
    pose_names = ['x', 'y', 'z', 'ax', 'ay', 'az', 'g']
    for n, p, in zip(pose_names, pose_array):
        rec.log(f'poses/{name}/components/{n}', rr.Scalars(p), static=static)

class Visualizer:
    dir_path: Path
    trajectory_length: int
    metadata: dict
    cameras: dict[str, np.ndarray]

    def __init__(self, urdf_path, robot_pose=None):
        """Initialize the visualizer without any Gradio integration."""
        rid = uuid.uuid4()
        self.rec = rr.RecordingStream(application_id="rerun_example", recording_id=rid)
        rr.spawn(recording=self.rec)
        self.rec.connect_grpc()

        # Log blueprint
        blueprint = build_blueprint()
        self.rec.send_blueprint(blueprint)

        self.rec.set_time("real_time", duration=0.0)

        # Log the URDF
        self.urdf_logger = URDFLogger(urdf_path, self.rec)
        self.urdf_logger.init()
        if robot_pose is not None:
            self.rec.log(self.urdf_logger.urdf.get_root(), rr.Transform3D(translation=robot_pose.pos, mat3x3=robot_pose.ori))

        # Log virtual cameras
        hand_path = '/'.join(self.urdf_logger.urdf.get_chain(root=self.urdf_logger.urdf.get_root(), tip='fr3_hand')[0::2])
        self.rec.log(f'{hand_path}/front_camera', rr.Transform3D(translation=np.array([0.4, 0.0, 0.0]), mat3x3=R.from_euler('zyx', [0, -np.pi/2, np.pi/2]).as_matrix()))
        self.rec.log(f'{hand_path}/front_camera', rr.Pinhole(fov_y=0.78, image_plane_distance=0.1, aspect_ratio=1.7777778))
        self.rec.log(f'{hand_path}/right_camera', rr.Transform3D(translation=np.array([0.0, -0.4, 0.0]), mat3x3=R.from_euler('xyz', [-np.pi/2, np.pi, 0]).as_matrix()))
        self.rec.log(f'{hand_path}/right_camera', rr.Pinhole(fov_y=0.78, image_plane_distance=0.1, aspect_ratio=1.7777778))


    def log(self,
            joints: dict[str, np.ndarray]={},
            images: dict[str, np.ndarray]={},
            depths: dict[str, np.ndarray]={},
            poses: dict[str, np.ndarray]={},
            cameras: dict[str, np.ndarray]={},
            trajectories: dict[str, np.ndarray]={},
            timestamp: float=0.0,
            static: bool=False,
        ):
        self.rec.set_time("real_time", duration=timestamp)
        
        # Use joints to move the urdf
        for joint_name, angle in joints.items():
            self.urdf_logger.log(joint_name, angle)

        # Log the camera images
        for name, image in images.items():
            self.rec.log(f"cameras/{name}/image", rr.Image(image), static=static)
        # Log the camera depths
        for name, depth in depths.items():
            self.rec.log(f"cameras/{name}/depth", rr.DepthImage(depth), static=static)

        # Log reference frames
        for name, pose in poses.items():
           log_frame(name, pose, self.rec, static=static)

        # Log cameras
        for name, camera in cameras.items():
            f = 0.5 * camera.height / math.tan(camera.fov * math.pi / 360)
            camera_matrix = np.array(((f, 0, camera.width / 2), (0, f, camera.height / 2), (0, 0, 1)))
            self.rec.log(f'cameras/{name}', rr.Pinhole(image_from_camera=camera_matrix), static=static)
            self.rec.log(f'cameras/{name}', rr.Transform3D(translation=camera.pos, mat3x3=camera.ori), static=static)

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        colormap = cm.get_cmap('viridis')
        # norm = mcolors.Normalize(vmin=0, vmax=len(trajectories))

        # Log trajectories
        for name, traj in trajectories.items():
            self.rec.log(f"trajectories/{name}", rr.LineStrips3D(strips=traj[...,:3]), static=static)
            # self.rec.log(f"trajectories/{name}/end", rr.Points3D(traj[-1, :3], colors=[255,0,0], radii=0.001), static=static)
            self.rec.log(f"trajectories/{name}/ori", rr.Transform3D(translation=traj[-1, :3], mat3x3=R.from_rotvec(traj[-1,3:6]).as_matrix(), axis_length=0.01), static=static)
            self.rec.log(f"trajectories/{name}", rr.LineStrips3D.from_fields(colors=colormap(traj[-1, -1])), static=static)

@dataclass
class Pose:
    pos: np.ndarray
    ori: np.ndarray
    grip: np.ndarray = 0.0

@dataclass
class Camera:
    pos: np.ndarray
    ori: np.ndarray
    fov: float = 45.0
    width: int = 240
    height: int = 240