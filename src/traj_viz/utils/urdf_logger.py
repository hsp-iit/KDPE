#!/usr/bin/env python3
"""
Modified version of the URDF logger
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr
import scipy.spatial.transform as st
import trimesh
from urdf_parser_py import urdf as urdf_parser
from scipy.spatial.transform import Rotation

import urdf_parser_py.xml_reflection.core as urdf_parser_core
urdf_parser_core.on_error = lambda x: x

def resolve_path(filepath, path: str) -> str:
    """Resolve a ROS path to an absolute path."""
    urdf_dir = Path(filepath).resolve().parent

    if path.startswith("package://"):
        package_name, relative_path = path[len("package://"):].split("/", 1)
        # Start from the URDF file path and traverse upwards until we find package.xml
        package_dir = urdf_dir
        while package_dir != package_dir.parent:
            if (package_dir / "package.xml").exists():
                package_path = package_dir
                break
            package_dir = package_dir.parent
        else:
            raise FileNotFoundError(f"Could not find package.xml for {package_name}")

        return str(package_path / relative_path)
    elif str(path).startswith("file://"):
        return str(urdf_dir / path[len("file://") :])
    else:
        return path



class URDFLogger:
    """Class to log a URDF to Rerun."""
    def __init__(self, filepath: str, rec, root_path: str = "", pos=[0, 0, 0], rpy=[0.0, 0.0, 0.0]) -> None:
        self.urdf = urdf_parser.URDF.from_xml_file(filepath)
        self.mat_name_to_mat = {mat.name: mat for mat in self.urdf.materials}
        self.entity_to_transform = {}
        self.root_path = root_path
        self.filepath = filepath
        self.rec = rec

        # self.urdf.joints[0].origin.xyz = pos
        # self.urdf.joints[0].origin.rpy = rpy


    def link_entity_path(self, link: urdf_parser.Link) -> str:
        """Return the entity path for the URDF link."""
        root_name = self.urdf.get_root()
        link_names = self.urdf.get_chain(root_name, link.name)[0::2]  # skip the joints
        return "/".join(link_names)

    def joint_entity_path(self, joint: urdf_parser.Joint) -> str:
        """Return the entity path for the URDF joint."""
        root_name = self.urdf.get_root()
        link_names = self.urdf.get_chain(root_name, joint.child)[0::2]  # skip the joints
        return "/".join(link_names)

    def init(self) -> None:
        """Log a URDF file to Rerun.
        """
        
        self.rec.log(self.root_path + "", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # default ROS convention

        for link in self.urdf.links:
            entity_path = self.link_entity_path(link)
            self.log_link(entity_path, link)

        for joint in self.urdf.joints:
            entity_path = self.joint_entity_path(joint)
            self.log_joint(entity_path, joint)

    def log_link(self, entity_path: str, link: urdf_parser.Link) -> None:
        # create one mesh out of all visuals
        for i, visual in enumerate(link.visuals):
            self.log_visual(entity_path + f"/visual_{i}", visual)

    def log_joint(self, entity_path: str, joint: urdf_parser.Joint) -> None:
        translation = rotation = None

        if joint.origin is not None and joint.origin.xyz is not None:
            translation = joint.origin.xyz

        if joint.origin is not None and joint.origin.rpy is not None:
            rotation = st.Rotation.from_euler("xyz", joint.origin.rpy).as_matrix()

        self.entity_to_transform[self.root_path + entity_path] = (translation, rotation)
        self.rec.log(self.root_path + entity_path, rr.Transform3D(translation=translation, mat3x3=rotation))

    def log_visual(self, entity_path: str, visual: urdf_parser.Visual) -> None:
        material = None
        if visual.material is not None:
            if visual.material.color is None and visual.material.texture is None:
                # use globally defined material
                material = self.mat_name_to_mat[visual.material.name]
            else:
                material = visual.material

        transform = np.eye(4)
        if visual.origin is not None and visual.origin.xyz is not None:
            transform[:3, 3] = visual.origin.xyz
        if visual.origin is not None and visual.origin.rpy is not None:
            transform[:3, :3] = st.Rotation.from_euler("xyz", visual.origin.rpy).as_matrix()

        if isinstance(visual.geometry, urdf_parser.Mesh):
            resolved_path = resolve_path(self.filepath, visual.geometry.filename)
            mesh_scale = visual.geometry.scale
            mesh_or_scene = trimesh.load_mesh(resolved_path)
            if mesh_scale is not None:
                transform[:3, :3] *= mesh_scale
        elif isinstance(visual.geometry, urdf_parser.Box):
            mesh_or_scene = trimesh.creation.box(extents=visual.geometry.size)
        elif isinstance(visual.geometry, urdf_parser.Cylinder):
            mesh_or_scene = trimesh.creation.cylinder(
                radius=visual.geometry.radius,
                height=visual.geometry.length,
            )
        elif isinstance(visual.geometry, urdf_parser.Sphere):
            mesh_or_scene = trimesh.creation.icosphere(
                radius=visual.geometry.radius,
            )
        else:
            self.rec.log(self.root_path +
                "",
                rr.TextLog("Unsupported geometry type: " + str(type(visual.geometry))),
            )
            mesh_or_scene = trimesh.Trimesh()

        mesh_or_scene.apply_transform(transform)

        if isinstance(mesh_or_scene, trimesh.Scene):
            scene = mesh_or_scene
            # use dump to apply scene graph transforms and get a list of transformed meshes
            for i, mesh in enumerate(scene.dump()):
                if material is not None:
                    if material.color is not None:
                        mesh.visual = trimesh.visual.ColorVisuals()
                        mesh.visual.vertex_colors = material.color.rgba
                    elif material.texture is not None:
                        texture_path = resolve_ros_path(material.texture.filename)
                        mesh.visual = trimesh.visual.texture.TextureVisuals(image=Image.open(texture_path))
                log_trimesh(self.root_path + entity_path+f"/{i}", mesh)
        else:
            mesh = mesh_or_scene
            if material is not None:
                if material.color is not None:
                    mesh.visual = trimesh.visual.ColorVisuals()
                    mesh.visual.vertex_colors = material.color.rgba
                elif material.texture is not None:
                    texture_path = resolve_ros_path(material.texture.filename)
                    mesh.visual = trimesh.visual.texture.TextureVisuals(image=Image.open(texture_path))
            self.log_trimesh(self.root_path + entity_path, mesh)


    def log_trimesh(self, entity_path: str, mesh: trimesh.Trimesh) -> None:
        vertex_colors = albedo_texture = vertex_texcoords = None
        if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
            vertex_colors = mesh.visual.vertex_colors
        elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            albedo_texture = mesh.visual.material.baseColorTexture
            if len(np.asarray(albedo_texture).shape) == 2:
                # If the texture is grayscale, we need to convert it to RGB.
                albedo_texture = np.stack([albedo_texture] * 3, axis=-1)
            vertex_texcoords = mesh.visual.uv
            # Trimesh uses the OpenGL convention for UV coordinates, so we need to flip the V coordinate
            # since Rerun uses the Vulkan/Metal/DX12/WebGPU convention.
            if vertex_texcoords is None:
                pass
            else:
                vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]
        else:
            # Neither simple color nor texture, so we'll try to retrieve vertex colors via trimesh.
            try:
                colors = mesh.visual.to_color().vertex_colors
                if len(colors) == 4:
                    # If trimesh gives us a single vertex color for the entire mesh, we can interpret that
                    # as an albedo factor for the whole primitive.
                    mesh_material = Material(albedo_factor=np.array(colors))
                else:
                    vertex_colors = colors
            except Exception:
                pass

        self.rec.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=vertex_colors,
                albedo_texture=albedo_texture,
                vertex_texcoords=vertex_texcoords,
            ),
            static=True,
        )


    def log(self, joint_name: str, value: float) -> None:
        """Logs an angle for the franka panda robot"""
        joint = self.urdf.joint_map[joint_name]

        if not joint.type == "revolute":
            raise ValueError(f"Joint {joint_name} is not a revolute joint.") 

        entity_path = self.urdf.get_chain(root=self.urdf.get_root(), tip=joint.child)[0::2]
        entity_path = '/'.join(entity_path)

        start_translation, start_rotation_mat = self.entity_to_transform[entity_path]

        # All angles describe rotations around the transformed z-axis.
        vec = np.array(np.array(np.round(joint.axis, 5)) * value)

        rot = Rotation.from_rotvec(vec).as_matrix()
        rotation_mat = start_rotation_mat @ rot

        self.rec.log(
            entity_path, rr.Transform3D(translation=start_translation, mat3x3=rotation_mat)
        )

    def get_joint_names(self, type):
        return [joint.name for joint in self.urdf.joints if joint.type == type]




if __name__ == "__main__":
    import rerun as rr
    from uuid import uuid4
    from scipy.spatial.transform import Rotation

    recording_id = uuid4()
    rr.init("DP_Trajectories", spawn=True)

    urdf_logger = URDFLogger('./assets/robots/ergoCubSN002/model.urdf', root_path='')
    urdf_logger.init()

    print(urdf_logger.get_joint_names('revolute'))

    urdf_logger.log('torso_yaw', -3.14/2)
    urdf_logger.log('neck_yaw', -3.14/2)
    urdf_logger.log('r_shoulder_yaw', 3.14/2)
    urdf_logger.log('r_shoulder_pitch', -3.14/2)


