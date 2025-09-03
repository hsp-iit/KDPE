# KDPE / traj_viz

Lightweight tooling to visualize robot rollout trajectories, images, depth, joint states and poses.

## 1. Prerequisites
Make sure you have:
- Git LFS (Large File Storage) (optional: to download the demo data). 
- pixi

If you do not have `pixi` installed: https://pixi.sh/

## 2. Clone (with dataset via Git LFS)
```bash
git clone https://github.com/hsp-iit/KDPE.git
cd KDPE
git lfs install
git lfs pull
```

## 3. Environment Setup
Install dependencies (creates/updates the pixi environment defined in `pyproject.toml`):
```bash
pixi install
```

## 5. Visualizer API (Core Logic)
`Visualizer` (see `src/traj_viz/visualizer.py`) exposes:

```python
Visualizer(
  urdf_path: str,
  robot_pose: Pose | None = None
)
```

Once constructed it creates an internal recording and logs a blueprint + URDF. You then call:

```python
visualizer.log(
  joints: dict[str, np.ndarray] = {},         # joint_name -> scalar angle
  images: dict[str, np.ndarray] = {},         # name -> HxWxC (uint8 RGB)
  depths: dict[str, np.ndarray] = {},         # name -> HxW or HxWx1 float32 depth
  poses: dict[str, Pose] = {},                # name -> Pose(pos, ori, grip)
  cameras: dict[str, Camera] = {},            # intrinsic/extrinsic virtual cams
  trajectories: dict[str, np.ndarray] = {},   # name -> (T, >=6) (xyz + rotvec + optional scalar)
  timestamp: float = 0.0,
  static: bool = False,
)
```



## 6. Example Usage (`inspect_rollout.py`)
```bash
pixi run python src/examples/inspect_rollout.py
```