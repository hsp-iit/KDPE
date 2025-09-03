from collections import defaultdict
import h5py
import numpy as np
from traj_viz.visualizer import Visualizer, Pose, Camera
from scipy.spatial.transform import Rotation as R

import time

class HDF5Episode:
    def __init__(self, dataset_path, episode):
        self.dataset = h5py.File(dataset_path, mode='r')[f'chunk0']
        actions = []
        obs = defaultdict(list)
        for k in sorted(self.dataset, key=lambda x: int(x[4:])):
            v = self.dataset[k]
            for kk, vv in v.items():
                if kk == 'action':
                    actions.append(vv[episode])
                elif kk == 'obs':
                    for kkk, vvv in vv.items():
                        obs[kkk].append(vvv[episode])

        actions = np.array(actions)
        obs = {k:np.concatenate(v, axis=0) for k,v in obs.items()}

        self.episode = episode

        self.episode_length = actions.shape[0]
        self.actions_denormalized = actions
        self.actions_normalized = actions
        self.observations = self.format_obs(obs)
 
    @property
    def actions(self):
        class Action:
            def __init__(self, denormalized, normalized):
                self._denormalized = denormalized
                self._normalized = normalized

            def __getitem__(self, index):
                if index < 0 or index >= len(self._denormalized):
                    raise IndexError("Index out of range")
                return {'denormalized': self._denormalized[index], 'normalized': self._normalized[index]}

        return Action(self.actions_denormalized, self.actions_normalized)

        
        
    def format_obs(self, obs):
        # Preload observations
        N = list(obs.values())[0].shape[0]

        observations = []
        for n in range(N):
            observations.append({k: v[n] for k,v in obs.items()})
        return observations


def main():
    dataset_path = 'data/rollout_compressed.hdf5'

    visualizer = Visualizer(urdf_path='data/models/franka_description/urdfs/fr3_franka_hand.urdf',
                            robot_pose=Pose(np.array([-0.56,  0.,  0.912]), R.from_quat([0, 0, 0, 1]).as_matrix()))

    visualizer.log(
        cameras={'agentview': Camera(fov=45.0, width=240, height=240, pos=np.array([np.array([0.483, 0.251, 1.214])]), ori=R.from_rotvec([-1.06585181, -1.92284687,  1.34639195]).as_matrix()),},
    )

    joints = [f'fr3_joint{i}' for i in range(1,9)]

    episode = HDF5Episode(dataset_path, 0)
    for i in range(episode.episode_length):
        actions = episode.actions[i]['denormalized']

        visualizer.log(
            trajectories={f'traj{i}': traj for i, traj in enumerate(actions)},
            timestamp=i * 8 * 0.05,
        )

        for j in range(8):
            obs = episode.observations[i * 8 + j]
            visualizer.log(
                joints=dict(zip(joints, obs['robot0_joint_pos'])),
                images={'agentview': obs['sideview_image'].transpose(1, 2, 0), 'eye_in_hand': obs['robot0_eye_in_hand_image'].transpose(1, 2, 0)},
                depths={'agentview': obs['sideview_depth'], 'eye_in_hand': obs['robot0_eye_in_hand_depth']},
                poses={'proprio': Pose(obs['robot0_eef_pos'], R.from_quat(obs['robot0_eef_quat']).as_matrix(), obs['robot0_gripper_qpos']),
                       'action': Pose(actions[0, -1, 0:3], R.from_rotvec(actions[0, -1, 3:6]).as_matrix(), actions[0, -1, 6:7])},
                timestamp=(i * 8 + j) * 0.05, 
            )
            # time.sleep(0.05)
    input()

    
if __name__ == "__main__":
    main()