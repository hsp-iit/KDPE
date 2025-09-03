import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
import torch
from pytorch3d.transforms import (quaternion_to_axis_angle, 
                                 quaternion_multiply, rotation_6d_to_matrix,
                                 euler_angles_to_matrix, matrix_to_euler_angles,
                                 matrix_to_quaternion, quaternion_to_matrix)

def _expand_variances(std, expected_dims, name, device):
    if isinstance(std, (int, float)):
        return torch.full((expected_dims,), std**2, device=device)
    elif isinstance(std, (list, tuple)):
        if len(std) == 1:
            return torch.full((expected_dims,), std[0]**2, device=device)
        elif len(std) == expected_dims:
            return torch.tensor(std, device=device).pow(2)
        else:
            raise ValueError(f"{name} length mismatch. Expected {expected_dims} or 1, got {len(std)}")
    else:
        raise TypeError(f"{name} must be a number, list, or tuple")

def compute_sigma(stds, feature_config, device):
    position_std, rotation_std, gripper_std = stds
    pos_dims = sum(s.stop - s.start for s in feature_config['pos'])
    pos_var = _expand_variances(position_std, pos_dims, 'position_std', device)
    
    rot_dims = 3 * len(feature_config['quat'])  # 3D per quaternion in axis-angle
    rot_var = _expand_variances(rotation_std, rot_dims, 'rotation_std', device)
    
    grip_dims = len(feature_config['grip'])
    grip_var = _expand_variances(gripper_std, grip_dims, 'gripper_std', device)
    
    variances = torch.cat([pos_var, rot_var, grip_var])
    invH_diag = 1.0 / variances
    detH = torch.exp(torch.sum(torch.log(variances)))  
    norm_factor = 1.0 / torch.sqrt((2 * torch.pi) ** variances.size(0) * detH)

    # Split inverse variances into components
    pos_invH = invH_diag[:pos_dims]
    rot_invH = invH_diag[pos_dims:pos_dims+rot_dims]
    grip_invH = invH_diag[pos_dims+rot_dims:]
    
    return pos_invH, rot_invH, grip_invH, norm_factor

def process_kernel_data(kernel_data, feature_config):
    # Extract and normalize kernel quaternions
    kernel_pos = torch.cat([kernel_data[..., s] for s in feature_config['pos']], dim=-1)
    kernel_quats = torch.cat([kernel_data[..., s] for s in feature_config['quat']], dim=-1)
    kernel_grip = torch.cat([kernel_data[..., s] for s in feature_config['grip']], dim=-1)
    
    # Normalize and invert quaternions
    kernel_quats = kernel_quats / torch.norm(kernel_quats, dim=-1, keepdim=True)
    mask = kernel_quats[..., 0] < 0
    kernel_quats = torch.where(mask.unsqueeze(-1), -kernel_quats, kernel_quats)
    kernel_quats_inv = kernel_quats * torch.tensor([1.0, -1.0, -1.0, -1.0], device=kernel_quats.device).repeat(len(feature_config['quat']))
    
    return torch.cat([kernel_pos, kernel_quats_inv, kernel_grip], dim=-1)

def process_query_data(query_data, feature_config):
    # Extract and normalize query quaternions
    query_pos = torch.cat([query_data[..., s] for s in feature_config['pos']], dim=-1)
    query_quats = torch.cat([query_data[..., s] for s in feature_config['quat']], dim=-1)
    query_grip = torch.cat([query_data[..., s] for s in feature_config['grip']], dim=-1)
    
    # Normalize quaternions
    query_quats = query_quats / torch.norm(query_quats, dim=-1, keepdim=True)
    mask = query_quats[..., 0] < 0
    query_quats = torch.where(mask.unsqueeze(-1), -query_quats, query_quats)
    
    return torch.cat([query_pos, query_quats, query_grip], dim=-1)

def _compute_pdf_chunk(query_data, kernel_data, pos_invH, rot_invH, grip_invH, norm_factor, feature_config):
    batch_size, M, _ = query_data.shape
    N = kernel_data.shape[1]
    num_quats = len(feature_config['quat'])

    # Position differences [batch, M, N, pos_dims]
    pos_diff = query_data[..., :pos_invH.shape[0]].unsqueeze(2) - \
               kernel_data[..., :pos_invH.shape[0]].unsqueeze(1)
    pos_quad = (pos_diff**2 * pos_invH).sum(dim=-1)

    # Quaternion processing
    quat_dims = 4*num_quats
    q1 = kernel_data[..., pos_invH.shape[0]:pos_invH.shape[0]+quat_dims].view(batch_size, N, num_quats, 4)
    q2 = query_data[..., pos_invH.shape[0]:pos_invH.shape[0]+quat_dims].view(batch_size, M, num_quats, 4)
    
    # Compute relative quaternions [batch, M, N, num_quats, 4]
    q_rel = quaternion_multiply(
        q2.unsqueeze(2),  # [batch, M, 1, num_quats, 4]
        q1.unsqueeze(1)   # [batch, 1, N, num_quats, 4]
    )
    
    # Ensure shortest path
    mask = q_rel[..., 0] < 0
    q_rel = torch.where(mask.unsqueeze(-1), -q_rel, q_rel)
    
    # Convert to axis-angle [batch, M, N, num_quats, 3]
    rot_diff = quaternion_to_axis_angle(q_rel)
    rot_diff = rot_diff.view(batch_size, M, N, -1)
    rot_quad = (rot_diff**2 * rot_invH).sum(dim=-1)

    # Gripper differences [batch, M, N, grip_dims]
    grip_diff = query_data[..., -grip_invH.shape[0]:].unsqueeze(2) - \
                kernel_data[..., -grip_invH.shape[0]:].unsqueeze(1)
    grip_quad = (grip_diff**2 * grip_invH).sum(dim=-1)

    # Combine quadratic terms
    quad = pos_quad + rot_quad + grip_quad
    densities = norm_factor * torch.exp(-0.5 * quad)
    return densities.mean(dim=-1)

def pdf(kernel_data, query_data, stds, 
        feature_config={'pos': [slice(0,3)],'quat': [slice(3,7)],'grip': [slice(7,8)]}, 
        chunk_size=100):
    device = kernel_data.device
    
    # Preprocess data
    query_data = process_query_data(query_data, feature_config)
    kernel_data = process_kernel_data(kernel_data, feature_config)
    pos_invH, rot_invH, grip_invH, norm_factor = compute_sigma(stds, feature_config, device)
    
    # Process in chunks
    batch_size = query_data.shape[0]
    results = []
    for i in range(0, batch_size, chunk_size):
        chunk = slice(i, min(i+chunk_size, batch_size))
        res = _compute_pdf_chunk(query_data[chunk], kernel_data[chunk], 
                                pos_invH, rot_invH, grip_invH, norm_factor, feature_config)
        results.append(res)
    
    return torch.cat(results, dim=0)

class OODKDE:
    def __init__(self, policy, **kwargs):
        print('Using filtering method: KDE Filter')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy = policy
        self.threshold = kwargs.get('threshold', None)
        self.seed = kwargs.get('seed', None)
        if isinstance(self.threshold, str):
            self.threshold = eval(self.threshold)
        if isinstance(self.seed, str):
            self.seed = eval(self.seed)

        seed = kwargs.get('seed', None)
        if isinstance(seed, str):
            seed = eval(seed)

        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(seed)


    def __call__(self, actions):
        denormalized_actions = self.policy.normalizer['action'].unnormalize(actions)

        s, e = self.policy.n_obs_steps - 1, self.policy.n_obs_steps - 1 + self.policy.n_action_steps
        denormalized_actions = denormalized_actions[:,:,s:e]

        b_dim, B = 0, denormalized_actions.shape[0]  # Batch dimension
        p_dim, P = 1, denormalized_actions.shape[1]  # Population dimension
        h_dim, H = 2, denormalized_actions.shape[2]  # Horizon dimension
        a_dim, A = 3, denormalized_actions.shape[3]  # Action dimension

        if A == 2: feature_config = {'pos': [slice(0,2)]}
        elif A == 10: feature_config = {'pos': [slice(0,3)], 'quat': [slice(3,7)], 'grip': [slice(7,8)]}
        elif A == 20: feature_config = {'pos': [slice(0,3), slice(8,11)], 'quat': [slice(3,7), slice(11,15)], 'grip': [slice(7,8), slice(15,16)]}
        else: raise ValueError(f"Unsupported action dimension: {A}")
        
        batched_actions = denormalized_actions.transpose(1,2).reshape(-1, P, A)

        to_quat = lambda x: matrix_to_quaternion(rotation_6d_to_matrix(x))

        # convert rotations from 6D to quaternion
        batched_actions = torch.cat(
            [
            torch.cat([
                batched_actions[...,slice(*(np.array([0,3]) + (i * 10)))],
                to_quat(batched_actions[...,slice(*(np.array([3,9]) + (i * 10)))]),
                batched_actions[...,slice(*(np.array([9,10]) + (i * 10)))]
            ], dim=-1)
            for i in range(len(feature_config['quat']))
        ], dim=-1)
        
        density = pdf(kernel_data=batched_actions,
            query_data=batched_actions,
            stds=[0.01, 0.05, 0.1],
            feature_config=feature_config,
        )

        density = density.reshape(B, H, P)
        trajectory_densities = density[:,-1]
        trajectory_densities = trajectory_densities / trajectory_densities.max(dim=1, keepdim=True).values

        top_trajectories = trajectory_densities.argmin(dim=1)
        worse_action =  actions[torch.arange(B), top_trajectories]

        return worse_action

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.generator is not None:
            state['generator_state'] = self.generator.get_state()
            del state['generator']
        return state

    def __setstate__(self, state):
        # If generator_state exists, use it to restore the exact state
        generator_state = state.pop('generator_state', None)
        self.__dict__.update(state)
        
        if generator_state is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.set_state(generator_state)