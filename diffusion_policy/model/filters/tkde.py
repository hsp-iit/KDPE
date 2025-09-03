import math
import torch

import torch
from pytorch3d.transforms import (quaternion_to_axis_angle, 
                                 quaternion_multiply)



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

def compute_pose_diff(query_data, kernel_data):
    batch_size, M, _ = query_data.shape
    pos_shape, rot_shape, grip_shape = 3, 4, 1
    N = kernel_data.shape[1]

    # Position differences [batch, M, N, pos_dims]
    pos_diff = query_data[..., :pos_shape].unsqueeze(2) - \
               kernel_data[..., :pos_shape].unsqueeze(1)

    # Quaternion processing
    q1 = kernel_data[..., pos_shape:pos_shape+rot_shape].view(batch_size, N, 4)
    q2 = query_data[..., pos_shape:pos_shape+rot_shape].view(batch_size, M, 4)
    
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

    # Gripper differences [batch, M, N, grip_dims]
    grip_diff = query_data[..., -grip_shape:].unsqueeze(2) - \
                kernel_data[..., -grip_shape:].unsqueeze(1)
    
    pose_diff = torch.cat([pos_diff, rot_diff, grip_diff], dim=-1)
    return pose_diff

def gaussian_kernel(x):
    quad = (x).pow(2).sum(dim=-1)
    densities = torch.exp(-0.5 * (quad))
    return densities
    

def compute_pdf(query_data, kernel_data, h):
    pose_diff = compute_pose_diff(query_data, kernel_data)
    densities = gaussian_kernel(pose_diff / h)
    return densities

def compute_normalization(h):
    det = torch.exp(torch.sum(torch.log(h)))  
    norm_factor = 1.0 / (math.sqrt((2 * torch.pi) ** h.size(0)) * det)
    return norm_factor

def expand_bandwidth(h, device='cpu'):
    position_std, rotation_std, gripper_std = h, h * 5, h * 20
    pos_shape, rot_shape, grip_shape = 3, 3, 1

    h = torch.tensor([position_std] * pos_shape + [rotation_std] * rot_shape + [gripper_std] * grip_shape, device=device)
    return h


class PoseKDE:
    def __init__(self, kernel_data, bandwidth):
        feature_config = {'pos': [slice(0,3)], 'quat': [slice(3,7)], 'grip': [slice(7,8)]}
        self.kernel_data = process_kernel_data(kernel_data, feature_config)
        self.bandwidth = bandwidth
        self.feature_config = feature_config
        self.device = self.kernel_data.device

        self.h = expand_bandwidth(self.bandwidth, kernel_data.device)
        self.norm_factor = compute_normalization(self.h)

    def __call__(self, query_data, chunk_size=100):
        
        # Preprocess data
        query_data = process_query_data(query_data, self.feature_config)
        
        # Process in chunks
        batch_size = query_data.shape[0]
        results = []
        for i in range(0, batch_size, chunk_size):
            chunk = slice(i, min(i+chunk_size, batch_size))
            # query_data [B,M,A], kernel_data [B,N,A] -> densities [B,M,N]
            # { exp(-0.5 * (xi - xj)^2/h^2) for i,j in M,N }
            raw_densities = compute_pdf(query_data[chunk], self.kernel_data[chunk], self.h)
            densities = raw_densities.mean(dim=-1)
            densities = densities * self.norm_factor
            results.append(densities)

        return torch.cat(results, dim=0)

class ConditionalPoseKDE:
    def __init__(self, kernel_data, kernel_cond, bandwidth):
        feature_config = {'pos': [slice(0,3)], 'quat': [slice(3,7)], 'grip': [slice(7,8)]}
        self.kernel_data = process_kernel_data(kernel_data, feature_config)
        self.kernel_cond = process_kernel_data(kernel_cond, feature_config)
        self.bandwidth = bandwidth
        self.feature_config = feature_config
        self.device = self.kernel_data.device

        self.h = expand_bandwidth(self.bandwidth, kernel_data.device)
        self.norm_factor = compute_normalization(self.h)

    def __call__(self, query_data, query_cond, chunk_size=100):
        
        # Preprocess data
        query_data = process_query_data(query_data, self.feature_config)
        query_cond = process_query_data(query_cond, self.feature_config)
        
        # Process in chunks
        batch_size = query_data.shape[0]
        results = []
        for i in range(0, batch_size, chunk_size):
            chunk = slice(i, min(i+chunk_size, batch_size))
            data_raw_densities = compute_pdf(query_data[chunk], self.kernel_data[chunk], self.h)
            cond_raw_densities = compute_pdf(query_cond[chunk], self.kernel_cond[chunk], self.h)

            joint_densities = (data_raw_densities * cond_raw_densities).mean(dim=-1) * (self.norm_factor ** 2)
            cond_densities = cond_raw_densities.mean(dim=-1) * self.norm_factor

            conditional_densities = joint_densities / cond_densities
            results.append(conditional_densities)
        
        return torch.cat(results, dim=0)
    
class MMKDE:
    def __init__(self, data_trajs, bandwidth):
        self.data_trajs = data_trajs
        self.bandwidth = bandwidth
    
    def __call__(self, query_trajs):

        density_0 = PoseKDE(
                        kernel_data=self.data_trajs[:,:,0],
                        bandwidth=self.bandwidth,
                    )(query_data=query_trajs[:,:,0])
        density_0 = torch.log(density_0)

        cond_densities = []
        for t in range(query_trajs.shape[2] - 1):

            density = ConditionalPoseKDE(
                        kernel_data=self.data_trajs[:,:,t+1],
                        kernel_cond=self.data_trajs[:,:,t], 
                        bandwidth=self.bandwidth,
                    )(query_data=query_trajs[:,:,t+1], query_cond=query_trajs[:,:,t])
            
            cond_densities.append(torch.log(density))

        cond_densities = torch.stack(cond_densities, -1)
        traj_densities = density_0 + cond_densities.sum(dim=-1)

        traj_densities = torch.exp(traj_densities - traj_densities.max(1, keepdim=True).values)
        return traj_densities
    
from pytorch3d.transforms import (quaternion_to_axis_angle, 
                                 quaternion_multiply, rotation_6d_to_matrix,
                                 matrix_to_quaternion)
    
class TKDE:
    def __init__(self, policy, **kwargs):
        print('Using filtering method: KDE Filter')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy = policy
        self.threshold = kwargs.get('threshold', None)
        self.seed = kwargs.get('seed', None)
        self.bandwidth = kwargs.get('bandwidth', None)
        if isinstance(self.threshold, str):
            self.threshold = eval(self.threshold)
        if isinstance(self.seed, str):
            self.seed = eval(self.seed)
        if isinstance(self.bandwidth, str):
            self.bandwidth = eval(self.bandwidth)

        self.generator = None
        if self.seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.seed)
    
    def __call__(self, actions):
        denormalized_actions = self.policy.normalizer['action'].unnormalize(actions)

        s, e = self.policy.n_obs_steps - 1, self.policy.n_obs_steps - 1 + self.policy.n_action_steps
        denormalized_actions = denormalized_actions[:,:,s:e]

        b_dim, B = 0, denormalized_actions.shape[0]  # Batch dimension
        p_dim, P = 1, denormalized_actions.shape[1]  # Population dimension
        h_dim, H = 2, denormalized_actions.shape[2]  # Horizon dimension
        a_dim, A = 3, denormalized_actions.shape[3]  # Action dimension

        if A != 10: ValueError(f"Unsupported action dimension: {A}")

        to_quat = lambda x: matrix_to_quaternion(rotation_6d_to_matrix(x))
        # convert rotations from 6D to quaternion
        batched_actions = torch.cat(
            [
                actions[...,0:3],
                to_quat(actions[...,3:9]),
                actions[...,9:10]
        ], dim=-1)
        
        trajectory_densities = MMKDE(batched_actions, bandwidth=self.bandwidth)(batched_actions)
        
        if self.threshold is None:
            top_trajectories = trajectory_densities.argmax(dim=1)
            best_actions =  actions[torch.arange(B), top_trajectories]
            return best_actions

        mask = trajectory_densities > float(self.threshold)

        if self.generator is None:
            # If no seed is provided, use the first trajectory above the threshold
            first_nonzero_indices = torch.argmax(mask.float(), dim=1)
            best_actions = actions[torch.arange(B), first_nonzero_indices]
            return best_actions

        probs = mask.float()
        probs /= probs.sum(dim=1, keepdim=True)
        sampled_indices = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(1)
        best_actions = actions[torch.arange(B), sampled_indices]

        return best_actions

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