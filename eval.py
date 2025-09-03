"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
from pathlib import Path
import click
import hydra
import torch
from omegaconf import open_dict
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from config import Config
from omegaconf import open_dict, DictConfig


def main():
    Path(Config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(Config.checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    with open_dict(cfg):
        cfg.policy.filter = DictConfig({
            'target': f'diffusion_policy.model.filters.{Config.Filter.name}',
            'bandwidth': Config.Filter.bandwidth,
            'threshold': Config.Filter.threshold,
            'seed': Config.Filter.seed
        })
        cfg.policy.population = int(Config.population)
        cfg.policy.n_action_steps = int(Config.horizon)

    workspace = cls(cfg, output_dir=Config.output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(Config.device)
    policy.to(device)
    policy.eval()

    cfg.task.env_runner.n_envs = 50
    cfg.task.env_runner.n_test = 100
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_action_steps = int(Config.horizon)

    if 'dataset_path' in cfg.task.env_runner and not Path(cfg.task.env_runner.dataset_path).exists():
        cfg.task.env_runner.dataset_path = f'data/datasets/robomimic/datasets/{Config.task[:-3]}/{Config.task[-2:]}/image_abs.hdf5'
    cfg.task.env_runner.test_start_seed = 4600000
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=Config.output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(Config.output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
