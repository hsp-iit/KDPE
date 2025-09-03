
import datetime
from pathlib import Path
from clearconf import BaseConfig
import json
import os

class Config(BaseConfig):
    checkpoint = 'step_79999.ckpt'
    model = 'cnn'
    task = 'tool_hang_ph'
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    checkpoint_path = '[eval]f"data/weights/diffusion_{cfg.model}_{cfg.task}/checkpoints/{cfg.checkpoint}"'
    output_dir = '[eval]f"data/evaluations/diffusion_{cfg.model}_{cfg.task}/{cfg.Filter.name}/{cfg.checkpoint}"'
    device = 'cuda:0'

    horizon = 8
    population = 100
    shift = 'none'
    tags = ''

    class Filter:
        name = 'kde'
        threshold = None
        seed = None
        bandwidth = 0.05

assert Config.shift in ['none', 'object_color']
print('Configuration Loaded')
Path(Config.output_dir).mkdir(parents=True, exist_ok=True)
with open(f'{Config.output_dir}/config.json', 'w') as f:
    json.dump(Config.to_dict(), f, indent=4)
