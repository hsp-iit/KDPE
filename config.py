
import datetime
from pathlib import Path
from clearconf import BaseConfig
import json
import os

class Config(BaseConfig):
    dataset = 'original'
    checkpoint = 'checkpoint1'
    model = 'transformer'
    task = 'can_mh'
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    checkpoint_path = '[eval]f"data/weights/diffusion_{cfg.model}_{cfg.task}/checkpoints/{cfg.dataset}/{cfg.checkpoint}.ckpt"'
    output_dir = os.environ.get('BULB_LOG_DIR', '[eval]f"data/evaluations/deubg/{cfg.start_time}"')
    device = 'cuda:0'

    horizon = 8 # 15, 9
    population = 100
    shift = 'none'  # flip, color_jitter, rotate
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
