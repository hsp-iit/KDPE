import torch
from config import Config as cfg
import numpy as np

        

def hsl_to_rgb(hsl_color):
    """
    Convert HSL color values to RGB color values.
    
    Parameters:
    h (float): Hue value (0-360)
    s (float): Saturation value (0-1)
    l (float): Lightness value (0-1)
    
    Returns:
    tuple: RGB values as integers (0-255)
    """
    h, s, l = hsl_color
    # Make sure h is in the range 0-360
    h = h % 360
    
    # Convert to values between 0 and 1
    h /= 360
    
    if s == 0:
        # Achromatic (gray)
        r = g = b = l
    else:
        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p
        
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    # Convert to 0-255 range
    r = round(r * 255)
    g = round(g * 255)
    b = round(b * 255)
    
    return np.array([r, g, b])

def random_color(hsl_color, noise, generator):
    clipped_range = (max(0, hsl_color[2] - (noise / 2)), min(1, hsl_color[2] + (noise / 2)))
    hsl_color[2] = torch.FloatTensor(1).uniform_(*clipped_range, generator=generator).item()

    return hsl_color



def shift_object_color(env, generator):
    model = env.env.sim.model

    if cfg.task.startswith('square'):
        for i in range(5):
            hsl_color = np.array([35,0.456,0.447])
            noisy_hsl_color = hsl_color
            noisy_hsl_color[2] = min(1,hsl_color[2] - (0.1))
            color_rgb = np.concatenate([hsl_to_rgb(noisy_hsl_color), [255]])

            obj = model.geom_name2id(f'SquareNut_g{i}_visual')
            model.geom_matid[obj] = -1
            model.geom_rgba[obj] =  color_rgb / 255

    elif cfg.task.startswith('coffee'):
        # coffee_pod_g0_visual
        hsl_color = np.array([0,0.0,0.835])
        noisy_hsl_color = hsl_color
        noisy_hsl_color[2] = min(1,hsl_color[2] - (0.1))
        color_rgb = np.concatenate([hsl_to_rgb(noisy_hsl_color), [255]])

        obj = model.geom_name2id(f'coffee_pod_g0_visual')
        model.geom_matid[obj] = -1
        model.geom_rgba[obj] =  color_rgb / 255

    elif cfg.task.startswith('stack'):
        # coffee_pod_g0_visual
        hsl_color = np.array([204,0.87,0.392])
        noisy_hsl_color = hsl_color
        noisy_hsl_color[2] = min(1,hsl_color[2] - (0.1))
        color_rgb = np.concatenate([hsl_to_rgb(noisy_hsl_color), [255]])

        obj = model.geom_name2id(f'cubeC_g0_vis')
        model.geom_matid[obj] = -1
        model.geom_rgba[obj] =  color_rgb / 255

    elif cfg.task.startswith('assembly'):
        hsl_color = np.array([354,0.512,0.475])
        noisy_hsl_color = hsl_color
        noisy_hsl_color[2] = min(1,hsl_color[2] - (0.1))
        color_rgb = np.concatenate([hsl_to_rgb(noisy_hsl_color), [255]])

        for name in [f'piece_1_{i}_{j}_{k}_vis' for i in [0,1,2] for j in range(3) for k in range(3) if (i==0) or (j==1)]:
            obj = model.geom_name2id(name)
            model.geom_matid[obj] = -1
            model.geom_rgba[obj] =  color_rgb / 255

    elif cfg.task.startswith('lift'):
        hsl_color = np.array([354,0.512,0.475])
        noisy_hsl_color = hsl_color
        noisy_hsl_color[2] = min(1,hsl_color[2] - (0.1))
        color_rgb = np.concatenate([hsl_to_rgb(noisy_hsl_color), [255]])

        obj = model.geom_name2id(f'cube_g0_vis')
        model.geom_matid[obj] = -1
        model.geom_rgba[obj] =  color_rgb / 255

    elif cfg.task.startswith('tool_hang'):
        hsl_color = np.array([167,0.054,0.673])
        noisy_hsl_color = hsl_color
        noisy_hsl_color[2] = min(1,hsl_color[2] - (0.1))
        color_rgb = np.concatenate([hsl_to_rgb(noisy_hsl_color), [255]])

        handle = model.geom_name2id('tool_handle_g0_vis')
        model.geom_matid[handle] = -1
        model.geom_rgba[handle] = color_rgb / 255
        for i in range(8):
            hole = model.geom_name2id(f'tool_hole1_hc_{i}_vis')
            model.geom_matid[hole] = -1
            model.geom_rgba[hole] = color_rgb / 255
        for i in range(8):
            hole = model.geom_name2id(f'tool_hole2_hc_{i}_vis')
            model.geom_matid[hole] = -1
            model.geom_rgba[hole] = color_rgb / 255

    elif cfg.task.startswith('can'):
        hsl_color = np.array([356,0.84,0.51])
        noisy_hsl_color = hsl_color
        noisy_hsl_color[2] = min(1,hsl_color[2] - (0.1))
        color_rgb = np.concatenate([hsl_to_rgb(noisy_hsl_color), [255]])

        obj = model.geom_name2id(f'Can_g0_visual')
        model.geom_matid[obj] = -1
        model.geom_rgba[obj] =  color_rgb / 255


class ShiftEnv:
    def __init__(self, seed=0):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __call__(self, env):
        if cfg.shift == 'object_color':
            shift_object_color(env, generator=self.generator)
