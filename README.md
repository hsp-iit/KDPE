<div align="center">

# KDPE: Kernel Density Policy Ensemble for Diffusion Action Generation

**(Code accompanying the KDPE method – submitted to CoRL 2025)**  
_Fork of Diffusion Policy with population sampling + density filtering for robust visuomotor control_

</div>

This is the code for our work **[KDPE: A Kernel Density Estimation Strategy for Diffusion Policy Trajectory Selection](https://arxiv.org/pdf/2508.10511)**, which is based on the original [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) implementation.

## Installation

To set up the environment, please use Conda and run:

```bash
conda env create --file conda_environment.yaml
conda activate kdpe
```

If you are running it on a remote cluster with no sudo privileges you might need to apply [this fix](https://github.com/openai/mujoco-py/issues/627)

## Datasets

The datasets should be organized in the following structure:

```
data/datasets/
├── tool_hang/
│   └── ph/
│       └── image_abs.hdf5
├── square/
│   ├── ph/
│   │   └── image_abs.hdf5
│   └── mh/
│       └── image_abs.hdf5
├── assembly/
│   └── image_abs.hdf5
...
```

The tasks comes from two different sources:
```
robotmimic = [lift, can, square, tool_hang]
mimicgen = [stack, coffee, assembly]
```
You can download the datasets from from the following links: [robomimic_dataset](https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip), [mimicgen_dataset](https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core).

`mimicgen` datasets use relative actions and need to be converted to absolute actions. You can use the following script for conversion:

```bash
python diffusion_policy/scripts/robomimic_dataset_conversion.py --input /path/to/your/dataset.hdf5 --output /path/to/your/output_dataset.hdf5
```

## Running Experiments

First, you need to train a model to obtain a checkpoint.

```bash
python train.py \
    --config-dir diffusion_policy/config \
    --config-name train_diffusion_unet_hybrid_workspace.yaml \
    task=tool_hang
```

This will produce a checkpoint named `step_79999.ckpt` under the path `data/weights/diffusion_cnn_tool_hang_ph/checkpoints`.

To test our method (KDPE), run:

```bash
python eval.py --model cnn --task tool_hang_ph --Filter.name kde
```

To test the baseline, run:

```bash
python eval.py --model cnn --task tool_hang_ph --Filter.name passall
```

<!-- ## License

The original Diffusion Policy code is licensed under the MIT License. The modifications introduced in this repository from commit `ad788214debba67fd201eb17a0dc31e64a2cdb45` onwards are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). -->

## Citation

If you find our method or visualizer useful, consider citing it.
