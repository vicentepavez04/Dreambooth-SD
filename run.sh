#!/bin/bash


weight=/path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt
text="a photo of a thermal face"
python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt $weight --prompt "${text}" 