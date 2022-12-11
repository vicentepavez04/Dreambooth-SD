#!/bin/bash

weight=backup/thermalface.ckpt
classifier=backup_classifier/cp-0028.hdf5
text="a man, thermalface, grey"
i=$$







CUDA_VISIBLE_DEVICES=2 python3.8 scripts/txt2img_with_classifier.py --outdir ./output_v2/ --seed $i \
                                                    --ddim_eta 0.0 --n_samples 1 \
                                                    --n_iter 1 --scale 10.0 \
                                                    --ddim_steps 50  --ckpt $weight \
                                                    --prompt "${text}" --classifier_path=$classifier