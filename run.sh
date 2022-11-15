#!/bin/bash


weight=backup/thermalface.ckpt
text="a photo of Barack Obama, grey, thermal, thermalface"

start=$$
END=$(( 20+$start ))
echo "start seed $start"
echo "end seed $END"
for i in $(seq $start $END); do
python scripts/stable_txt2img.py --seed $i --ddim_eta 0.0 --n_samples 3 --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt $weight --prompt "$>
done
