#! /bin/bash

echo "Start draw weight distribution for tiny"
python3 main_no_dist.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --plot_weight_distribution --resume ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth --zip --plot_type histogram >> draw_weight_distribution.log 2>&1
python3 main_no_dist.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --plot_weight_distribution --resume ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth --zip --plot_type box >> draw_weight_distribution.log 2>&1
echo "End draw weight distribution for tiny"

echo "Start draw weight distribution for small"
python3 main_no_dist.py --cfg configs/vssm/vmambav2v_small_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --plot_weight_distribution --resume ckpt/vssm1_small_0229s_ckpt_epoch_240.pth --zip --plot_type histogram >> draw_weight_distribution.log 2>&1
python3 main_no_dist.py --cfg configs/vssm/vmambav2v_small_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --plot_weight_distribution --resume ckpt/vssm1_small_0229s_ckpt_epoch_240.pth --zip --plot_type box >> draw_weight_distribution.log 2>&1
echo "End draw weight distribution for small"

echo "Start draw weight distribution for base"
python3 main_no_dist.py --cfg configs/vssm/vmambav2v_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --plot_weight_distribution --resume ckpt/vssm1_base_0229s_ckpt_epoch_225.pth --zip --plot_type histogram >> draw_weight_distribution.log 2>&1
python3 main_no_dist.py --cfg configs/vssm/vmambav2v_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --plot_weight_distribution --resume ckpt/vssm1_base_0229s_ckpt_epoch_225.pth --zip --plot_type box >> draw_weight_distribution.log 2>&1
echo "End draw weight distribution for base"