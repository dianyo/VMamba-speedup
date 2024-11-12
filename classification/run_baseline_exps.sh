#!/bin/bash
# export ONE_DATA=1
# ConvNextv2
echo "Running ConvNextv2 baselines"
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/convnextv2-tiny-1k-224 > baseline_outputs/convnextv2-tiny-1k-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/convnextv2-base-1k-224 > baseline_outputs/convnextv2-base-1k-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/convnextv2-large-1k-224 > baseline_outputs/convnextv2-large-1k-224.log 2>&1

# EfficientNetv2
echo "Running EfficientNetv2 baselines"
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name tf_efficientnetv2_s.in1k > baseline_outputs/efficientnetv2_s.in1k.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name tf_efficientnetv2_m.in1k > baseline_outputs/efficientnetv2_m.in1k.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name tf_efficientnetv2_l.in1k > baseline_outputs/efficientnetv2_l.in1k.log 2>&1

# DeiT
echo "Running DeiT baselines"
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/deit-tiny-patch16-224 > baseline_outputs/deit-tiny-patch16-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/deit-small-patch16-224 > baseline_outputs/deit-small-patch16-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/deit-base-patch16-224 > baseline_outputs/deit-base-patch16-224.log 2>&1

# SwinTransformer
echo "Running SwinTransformer baselines"
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name microsoft/swin-tiny-patch4-window7-224 > baseline_outputs/swin-tiny-patch4-window7-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name microsoft/swin-small-patch4-window7-224 > baseline_outputs/swin-small-patch4-window7-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name microsoft/swin-base-patch4-window7-224 > baseline_outputs/swin-base-patch4-window7-224.log 2>&1

# ========================
# QUATERMAP
export QUATERMAP=1

# ConvNext
echo "Running ConvNextv2 quatermap baselines"
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/convnextv2-tiny-1k-224 > baseline_outputs/quatermap-convnextv2-tiny-1k-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/convnextv2-base-1k-224 > baseline_outputs/quatermap-convnextv2-base-1k-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/convnextv2-large-1k-224 > baseline_outputs/quatermap-convnextv2-large-1k-224.log 2>&1

# EfficientNetv2
echo "Running EfficientNetv2 quatermap baselines"
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name tf_efficientnetv2_s.in1k > baseline_outputs/quatermap-efficientnetv2_s.in1k.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name tf_efficientnetv2_m.in1k > baseline_outputs/quatermap-efficientnetv2_m.in1k.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name tf_efficientnetv2_l.in1k > baseline_outputs/quatermap-efficientnetv2_l.in1k.log 2>&1

# DeiT
echo "Running DeiT quatermap baselines"
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/deit-tiny-patch16-224 > baseline_outputs/quatermap-deit-tiny-patch16-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/deit-small-patch16-224 > baseline_outputs/quatermap-deit-small-patch16-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name facebook/deit-base-patch16-224 > baseline_outputs/quatermap-deit-base-patch16-224.log 2>&1

# SwinTransformer
echo "Running SwinTransformer quatermap baselines"
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name microsoft/swin-tiny-patch4-window7-224 > baseline_outputs/quatermap-swin-tiny-patch4-window7-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name microsoft/swin-small-patch4-window7-224 > baseline_outputs/quatermap-swin-small-patch4-window7-224.log 2>&1
python3 main_baseline.py --cfg configs/other_baselines.yaml --data-path /joe/data/ImageNet-Zip/ --output /tmp/ --zip --model_name microsoft/swin-base-patch4-window7-224 > baseline_outputs/quatermap-swin-base-patch4-window7-224.log 2>&1