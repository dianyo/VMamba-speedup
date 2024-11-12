#!/bin/bash
echo "Running VMamba baselines for segmentation"

echo "Running tiny"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth

echo "Running small"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth

echo "Running base"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth

echo "Running VMamba Quatermap for segmentation"
export QUATERMAP=1
echo "Running tiny with quatermap"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth

echo "Running small with quatermap"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth

echo "Running base with quatermap"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
