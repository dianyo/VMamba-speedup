import os
import subprocess
import sys

model_type = os.environ.get("MODEL_TYPE", "base")
layer_names = {
    "base": [
        "backbone.layers.0.blocks.0.op", "backbone.layers.0.blocks.1.op", "backbone.layers.1.blocks.0.op", "backbone.layers.1.blocks.1.op",
        "backbone.layers.2.blocks.0.op", "backbone.layers.2.blocks.1.op", "backbone.layers.2.blocks.2.op", "backbone.layers.2.blocks.3.op",
        "backbone.layers.2.blocks.4.op", "backbone.layers.2.blocks.5.op", "backbone.layers.2.blocks.6.op", "backbone.layers.2.blocks.7.op",
        "backbone.layers.2.blocks.8.op", "backbone.layers.2.blocks.9.op", "backbone.layers.2.blocks.10.op", "backbone.layers.2.blocks.11.op",
        "backbone.layers.2.blocks.12.op", "backbone.layers.2.blocks.13.op", "backbone.layers.2.blocks.14.op", "backbone.layers.2.blocks.15.op",
        "backbone.layers.2.blocks.16.op", "backbone.layers.2.blocks.17.op", "backbone.layers.2.blocks.18.op", "backbone.layers.2.blocks.19.op",
        "backbone.layers.3.blocks.0.op", "backbone.layers.3.blocks.1.op"
    ],
    "small": [
        "backbone.layers.0.blocks.0.op", "backbone.layers.0.blocks.1.op", "backbone.layers.1.blocks.0.op", "backbone.layers.1.blocks.1.op",
        "backbone.layers.2.blocks.0.op", "backbone.layers.2.blocks.1.op", "backbone.layers.2.blocks.2.op", "backbone.layers.2.blocks.3.op",
        "backbone.layers.2.blocks.4.op", "backbone.layers.2.blocks.5.op", "backbone.layers.2.blocks.6.op", "backbone.layers.2.blocks.7.op",
        "backbone.layers.2.blocks.8.op", "backbone.layers.2.blocks.9.op", "backbone.layers.2.blocks.10.op", "backbone.layers.2.blocks.11.op",
        "backbone.layers.2.blocks.12.op", "backbone.layers.2.blocks.13.op", "backbone.layers.2.blocks.14.op", "backbone.layers.2.blocks.15.op",
        "backbone.layers.2.blocks.16.op", "backbone.layers.2.blocks.17.op", "backbone.layers.2.blocks.18.op", "backbone.layers.2.blocks.19.op",
        "backbone.layers.3.blocks.0.op", "backbone.layers.3.blocks.1.op"
    ],
    "tiny": [
        "backbone.layers.0.blocks.0.op", "backbone.layers.0.blocks.1.op", "backbone.layers.1.blocks.0.op", "backbone.layers.1.blocks.1.op",
        "backbone.layers.2.blocks.0.op", "backbone.layers.2.blocks.1.op", "backbone.layers.2.blocks.2.op", "backbone.layers.2.blocks.3.op",
        "backbone.layers.2.blocks.4.op", "backbone.layers.2.blocks.5.op", "backbone.layers.2.blocks.6.op", "backbone.layers.2.blocks.7.op",
        "backbone.layers.3.blocks.0.op", "backbone.layers.3.blocks.1.op"
    ]
}

for i, selected_layer in enumerate(layer_names[model_type]):
    print(f"Running layer: {selected_layer}")
    print("-----------------------------------")
    print("-----------------------------------")
    os.environ["SELECTED_LAYERS"] = selected_layer
    if model_type == "base":
        # python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
        subprocess.run([
            "python3", "tools/test.py",
            "configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py",
            "ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth",
            "--cfg-options", "model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth"
        ], stdout=sys.stdout, stderr=sys.stderr)
    
    elif model_type == "small":
        # python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth
        subprocess.run([
            "python3", "tools/test.py",
            "configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py",
            "ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth",
            "--cfg-options", "model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth"
        ], stdout=sys.stdout, stderr=sys.stderr)
    
    else:
        # python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth
        subprocess.run([
            "python3", "tools/test.py",
            "configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py",
            "ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth",
            "--cfg-options", "model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth"
        ], stdout=sys.stdout, stderr=sys.stderr)
