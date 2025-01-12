import os
import subprocess
import sys
import pickle

model_type = os.environ.get("MODEL_TYPE", "base")
layers = pickle.load(open(f"layer_select_each_layer_by_acc.pkl", "rb"))

selected_layers = []
for i, selected_layer in enumerate(reversed(layers[model_type][0])):
    selected_layers.append(selected_layer)
    if not len(selected_layers) in [2, 4, 8]:
        continue
    print(f"Running layer: {selected_layers}")
    print("-----------------------------------")
    print("-----------------------------------")
    os.environ["SELECTED_LAYERS"] = ",".join(selected_layers)
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
