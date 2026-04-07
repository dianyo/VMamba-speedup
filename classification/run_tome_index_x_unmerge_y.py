import os
import subprocess
import sys
import pickle

model_type = os.environ.get("MODEL_TYPE", "base")

# tome_numbers = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
tome_numbers = [64]
tome_ks = [2, 3]
for i, tome_n in enumerate(tome_numbers):
    for tome_k in tome_ks:
        os.environ["TOME_N"] = str(tome_n)
        os.environ["TOME_K"] = str(tome_k)
        print(f"TOME_N: {tome_n}, TOME_K: {tome_k}")
        # if model_type == "base":
        if False:
        # python3 main_no_dist.py --cfg configs/vssm/vmambav2v_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/  --resume ckpt/vssm1_base_0229s_ckpt_epoch_225.pth --zip
            subprocess.run([
                "python3", "main_no_dist.py", 
                "--cfg", "configs/vssm/vmambav2_base_224.yaml", 
                "--data-path", "/joe/data/ImageNet-Zip", 
                "--output", "/tmp/", 
                "--resume", "ckpt/vssm_base_0229_ckpt_epoch_237.pth", 
                "--zip",
                "--throughput"
            ], stdout=sys.stdout, stderr=sys.stderr)
        
        if True:
            # python3 main_no_dist.py --cfg configs/vssm/vmambav2_small_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_small_0229_ckpt_epoch_222.pth --zip
            subprocess.run([
                "python3", "main_no_dist.py", 
                "--cfg", "configs/vssm/vmambav2_small_224.yaml", 
                "--data-path", "/joe/data/ImageNet-Zip", 
                "--output", "/tmp/", 
                "--resume", "ckpt/vssm_small_0229_ckpt_epoch_222.pth", 
                "--zip",
                "--throughput"
            ], stdout=sys.stdout, stderr=sys.stderr)
        
        if True:
            # python3 main_no_dist.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth --zip
            # python3 main_no_dist.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/  --resume ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth --zip
            subprocess.run([
                "python3", "main_no_dist.py", 
                "--cfg", "configs/vssm/vmambav2v_tiny_224.yaml", 
                "--data-path", "/joe/data/ImageNet-Zip", 
                "--output", "/tmp/", 
                "--resume", "ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth", 
                "--zip",
                "--throughput"
            ], stdout=sys.stdout, stderr=sys.stderr)