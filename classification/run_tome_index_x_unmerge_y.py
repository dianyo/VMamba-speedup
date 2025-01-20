import os
import subprocess
import sys
import pickle

model_type = os.environ.get("MODEL_TYPE", "base")

# tome_numbers = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
tome_numbers = [64]
for i, tome_n in enumerate(tome_numbers):
    os.environ["TOME_N"] = str(tome_n)
    if model_type == "base":
        # python3 main_no_dist.py --cfg configs/vssm/vmambav2v_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/  --resume ckpt/vssm1_base_0229s_ckpt_epoch_225.pth --zip
        subprocess.run([
            "python3", "main_no_dist.py", 
            "--cfg", "configs/vssm/vmambav2v_base_224.yaml", 
            "--data-path", "/joe/data/ImageNet-Zip", 
            "--output", "/tmp/", 
            "--resume", "ckpt/vssm1_base_0229s_ckpt_epoch_225.pth", 
            "--zip",
            "--throughput"
        ], stdout=sys.stdout, stderr=sys.stderr)
    
    elif model_type == "small":
        # python3 main_no_dist.py --cfg configs/vssm/vmambav2v_small_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/  --resume ckpt/vssm1_small_0229s_ckpt_epoch_240.pth --zip
        subprocess.run([
            "python3", "main_no_dist.py", 
            "--cfg", "configs/vssm/vmambav2v_small_224.yaml", 
            "--data-path", "/joe/data/ImageNet-Zip", 
            "--output", "/tmp/", 
            "--resume", "ckpt/vssm1_small_0229s_ckpt_epoch_240.pth", 
            "--zip",
            "--throughput"
        ], stdout=sys.stdout, stderr=sys.stderr)
    
    else:
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
