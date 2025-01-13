#! /bin/bash
echo "Running VMamba baselines"
image_sizes=(320 480)

for image_size in ${image_sizes[@]}
do
    export IMG_SIZE=$image_size
    # echo "Running tiny"
    # python3 main_no_dist.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth --zip > quatermap_results/vmamba_outputs/vmambav2v_tiny_224.log 2>&1

    # echo "Running small"
    # python3 main_no_dist.py --cfg configs/vssm/vmambav2_small_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_small_0229_ckpt_epoch_222.pth --zip > quatermap_results/vmamba_outputs/vmambav2v_small_224.log 2>&1


    echo "Running base with image size $image_size"
    python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip > quatermap_results/vmamba_outputs/vmambav2v_base_${image_size}.log 2>&1
done

export QUATERMAP=1

for image_size in ${image_sizes[@]}
do
    export IMG_SIZE=$image_size
    # echo "Running tiny with quatermap"
    # python3 main_no_dist.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth --zip > quatermap_results/vmamba_outputs/quatermap_vmambav2v_tiny_224.log 2>&1

    # echo "Running small with quatermap"
    # python3 main_no_dist.py --cfg configs/vssm/vmambav2_small_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_small_0229_ckpt_epoch_222.pth --zip > quatermap_results/vmamba_outputs/quatermap_vmambav2v_small_224.log 2>&1

    echo "Running base with quatermap"
    python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip > quatermap_results/vmamba_outputs/quatermap_vmambav2v_base_${image_size}.log 2>&1
done