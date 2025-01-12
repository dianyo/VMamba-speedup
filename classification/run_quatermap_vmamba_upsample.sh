export QUATERMAP=1

up_smapling_methods=("nearest" "bilinear" "bicubic")
# for i in "${up_smapling_methods[@]}"
# do
#     echo "-----------------------------------"
#     export UPSAMPLE_MODE=$i
#     echo "Running base with quatermap upsampling method: $i"
#     file_name="quatermap_results/vmamba_outputs/quatermap_vmambav2v_base_224_upsample_$i.log"
#     python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip > $file_name 2>&1
# done

export ONE_DATA=1
for i in "${up_smapling_methods[@]}"
do
    echo "-----------------------------------"
    export UPSAMPLE_MODE=$i
    echo "Running one_data (flops and throughput) with quatermap upsampling method: $i"
    file_name="quatermap_results/vmamba_outputs/quatermap_vmambav2v_224_upsample_${i}_flops.log"
    python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip --flops > $file_name 2>&1

    file_name="quatermap_results/vmamba_outputs/quatermap_vmambav2v_224_upsample_${i}_throughput.log"
    python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip --throughput > $file_name 2>&1
done
