#!/bin/bash
export ONE_DATA=1
export QUATERMAP=1

export QUATERMAP_STRATEGY="none_first_layer"
echo -e "\n\n==================================="
echo "Running base with quatermap strategy $QUATERMAP_STRATEGY"
for i in {2..2}
do
    echo "-----------------------------------"
    export QUATERMAP_FREQ=$i
    echo "Running base with quatermap strategy $QUATERMAP_STRATEGY and freq $QUATERMAP_FREQ"
    # echo "Running tiny"
    file_name="quatermap_results/vmamba_outputs/${QUATERMAP_STRATEGY}_vmambav2v_tiny_224_freq_${i}_throughput.log"
    python3 main_no_dist.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth --zip --throughput > $file_name 2>&1

    # echo "Running small"
    file_name="quatermap_results/vmamba_outputs/${QUATERMAP_STRATEGY}_vmambav2v_small_224_freq_${i}_throughput.log"
    python3 main_no_dist.py --cfg configs/vssm/vmambav2_small_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_small_0229_ckpt_epoch_222.pth --zip --throughput > $file_name 2>&1
    
    # file_name="quatermap_results/vmamba_outputs/${QUATERMAP_STRATEGY}_vmambav2v_base_224_freq_${i}_throughput.log"
    # python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip --throughput > $file_name 2>&1
done
# unset QUATERMAP_FREQ

# export QUATERMAP_STRATEGY="deepest_layer"
# echo -e "\n\n==================================="
# echo "Running base with quatermap strategy $QUATERMAP_STRATEGY"
# file_name="quatermap_results/vmamba_outputs/${QUATERMAP_STRATEGY}_vmambav2v_base_224_throughput.log"
# python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip --throughput > $file_name 2>&1

# export QUATERMAP_STRATEGY="full_layer"
# echo -e "\n\n==================================="
# echo "Running base with quatermap strategy $QUATERMAP_STRATEGY"
# for i in {0..3}
# do
#     echo "-----------------------------------"
#     layer_name="layers.$i"
#     export QUATERMAP_LAYER=$layer_name
#     echo "Running base with quatermap strategy $QUATERMAP_STRATEGY and layer $QUATERMAP_LAYER"
#     file_name="quatermap_results/vmamba_outputs/${QUATERMAP_STRATEGY}_vmambav2v_base_224_${QUATERMAP_LAYER}_throughput.log"
#     python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip --throughput > $file_name 2>&1
# done
# unset QUATERMAP_LAYER

# export QUATERMAP_STRATEGY="all_layers"
# echo -e "\n\n==================================="
# echo "Running base with quatermap strategy $QUATERMAP_STRATEGY"
# for i in {1..4}
# do
#     echo "-----------------------------------"
#     export QUATERMAP_FREQ=$i
#     echo "Running base with quatermap strategy $QUATERMAP_STRATEGY and freq $QUATERMAP_FREQ"
#     file_name="quatermap_results/vmamba_outputs/${QUATERMAP_STRATEGY}_vmambav2v_base_224_freq_${i}_throughput.log"
#     python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip --throughput > $file_name 2>&1
# done

