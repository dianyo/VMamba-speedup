export QUATERMAP=1
for i in {2..4}
do
    start=$((2 * i))
    for j in $(seq $start 8)
    do
        echo "-----------------------------------"
        export QUATERMAP_INTERVAL=$j
        export QUATERMAP_N_OUT_OF_INTERVAL=$i
        echo "Running base with quatermap interval $QUATERMAP_N_OUT_OF_INTERVAL out of $QUATERMAP_INTERVAL"
        file_name="quatermap_results/vmamba_outputs/quatermap_vmambav2v_base_224_$i"_out_of_"$j.log"
        python3 main_no_dist.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /joe/data/ImageNet-Zip --output /tmp/ --resume ckpt/vssm_base_0229_ckpt_epoch_237.pth --zip > $file_name 2>&1
    done
done

