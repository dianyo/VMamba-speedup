export PYTHONPATH=$PYTHONPATH:/joe/VMamba/segmentation

# run tiny model
# compress_list=(384 768 1536 3072)
# compress_list=(768 1536 3072)
# for i in ${compress_list[@]}; do
#     export COMPRESS_DIM=$i
#     python3 tools/test.py \
#         configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
#         ckpts/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth
# done
# unset COMPRESS_DIM
# python3 tools/test.py \
#     configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
#     ckpts/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth

# # run small model
# compress_list=(768 1536 3072 6144)
# for i in ${compress_list[@]}; do
#     export COMPRESS_DIM=$i
#     python3 tools/test.py \
#         configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
#         ckpts/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth
# done
# unset COMPRESS_DIM
# python3 tools/test.py \
#     configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
#     ckpts/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth

# run base model
# compress_list=(1024 2048 4096 8192)
compress_list=(4096 8192)
for i in ${compress_list[@]}; do
    export COMPRESS_DIM=$i
    python3 tools/test.py \
        configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py \
        ckpts/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
done
unset COMPRESS_DIM
python3 tools/test.py \
    configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py \
    ckpts/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
# python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py ckpts/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth
