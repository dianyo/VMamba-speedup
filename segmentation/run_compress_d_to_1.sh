export PYTHONPATH=$PYTHONPATH:/joe/VMamba/segmentation
# Run compress before scan
# compress_dim_list=(384 768 1536 3072)
# # compress_dim_list=(3072)
# for compress_dim in ${compress_dim_list[@]}; do
#     export COMPRESS_DIM=$compress_dim
#     echo "COMPRESS_DIM: $compress_dim"
#     echo "tiny"
#     python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth
# done

# compress_dim_list=(768 1536 3072 6144)
# # compress_dim_list=(6144)
# for compress_dim in ${compress_dim_list[@]}; do
#     export COMPRESS_DIM=$compress_dim
#     echo "COMPRESS_DIM: $compress_dim"
#     echo "small"
#     python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth
# done

# compress_dim_list=(1024 2048 4096 8196)
# # compress_dim_list=(8196)
# for compress_dim in ${compress_dim_list[@]}; do
#     export COMPRESS_DIM=$compress_dim
#     echo "COMPRESS_DIM: $compress_dim"
#     echo "base"
#     python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
# done

# # Run baseline
unset COMPRESS_DIM
echo "tiny"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth
echo "small"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth
echo "base"
python3 tools/test.py configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth