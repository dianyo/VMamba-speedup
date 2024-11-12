# Baseline
echo "Running Swin baselines for segmentation"

echo "Running tiny"
python3 tools/test.py configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py ckpt/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth

echo "Running small"
python3 tools/test.py configs/swin/swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py ckpt/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth

echo "Running base"
python3 tools/test.py configs/swin/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py ckpt/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth

echo "Running VMamba Quatermap for segmentation"
export QUATERMAP=1
echo "Running tiny with quatermap"
python3 tools/test.py configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py ckpt/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth

echo "Running small with quatermap"
python3 tools/test.py configs/swin/swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py ckpt/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth

echo "Running base with quatermap"
python3 tools/test.py configs/swin/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py ckpt/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth  --cfg-options model.backbone.pretrained=ckpt/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth
