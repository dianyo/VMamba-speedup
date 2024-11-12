# CNN: ConvNext, EfficientNet
# ViT: DeiT, SwinTransformer
# facebook/convnextv2-tiny-1k-224, timm/tf_efficientnetv2_m.in1k
# facebook/deit-base-patch16-224, microsoft/swin-tiny-patch4-window7-224

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from transformers.models.convnextv2.modeling_convnextv2 import (
    ConvNextV2Layer,
    ConvNextV2ForImageClassification,
)
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification,
    ViTSelfAttention,
)
from transformers.models.swin.modeling_swin import (
    SwinForImageClassification,
    SwinSelfAttention,
)
from tqdm import tqdm
import argparse
import os
import time
from timm.utils import accuracy, AverageMeter
from utils.utils import reduce_tensor
from utils.logger import create_logger
import timm
from timm.models._efficientnet_blocks import ConvBnAct, InvertedResidual, EdgeResidual
from timm.models.efficientnet import EfficientNet

from data.build import build_dataset
from config import get_config
import math
import sys

from typing import Optional
import types

TESTING_MODEL = "convnextv2"
TIMM_MODELS = [
    "tf_efficientnetv2_l.in1k",
    "tf_efficientnetv2_b0.in1k",
    "tf_efficientnetv2_m.in1k",
    "tf_efficientnetv2_s.in1k",
    "tf_efficientnetv2_s.in21k",
    "tf_efficientnetv2_l.in21k",
]
layer_info = {}


@torch.no_grad()
def validate(config, data_loader, model, is_timm_model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    after_warmup_batch_time = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        end = time.time()
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            if is_timm_model:
                output = model(images)
            else:
                output = model(images, return_dict=True).logits

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            # if idx % 5 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    end = time.time()
    logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def quatermap_conv_forward_pre_hook(module, input):
    return input[0][:, :, ::2, ::2]
    # return input[0]


def quatermap_conv_forward_hook(module, input, output):
    global layer_info
    H, W = layer_info[id(module)]
    output = F.interpolate(output, size=(H, W), mode="nearest")
    # print(output.shape)
    return output


def quatermap_vit_forward_pre_hook(module, input):
    global layer_info
    B, L, C = input[0].shape
    cls_token = input[0][:, 0, :]
    feature_map_dim = int(math.sqrt(int(L - 1)))
    layer_info[id(module)] = (feature_map_dim, feature_map_dim)
    feature_map = input[0][:, 1:, :].reshape(
        -1, feature_map_dim, feature_map_dim, input[0].shape[2]
    )
    feature_map = feature_map[:, ::2, ::2, :].reshape(B, -1, C)
    new_input = torch.cat([cls_token.unsqueeze(1), feature_map], dim=1)
    # return input[0]
    return new_input


def quatermap_vit_forward_hook(module, input, output):
    global layer_info
    H, W = layer_info[id(module)]
    cls_token = output[0][:, 0, :]
    feature_map = (
        output[0][:, 1:, :]
        .reshape(-1, H // 2, W // 2, output[0].shape[2])
        .permute(0, 3, 1, 2)
    )
    feature_map = F.interpolate(feature_map, size=(H, W), mode="nearest")
    feature_map = feature_map.permute(0, 2, 3, 1).reshape(-1, H * W, output[0].shape[2])
    new_output = torch.cat([cls_token.unsqueeze(1), feature_map], dim=1)
    return (new_output,)


def quatermap_swin_forward_pre_hook(module, input):
    global layer_info
    B, L, C = input[0].shape
    feature_map_dim = int(math.sqrt(int(L)))

    # if feature_map_dim <= module.window_size:
    #     layer_info[id(module)] = (feature_map_dim, feature_map_dim)
    #     return input
    layer_info[id(module)] = (feature_map_dim, feature_map_dim)
    feature_map = input[0].reshape(
        -1, feature_map_dim, feature_map_dim, input[0].shape[2]
    )
    new_input = feature_map[:, ::2, ::2, :]
    new_input_dim = (new_input.shape[1], new_input.shape[2])
    new_input = new_input.reshape(B, -1, C)
    new_attention_mask = input[1]
    if new_attention_mask is not None:
        new_attention_mask = new_attention_mask.reshape(
            -1, feature_map_dim, feature_map_dim, feature_map_dim, feature_map_dim
        )
        new_attention_mask = new_attention_mask[:, ::2, ::2, ::2, ::2]
        new_attention_mask = new_attention_mask.reshape(
            -1,
            new_attention_mask.shape[1] * new_attention_mask.shape[2],
            new_attention_mask.shape[3] * new_attention_mask.shape[4],
        )
    return (new_input, new_attention_mask, input[2], input[3])


def quatermap_swin_forward_hook(module, input, output):
    global layer_info
    H, W = layer_info[id(module)]
    dim = int(math.sqrt(output[0].shape[1]))
    feature_map = (
        output[0].reshape(-1, dim, dim, output[0].shape[2]).permute(0, 3, 1, 2)
    )
    feature_map = F.interpolate(feature_map, size=(H, W), mode="nearest")
    new_output = feature_map.permute(0, 2, 3, 1).reshape(-1, H * W, output[0].shape[2])
    return (new_output,)


def patch_swin_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
):
    # print("Patched SwinSelfAttention -------------------")
    # print("hidden_states", hidden_states.shape)
    batch_size, dim, num_channels = hidden_states.shape
    mixed_query_layer = self.query(hidden_states)

    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)
    # print("key_layer", key_layer.shape)
    # print("value_layer", value_layer.shape)
    # print("query_layer", query_layer.shape)
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    # print("attention_scores", attention_scores.shape)
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ]
    # print("self.relative_position_bias_table", self.relative_position_bias_table.shape)
    # print("self.relative_position_index", self.relative_position_index)
    # print("self.relative_position_index", self.relative_position_index.shape)
    # print("relative_position_bias", relative_position_bias.shape)
    relative_position_bias = relative_position_bias.view(
        self.window_size[0] * self.window_size[1],
        self.window_size[0] * self.window_size[1],
        -1,
    )

    if (
        self.window_size[0] != attention_scores.shape[-2]
        or self.window_size[1] != attention_scores.shape[-1]
    ):
        relative_position_bias = relative_position_bias.view(
            self.window_size[0],
            self.window_size[1],
            self.window_size[0],
            self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias[
            ::2, ::2, ::2, ::2, :
        ].contiguous()
        relative_position_bias = relative_position_bias.view(
            relative_position_bias.shape[0] * relative_position_bias.shape[1],
            relative_position_bias.shape[2] * relative_position_bias.shape[3],
            -1,
        )
        # print("relative_position_bias", relative_position_bias.shape)

    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in SwinModel forward() function)
        mask_shape = attention_mask.shape[0]
        attention_scores = attention_scores.view(
            batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
        )
        attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
        attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (
        (context_layer, attention_probs) if output_attentions else (context_layer,)
    )
    return outputs


def recording_forward_hook(module, input, output):
    global layer_info
    layer_info[id(module)] = (output.shape[2], output.shape[3])
    return output


def apply_quatermap(model):
    layer = 0
    efficientnet_should_apply = False
    convnext_should_apply = False
    for name, module in model.named_modules():
        if isinstance(model, EfficientNet):
            if (
                isinstance(module, ConvBnAct)
                or isinstance(module, InvertedResidual)
                or isinstance(module, EdgeResidual)
            ):
                layer += 1
                if layer > 2 and layer % 3 == 0:
                    efficientnet_should_apply = True
                else:
                    efficientnet_should_apply = False
            if isinstance(module, Conv2d) and efficientnet_should_apply:
                print(f"Registering hook for {model.__class__.__name__} layer {name}")
                module.register_forward_pre_hook(quatermap_conv_forward_pre_hook)
                module.register_forward_hook(quatermap_conv_forward_hook)
        elif isinstance(model, ConvNextV2ForImageClassification):
            if isinstance(module, ConvNextV2Layer):
                layer += 1
                if layer > 2 and layer % 3 == 0:
                    convnext_should_apply = True
                else:
                    convnext_should_apply = False

            if isinstance(module, Conv2d) and convnext_should_apply:
                print(f"Registering hook for {model.__class__.__name__} layer {name}")
                module.register_forward_pre_hook(quatermap_conv_forward_pre_hook)
                module.register_forward_hook(quatermap_conv_forward_hook)
        elif isinstance(model, ViTForImageClassification) and isinstance(
            module, ViTSelfAttention
        ):
            layer += 1
            if layer > 2 and layer % 3 == 0:
                print(f"Registering hook for {model.__class__.__name__} layer {name}")
                module.register_forward_pre_hook(quatermap_vit_forward_pre_hook)
                module.register_forward_hook(quatermap_vit_forward_hook)
        elif isinstance(model, SwinForImageClassification) and isinstance(
            module, SwinSelfAttention
        ):
            layer += 1
            if layer > 2 and layer % 3 == 0:
                print(f"Registering hook for {model.__class__.__name__} layer {name}")
                module.forward = types.MethodType(patch_swin_attention_forward, module)
                module.register_forward_pre_hook(quatermap_swin_forward_pre_hook)
                module.register_forward_hook(quatermap_swin_forward_hook)
    # sys.exit(0)
    # for m in model.modules():


def apply_recording_hook(model):
    hooks = []
    for name, module in model.named_modules():
        # ConvNextV2
        if (
            isinstance(model, EfficientNet)
            and (
                isinstance(module, Conv2d)
                # or isinstance(module, InvertedResidual)
                # or isinstance(module, EdgeResidual)
            )
        ) or (
            isinstance(model, ConvNextV2ForImageClassification)
            and isinstance(module, Conv2d)
        ):
            hooks.append(module.register_forward_hook(recording_forward_hook))
    return hooks


def remove_recording_hook(hooks):
    for hook in hooks:
        hook.remove()


def main(args, config):
    # load dataset
    dataset_val, _ = build_dataset(is_train=False, config=config)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    is_timm_model = False
    # load model
    if args.model_name in TIMM_MODELS:
        model = timm.create_model(args.model_name, pretrained=True)
        is_timm_model = True
    else:
        model = AutoModelForImageClassification.from_pretrained(args.model_name)
    model.eval()
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if os.environ.get("QUATERMAP", False):
        hooks = apply_recording_hook(model)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model(dummy_input)
        # remove hooks
        remove_recording_hook(hooks)
        apply_quatermap(model)

    validate(config, data_loader_val, model, is_timm_model)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on different models")
    parser.add_argument(
        "--cfg", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size for single GPU"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/dataset/ImageNet_ILSVRC2012",
        help="path to dataset",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument(
        "--tag",
        default=time.strftime("%Y%m%d%H%M%S", time.localtime()),
        help="tag of experiment",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    parser.add_argument(
        "--fused_layernorm", action="store_true", help="Use fused layernorm."
    )
    parser.add_argument(
        "--optim", type=str, help="overwrite optimizer if provided, can be adamw/sgd."
    )

    # Weight analysis
    parser.add_argument(
        "--plot_weight_distribution",
        action="store_true",
        help="plot weight distribution",
    )
    parser.add_argument(
        "--bitwidth", type=int, default=32, help="bitwidth for weight distribution plot"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="histogram",
        help="plot type for weight distribution plot",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/convnextv2-tiny-1k-224",
        help="Model name to use",
    )
    args, unparsed = parser.parse_known_args()
    args.batch_size = int(os.environ.get("BATCH_SIZE", args.batch_size))
    args.img_size = int(os.environ.get("IMG_SIZE", 224))
    config = get_config(args)

    return args, config


if __name__ == "__main__":
    args, config = parse_args()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{args.model_name}")
    main(args, config)
