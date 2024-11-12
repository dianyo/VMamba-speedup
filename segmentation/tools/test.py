# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import model
from plot_utils import save_module_id_mapping
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from mmseg.models.backbones.swin import WindowMSA

layer_info = {}
def quatermap_swin_forward_pre_hook(module, input, kwargs):
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
    new_input = new_input.reshape(B, -1, C)
    new_args = [new_input]
    new_kwargs = {}
    if kwargs['mask'] is not None:
        new_attention_mask = kwargs['mask']
        new_attention_mask = new_attention_mask.reshape(
            -1, feature_map_dim, feature_map_dim, feature_map_dim, feature_map_dim
        )
        new_attention_mask = new_attention_mask[:, ::2, ::2, ::2, ::2]
        new_attention_mask = new_attention_mask.reshape(
            -1,
            new_attention_mask.shape[1] * new_attention_mask.shape[2],
            new_attention_mask.shape[3] * new_attention_mask.shape[4],
        )
        new_kwargs['mask'] = new_attention_mask
    return (new_args, new_kwargs)

def quatermap_swin_forward_hook(module, input, output):
    global layer_info
    H, W = layer_info[id(module)]
    dim = int(math.sqrt(output.shape[1]))
    feature_map = (
        output.reshape(-1, dim, dim, output.shape[2]).permute(0, 3, 1, 2)
    )
    feature_map = F.interpolate(feature_map, size=(H, W), mode="nearest")
    new_output = feature_map.permute(0, 2, 3, 1).reshape(-1, H * W, output.shape[2])
    return new_output


def patch_swin_attention_forward(
    self,
    x, mask=None
):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                C // self.num_heads).permute(2, 0, 3, 1, 4)
    # make torchscript happy (cannot use tensor as tuple)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH

    if (
        self.window_size[0] != attn.shape[-2]
        or self.window_size[1] != attn.shape[-1]
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

    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B // nW, nW, self.num_heads, N,
                            N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)

    attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def apply_quatermap(model):
    layer = 0
    for name, module in model.named_modules():
        if isinstance(
            module, WindowMSA
        ):
            layer += 1
            if layer > 2 and layer % 3 == 0:
                print(f"Registering hook for {model.__class__.__name__} layer {name}")
                module.forward = types.MethodType(patch_swin_attention_forward, module)
                module.register_forward_pre_hook(quatermap_swin_forward_pre_hook, with_kwargs=True)
                module.register_forward_hook(quatermap_swin_forward_hook)

# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    save_module_id_mapping(runner.model)
    if os.getenv("QUATERMAP", False):
        apply_quatermap(runner.model)
    
    # start testing
    runner.test()


if __name__ == '__main__':
    main()
