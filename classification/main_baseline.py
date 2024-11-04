# CNN: ConvNext, EfficientNet
# ViT: DeiT, SwinTransformer

import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import os
import time
from timm.utils import accuracy, AverageMeter
from utils.utils import reduce_tensor
from utils.logger import create_logger
import timm

from data.build import build_dataset
from config import get_config


TIMM_MODELS = [
    "tf_efficientnetv2_l.in1k",
    "tf_efficientnetv2_b0.in1k",
    "tf_efficientnetv2_m.in1k",
    "tf_efficientnetv2_s.in1k",
    "tf_efficientnetv2_s.in21k",
    "tf_efficientnetv2_l.in21k"
]


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
