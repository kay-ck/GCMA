import torch
from gluoncv.torch.data import VideoClsDataset

from transforms_ucf101 import *


def custom_dataset(args, cfg):
    cfg.CONFIG.VAL.BATCH_SIZE = args.batch_size
    cfg.CONFIG.DATA.VAL_ANNO_PATH = "kinetics400/anno/val.csv"
    # cfg.CONFIG.DATA.VAL_ANNO_PATH = "kinetics400/anno/val400.csv"
    cfg.CONFIG.DATA.VAL_DATA_PATH = "kinetics400/videos/val/"
    val_loader = build_dataloader_val(cfg)
    return val_loader, len(val_loader)


def build_dataloader_val(cfg):
    """Build dataloader for testing"""
    val_dataset = VideoClsDataset(anno_path=cfg.CONFIG.DATA.VAL_ANNO_PATH,
                                   data_path=cfg.CONFIG.DATA.VAL_DATA_PATH,
                                   mode="validation",
                                   clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                   frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                   num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                                   num_crop=cfg.CONFIG.DATA.NUM_CROP,
                                   keep_aspect_ratio=cfg.CONFIG.DATA.KEEP_ASPECT_RATIO,
                                   crop_size=cfg.CONFIG.DATA.CROP_SIZE,
                                   short_side_size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE,
                                   new_height=cfg.CONFIG.DATA.NEW_HEIGHT,
                                   new_width=cfg.CONFIG.DATA.NEW_WIDTH)

    if cfg.DDP_CONFIG.DISTRIBUTED:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
        num_workers=4, sampler=val_sampler, pin_memory=True)

    return val_loader