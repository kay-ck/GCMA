import argparse
import random
import torch
import numpy as np
import torchvision.utils as vutils
from gluoncv.torch.engine.config import get_cfg_defaults
from torch.nn.parallel import DistributedDataParallel

from dataset_kinetics import custom_dataset
from model import custom_model, CONFIG_PATH
from generators import custom_gan


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args_parser():
    parser = argparse.ArgumentParser(description="Cross Modal Transferability", add_help=False)
    parser.add_argument("--batch_size", type=int, default=13, help="Batch Size")
    parser.add_argument("--eps", type=int, default=10, help="Perturbation Budget")
    parser.add_argument("--model_type", type=str, default= "res152",  help ="Model against GAN is trained: res152, dense201, squeeze1_1, shufflev2_x1.0")
    parser.add_argument("--model_t",type=str, default= "nl5_50",  help ="Model under attack: nl5_50/101, slowfast_50/101, tpn_50/101" )
    parser.add_argument("--local_rank", default=0, type=int, help="node rank for distributed training")
    return parser


def main(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(CONFIG_PATH[args.model_t])

    # Normalize (0-1)
    eps = args.eps / 255

    setup_seed(0)

    # GPU
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", args.local_rank)

    netG = custom_gan(args).to(device)
    netG.load_state_dict(torch.load("checkpoints/netG_{}_gcma.pth".format(args.model_type), map_location="cpu"))
    netG = DistributedDataParallel(netG, device_ids=[args.local_rank], output_device=args.local_rank)
    netG.eval()

    model_t = custom_model(cfg).to(device)
    model_t = DistributedDataParallel(model_t, device_ids=[args.local_rank], output_device=args.local_rank)
    model_t.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    def normalize(t):
        t[:, 0, :, :, :] = (t[:, 0, :, :, :] - mean[0]) / std[0]
        t[:, 1, :, :, :] = (t[:, 1, :, :, :] - mean[1]) / std[1]
        t[:, 2, :, :, :] = (t[:, 2, :, :, :] - mean[2]) / std[2]
        return t

    inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    inv_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    def inv_normalize(t):
        t[:, 0, :, :, :] = (t[:, 0, :, :, :] - inv_mean[0]) / inv_std[0]
        t[:, 1, :, :, :] = (t[:, 1, :, :, :] - inv_mean[1]) / inv_std[1]
        t[:, 2, :, :, :] = (t[:, 2, :, :, :] - inv_mean[2]) / inv_std[2]
        return t

    val_loader, val_size = custom_dataset(args, cfg)
    print("Testing data size:", val_size)

    # Evaluation
    correct = torch.zeros(1).to(device)
    incorrect = torch.zeros(1).to(device)
    total = torch.zeros(1).to(device)
    for i, (vid, label, _) in enumerate(val_loader):
        with torch.no_grad():
            vid, label = inv_normalize(vid).to(device), label.to(device)
            B, C, T, H, W = vid.size()

            vid_out = model_t(normalize(vid.clone().detach())).argmax(dim=-1)
            correct += torch.sum(vid_out == label).item()
            total += B

            # Untargeted Adversary
            pert = netG(vid.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)) * 2 - 1
            pert = pert.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

            # Projection
            adv = pert * eps + vid
            adv = torch.clamp(adv, 0.0, 1.0)

            adv_out = model_t(normalize(adv.clone().detach())).argmax(dim=-1)
            incorrect += torch.sum((adv_out != label) & (vid_out == label)).item()

            if args.local_rank == 0:
                for t in range(T):
                    vutils.save_image(vutils.make_grid(adv[0, :, t, :, :], normalize=True, scale_each=True), "./adv/adv_{}.png".format(t))
                    vutils.save_image(vutils.make_grid(vid[0, :, t, :, :], normalize=True, scale_each=True), "./ori/ori_{}.png".format(t))
            print("At Batch:{}\t l_inf:{}".format(i, (vid - adv).max() * 255))

    torch.distributed.all_reduce(correct)
    torch.distributed.all_reduce(incorrect)
    torch.distributed.all_reduce(total)
    print("Clean:{0:}\t Adversarial:{1:}\t Total:{2:}\t ACC:{3:}\t ASR:{4:}".format(correct, incorrect, total, correct/total, incorrect / correct))
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)
