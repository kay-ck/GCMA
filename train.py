import argparse
import cv2
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from generators import custom_gan


sigma = {"res152": 10, "dense201": 8, "squeeze1_1": 12, "shufflev2_x1.0": 8}


internal_feature_in = []
internal_feature_out = []
def forward_hook(module, input, output):
    if len(internal_feature_in) == 0:
        internal_feature_in.append(input)
        internal_feature_out.append(output)
    else:
        internal_feature_in[0] = input
        internal_feature_out[0] = output
    return None


def get_args_parser():
    parser = argparse.ArgumentParser(description="Cross Modal Transferability", add_help=False)
    parser.add_argument("--train_dir", default="imagenet_2012/train", help="imagenet")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of trainig samples/batch")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Initial learning rate for adam")
    parser.add_argument("--eps", type=int, default=10, help="Perturbation Budget")
    parser.add_argument("--model_type", type=str, default="res152", help="Model against GAN is trained: res152, dense201, squeeze1_1, shufflev2_x1.0")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank for distributed training")
    return parser


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def warp(x, flo, padding_mode="border"):
    B, _, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flo
    
    # Scale grid to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1).cuda()
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode="nearest")
    return output


def random_optical_flow(t, model_type):
    B, _, H, W = t.size()
    flows = []
    for b in range(B):
        flow = np.random.normal(0, sigma[model_type], size = [H // 100, W // 100, 2])
        flow = cv2.resize(flow, (H, W))
        flow[:, :, 0] += random.randint(-10, 10)
        flow[:, :, 1] += random.randint(-10, 10)
        flow = cv2.blur(flow, (100, 100))
        flows.append(torch.from_numpy(flow.transpose((2, 0, 1))).float())
    return torch.stack(flows, 0)


def main(args):
    # Normalize (0-1)
    eps = args.eps / 255
    warmup_eps = 4 / 255

    setup_seed(0)

    # GPU
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", args.local_rank)

    ####################
    # Model and Generator
    ####################
    if args.model_type == "res152":
        model = models.resnet152(pretrained=True).to(device)
        model.layer3.register_forward_hook(hook=forward_hook)
    elif args.model_type == "dense201":
        model = models.densenet201(pretrained=True).to(device)
        for name, module in model.features.named_modules(): 
            if name == "transition3":
                module.register_forward_hook(hook=forward_hook)
    elif args.model_type == "squeeze1_1":
        model = models.squeezenet1_1(pretrained=True).to(device)
        model.features[10].register_forward_hook(hook=forward_hook)
    elif args.model_type == "shufflev2_x1.0":
        model = models.shufflenet_v2_x1_0(pretrained=True).to(device)
        model.stage3.register_forward_hook(hook=forward_hook)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    netG = custom_gan(args)
    netG = nn.SyncBatchNorm.convert_sync_batchnorm(netG).to(device)
    netG = DistributedDataParallel(netG, device_ids=[args.local_rank], output_device=args.local_rank)

    # Optimizer
    optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Input dimensions
    scale_size = 256
    img_size = 224

    # Data
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    def normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
        return t

    train_dir = args.train_dir
    train_set = datasets.ImageFolder(train_dir, data_transform)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=train_sampler, pin_memory=True)
    train_size = len(train_set)
    print("Training data size:", train_size)

    # Loss
    cos_sim = nn.CosineSimilarity()

    # Training
    print("Model: {} \t Distribution: {} \t Saving instances: {}".format(args.model_type, args.train_dir, args.epochs))
    netG.train()
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0
        for step, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            B, C, H, W = img.size()

            flow = random_optical_flow(img, args.model_type)
            next_img = warp(img, flow)
            imgs = torch.cat((img, next_img), 0)

            perts = netG(imgs) * 2 - 1

            # Projection
            if step < 250:
                advs = perts * warmup_eps + imgs
            else:
                advs = perts * eps + imgs
            advs = torch.clamp(advs, 0.0, 1.0)

            optimG.zero_grad()

            # Gradient accent (Untargeted Attack)
            adv, next_adv = torch.split(advs, B, 0)
            adv_next = warp(adv, flow)
            cat_in = torch.cat([next_img, next_adv, adv_next], 0)
            cat_out = model(normalize(cat_in.clone()))
            next_img_fea = internal_feature_out[0][0 : B, :, :, :]
            next_adv_fea = internal_feature_out[0][B : 2 * B, :, :, :]
            adv_next_fea = internal_feature_out[0][2 * B : 3 * B, :, :, :]
            print(cos_sim(next_img_fea.view(B, -1), next_adv_fea.view(B, -1)).mean(), cos_sim(next_adv_fea.view(B, -1), adv_next_fea.view(B, -1)).mean())

            loss = cos_sim(next_img_fea.view(B, -1), adv_next_fea.view(B, -1)).mean() - cos_sim(next_adv_fea.view(B, -1), adv_next_fea.view(B, -1)).mean()

            loss.backward()
            optimG.step()

            if step != 0 and step % 10 == 0:
                print("Epoch: {0} \t Batch: {1} \t loss: {2:.5f}".format(epoch, step, running_loss / 10))
                running_loss = 0
            running_loss += loss.item()
        if args.local_rank == 0:
            torch.save(netG.module.state_dict(), "netG_{}_gcma.pth".format(args.model_type))
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)