import os
from gluoncv.torch.model_zoo import get_model


CONFIG_ROOT_KINETICS = "kinetics400/config"
CKPT_ROOT_UCF = "ucf101/ckpt"
CONFIG_PATH = {
    "nl5_50": os.path.join(CONFIG_ROOT_KINETICS, "i3d_nl5_resnet50_v1_kinetics400.yaml"),
    "slowfast_50": os.path.join(CONFIG_ROOT_KINETICS, "slowfast_8x8_resnet50_kinetics400.yaml"),
    "tpn_50": os.path.join(CONFIG_ROOT_KINETICS, "tpn_resnet50_f32s2_kinetics400.yaml"),
    "nl5_101": os.path.join(CONFIG_ROOT_KINETICS, "i3d_nl5_resnet101_v1_kinetics400.yaml"),
    "slowfast_101": os.path.join(CONFIG_ROOT_KINETICS, "slowfast_8x8_resnet101_kinetics400.yaml"),
    "tpn_101": os.path.join(CONFIG_ROOT_KINETICS, "tpn_resnet101_f32s2_kinetics400.yaml"),
    }
CKPT_PATH = {
    "nl5_50": os.path.join(CKPT_ROOT_UCF, "i3d_resnet50.pth"),
    "slowfast_50": os.path.join(CKPT_ROOT_UCF, "slowfast_resnet50.pth"),
    "tpn_50": os.path.join(CKPT_ROOT_UCF, "tpn_resnet50.pth"),
    "nl5_101": os.path.join(CKPT_ROOT_UCF, "i3d_resnet101.pth"),
    "slowfast_101": os.path.join(CKPT_ROOT_UCF, "slowfast_resnet101.pth"),
    "tpn_101": os.path.join(CKPT_ROOT_UCF, "tpn_resnet101.pth")
}
    

def custom_model(cfg):
    cfg.CONFIG.MODEL.PRETRAINED = True
    model = get_model(cfg)
    return model