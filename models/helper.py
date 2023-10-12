import torch
from collections import OrderedDict

from misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    create_dir,
    str2bool,
    get_preprocessing
)

def create_new_state_dict(checkpoint, keyword='net'):

    new_state_dict = OrderedDict()
    for k, v in checkpoint[keyword].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def get_model(args):
    preprocessing = get_preprocessing(args)
    if args.dataset == 'imagenet':
        if args.model == 'wrn50-2':
            import torchvision.models as models
            model = models.wide_resnet50_2(weights='DEFAULT')
        elif args.model == 'vgg16':
            import torchvision.models as models
            model = models.vgg16(weights='DEFAULT')

    elif args.dataset == 'cifar10':
        if args.model == 'wrn28-10':
            from models.wide_residual_distr import WideResNet as WideResNet96
            from models.wide_residual_distr import WideBasic as WideBasic96
            model = WideResNet96(num_classes=10, block=WideBasic96, depth=28, widen_factor=10, dropout=0.3, preprocessing=[])
            ckpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/wide-resnet-28x10_2022-04-17_14:12:50.pt')
            model.load_state_dict(ckpt['model_state_dict'],  strict=True)
        elif args.model == 'vgg16':
            from models.vgg_cif10 import VGG
            from models.helper import create_new_state_dict
            model = VGG('VGG16', preprocessing=[])
            ckpt = torch.load('/home/lorenzp/adversialml/src/checkpoint/vgg16/vgg_cif10.pth')
            new_state_dict = create_new_state_dict(ckpt)
            model.load_state_dict(new_state_dict)

    elif args.dataset == 'cifar100':
        if args.model == 'wrn28-10':
            from models.wide_residual_distr import WideResNet as WideResNet96
            from models.wide_residual_distr import WideBasic as WideBasic96
            model = WideResNet96(num_classes=100, block=WideBasic96, depth=28, widen_factor=10, dropout=0.3, preprocessing=[])
            ckpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar100/wide-resnet-28x10_2022-04-17_14:13:21.pt')
            model.load_state_dict(ckpt['model_state_dict'],  strict=True)
        elif args.model == 'vgg16':
            from models.vgg import vgg16_bn
            model = vgg16_bn(num_class=100, preprocessing={})
            ckpt = torch.load('/home/lorenzp/adversialml/src/checkpoint/vgg16/vgg_cif100.pth')
            model.load_state_dict(ckpt)

    
    return model, preprocessing



