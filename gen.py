"""
The spatial attack is a very special attack because it tries to find adversarial
perturbations using a set of translations and rotations rather then in an Lp ball.
It therefore has a slightly different interface.
https://github.com/bethgelab/foolbox/blob/master/examples/spatial_attack_pytorch_resnet18.py
"""
import os
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import foolbox as fb
import numpy as np
import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import copy
import argparse
from tqdm import tqdm
import cfg


from misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    create_dir,
    str2bool,
    get_preprocessing
)

DEEPFOOL = ['fgsm', 'bim', 'pgd', 'df', 'cw']
AUTOATTACK = ['aa', 'apgd-ce']

def pred(fmodel, inputs, labels):
    (inputs_, labels_), restore_types = ep.astensors_(inputs, labels)
    del inputs, labels

    predictions = fmodel(inputs_).argmax(axis=-1)
    correct_predicted = (predictions == labels_)

    return inputs_[correct_predicted], labels_[correct_predicted], restore_types


def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    # model = models.resnet18(pretrained=True).eval()
    parser = argparse.ArgumentParser("gen")
    parser.add_argument("--att",  default="fgsm", choices=['fgsm', 'bim', 'pgd', 'df', 'cw'], help="")
    parser.add_argument("--dataset",  default="imagenet", choices=['imagenet', 'cifar10', 'cifar100'], help="")
    parser.add_argument("--model",  default="wrn50-2", choices=['wrn50-2', 'wrn28-10', 'vgg16'], help="")
    parser.add_argument("--save_nor",  default="normalos.pt", help="")
    parser.add_argument("--save_adv",  default="adverlos.pt", help="")
    parser.add_argument("--eps",  default=8/255, help="")
    parser.add_argument("--norm",  default="Linf", choices=['Linf', 'L2', 'L1'], help="")
    parser.add_argument("--version",  default="standard", help="")
    parser.add_argument("--bs",   default=16, help="")
    parser.add_argument("--max_counter",  default=2000, help="")
    parser.add_argument("--debug",  default=True, type=str2bool, help="")

    parser.add_argument('--save_json', default="", help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_json', default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    args = args_handling(args, parser, "./configs/gen")
    args.eps = convert_to_float(args.eps)
    print_args(args)

    base_pth = os.path.join(cfg.workspace, 'data/gen', args.dataset, args.model, args.att)
    base_log_pth = os.path.join(base_pth, 'logs')
    create_dir(base_pth)
    create_dir(base_log_pth)

    if args.dataset == 'imagenet':
        preprocessing = get_preprocessing(args)
        if args.model == 'wrn50-2':
            import torchvision.models as models
            model = models.wide_resnet50_2(weights='DEFAULT')
        elif args.model == 'vgg16':
            import torchvision.models as models
            model = models.vgg16(weights='DEFAULT')

    elif args.dataset == 'cifar10':
        preprocessing = get_preprocessing(args)
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
        preprocessing = get_preprocessing(args)
        if args.model == 'wrn28-10':
            from models.wide_residual_distr import WideResNet as WideResNet96
            from models.wide_residual_distr import WideBasic as WideBasic96
            model = WideResNet96(num_classes=100, block=WideBasic96, depth=28, widen_factor=10, dropout=0.3, preprocessing=[])
            ckpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar100/wide-resnet-28x10_2022-04-17_14:13:21.pt')
            model.load_state_dict(ckpt['model_state_dict'],  strict=True)
        elif args.model == 'vgg16':
            from models.vgg import vgg16_bn
            from models.helper import create_new_state_dict
            model = VGG('VGG16', preprocessing=[])
            ckpt = torch.load('/home/lorenzp/adversialml/src/checkpoint/vgg16/vgg_cif100.pth')
            new_state_dict = create_new_state_dict(ckpt)
            model.load_state_dict(new_state_dict)

    model = model.eval()
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    if args.dataset == 'imagenet':
        transform_list = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        transform = transforms.Compose(transform_list)
        
        dataset_dir_path = os.path.join(cfg.DATASET_BASE_PTH,"ImageNet/val")
        data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(dataset_dir_path, transform), 
            batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True
        )

    elif args.dataset == 'cifar10':
        transform_list = [transforms.ToTensor()] 
        transform = transforms.Compose(transform_list)
        
        dataset_dir_path = os.path.join(cfg.DATASET_BASE_PTH,"cifar10") 

        dataset = datasets.CIFAR10(root=dataset_dir_path, train=False, transform=transform, download=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=8)

    elif args.dataset == 'cifar100':
        transform_list = [transforms.ToTensor()] 
        transform = transforms.Compose(transform_list)
        
        dataset_dir_path =  os.path.join(cfg.DATASET_BASE_PTH,"cifar100") 

        dataset = datasets.CIFAR100(root=dataset_dir_path, train=False, transform=transform, download=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=8)

    if args.att == 'fgsm':
        attack = fa.FGSM()
    elif args.att == 'bim':
        attack = fa.LinfBasicIterativeAttack()
    elif args.att == 'pgd':
        attack = fa.PGD()
    elif args.att == 'df':
        attack = fa.L2DeepFoolAttack()
        args.eps = None
    elif args.att == 'cw':
        attack = fa.L2CarliniWagnerAttack(steps=1000)
        args.eps = None
    elif args.att in AUTOATTACK:
        from submodules.autoattack.autoattack import AutoAttack as AutoAttack_mod
        adversary = AutoAttack_mod(fmodel, norm=args.norm, eps=args.eps, log_path=os.path.join(base_log_pth, args.load_json.split('/')[-1]), version=args.version)
        if args.version == 'individual':
            breakpoint()
            adversary.attacks_to_run = [ args.att ]

    # report the success rate of the attack (percentage of samples that could
    # be adversarially perturbed) and the robust accuracy (the remaining accuracy
    # of the model when it is attacked)
    (images, labels), restore_type = ep.astensors_(*samples(fmodel, dataset=args.dataset, batchsize=args.bs))
    clean_acc = accuracy(fmodel, images, labels) * 100
    print(f"clean accuracy:  {clean_acc:.1f} %")
    print(images.shape)
    if args.att in DEEPFOOL:
        criterion = fb.criteria.Misclassification(labels)
        xp_, x_, success = attack(fmodel, images, criterion=criterion, epsilons=args.eps)
        suc = success.float32().mean().item() * 100
        print(
            f"attack success:  {suc:.1f} %"
            ""
        )
        print(
            f"robust accuracy: {100 - suc:.1f} %"
            ""
        )

    counter = 0
    total_success = []
    normalos = []
    adverlos = []
    
    with torch.no_grad():
        for it, (img, lab) in tqdm(enumerate(data_loader), total=round((args.max_counter*2)/args.bs)):
    
            img_cu, lab_cu, restore_type = pred(fmodel, img.cuda(non_blocking=True), lab.cuda(non_blocking=True))

            clean_acc = accuracy(fmodel, img_cu, lab_cu) * 100
            if args.debug:
                print("clean_acc: ", clean_acc)

            if args.att in DEEPFOOL:
                lab_cu = fb.criteria.Misclassification(lab_cu)
                xp_, x_, success = attack(fmodel, img_cu, lab_cu, epsilons=args.eps)
                suc = success.float32().mean().item() * 100
                
                nor = restore_type(img_cu[torch.where(restore_type(success).int().cpu().squeeze() == 1)[0]]).cpu()
                adv = restore_type(x_[torch.where(restore_type(success).int().cpu().squeeze() == 1)[0]]).cpu()

            elif args.att in AUTOATTACK:
                img_cu = torch.squeeze(restore_type(img_cu))
                lab_cu = torch.squeeze(restore_type(lab_cu))

                if args.bs == 1:
                    img_cu = torch.unsqueeze(img_cu, 0)
                    lab_cu = torch.unsqueeze(lab_cu, 0)

                if args.version == 'standard':
                    x_, y_, max_nr, success = adversary.run_standard_evaluation(img_cu, lab_cu, bs=args.bs, return_labels=True)
                else: 
                    adv_complete = adversary.run_standard_evaluation_individual(img_cu, lab_cu, bs=args.bs, return_labels=True)
                    x_, y_, max_nr, success = adv_complete[ args.att ]
                suc = success.float().mean().item() * 100

                nor = img_cu[torch.where(success.int().cpu().squeeze() == 1)[0]].cpu()
                adv = x_[torch.where(success.int().cpu().squeeze() == 1)[0]].cpu()

            normalos.append(nor)
            adverlos.append(adv)

            total_success.append(suc)
            counter = counter + nor.shape[0]
            
            if counter >= args.max_counter:
                break
    
    normalos = torch.vstack(normalos)
    adverlos = torch.vstack(adverlos)

    print("counter", counter, ", ", normalos.shape, np.mean(total_success))

    torch.save(normalos, os.path.join(base_pth, args.save_nor))
    torch.save(adverlos, os.path.join(base_pth, args.save_adv))

 
if __name__ == "__main__":
    main()