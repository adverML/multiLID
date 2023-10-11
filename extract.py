"""
The spatial attack is a very special attack because it tries to find adversarial
perturbations using a set of translations and rotations rather then in an Lp ball.
It therefore has a slightly different interface.
https://github.com/bethgelab/foolbox/blob/master/examples/spatial_attack_pytorch_resnet18.py
"""
import os
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import foolbox as fb
import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset, TensorDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import copy
import numpy as np
import argparse

import cfg

from defenses.multiLID import (multiLID, LID)

from misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    create_dir
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--att",  default="fgsm", choices=['fgsm', 'bim', 'pgd', 'df', 'cw'], help="")
    parser.add_argument("--defense",    default="multilid", choices=['multilid', 'lid'], help="")
    parser.add_argument("--dataset",  default="imagenet", choices=['imagenet', 'cifar10', 'cifar100'], help="")
    parser.add_argument("--model",  default="wrn50-2", choices=['wrn50-2', 'wrn28-10', 'vgg16'], help="")
    parser.add_argument("--load_nor",   default="normalos_8255.pt", help="")
    parser.add_argument("--load_adv",   default="adverlos_8255.pt", help="")
    parser.add_argument("--save_nor",   default="normalos_8255.pt", help="")
    parser.add_argument("--save_adv",   default="adverlos_8255.pt", help="")
    parser.add_argument("--eps",        default="8/255", help="")
    parser.add_argument("--k",          default=30, type=int, help="")
    parser.add_argument("--normalize",  default="imagenet", help="")
    parser.add_argument("--nr_samples", default=2000, type=int, help="")

    parser.add_argument('--save_json', default="", help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_json', default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    args = args_handling(args, parser, "configs/extract")
    args.eps = convert_to_float(args.eps)
    print_args(args)


    nor = torch.load(os.path.join(cfg.workspace, 'data/gen', args.dataset, args.model, args.att, args.load_nor))[:args.nr_samples]
    adv = torch.load(os.path.join(cfg.workspace, 'data/gen', args.dataset, args.model, args.att, args.load_adv))[:args.nr_samples]

    # instantiate a model (could also be a TensorFlow or JAX model)
    # model = models.resnet18(pretrained=True).eval()
    model =  models.wide_resnet50_2(weights='DEFAULT').eval()

    preprocessing = {}
    if args.normalize == "imagenet":
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

        mean = torch.from_numpy(np.asarray(preprocessing['mean']))
        std  = torch.from_numpy(np.asarray(preprocessing['std']))

    dataset = torch.utils.data.TensorDataset(nor, adv)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )
    
    normalos_nor = []
    adverlos_nor = []
    
    # train_nodes, eval_nodes = get_graph_node_names(model)
    fe = create_feature_extractor(model, return_nodes=args.layers)

    for nor, adv in data_loader:
        
        if not args.normalize == None:
            nor[:,0] = (nor[:,0] - mean[0]) / std[0]
            nor[:,1] = (nor[:,1] - mean[1]) / std[1]
            nor[:,2] = (nor[:,2] - mean[2]) / std[2]

            adv[:,0] = (adv[:,0] - mean[0]) / std[0]
            adv[:,1] = (adv[:,1] - mean[1]) / std[1]
            adv[:,2] = (adv[:,2] - mean[2]) / std[2]

        normalos_nor.append(nor)
        adverlos_nor.append(adv)

    nor = torch.vstack(normalos_nor)
    adv = torch.vstack(adverlos_nor)

    if args.defense == 'lid':
        lid, lid_adv = LID(     nor, adv, feature_extractor=fe, lid_dim=len(args.layers), k=args.k, batch_size=100, device='cpu')   
    elif args.defense == 'multilid':
        lid, lid_adv = multiLID(nor, adv, feature_extractor=fe, lid_dim=len(args.layers), k=args.k, batch_size=100, device='cpu')

    normalos_nor = torch.from_numpy(lid)
    adverlos_nor = torch.from_numpy(lid_adv)
    
    base_pth = os.path.join(cfg.workspace, 'data/extract', args.dataset, args.model, args.defense, args.att, 'k'+str(args.k))
    create_dir(base_pth)
    torch.save(normalos_nor, os.path.join(base_pth, args.save_nor))
    torch.save(adverlos_nor, os.path.join(base_pth, args.save_adv))


if __name__ == "__main__":
    main()