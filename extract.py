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

from models.helper import get_model
from defenses.multiLID import (multiLID, LID)

from misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    str2bool,
    create_dir,
    create_log_file,
    save_log
)

def main() -> None:
    parser = argparse.ArgumentParser("extract")
    parser.add_argument("--run_nr",     default="run_1", help="")
    parser.add_argument("--att",        default="fgsm", choices=['fgsm', 'bim', 'pgd', 'df', 'cw'], help="")
    parser.add_argument("--defense",    default="multilid", choices=['multilid', 'lid'], help="")
    parser.add_argument("--dataset",    default="imagenet", choices=['imagenet', 'cifar10', 'cifar100'], help="")
    parser.add_argument("--model",      default="wrn50-2", choices=['wrn50-2', 'wrn28-10', 'vgg16'], help="")
    parser.add_argument("--load_nor",   default="normalos_8255.pt", help="")
    parser.add_argument("--load_adv",   default="adverlos_8255.pt", help="")
    parser.add_argument("--save_nor",   default="normalos_8255.pt", help="")
    parser.add_argument("--save_adv",   default="adverlos_8255.pt", help="")
    parser.add_argument("--eps",        default="8/255", help="")
    parser.add_argument("--k",          default=30, type=int, help="")
    parser.add_argument("--normalize",  default=True, type=str2bool, help="")
    parser.add_argument("--nr_samples", default=2000, type=int, help="")

    parser.add_argument('--save_json', default="", help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_json', default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    args = args_handling(args, parser, "configs/extract")
    args.eps = convert_to_float(args.eps)
    print_args(args)

    print("Create paths!")
    base_pth = os.path.join(cfg.workspace, 'data/gen', args.run_nr, args.dataset, args.model, args.att)
    create_dir(base_pth)
    log_pth = os.path.join(base_pth, 'logs')
    log = create_log_file(args, log_pth)

    print("Load data")
    nor = torch.load(os.path.join(base_pth, args.load_nor))[:args.nr_samples]
    adv = torch.load(os.path.join(base_pth, args.load_adv))[:args.nr_samples]


    print("Load model and dataloader")
    model, preprocessing = get_model(args)
    model = model.eval()

    mean = torch.from_numpy(np.asarray(preprocessing['mean']))
    std  = torch.from_numpy(np.asarray(preprocessing['std']))

    dataset = torch.utils.data.TensorDataset(nor, adv)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )
    
    normalos_nor = []
    adverlos_nor = []
    
    fe = None
    activation = None
    if args.dataset == 'imagenet':
        # train_nodes, eval_nodes = get_graph_node_names(model)
        fe = create_feature_extractor(model, return_nodes=args.layers)
    else:
        from defenses.helper_extract import registrate_whitebox_features
        get_layer_feature_maps, activation = registrate_whitebox_features(args, model)
        def feature_extractor(args, model, batch, activation):
            feat_img = model(batch)
            if args.model == 'wrn28-10':
                X_act = get_layer_feature_maps(activation, args.layers)
            else:
                X_act = get_layer_feature_maps(batch, args.layers)
            return X_act
        fe = feature_extractor

    print("Normalize images!")
    for nor, adv in data_loader:
        if args.normalize:
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

    print("Calculate Features")
    if args.defense == 'lid':
        lid, lid_adv = LID(     args, nor, adv, feature_extractor=fe, model=model, activation=activation, lid_dim=len(args.layers), k=args.k, batch_size=100, device='cpu')   
    elif args.defense == 'multilid':
        lid, lid_adv = multiLID(args, nor, adv, feature_extractor=fe, model=model, activation=activation, lid_dim=len(args.layers), k=args.k, batch_size=100, device='cpu')

    normalos_nor = torch.from_numpy(lid)
    adverlos_nor = torch.from_numpy(lid_adv)
    
    base_pth = os.path.join(cfg.workspace, 'data/extract', args.run_nr, args.dataset, args.model, args.defense, args.att, 'k'+str(args.k))
    create_dir(base_pth)
    torch.save(normalos_nor, os.path.join(base_pth, args.save_nor))
    torch.save(adverlos_nor, os.path.join(base_pth, args.save_adv))
    
    save_log(args, log, log_pth)


if __name__ == "__main__":
    main()