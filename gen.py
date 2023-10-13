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
    create_log_file,
    save_log
)

from models.helper import get_model

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
    parser.add_argument("--run_nr",  default="run_1", help="")
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
    parser.add_argument("--shuffle",  default=False, type=str2bool, help="")

    parser.add_argument('--save_json', default="", help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_json', default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    args = args_handling(args, parser, "./configs/gen")
    args.eps = convert_to_float(args.eps)
    print_args(args)

    print("Create paths!")
    base_pth = os.path.join(cfg.workspace, 'data/gen', args.run_nr, args.dataset, args.model, args.att)
    create_dir(base_pth)
    log_pth = os.path.join(base_pth, 'logs')
    log = create_log_file(args, log_pth)

    print("Load model and data")
    model, preprocessing = get_model(args)
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
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=args.shuffle, num_workers=8)

    elif args.dataset == 'cifar100':
        transform_list = [transforms.ToTensor()] 
        transform = transforms.Compose(transform_list)
        
        dataset_dir_path =  os.path.join(cfg.DATASET_BASE_PTH,"cifar100") 

        dataset = datasets.CIFAR100(root=dataset_dir_path, train=False, transform=transform, download=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=args.shuffle, num_workers=8)

    print("Prepare attack")
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
        adversary = AutoAttack_mod(fmodel, norm=args.norm.capitalize(), eps=args.eps, 
                                    log_path=os.path.join(log_pth, args.load_json.split('/')[-1]).replace("json", "log"),  verbose=args.debug, version=args.version)
        if args.version == 'individual':
            adversary.attacks_to_run = [ args.att ]

    # report the success rate of the attack (percentage of samples that could
    # be adversarially perturbed) and the robust accuracy (the remaining accuracy
    # of the model when it is attacked)
    (images, labels), restore_type = ep.astensors_(*samples(fmodel, dataset=args.dataset, batchsize=20))
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

    print("Generate data")
    counter = 0
    clean_acc_list = []
    total_success = []
    normalos = []
    adverlos = []
    for it, (img, lab) in tqdm(enumerate(data_loader), total=round((args.max_counter)/args.bs)):

        img_cu, lab_cu = img.cuda(non_blocking=True), lab.cuda(non_blocking=True)
        clean_acc_list.append(accuracy(fmodel, img_cu, lab_cu) * 100)
        
        img_cu, lab_cu, restore_type = pred(fmodel, img_cu, lab_cu) # select only correct predicted

        if args.debug:
            print("clean_acc for att: ", np.mean(clean_acc_list))
            print("accuracy model: {:.2f}".format( img_cu.shape[0]/img.shape[0]) )

        if args.att in DEEPFOOL:
            lab_cu = fb.criteria.Misclassification(lab_cu)
            xp_, x_, success = attack(fmodel, img_cu, lab_cu, epsilons=args.eps)
            suc = success.float32().mean().item() * 100
            
            nor = restore_type(img_cu[torch.where(restore_type(success).int().cpu().squeeze() == 1)[0]]).cpu()
            adv = restore_type(x_[torch.where(restore_type(success).int().cpu().squeeze() == 1)[0]]).cpu()

        elif args.att in AUTOATTACK:
            with torch.no_grad():
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

    asr = np.mean(total_success)
    print(args.att, f", counter {counter}", normalos.shape, asr)

    log['final_nr_samples'] = counter
    log['asr'] = asr
    log['clean_acc'] = 0 if len(clean_acc_list) == None else np.mean(clean_acc_list)

    save_log(args, log, log_pth)

    torch.save(normalos, os.path.join(base_pth, args.save_nor))
    torch.save(adverlos, os.path.join(base_pth, args.save_adv))

 
if __name__ == "__main__":
    main()