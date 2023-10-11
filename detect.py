"""
The spatial attack is a very special attack because it tries to find adversarial
perturbations using a set of translations and rotations rather then in an Lp ball.
It therefore has a slightly different interface.
https://github.com/bethgelab/foolbox/blob/master/examples/spatial_attack_pytorch_resnet18.py
"""
import os

import foolbox as fb
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import copy
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
import argparse

from misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    create_dir
)

def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--att",  default="fgsm", choices=['fgsm', 'bim', 'pgd', 'df', 'cw'], help="")
    parser.add_argument("--defense",    default="multilid", choices=['multilid', 'lid'], help="")
    parser.add_argument("--dataset",  default="imagenet", choices=['imagenet', 'cifar10', 'cifar100'], help="")
    parser.add_argument("--model",  default="wrn50-2", choices=['wrn50-2', 'wrn28-10', 'vgg16'], help="")
    parser.add_argument("--load_nor",   default="normalos_8255.pt", help="save_gen_nor")
    parser.add_argument("--load_adv",   default="adverlos_8255.pt", help="save_gen_adv")
    parser.add_argument("--eps",        default="8/255", help="")
    parser.add_argument("--k",          default=30, help="")
    parser.add_argument("--nr_samples", default=2000, help="")
    parser.add_argument("--clf",        default='rf', choices=['rf', 'lr'], help="")
    parser.add_argument("--tr_size",    default=0.72, help="")

    parser.add_argument('--save_json', default="", help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_json', default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    args = args_handling(args, parser, "configs/detect")
    args.eps = convert_to_float(args.eps)
    print_args(args)

    base_pth = os.path.join(cfg.workspace, 'data/extract', args.dataset, args.model, args.defense, args.att, 'k'+str(args.k))
    normalos_fe = torch.load(os.path.join(base_pth, args.load_nor)).numpy()
    adverlos_fe = torch.load(os.path.join(base_pth, args.load_adv)).numpy()


    print("feature_method", args.defense, 'classifier', args.clf)

    # if isinstance(normalos_fe, np.ndarray):
    #     normalos_fe = torch.from_numpy(normalos_fe)
    #     adverlos_fe = torch.from_numpy(adverlos_fe)

    if len(normalos_fe.shape) > 2:
        normalos_fe = normalos_fe.reshape((normalos_fe.shape[0], -1))
        adverlos_fe = adverlos_fe.reshape((adverlos_fe.shape[0], -1))

    num_samples = int(normalos_fe.shape[0] * args.tr_size)
    print("num sapmpels: ", num_samples)
    X_train_nor = normalos_fe[:num_samples]
    X_train_adv = adverlos_fe[:num_samples]

    X_test_nor = normalos_fe[num_samples:]
    X_test_adv = adverlos_fe[num_samples:]

    X_train = np.vstack((X_train_nor, X_train_adv))
    y_train = np.concatenate((np.zeros(X_train_nor.shape[0]), np.ones(X_train_adv.shape[0])), axis=0)
    X_test  = np.vstack((X_test_nor, X_test_adv))
    y_test = np.concatenate((np.zeros(X_test_nor.shape[0]), np.ones(X_test_adv.shape[0])), axis=0)

    if args.clf == 'lr':
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        clf = LogisticRegressionCV(n_jobs=-1).fit(X_train, y_train)
        X_test = scaler.transform(X_test)

    elif args.clf == 'rf':
         clf = RandomForestClassifier(n_estimators=300, n_jobs=-1).fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    y_label_pred = clf.predict(X_test)

    # AUC
    _, _, auc_score = compute_roc(y_test, y_pred, plot=False)
    precision = precision_score(y_test, y_label_pred)
    recall = recall_score(y_test, y_label_pred)
    acc = accuracy_score(y_test, y_label_pred)

    print("auc_score: ", auc_score, "acc: ", acc)


if __name__ == "__main__":
    main()