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
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.preprocessing import scale
import argparse

import cfg

from misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    create_dir,
    create_log_file,
    save_log
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


def perf_measure(y_actual, y_hat):
    """
    https://shouland.com/false-positive-rate-test-sklearn-code-example
    """
    TP = 0; FP = 0; TN = 0; FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return (TP, FP, TN, FN)


def show_results(args, log, y_test, y_label_pred, y_pred):
    
    TP, FP, TN, FN = perf_measure(y_test, y_label_pred)
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP) 
    FNR = FN / (FN + TP)

    # auc = round(100*roc_auc_score(y_test, y_hat_pr), 2)
    _, _, auc = compute_roc(y_test, y_pred, plot=False)
    PRECISION = precision_score(y_test, y_label_pred)
    F1 = f1_score(y_test, y_label_pred)
    RECALL = recall_score(y_test, y_label_pred)
    ACC = accuracy_score(y_test, y_label_pred)

        
    f1  = round(100*F1, 2)
    pre = round(100*PRECISION, 2)
    acc = round(100*ACC, 2)
    tpr = round(100*TPR, 2)
    tnr = round(100*TNR, 2)
    fnr = round(100*FNR, 2)

    log['F1'] = str(f1)
    log['PREC'] = str(pre)
    log['ACC']  = str(acc)
    log['AUC']  = str(auc)
    log['TPR']  = str(tpr) # True positive rate/adversarial detetcion rate/recall/sensitivity is 
    log['TNR']  = str(tnr) # True negative rate/normal detetcion rate/selectivity is 
    log['FNR']  = str(fnr)

    print(f"{args.defense}, {args.clf}, {args.dataset}, auc: {auc}, acc: {acc}, pre: {pre}, f1: {f1}, tpr: {tpr}, tnr: {tnr}, fnr: {fnr}")

    return log


def main() -> None:
    parser = argparse.ArgumentParser("extract")
    parser.add_argument("--run_nr",  default="run_1", help="")
    parser.add_argument("--att",        default="fgsm", choices=['fgsm', 'bim', 'pgd', 'df', 'cw'], help="")
    parser.add_argument("--defense",    default="multilid", choices=['multilid', 'lid'], help="")
    parser.add_argument("--dataset",    default="imagenet", choices=['imagenet', 'cifar10', 'cifar100'], help="")
    parser.add_argument("--model",      default="wrn50-2", choices=['wrn50-2', 'wrn28-10', 'vgg16'], help="")
    parser.add_argument("--load_nor",   default="normalos_8255.pt", help="save_gen_nor")
    parser.add_argument("--load_adv",   default="adverlos_8255.pt", help="save_gen_adv")
    parser.add_argument("--eps",        default="8/255", help="")
    parser.add_argument("--k",          default=30, help="")
    parser.add_argument("--nr_samples", default=2000, help="")
    parser.add_argument("--clf",        default='rf', choices=['rf', 'lr'], help="")
    parser.add_argument("--tr_size",    default=0.72, help="")

    parser.add_argument('--save_json',  default="", help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_json',  default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    args = args_handling(args, parser, "configs/detect")
    args.eps = convert_to_float(args.eps)
    print_args(args)

    base_pth = os.path.join(cfg.workspace, 'data/extract', args.run_nr, args.dataset, args.model, args.defense, args.att, 'k'+str(args.k))
    normalos_fe = torch.load(os.path.join(base_pth, args.load_nor)).numpy()
    adverlos_fe = torch.load(os.path.join(base_pth, args.load_adv)).numpy()

    print("Create paths!")
    base_pth = os.path.join(cfg.workspace, 'data/extract', args.run_nr, args.dataset, args.model, args.defense, args.att, 'k'+str(args.k))
    base_pth_det = os.path.join(cfg.workspace, 'data/detect', args.run_nr, args.dataset, args.model, args.att, args.clf)
    create_dir(base_pth_det)
    log_pth = os.path.join(base_pth_det, 'logs')
    log = create_log_file(args, log_pth)
    log['timestamp'] =  datetime.now().strftime("%Y-%m-%d-%H-%M")


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

    y_label_pred = clf.predict(X_test)
    y_pred = clf.predict_proba(X_test)[:, 1]

    log = show_results(args, log, y_test, y_label_pred, y_pred)

    save_log(args, log, log_pth)

if __name__ == "__main__":
    main()