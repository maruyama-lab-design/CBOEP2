#!/usr/bin/env python3

import argparse, os, sys, time, shutil, tqdm
import warnings, json, gzip
import numpy as np
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

import epi_models
import epi_dataset
import misc_utils


import functools
from sklearn import metrics

import pandas as pd 
import pickle
import glob

# tensorboard
from torch.utils.tensorboard import SummaryWriter

def model_summary(model):
    """
    model: pytorch model
    """
    import torch
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}


def predict(model: nn.Module, data_loader: DataLoader, device=torch.device('cuda'), use_weighted_bce=True):
    model.eval()
    result, true_label = list(), list()
    for batch_idx, (feats, dists, enh_idxs, prom_idxs, labels) in enumerate(data_loader):

        feats, labels = feats.to(device), labels.to(device)
        pred, pred_dists, att = model(feats, return_att=True, enh_idx=enh_idxs, prom_idx=prom_idxs, batch_idx=batch_idx)
        del att
        pred = pred.detach().cpu().numpy()
        if use_weighted_bce == True:
            # when using weighted bce, sigmoid is not included in the model
            # so we need to add sigmoid here
            pred = nn.Sigmoid()(torch.Tensor(pred)).numpy()
        labels = labels.detach().cpu().numpy()
        result.append(pred)
        true_label.append(labels)

    result = np.concatenate(result, axis=0)
    true_label = np.concatenate(true_label, axis=0)

    return (result.squeeze(), true_label.squeeze())



def train(
        model_class, model_params, 
        optimizer_class, optimizer_params, 
        dataset, groups,
        num_epoch, patience, batch_size, num_workers,
        model_dir,
        checkpoint_prefix, device, 
        cv_chroms,
        use_scheduler=False,
        use_weighted_bce=True,
        add_dist_loss=False,
        save_all_epoch=False,
        writer=None) -> nn.Module:
    print(f"Training {model_class.__name__} model...")

    wait = 0
    best_epoch, best_val_auc, best_val_aupr, best_balanced_acc = -999, -999, -999, -999
    best_loss = 999
    all_epoch_results = {
        "AUC": np.zeros((0, 6)), # 5-fold + average
        "AUPR": np.zeros((0, 6)), # 5-fold + average
        "Balanced accuracy": np.zeros((0, 6)), # 5-fold + average
    }
    loss_dict = {"epochs": [], "train_loss": [], "valid_loss": []}
    log_list = []

    for epoch_idx in range(num_epoch):

        now_epoch_results = {
            "AUC": np.zeros(6), # 5-fold + average
            "AUPR": np.zeros(6), # 5-fold + average
            "Balanced accuracy": np.zeros(6), # 5-fold + average
        }


        for fold_idx, cv_chrom_dict in enumerate(cv_chroms):
            print(f"epoch: {epoch_idx}, fold: {fold_idx+1}")
            train_chroms = cv_chrom_dict["train_chroms"]
            valid_chroms = cv_chrom_dict["valid_chroms"]

            train_idx, valid_idx = [], []
            for idx, chrom in enumerate(groups):
                if chrom in train_chroms:
                    train_idx.append(idx)
                elif chrom in valid_chroms:
                    valid_idx.append(idx)
      

            # define loss function
            mse_loss = nn.MSELoss() # mean squared error
            if use_weighted_bce == True:
                # get train labels
                labels = np.array(dataset.metainfo["label"])[train_idx]
                pos_cnt = np.sum(labels)
                neg_cnt = len(labels) - pos_cnt
                pos_weight = torch.tensor([neg_cnt/pos_cnt]).to(device)
                bce_logits_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # weighted binary cross entropy
            else:
                bce_loss = nn.BCELoss() # binary cross entropy

            
            os.makedirs(model_dir, exist_ok = True)

            with open(os.path.join(model_dir, "log.txt"), "a") as f:
                print(
                    f"epochs: {epoch_idx}, fold: {fold_idx+1},  \
                    training size: {len(train_idx)}({misc_utils.count_unique_itmes(groups[train_idx])}), \
                    validation size: {len(valid_idx)}({misc_utils.count_unique_itmes(groups[valid_idx])})", file=f
                )
            
            train_loader = DataLoader(Subset(dataset, indices=train_idx), shuffle=True, batch_size=batch_size, num_workers=num_workers)
            sample_idx = np.random.permutation(train_idx)[0:1024] # for train loss
            sample_loader = DataLoader(Subset(dataset, indices=sample_idx), shuffle=False, batch_size=batch_size, num_workers=num_workers)
            valid_loader = DataLoader(Subset(dataset, indices=valid_idx), shuffle=False, batch_size=batch_size, num_workers=num_workers)
            checkpoint = f"{model_dir}/{checkpoint_prefix}_fold{fold_idx+1}.pt"

            if epoch_idx == 0:
                model = model_class(**model_params).to(device)
                optimizer = optimizer_class(model.parameters(), **optimizer_params)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)
            else:
                # state_dict = torch.load(checkpoint) # FutureWarning @2024-10-28.
                state_dict = torch.load(checkpoint, weights_only=True)

                model.load_state_dict(state_dict["model_state_dict"])
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                scheduler.load_state_dict(state_dict["scheduler_state_dict"])

            model.train()
            for feats, dists, enh_idxs, prom_idxs, labels in tqdm.tqdm(train_loader): # train by batch
                feats, dists, labels = feats.to(device), dists.to(device) ,labels.to(device)
                prom_idxs = prom_idxs.clamp(0, feats.shape[2] - 1) # prom_idxs is restricted from 0 to feats.shape[2] - 1. 
                if hasattr(model, "att_C"):
                    pred, pred_dists, att = model(feats, return_att=True, enh_idx=enh_idxs, prom_idx=prom_idxs)
                    attT = att.transpose(1, 2)
                    identity = torch.eye(att.size(1)).to(device)
                    identity = Variable(identity.unsqueeze(0).expand(labels.size(0), att.size(1), att.size(1)))
                    penal = model.l2_matrix_norm(torch.matmul(att, attT) - identity)

                    if add_dist_loss == True:
                        loss = bce_loss(pred, labels) + mse_loss(pred_dists, dists) + (model.att_C * penal / labels.size(0)).type(torch.cuda.FloatTensor)
                    else:
                        if use_weighted_bce == True:
                            loss = bce_logits_loss(pred, labels) + (model.att_C * penal / labels.size(0)).type(torch.cuda.FloatTensor)
                        else:
                            loss = bce_loss(pred, labels) + (model.att_C * penal / labels.size(0)).type(torch.cuda.FloatTensor)

                    del penal, identity
                else:
                    pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
                    loss = bce_loss(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, checkpoint)

            if save_all_epoch == True:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                }, os.path.join(model_dir, f"epoch{epoch_idx}_fold{fold_idx+1}.pt"))
            
            model.eval()
            train_loss, valid_loss = None, None
            train_pred, train_true = predict(model, sample_loader, device=device, use_weighted_bce=use_weighted_bce)
            tra_AUC, tra_AUPR, tra_balanced_acc, tra_rec, tra_spec, tra_MCC = misc_utils.evaluator(train_true, train_pred, out_keys=["AUC", "AUPR", "Balanced accuracy", "Recall", "Specificity", "MCC"])
            valid_pred, valid_true = predict(model, valid_loader, device=device, use_weighted_bce=use_weighted_bce)
            val_AUC, val_AUPR, val_balanced_acc, val_rec, val_spec, val_MCC = misc_utils.evaluator(valid_true, valid_pred, out_keys=["AUC", "AUPR", "Balanced accuracy", "Recall", "Specificity", "MCC"])

            train_loss = metrics.log_loss(train_true, train_pred.astype(np.float64))
            valid_loss = metrics.log_loss(valid_true, valid_pred.astype(np.float64))
            writer.add_scalar('train-loss', train_loss, epoch_idx)
            writer.add_scalar('valid-loss', valid_loss, epoch_idx)

            log_tra_txt = f"  - train...\nloss={train_loss:.4f}\tAUPR={tra_AUPR:.4f}\tBalanced accuracy={tra_balanced_acc:.4f}\tRecall={tra_rec:.4f}\tSpecificity={tra_spec:.4f}\tMCC={tra_MCC:.4f}\t"
            log_val_txt = f"  - valid...\nloss={valid_loss:.4f}\tAUPR={val_AUPR:.4f}\tBalanced accuracy={val_balanced_acc:.4f}\tRecall={val_rec:.4f}\tSpecificity={val_spec:.4f}\tMCC={val_MCC:.4f}\t"
            print(log_tra_txt)
            print(log_val_txt)

            with open(os.path.join(model_dir, "log.txt"), "a") as f:
                print(f"___epoch{epoch_idx} fold{fold_idx+1}___", file=f)
                print(log_tra_txt, file=f)
                print(log_val_txt, file=f)

            log_list.append([train_loss, valid_loss, tra_AUPR, val_AUPR, tra_balanced_acc, val_balanced_acc, tra_rec, val_rec, tra_spec, val_spec, tra_MCC, val_MCC])

            now_epoch_results["AUC"][fold_idx] = val_AUC
            now_epoch_results["AUPR"][fold_idx] = val_AUPR
            now_epoch_results["Balanced accuracy"][fold_idx] = val_balanced_acc

            # __end of fold__

        # __end of epoch__

        # ___ is now epoch better than before? ___
        now_epoch_results["AUC"][-1] = now_epoch_results["AUC"][:-1].mean()
        now_epoch_results["AUPR"][-1] = now_epoch_results["AUPR"][:-1].mean()
        now_epoch_results["Balanced accuracy"][-1] = now_epoch_results["Balanced accuracy"][:-1].mean()

        # if best_val_auc < now_epoch_results["AUC"][-1] and best_val_aupr < now_epoch_results["AUPR"][-1]:
        if best_val_aupr < now_epoch_results["AUPR"][-1]:
            wait = 0
            best_val_auc = now_epoch_results["AUC"][-1]
            best_val_aupr = now_epoch_results["AUPR"][-1]
            best_epoch = epoch_idx
            print("Best epoch {}\t({})".format(best_epoch, time.asctime()))
            for fold_idx in range(len(cv_chroms)):
                shutil.copyfile(os.path.join(model_dir, f"checkpoint_fold{fold_idx+1}.pt"), os.path.join(model_dir, f"best_epoch_fold{fold_idx+1}.pt"))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopped ({})".format(time.asctime()))
                print("Best epoch/AUC/AUPR: {}\t{:.4f}\t{:.4f}".format(best_epoch, best_val_auc, best_val_aupr))
                break
            else:
                print("Wait{} ({})".format(wait, time.asctime()))

       
        all_epoch_results["AUC"] = np.concatenate([all_epoch_results["AUC"], now_epoch_results["AUC"].reshape(1, -1)], 0)
        all_epoch_results["AUPR"] = np.concatenate([all_epoch_results["AUPR"], now_epoch_results["AUPR"].reshape(1, -1)], 0)
        all_epoch_results["Balanced accuracy"] = np.concatenate([all_epoch_results["Balanced accuracy"], now_epoch_results["Balanced accuracy"].reshape(1, -1)], 0)



def test(model_class, model_params, 
        optimizer_class, optimizer_params, 
        dataset, groups, batch_size, num_workers,
        model_dir, pred_dir, device, 
        cv_chroms,
        use_weighted_bce=True):

    for fold_idx in range(len(cv_chroms)):
        model_path = os.path.join(model_dir, f"best_epoch_fold{fold_idx+1}.pt")

        print(f"loading {model_path}...")
        model = model_class(**model_params, use_weighted_bce=use_weighted_bce).to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        # state_dict = torch.load(model_path) # FutureWarning @2024-10-28.
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        test_idx = []
        test_chroms = cv_chroms[fold_idx]["test_chroms"]
        for idx, chrom in enumerate(groups):
            if chrom in test_chroms:
                test_idx.append(idx)
        chroms = np.array(dataset.metainfo["chrom"])[test_idx]
        distances = np.array(dataset.metainfo["dist"])[test_idx]
        enh_names = np.array(dataset.metainfo["enh_name"])[test_idx]
        prom_names = np.array(dataset.metainfo["prom_name"])[test_idx]
        test_loader = DataLoader(Subset(dataset, indices=test_idx), shuffle=False, batch_size=batch_size, num_workers=num_workers)
        model.eval()
        test_pred, test_true = predict(model, test_loader,  device=device, use_weighted_bce=use_weighted_bce)
        # AUC, AUPR, F_in, pre, rec, MCC = misc_utils.evaluator(test_true, test_pred, out_keys=["AUC", "AUPR", "F1", "precision", "recall", "MCC"])

        pred_path = os.path.join(pred_dir, f"fold{fold_idx+1}.txt")
        os.makedirs(pred_dir, exist_ok=True)
        np.savetxt(
                os.path.join(pred_path),
                np.concatenate((
                    test_true.reshape(-1, 1).astype(int).astype(str),
                    test_pred.reshape(-1, 1).round(4).astype(str),
                    chroms.reshape(-1, 1),
                    distances.reshape(-1, 1).astype(int).astype(str),
                    enh_names.reshape(-1, 1),
                    prom_names.reshape(-1, 1)
                ), axis=1),
                delimiter='\t',
                fmt="%s",
                comments="",
                header="true\tpred\tchrom\tdistance\tenhancer_name\tpromoter_name"
        )

        print(f"Prediction results are saved in {pred_path} !!")





def common_setting_of_train_test(config="opt.json", gpu=0, seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = json.load(open(os.path.join(os.path.dirname(__file__), config)))
    optimizer_params = {'lr': config["train_opts"]["learning_rate"], 'weight_decay': 1e-8}

    if gpu >= 0:
        print(f'torch.cuda.is_available() {torch.cuda.is_available()}')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    return device, config, optimizer_params



# def main(do_train, do_test, train_data_path, test_data_path, pred_dir="preds", model_dir="models", tensorboard_dir="tensorboard", 
#          gpu=0, seed=2020, use_mask=False, use_weighted_bce=True, use_dist_loss=False, config="opt.json"):

def do_train(train_data_path, model_dir="models", tensorboard_dir="tensorboard", 
        use_mask=False, use_weighted_bce=False, use_dist_loss=False, 
        config="opt.json", gpu=0, seed=2020):
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # 非同期操作を強制同期
    
    device, config, optimizer_params = common_setting_of_train_test(config, gpu, seed)

    writer = SummaryWriter(tensorboard_dir)

    train_data = epi_dataset.EPIDataset(
        datasets=train_data_path,
        feats_config=config["feats_config"],
        feats_order=config["feats_order"], 
        seq_len=config["seq_len"], 
        bin_size=config["bin_size"], 
        use_mark=False,
        use_mask=use_mask,
        sin_encoding=False,
        rand_shift=False,
    )

    config["model_opts"]["in_dim"] = train_data.feat_dim
    config["model_opts"]["seq_len"] = config["seq_len"] // config["bin_size"]
    chroms = train_data.metainfo["chrom"]
    model_class = getattr(epi_models, config["model_opts"]["model"])

    train(
        model_class=model_class,
        model_params=config["model_opts"],
        optimizer_class=torch.optim.Adam,
        optimizer_params=optimizer_params,
        dataset=train_data,
        groups=chroms,
        num_epoch=config["train_opts"]["num_epoch"],
        patience=config["train_opts"]["patience"], 
        batch_size=config["train_opts"]["batch_size"], 
        num_workers=config["train_opts"]["num_workers"],
        model_dir=model_dir,
        checkpoint_prefix="checkpoint",
        device=device,
        cv_chroms = [config["train_opts"][f"fold_{i+1}"] for i in range(5)],
        use_scheduler=config["train_opts"]["use_scheduler"],
        add_dist_loss=use_dist_loss,
        use_weighted_bce=use_weighted_bce,
        save_all_epoch=False,
        writer=writer
    )



def do_test(test_data_path, pred_dir="preds", model_dir="models", 
        use_mask=False, 
        config="opt.json", gpu=0, seed=2020):
    
    device, config, optimizer_params = common_setting_of_train_test(config, gpu, seed)

    test_data = epi_dataset.EPIDataset(
        datasets=test_data_path,
        feats_config=config["feats_config"],
        feats_order=config["feats_order"], 
        seq_len=config["seq_len"], 
        bin_size=config["bin_size"], 
        use_mark=False,
        use_mask=use_mask,
        sin_encoding=False,
        rand_shift=False,
    )

    config["model_opts"]["in_dim"] = test_data.feat_dim
    config["model_opts"]["seq_len"] = config["seq_len"] // config["bin_size"]
    chroms = test_data.metainfo["chrom"]
    model_class = getattr(epi_models, config["model_opts"]["model"])

    test(
        model_class=model_class,
        model_params=config["model_opts"],
        optimizer_class=torch.optim.Adam,
        optimizer_params=optimizer_params,
        dataset=test_data,
        groups=chroms,
        batch_size=config["train_opts"]["batch_size"],
        num_workers=config["train_opts"]["num_workers"],
        model_dir=model_dir,
        pred_dir=pred_dir,
        device=device, 
        cv_chroms = [config["train_opts"][f"fold_{i+1}"] for i in range(5)]
    )

# def parse_arguments():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument('--do_train', action='store_true', help="Training models")
#     p.add_argument('--do_test', action='store_true', help="Test models")

#     p.add_argument('--train_data_path', required=True, help="EPI data for training")
#     p.add_argument('--test_data_path', required=True, help="EPI data for testing")

#     p.add_argument('--pred_dir', default="./preds/")
#     p.add_argument('--model_dir', default="./models/")
#     p.add_argument('--tensorboard_dir', default="./tensorboard/")

#     p.add_argument('--gpu', default=0, type=int, help="GPU ID, (-1 for CPU)")
#     p.add_argument('--seed', type=int, default=2020, help="Random seed")

#     p.add_argument('--use_mask', default=False, action='store_true')
#     p.add_argument('--use_weighted_bce', action='store_true')
#     p.add_argument('--use_dist_loss', action='store_true')
#     p.add_argument('--config', default="opt.json")
#     args = p.parse_args()
#     return args

# if __name__ == "__main__":
#     args = parse_arguments()
#     main(args.do_train, args.do_test, args.train_data_path, args.test_data_path, 
#          args.pred_dir, args.model_dir, args.tensorboard_dir, 
#          args.gpu, args.seed, args.use_mask, args.use_weighted_bce, args.use_dist_loss, args.config)




    