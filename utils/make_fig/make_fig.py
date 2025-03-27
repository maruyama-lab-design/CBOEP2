import pandas as pd
import numpy as np
import os
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pathlib

import glob

"""
The functions in this file are called by make_all.py and other scripts in the make_fig directory 
to create figures for manuscripts.
"""


def get_prediction_score(metric, true, pred, prob):
    if metric == "Recall":
        return get_recall(true, pred)
    elif metric == "Specificity":
        return get_specificity(true, pred)
    elif metric == "Balanced accuracy":
        return get_balanced_accuracy(true, pred)
    elif metric == "AUC":
        return get_roc_auc(true, prob)
    elif metric == "AUPR":
        return get_pr_auc(true, prob)
    elif metric == "AUPR-ratio":
        return get_pr_auc_ratio(true, prob)
    elif metric == "F1":
        return metrics.f1_score(true, pred)
    elif metric == "MCC":
        return metrics.matthews_corrcoef(true, pred)
    elif metric == "Precision":
        return metrics.precision_score(true, pred)

def get_specificity(true, pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    return tn / (tn + fp)

def get_recall(true, pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    return tp / (tp + fn)

def get_balanced_accuracy(true, pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return (recall + specificity) / 2

def get_roc_auc(true, prob):
    return metrics.roc_auc_score(true, prob)

def get_pr_auc(true, prob):
    # return metrics.average_precision_score(true, prob)
    precision, recall, thresholds = metrics.precision_recall_curve(true, prob)
    auc = metrics.auc(recall, precision)
    return auc

def get_pr_auc_ratio(true, prob):
    # PR-AUC / positive proportion
    return get_pr_auc(true, prob) / (np.sum(true) / len(true))


def get_scores(result_file):
    if os.path.exists(result_file) == False:
        return -1, -1, -1
    result_df = pd.read_table(result_file, sep="\t")
    true = result_df["true"].to_list()
    prob = result_df["pred"].to_list()
    pred =  list(map(round, prob))
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()

    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = (recall + specificity) / 2

    return recall, specificity, balanced_accuracy


def plot_curve(
        metric="AUC", result_dir="", title="", outname="",
        test_cells=["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
    ):
    plt.figure()
    for test_cell in test_cells:
        if len(glob.glob(f"{result_dir}/*{test_cell}*.txt")) == 0:
            result_file = result_dir + f"-{test_cell}.csv"
        else:
            result_file = glob.glob(f"{result_dir}/*{test_cell}.txt")[0]

        if os.path.exists(result_file) == False:
            continue

        result_df = pd.read_table(result_file, sep="\t")
        true = result_df["true"].to_list()
        prob = result_df["pred"].to_list()
        pred =  list(map(round, prob))

        if metric == "AUC":
            fpr, tpr, thresholds = metrics.roc_curve(true, prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{test_cell} (AUC={auc:.3f})")
        elif metric == "AUPR":
            precision, recall, thresholds = metrics.precision_recall_curve(true, prob)
            auc = metrics.auc(recall, precision)
            plt.plot(recall, precision, label=f"{test_cell} (AUPR={auc:.3f})")
        
    if metric == "AUC":
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    elif metric == "AUPR":
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    plt.title(title)
    plt.legend()
    plt.savefig(outname)





def make_result_csv(
        metric = "Balanced accuracy",
        train_cell = "GM12878",
        test_cells = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"],
        result_dirs = [],
        columns = [],
        cv = False,
        outname = ""
):
    
    if cv == True:
        result_summary = pd.DataFrame(
            data=np.zeros((len(test_cells)+1, len(columns)*6)),
            columns=[f"{column}_fold{fold_idx}" for column in columns for fold_idx in [1, 2, 3, 4, 5, "_combined"]],
            index=test_cells + ["average"]
        )
    else:
        result_summary = pd.DataFrame(
            data=np.zeros((len(test_cells)+1, len(columns))),
            columns=columns,
            index=test_cells + ["average"]
        )

    for (result_dir, column) in zip(result_dirs, columns):
        for test_cell in test_cells:
            if cv == True:
                trues = []
                probs = []
                preds = []
                for fold_idx in range(5):
                    result_file = glob.glob(f"{result_dir}/*{test_cell}_fold{fold_idx+1}.txt")[0]
                    result_df = pd.read_table(result_file, sep="\t")
                    true = result_df["true"].to_list()
                    prob = result_df["pred"].to_list()
                    pred =  list(map(round, prob))

                    if metric == "Recall":
                        result_summary.loc[test_cell, column + f"_fold{fold_idx+1}"] = get_recall(true, pred)
                    elif metric == "Specificity":
                        result_summary.loc[test_cell, column + f"_fold{fold_idx+1}"] = get_specificity(true, pred)
                    elif metric == "Balanced accuracy":
                        result_summary.loc[test_cell, column + f"_fold{fold_idx+1}"] = get_balanced_accuracy(true, pred)
                    elif metric == "AUC":
                        result_summary.loc[test_cell, column + f"_fold{fold_idx+1}"] = get_roc_auc(true, prob)
                    elif metric == "AUPR":
                        result_summary.loc[test_cell, column + f"_fold{fold_idx+1}"] = get_pr_auc(true, prob)
                    elif metric == "AUPR-ratio":
                        result_summary.loc[test_cell, column + f"_fold{fold_idx+1}"] = get_pr_auc_ratio(true, prob)
                    trues += true
                    probs += prob
                    preds += pred

                if metric == "Recall":
                    result_summary.loc[test_cell, column + f"_fold_combined"] = get_recall(trues, preds)
                elif metric == "Specificity":
                    result_summary.loc[test_cell, column + f"_fold_combined"] = get_specificity(trues, preds)
                elif metric == "Balanced accuracy":
                    result_summary.loc[test_cell, column + f"_fold_combined"] = get_balanced_accuracy(trues, preds)
                elif metric == "AUC":
                    result_summary.loc[test_cell, column + f"_fold_combined"] = get_roc_auc(trues, probs)
                elif metric == "AUPR":
                    result_summary.loc[test_cell, column + f"_fold_combined"] = get_pr_auc(trues, probs)
                elif metric == "AUPR-ratio":
                    result_summary.loc[test_cell, column + f"_fold_combined"] = get_pr_auc_ratio(trues, probs)

            else:
                if len(glob.glob(f"{result_dir}/*{test_cell}*.txt")) == 0:
                    result_file = result_dir + f"-{test_cell}.csv"
                else:
                    result_file = glob.glob(f"{result_dir}/*{test_cell}.txt")[0]

                if os.path.exists(result_file) == False:
                    continue

                result_df = pd.read_table(result_file, sep="\t")
                true = result_df["true"].to_list()
                prob = result_df["pred"].to_list()
                pred =  list(map(round, prob))

                if metric == "Recall":
                    result_summary.loc[test_cell, column] = get_recall(true, pred)
                elif metric == "Specificity":
                    result_summary.loc[test_cell, column] = get_specificity(true, pred)
                elif metric == "Balanced accuracy":
                    result_summary.loc[test_cell, column] = get_balanced_accuracy(true, pred)
                elif metric == "AUC":
                    result_summary.loc[test_cell, column] = get_roc_auc(true, prob)
                elif metric == "AUPR":
                    result_summary.loc[test_cell, column] = get_pr_auc(true, prob)
                elif metric == "AUPR-ratio":
                    result_summary.loc[test_cell, column] = get_pr_auc_ratio(true, prob)
        
        if cv == True:
            for fold_idx in [1, 2, 3, 4, 5, "_combined"]:
                result_summary.loc["average", column + f"_fold{fold_idx}"] = np.mean(result_summary[:-1][column + f"_fold{fold_idx}"].values)
        else:
            result_summary.loc["average", column] = np.mean(result_summary[:-1][column].values)


    result_summary.to_csv(f"{outname}", sep="\t")


def plot_prob_by_dist(infile, outfile):
    dmax=500000

    df = pd.read_csv(infile, sep="\t", usecols=["true", "pred", "distance"])
    p_df = df[df["true"]==1]
    n_df = df[df["true"]==0]

    df = df.query("distance <= @dmax")

    plt.figure()
    plt.scatter(p_df["distance"], p_df["pred"], label="Actual positive", color="red", s=0.5, alpha=0.3)
    plt.scatter(n_df["distance"], n_df["pred"], label="Actual negative", color="blue", s=0.5, alpha=0.3)

    plt.xlabel("EPI distance")
    plt.ylabel("Prediction probability")
    plt.hlines(0.5, 0, 2500000, color='gray', linestyles='dotted')
    plt.xlim((0, dmax))
    plt.ylim((0, 1))
    plt.legend()

    plt.savefig(outfile)

def get_class_balance(df):
    r = {}
    for _, row in df.iterrows():
        e = row["enhancer_name"]
        p = row["promoter_name"]
        if e not in r.keys():
            r[e] = {
                "+": 0,
                "-": 0
            }
        if p not in r.keys():
            r[p] = {
                "+": 0,
                "-": 0
            }
        
        if row["label"] == 1:
            r[e]["+"] += 1
            r[p]["+"] += 1
        else:
            r[e]["-"] += 1
            r[p]["-"] += 1
    
    class_balance = 0
    for k, v in r.items():
        class_balance += min(v["+"], v["-"]) / max(v["+"], v["-"])
    class_balance /= len(r.keys())
    return class_balance


def make_prob_hist(
        indir = "",
        test_cell = "GM12878",
        n_bins = 20,
        prob_range = (0, 1),
        colors = ["red", "blue"],
        width = 0.75, # width of bar
        outfile = "",
        folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
):
    
    if folds == ["combined"]:
        folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
    
    # folds を　まとめて ロード
    dfs = []
    for fold in folds:
        if len(glob.glob(os.path.join(indir, f"*{test_cell}_{fold}.txt"))) == 0:
            # TargetFinder の場合
            if len(glob.glob(os.path.join(indir, f"*{test_cell}_{fold}.csv"))) == 1:
                file = glob.glob(os.path.join(indir, f"*{test_cell}_{fold}.csv"))[0]
            else:
                continue
        # TransEPI の場合
        elif len(glob.glob(os.path.join(indir, f"*{test_cell}_{fold}.txt"))) == 1:
            file = glob.glob(os.path.join(indir, f"*{test_cell}_{fold}.txt"))[0]
        else:
            continue
        # dfs.append(pd.read_csv(f"{indir}/{fold}.csv", sep="\t"))
        dfs.append(pd.read_csv(file, sep="\t"))
    df = pd.concat(dfs, axis=0)

    bin_list = [prob_range[0] + (prob_range[1] - prob_range[0]) / n_bins * i for i in range(n_bins+1)]

    actual_positive = df[df["true"]==1]["pred"].to_list()
    actual_negative = df[df["true"]==0]["pred"].to_list()


    actual_positive_ratio_by_bin = {}
    actual_negative_ratio_by_bin = {}
    # p_tmp, n_tmp = [], []
    for i in range(n_bins):
        actual_positive_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f})"] = len([p for p in actual_positive if bin_list[i] <= p < bin_list[i+1]]) / len(actual_positive)
        actual_negative_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f})"] = - (len([p for p in actual_negative if bin_list[i] <= p < bin_list[i+1]]) / len(actual_negative))
        # print([p for p in actual_positive if bin_list[i] <= p < bin_list[i+1]])
        # p_tmp += [p for p in actual_positive if bin_list[i] <= p < bin_list[i+1]]
        # n_tmp += [p for p in actual_negative if bin_list[i] <= p < bin_list[i+1]]

    # 1.0も含める
    actual_positive_ratio_by_bin.pop(f"[{bin_list[-2]:.3f}, {bin_list[-1]:.3f})")
    actual_negative_ratio_by_bin.pop(f"[{bin_list[-2]:.3f}, {bin_list[-1]:.3f})")
    actual_positive_ratio_by_bin[f"[{bin_list[-2]:.3f}, {bin_list[-1]:.3f}]"] = len([p for p in actual_positive if bin_list[-2] <= p <= bin_list[-1]]) / len(actual_positive)
    actual_negative_ratio_by_bin[f"[{bin_list[-2]:.3f}, {bin_list[-1]:.3f}]"] = - (len([p for p in actual_negative if bin_list[-2] <= p <= bin_list[-1]]) / len(actual_negative))

    
    # check if the sum of the ratio is 1
    print(sum(actual_positive_ratio_by_bin.values()))
    print(sum(actual_negative_ratio_by_bin.values())) 

    plt.figure()
    plt.bar(np.arange(n_bins), actual_positive_ratio_by_bin.values(), width=width, align="center", color=colors[0], linewidth=0.4, edgecolor="black", label="Actual positive")
    plt.bar(np.arange(n_bins), actual_negative_ratio_by_bin.values(), width=width, align="center", color=colors[1], linewidth=0.4, edgecolor="black", label="Actual negative")
    plt.axhline(0, color='gray', linewidth=0.4)
    plt.xticks(np.arange(n_bins), actual_positive_ratio_by_bin.keys(), rotation=90)
    plt.yticks(np.arange(-1, 1.1, 0.2), [f"{abs(i):.1f}" for i in np.arange(-1, 1.1, 0.2)])
    plt.xlabel("Prediction probability")
    plt.legend()
    plt.ylim((-1, 1))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")


def make_prob_line_graph(
        indirs = [],
        labels = [],
        test_cell = "GM12878",
        n_bins = 100,
        custom_bin_list = None,
        prob_range = (0, 1),
        colors = [
            "#7f7fff",
            "#ff7f7f",
            "#7fffbf"
        ],
        markers = [
            "x",
            "+",
            "v"
        ],
        ylim = (-1.0, 1.0),
        outfile = "",
        folds = ["fold1", "fold2", "fold3", "fold4", "fold5"],
        accumulate = False,
        # title = ""
):
    
    plt.figure()

    for _, (indir, label) in enumerate(zip(indirs, labels)):
        if folds == ["combined"]:
            folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]

        dfs = []
        for fold in folds:
            file = os.path.join(indir, f"{test_cell}/{fold}.txt")
            dfs.append(pd.read_csv(file, sep="\t"))
        if len(dfs) == 0:
            return
        df = pd.concat(dfs, axis=0)

        if custom_bin_list is not None:
            bin_list = custom_bin_list
            n_bins = len(bin_list) - 1
        else:
            bin_list = [prob_range[0] + (prob_range[1] - prob_range[0]) / n_bins * i for i in range(n_bins+1)]

        actual_positive = df[df["true"]==1]["pred"].to_list()
        actual_negative = df[df["true"]==0]["pred"].to_list()


        actual_positive_ratio_by_bin = {}
        actual_negative_ratio_by_bin = {}
        for i in range(n_bins):
            if i == 0:
                actual_positive_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f})"] = len([p for p in actual_positive if bin_list[i] <= p < bin_list[i+1]]) / len(actual_positive)
                actual_negative_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f})"] = - (len([p for p in actual_negative if bin_list[i] <= p < bin_list[i+1]]) / len(actual_negative))
            elif i != n_bins-1:
                if accumulate:
                    actual_positive_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f})"] = actual_positive_ratio_by_bin[f"[{bin_list[i-1]:.3f}, {bin_list[i]:.3f})"] + len([p for p in actual_positive if bin_list[i] <= p < bin_list[i+1]]) / len(actual_positive)
                    actual_negative_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f})"] = actual_negative_ratio_by_bin[f"[{bin_list[i-1]:.3f}, {bin_list[i]:.3f})"] - (len([p for p in actual_negative if bin_list[i] <= p < bin_list[i+1]]) / len(actual_negative))
                else:
                    actual_positive_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f})"] = len([p for p in actual_positive if bin_list[i] <= p < bin_list[i+1]]) / len(actual_positive)
                    actual_negative_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f})"] = - (len([p for p in actual_negative if bin_list[i] <= p < bin_list[i+1]]) / len(actual_negative))
            elif i == n_bins-1:
                if accumulate:
                    actual_positive_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f}]"] = actual_positive_ratio_by_bin[f"[{bin_list[i-1]:.3f}, {bin_list[i]:.3f})"] + len([p for p in actual_positive if bin_list[i] <= p <= bin_list[i+1]]) / len(actual_positive)
                    actual_negative_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f}]"] = actual_negative_ratio_by_bin[f"[{bin_list[i-1]:.3f}, {bin_list[i]:.3f})"] - (len([p for p in actual_negative if bin_list[i] <= p <= bin_list[i+1]]) / len(actual_negative))
                else:
                    actual_positive_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f}]"] = len([p for p in actual_positive if bin_list[i] <= p <= bin_list[i+1]]) / len(actual_positive)
                    actual_negative_ratio_by_bin[f"[{bin_list[i]:.3f}, {bin_list[i+1]:.3f}]"] = - (len([p for p in actual_negative if bin_list[i] <= p <= bin_list[i+1]]) / len(actual_negative))

        plt.plot(np.arange(n_bins), actual_positive_ratio_by_bin.values(), label=label, color=colors[_], linewidth=1, marker=markers[_], markersize=3, alpha=0.6)
        plt.plot(np.arange(n_bins), actual_negative_ratio_by_bin.values(), color=colors[_], linewidth=1, marker=markers[_], markersize=3, alpha=0.6)

    
    # plt.axhspan(0, 1, color='red', alpha=0.05) # Background color
    # plt.axhspan(-1, 0, color='blue', alpha=0.05) # Background color
    plt.axhline(0, color='black', linewidth=1.0)
    for i in range(n_bins):
        plt.axvline(i, color='gray', linestyle='dotted', linewidth=0.2)

    plt.xticks(np.arange(n_bins), actual_positive_ratio_by_bin.keys(), rotation=0)
    
    # plt.draw()
    xlocs, xlabs = plt.xticks()
    ylocs, ylabs = plt.yticks()

    # absoulte value. 
    new_ylabels = []
    for i in range(len(ylocs)):
        y = ylocs[i]
        new_ylabels.append(round(abs(y), 3))
    plt.yticks(ylocs, new_ylabels)
    

    # plt.yticks(np.arange(-1, 1.1, 0.2), [f"{abs(i):.1f}" for i in np.arange(-1, 1.1, 0.2)])
    # plt.xlabel("Bin of prediction probability")
    plt.legend(loc="upper center", fontsize=8, ncol=4)
    plt.ylim(ylim)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    # plt.title(f"{title}")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")

def dataset_score_function(df):
    enh_set = set(df["enhancer_name"].to_list())
    prm_set = set(df["promoter_name"].to_list())
    enh_cnt = len(enh_set)
    prm_cnt = len(prm_set)

    score = 0
    # Count the occurrences in the positive and negative sets of each enhancer 
    for enh_name, enh_df in df.groupby("enhancer_name"):
        f_plus = len(enh_df[enh_df["label"]==1])
        f_minus = len(enh_df[enh_df["label"]==0])
        score += (f_plus - f_minus) ** 2
    
    # Count the occurrences in the positive and negative sets of each promoter 
    for prm_name, prm_df in df.groupby("promoter_name"):
        f_plus = len(prm_df[prm_df["label"]==1])
        f_minus = len(prm_df[prm_df["label"]==0])
        score += (f_plus - f_minus) ** 2
    
    score /= (enh_cnt + prm_cnt)
    score = score ** 0.5
    return score
    


def make_dataset_score_bar(
        score_func = dataset_score_function,
        indirs = [],
        labels = [],
        colors = [],
        width = 0.25, # width of bar
        interval = 0.5, # interval between each test cell
        outfile = "",
        ylim = (0.0, 8.0),
        cells = ["GM12878"] # , "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
):
    plt.figure()
    x = 0 # now x position
    x_ticks = []
    for i, cell in enumerate(cells):
        for j, (indir, label) in enumerate(zip(indirs, labels)):
            infile = indir + f"/{cell}.csv"
            df = pd.read_csv(infile)
            score = score_func(df)
            print(f"{label} {cell}: {score}")
            if i == 0:
                plt.bar(x, score, width=width*0.90, label=label, align="center", linewidth=0.4, edgecolor="black", color=colors[j])
            else:
                plt.bar(x, score, width=width*0.90, align="center", linewidth=0.4, edgecolor="black", color=colors[j])
            if j == len(labels)//2:
                x_ticks.append(x)
            x += width
        x += interval

    x -= width

    plt.grid(axis="y", color="gray", linestyle="dotted", linewidth=0.4)
    # plt.xticks(x_ticks, cells)
    plt.ylim(ylim)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.legend(fontsize=8, loc="upper center", ncol=3)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")



def make_result_bar(
        indirs = [],
        labels = [],
        colors = [],
        title = "",
        width = 0.25, # width of bar
        interval = 0.5, # interval between each test cell
        outfile = "",
        metric = "Balanced accuracy",
        test_cells = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"],
        n_fold = -1,
        legend = {
            "fontsize": 8,
            "loc": "upper center",
            "ncol": 3}
            ):
    
    plt.figure()
    x = 0 # now x position
    x_ticks = []
    average_score = {label: np.zeros(len(test_cells)) for label in labels}
    for i, test_cell in enumerate(test_cells):                       # ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]
        for j, (indir, label) in enumerate(zip(indirs, labels)):     # ["unfilteredBENGI-N", "filteredBENGI-N", "CBMF-N", "CBGS-N"]

            if n_fold == -1: # no cross validation
                if len(glob.glob(f"{indir}/*{test_cell}*.txt")) == 0:
                    infile = indir + f"-{test_cell}.csv"
                elif len(glob.glob(f"{indir}/*{test_cell}*.txt")) == 1:
                    infile = glob.glob(f"{indir}/*{test_cell}.txt")[0]

                if os.path.exists(infile):
                    df = pd.read_table(infile, sep="\t")
                    true = df["true"].to_list()
                    prob = df["pred"].to_list()

                    pred =  list(map(round, prob))
                    score = get_prediction_score(metric, true, pred, prob)
                    if i == 0:
                        plt.bar(x, score, width=width*0.90, label=label, align="center", color=colors[j], 
                                linewidth=0.4, edgecolor="black", capsize=3)
                    else:
                        plt.bar(x, score, width=width*0.90,              align="center", color=colors[j], 
                                linewidth=0.4, edgecolor="black", capsize=3)
                else:
                    score = 0

            else: # cross validation
                score_list = np.zeros(n_fold)
                for fold_idx in range(1, n_fold+1):
                    infile = pathlib.Path(f"{indir}/{test_cell}/fold{fold_idx}.txt")
                    if not infile.exists():
                        print(f'No such input file: {infile}')

                    if os.path.exists(infile):
                        df = pd.read_table(infile, sep="\t")
                        # print(df)
                        true = df["true"].to_list()
                        prob = df["pred"].to_list()
                        pred =  list(map(round, prob))
                        score_list[fold_idx-1] =  get_prediction_score(metric, true, pred, prob)
                    else:
                        print(f'No such input file: {infile}')
                        score_list[fold_idx-1] = 0
                score = np.mean(score_list) 
                _yerr = np.std(score_list) / np.sqrt(n_fold)
                if i == 0:
                    plt.bar(x, score, width=width*0.90, label=label, align="center", color=colors[j], 
                            yerr=_yerr, linewidth=0.4, edgecolor="black", capsize=3)
                else:
                    plt.bar(x, score, width=width*0.90,              align="center", color=colors[j], 
                            yerr=_yerr, linewidth=0.4, edgecolor="black", capsize=3)
            print('{: <20}: {: <10}: {: <12}: {:>.3}'.format(label, test_cell, metric, score))


            if j == len(labels)//2:
                x_ticks.append(x)
            x += width
            average_score[label][i] = score
        x += interval
    
    # plot average
    print()
    for j, label in enumerate(labels):
        _yerr = np.std(average_score[label]) / np.sqrt(len(test_cells))
        _mean = np.mean(average_score[label])
        plt.bar(x, _mean, width=width*0.90, align="center", color=colors[j], 
                yerr=_yerr, linewidth=0.4, edgecolor="black", capsize=3)
        if j == len(labels)//2:
            x_ticks.append(x)
        x += width
        print('{: <20}: {: <10}: {: <12}: {: >.3} +- {:>.3}'.format(label, 'Average', metric, _mean, _yerr))
    print()


    x -= width
    if metric == "MCC":
        # plt.ylim((-1, 1.19))
        plt.ylim((-0.1, 1.19))
    else:
        plt.ylim((0, 1.19))
    # dot horizontal line by 0.2
    for i in range(1, 6):
        plt.axhline(0.2*i, color='gray', linestyle='dotted', linewidth=0.4)
    plt.xticks(x_ticks, test_cells + ["average"])
    # plt.ylabel(metric)
    # plt.title(title)
    plt.legend(**legend)
    plt.tick_params(bottom=False)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")



def make_pr_curve(
        indirs = [],
        labels = [],
        colors = [],
        outfile = "",
        folds = ["combined"],
        test_cell = "GM12878",
        title = "",
        marker_by_threshold = {0.5: "x"},
        ylim = (0.0, 1.0)
):
    plt.figure()

    for _, (indir, label) in enumerate(zip(indirs, labels)):
        if folds == ["combined"]:
            folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]

        dfs = []
        for fold in folds:
            file = f"{indir}/{test_cell}/{fold}.txt"

            dfs.append(pd.read_csv(file, sep="\t"))
        if len(dfs) == 0:
            return
        df = pd.concat(dfs, axis=0)

        true = df["true"].to_list()
        prob = df["pred"].to_list()

        precision, recall, thresholds = metrics.precision_recall_curve(true, prob)
        # print(f"thresholds of {label}: {thresholds[:10]}")
        # print(f"precision of {label}: {precision[:10]}")
        # print("\n\n")
        plt.plot(recall, precision, label=labels[_], color=colors[_], linewidth=1.0)

        # plot the points displaying the threshold. 
        for threshold, marker in marker_by_threshold.items():
            pred = list(map(round, [0 if p < threshold else 1 for p in prob]))
            precision = metrics.precision_score(true, pred)
            recall = metrics.recall_score(true, pred)
            plt.scatter(recall, precision, color=colors[_], marker=marker, s=20)
            plt.text(recall, precision+0.01, f"{threshold:.2f}", fontsize=5, color=colors[_], horizontalalignment='center', verticalalignment='bottom')

    # Showing grids.
    for i in range(1, 6):
        plt.axhline(0.2*i, color='gray', linestyle='dotted', linewidth=0.2)
        plt.axvline(0.2*i, color='gray', linestyle='dotted', linewidth=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower center", fontsize=8, ncol=4)
    # plt.title(title)
    # The background is colored gray. 
    # plt.axvspan(0, 1, color='gray', alpha=0.05)
    plt.xlim((0, 1))
    plt.ylim(ylim)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=1000, bbox_inches="tight")