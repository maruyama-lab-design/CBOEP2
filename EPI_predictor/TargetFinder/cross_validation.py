import pandas as pd
import numpy as np
import os
import io
import subprocess
import tempfile
from glob import glob

import argparse

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix

import warnings, json, gzip

n_folds = 5

def specificity_score(true, pred):
	tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
	return tn / (tn + fp)


def get_classifier(tree, depth, lr):
	return GradientBoostingClassifier(n_estimators = tree, learning_rate = lr, max_depth = depth, max_features ='log2', random_state = 2023, verbose=1)


def get_weights(y):

	weights_dic = {
		0: 1 / (np.sum(y==0) / len(y)),
		1: 1 / (np.sum(y==1) / len(y))
	}

	weights_arr = np.zeros(len(y))

	for i in range(len(y)):
		weights_arr[i] = weights_dic[y[i]]

	return weights_arr


def train(df, use_window, train_chroms, gbdt_tree, gbdt_depth, gbdt_lr):

	_nonpredictors = ["bin","enhancer_chrom","enhancer_distance_to_promoter","enhancer_end","enhancer_name","enhancer_start","label","promoter_chrom","promoter_end","promoter_name","promoter_start","window_end","window_start","window_chrom","window_name","interactions_in_window","active_promoters_in_window"]
	nonpredictors = [f for f in _nonpredictors if f in df.columns]
	if not use_window:
		nonpredictors +=  [f for f in df.columns if "(window)" in f]
	print(nonpredictors)

	train_df = df[df["enhancer_chrom"].isin(train_chroms)]

	x_train = train_df.drop(columns=nonpredictors).values
	y_train = train_df["label"].values.flatten()
	
	weights = get_weights(y_train)
	classifier = get_classifier(gbdt_tree, gbdt_depth, gbdt_lr)

	x_train = np.nan_to_num(x_train, nan=0, posinf=0)

	print(f"train data size: {x_train.shape}")
	classifier.fit(x_train, y_train, sample_weight=weights) # train

	return classifier



def test(classifier, df, use_window, test_chroms, fold_idx, pred_dir):
	_nonpredictors = ["bin","enhancer_chrom","enhancer_distance_to_promoter","enhancer_end","enhancer_name","enhancer_start","label","promoter_chrom","promoter_end","promoter_name","promoter_start","window_end","window_start","window_chrom","window_name","interactions_in_window","active_promoters_in_window"]
	nonpredictors = [f for f in _nonpredictors if f in df.columns]
	if not use_window:
		nonpredictors +=  [f for f in df.columns if "(window)" in f]

	metrics = {
		"balanced accuracy": -99,
		"recall": -99,
		"specificity": -99,
		"AUC": -99,
		"MCC": -99,
		"F1": -99
	}

	output_path = os.path.join(pred_dir, f"fold{fold_idx+1}.txt")
	test_df = df[df["enhancer_chrom"].isin(test_chroms)]

	x_test = test_df.drop(columns=nonpredictors).values
	x_test = np.nan_to_num(x_test, nan=0, posinf=0)
	y_test = test_df["label"].values.flatten()

	print(f"test data size: {x_test.shape}")
	y_pred = classifier.predict_proba(x_test) # predict
	y_pred = [prob[1] for prob in y_pred]

	result_df = pd.DataFrame(
		{
			"true": y_test,
			"pred": y_pred,
			"chrom": test_df["enhancer_chrom"],
			"distance": test_df["enhancer_distance_to_promoter"],
			"enhancer_name": test_df["enhancer_name"],
			"promoter_name": test_df["promoter_name"]
		}
	)

	if not os.path.exists(os.path.dirname(output_path)):
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
	result_df.to_csv(output_path, index=False, sep="\t")

	true = result_df["true"].to_list()
	prob = result_df["pred"].to_list()
	pred =  list(map(round, prob))

	metrics["balanced accuracy"] = balanced_accuracy_score(true, pred)
	metrics["recall"] = recall_score(true, pred)
	metrics["specificity"] = specificity_score(true, pred)
	metrics["AUC"] = roc_auc_score(true, prob)
	metrics["F1"] = f1_score(true, pred)
	metrics["MCC"] = matthews_corrcoef(true, pred)

	print(metrics)


def main(train_EPI, test_EPI, pred_dir, use_window):
	config = json.load(open(os.path.join(os.path.dirname(__file__), "opt.json")))
	train_chroms_dict = config["train_opt"]["train_chroms_dict"]
	test_chroms_dict = config["train_opt"]["test_chroms_dict"]

	train_df = pd.read_csv(train_EPI)
	test_df = pd.read_csv(test_EPI)

	if not os.path.exists(pred_dir):
		os.makedirs(pred_dir, exist_ok=True)

	# match feature order in train and test
	train_columns = set(list(train_df.columns))
	test_columns = set(list(test_df.columns))
	print(f"train columns: {len(train_columns)}")
	print(f"test columns: {len(test_columns)}")
	union_columns = train_columns | test_columns
	print(f"union columns: {len(union_columns)}")
	train_df[list(union_columns - train_columns)] = 0
	test_df[list(union_columns - test_columns)] = 0
	train_df = train_df[list(test_df.columns)]

	assert list(train_df.columns) == list(test_df.columns), "feature order error"

	for fold_idx in range(len(train_chroms_dict)):
		train_chroms_ = train_chroms_dict[f"fold{fold_idx+1}"]
		test_chroms_ = test_chroms_dict[f"fold{fold_idx+1}"]
		classifier = train(
			df=train_df, 
			use_window=use_window, 
			train_chroms=train_chroms_,
			gbdt_tree=config["model_opt"]["gbdt_tree"],
			gbdt_depth=config["model_opt"]["gbdt_depth"],
			gbdt_lr=config["model_opt"]["gbdt_alpha"])
		
		test(
			classifier=classifier, 
	   		df=test_df, 
			use_window=use_window, 
			test_chroms=test_chroms_, 
			fold_idx=fold_idx, 
			pred_dir=pred_dir)


def parse_arguments():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("--train_EPI", required=True)
	p.add_argument("--test_EPI", required=True)
	p.add_argument("--pred_dir", default="./preds/")
	p.add_argument("--use_window", action="store_true")
	args = p.parse_args()
	return args

if __name__ == "__main__":
	args = parse_arguments()
	main(args.train_EPI, args.test_EPI, args.pred_dir, args.use_window)