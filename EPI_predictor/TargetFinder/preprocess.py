# Here, features are added to the pair data.
# The addition of the window regions is also executed here.

#!/usr/bin/env python

import chromatics
import os
import numpy as np
import pandas as pd
# import sys

from glob import glob
import math
import argparse

# import json


def insert_window(df):
	cell = df.at[0, "enhancer_name"].split("|")[0]
	df["window_chrom"] = df["enhancer_chrom"]
	df["window_start"] = np.where(df["enhancer_end"] < df["promoter_end"], df["enhancer_end"], df["promoter_end"])
	df["window_end"] = np.where(df["enhancer_start"] > df["promoter_start"], df["enhancer_start"], df["promoter_start"])
	df["window_name"] = cell + "|" + df["window_chrom"] + ":" + df["window_start"].apply(str) + "-" +  df["window_end"].apply(str)
	assert (df["window_end"] >= df["window_start"]).all(), df[df["window_end"] <= df["window_start"]].head()
	return df


def preprocess_features(cell):

	# peaks_fn = 'peaks.bed.gz'
	peaks_fn = os.path.join('tmp', f'{cell}', 'peaks.bed.gz')
	os.makedirs(os.path.dirname(peaks_fn), exist_ok=True)

	# methylation_fn = 'methylation.bed.gz'
	methylation_fn = os.path.join('tmp', f'{cell}', 'methylation.bed.gz')
	os.makedirs(os.path.dirname(methylation_fn), exist_ok=True)

	# cage_fn = 'cage.bed.gz'
	cage_fn = os.path.join('tmp', f'{cell}', 'cage.bed.gz')
	os.makedirs(os.path.dirname(cage_fn), exist_ok=True)

	generators = []

	# preprocess peaks
	peaks_dir = os.path.join(os.path.dirname(__file__), "input_features", cell, "peaks")
	if os.path.exists(peaks_dir):
		print(f"preprocess peaks...")
		assays = []
		for name, filename, source, accession in pd.read_csv(os.path.join(peaks_dir, "filenames.csv")).itertuples(index = False):
			columns = chromatics.narrowpeak_bed_columns if filename.endswith('narrowPeak') else chromatics.broadpeak_bed_columns
			assay_df = chromatics.read_bed(os.path.join(peaks_dir, f"{filename}.gz"), names = columns, usecols = chromatics.generic_bed_columns + ['signal_value'])
			assay_df['name'] = name
			assays.append(assay_df)
		peaks_df = pd.concat(assays, ignore_index = True)
		chromatics.write_bed(peaks_df, peaks_fn, compression = 'gzip')
		generators.append((chromatics.generate_average_signal_features, peaks_fn))

	# preprocess methylation
	methylation_dir = os.path.join(os.path.dirname(__file__), "input_features", cell, "methylation")
	if os.path.exists(methylation_dir):
		print(f"preprocess methylation...")
		assays = [chromatics.read_bed(_, names = chromatics.methylation_bed_columns, usecols = chromatics.generic_bed_columns + ['mapped_reads', 'percent_methylated']) for _ in glob(os.path.join(methylation_dir, f"*.bed.gz"))]
		methylation_df = pd.concat(assays, ignore_index = True).query('mapped_reads >= 10 and percent_methylated > 0')
		methylation_df['name'] = 'Methylation'
		del methylation_df['mapped_reads']
		chromatics.write_bed(methylation_df, methylation_fn, compression = 'gzip')
		generators.append((chromatics.generate_average_signal_features, methylation_fn))

	# preprocess cage
	cage_dir = os.path.join(os.path.dirname(__file__), "input_features", cell, "cage")
	if os.path.exists(cage_dir):
		print(f"preprocess cage...")
		cage_df = chromatics.read_bed(glob(os.path.join(cage_dir, f"*.bed.gz"))[0], names = chromatics.cage_bed_columns, usecols = chromatics.cage_bed_columns[:5])
		cage_df['name'] = 'CAGE'
		chromatics.write_bed(cage_df, cage_fn, compression = 'gzip')
		generators.append((chromatics.generate_average_signal_features, cage_fn))

	return generators




def generate_features(use_window, generators, infile, outfile):
	pairs_df = pd.read_csv(infile) # load EPI data

	if use_window == 1:
		regions = ["enhancer", "promoter", "window"]
		if "window_name" not in pairs_df.columns:
			pairs_df = insert_window(pairs_df)
	else:
		regions = ["enhancer", "promoter"]

	assert pairs_df.duplicated().sum() == 0
	training_df = chromatics.generate_training(pairs_df, regions, generators, chunk_size = 2**14, n_jobs = 1)

	# save
	training_df.to_csv(outfile, index=False)



# To save memory, split the input dataframe in advance.
def data_split(infile, n_split = 1):
	df = pd.read_csv(infile)
	cnt = math.ceil(len(df) / n_split)
	start = 0
	end = start + cnt
	for i in range(n_split):
		sub_df = df[start:end]
		outfile = infile.replace(".csv", f"_{i}.csv")
		sub_df.to_csv(outfile, index=False)
		start = end
		end = min(start + cnt, len(df))

def main(infile, outfile, cell, use_window, data_split_size):
	generators = preprocess_features(cell)

	from pathlib import Path
	relative_path = Path(outfile)
	absolute_path = relative_path.resolve()
	parent_directory = absolute_path.parent
	os.makedirs(parent_directory, exist_ok=True)

	data_split(infile, data_split_size)
	df = pd.DataFrame()
	for i in range(data_split_size):
		print(f"{i} / {data_split_size}")
		generate_features(use_window, generators, infile.replace(".csv", f"_{i}.csv"), outfile.replace(".csv", f"_{i}.csv"))
		sub_df = pd.read_csv(outfile.replace(".csv", f"_{i}.csv"))
		df = pd.concat([df, sub_df])
		os.remove(infile.replace(".csv", f"_{i}.csv"))
		os.remove(outfile.replace(".csv", f"_{i}.csv"))
	df.to_csv(outfile, index=False)

def parse_arguments():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# p.add_argument("--use_config", type=int, default=0)
	p.add_argument("-i", "--infile", help="input file path")	
	p.add_argument("-o", "--outfile", help="output file path")
	p.add_argument("--cell", required=True, help="cell type")
	p.add_argument("--use_window", action="store_true")
	p.add_argument("--data_split_size", type=int, default=20)
	args = p.parse_args()
	return args

if __name__ == "__main__":
	args = parse_arguments()
	main(args.infile, args.outfile, args.cell, args.use_window, args.data_split_size)

	


		

		

