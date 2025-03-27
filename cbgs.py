import numpy as np
import os
import argparse
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

def get_distance(enh, prm):
	# region name must be "GM12878|chr16:88874-88924"
	# map(function, iterable) applies function to every item of iterable and returns a list of the results.
	enh_start, enh_end = map(int, enh.split(":")[1].split("-")) 
	prm_start, prm_end = map(int, prm.split(":")[1].split("-"))
	enh_pos = (enh_start + enh_end) // 2
	prm_pos = (prm_start + prm_end) // 2
	return abs(enh_pos - prm_pos)


def get_all_neg(df, pos_df, dmin, dmax):
	enhs, prms = set(df["enhancer_name"].to_list()), set(df["promoter_name"].to_list())
	all_neg = set()
	for enh in enhs:
		for prm in prms:
			dist = get_distance(enh, prm)
			if dist > dmax or dist < dmin:
				continue
			assert f"{enh}={prm}" not in all_neg
			all_neg.add(f"{enh}={prm}")
	
	for _, row in pos_df.iterrows(): # remove positive pairs from all negative candidates
		enh, prm = row["enhancer_name"], row["promoter_name"] 
		if f"{enh}={prm}" in all_neg:
			all_neg.remove(f"{enh}={prm}")
		# else:
		# 	print(get_distance(enh, prm))
	
	return all_neg


def get_region_freq_dict(pos_df, neg_cand_set):
	"""It is assumed that all enhancers and promoters of negative pairs also exist in positive pairs.
	Args:
		pos_df (pd.DataFrame): positive pairs
		neg_cand_set (set): negative pairs"""
	
	enh_freq, prm_freq = {}, {}
	for _, row in pos_df.iterrows():
		enh = row["enhancer_name"]
		prm = row["promoter_name"]

		if enh in enh_freq:
			enh_freq[enh]["+"] += 1
		else:
			enh_freq[enh] = {
				"+": 1,
				"-": 0
			}
			
		if prm in prm_freq:
			prm_freq[prm]["+"] += 1
		else:
			prm_freq[prm] = {
				"+": 1,
				"-": 0
			}
	for neg in neg_cand_set:
		enh, prm = neg.split("=")
		enh_freq[enh]["-"] += 1
		prm_freq[prm]["-"] += 1

	return enh_freq, prm_freq

def get_freq(org_df_by_chrom, pos_df, neg_cand_set):
    """
    Calculates frequency counts for enhancer and promoter occurrences 
    in positive and negative datasets.

    Parameters:
        org_df_by_chrom (DataFrame): Original DataFrame with enhancer and promoter names.
        pos_df (DataFrame): DataFrame containing positive examples.
        neg_cand_set (set): Set of negative candidates in "enh=prm" format.

    Returns:
        enh_freq (dict): Frequency of enhancers (+ and - counts).
        prm_freq (dict): Frequency of promoters (+ and - counts).
    """
    from collections import defaultdict

    # Initialize frequency dictionaries with default values.
    enh_freq = defaultdict(lambda: {"+": 0, "-": 0})
    prm_freq = defaultdict(lambda: {"+": 0, "-": 0})

    # Initialize keys from the original DataFrame.
    for row in org_df_by_chrom.itertuples(index=False):
        enh_freq[row.enhancer_name]  # Ensure default keys exist.
        prm_freq[row.promoter_name]  # Ensure default keys exist.

    # Update frequencies based on the positive dataset.
    for row in pos_df.itertuples(index=False):
        enh_freq[row.enhancer_name]["+"] += 1
        prm_freq[row.promoter_name]["+"] += 1

    # Update frequencies based on the negative candidates.
    for neg in neg_cand_set:
        try:
            enh, prm = neg.split("=")
            enh_freq[enh]["-"] += 1
            prm_freq[prm]["-"] += 1
        except ValueError:
            raise ValueError(f"Invalid format in neg_cand_set: {neg}. Expected 'enh=prm'.")

    return dict(enh_freq), dict(prm_freq)



def make_neg_df(neg_cand_set):
	labels = [0] * len(neg_cand_set)
	enhs = [x.split("=")[0] for x in neg_cand_set]
	prms = [x.split("=")[1] for x in neg_cand_set]
	enh_chorms = [x.split("|")[1].split(":")[0] for x in enhs]
	prm_chorms = [x.split("|")[1].split(":")[0] for x in prms]
	enh_starts = [int(x.split("|")[1].split(":")[1].split("-")[0]) for x in enhs]
	enh_ends = [int(x.split("|")[1].split(":")[1].split("-")[1]) for x in enhs]
	prm_starts = [int(x.split("|")[1].split(":")[1].split("-")[0]) for x in prms]
	prm_ends = [int(x.split("|")[1].split(":")[1].split("-")[1]) for x in prms]
	dists = [get_distance(enh, prm) for (enh, prm) in zip(enhs, prms)]
	neg_df = pd.DataFrame(
		columns=["label", "enhancer_distance_to_promoter",
		"enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name",
		"promoter_chrom", "promoter_start", "promoter_end", "promoter_name"
		]
	)
	neg_df["label"] = labels
	neg_df["label"] = neg_df["label"].astype(int)
	neg_df["enhancer_distance_to_promoter"] = dists
	neg_df["enhancer_distance_to_promoter"] = neg_df["enhancer_distance_to_promoter"].astype(int)
	neg_df["enhancer_chrom"] = enh_chorms
	neg_df["enhancer_start"] = enh_starts
	neg_df["enhancer_start"] = neg_df["enhancer_start"].astype(int)
	neg_df["enhancer_end"] = enh_ends
	neg_df["enhancer_end"] = neg_df["enhancer_end"].astype(int)
	neg_df["enhancer_name"] = enhs

	neg_df["promoter_chrom"] = prm_chorms
	neg_df["promoter_start"] = prm_starts
	neg_df["promoter_start"] = neg_df["promoter_start"].astype(int)
	neg_df["promoter_end"] = prm_ends
	neg_df["promoter_end"] = neg_df["promoter_end"].astype(int)
	neg_df["promoter_name"] = prms

	return neg_df


def calc_sub_score_by_freq(pos_freq, neg_freq):
	# x = ((pos_freq - neg_freq) / max([pos_freq, neg_freq])) ** 2
	x = (pos_freq - neg_freq)** 2
	return x

def calc_prob(T, enh_freq_cand_enh, prm_freq_cand_prm, offset, alpha, beta, pos_size, neg_size):
	x0 = calc_sub_score_by_freq(enh_freq_cand_enh["+"], enh_freq_cand_enh["-"] + offset)
	x0 += calc_sub_score_by_freq(prm_freq_cand_prm["+"], prm_freq_cand_prm["-"] + offset)
	
	x1 = beta * (alpha * pos_size - (neg_size + offset)) ** 2
	# print(f'x0, x1 = {x0}, {x1}')
	x = x0 + x1
	score = math.exp(-x/T)
	if score == 0.0:
		print(f"calc_prob retunrs 0.0: {x0}, {x1}")
	return score


def calc_score(core_score, alpha, beta, pos_size, neg_size):
	x =  core_score + beta * (alpha * pos_size - neg_size) ** 2
	return x


def CBGS(input, outdir, cell_type, dmin, dmax, alpha, beta, T, iterations, concat, fig):
	if dmax != "INF": # if dmax is not INF, convert to int
		dmax = int(dmax)

	os.makedirs(os.path.join(outdir), exist_ok=True)

	org_df = pd.read_csv(input)
	neg_training_df = pd.DataFrame() # negative training instances.

	plt.figure()

	for chrom, org_df_by_chrom in org_df.groupby("enhancer_chrom"):
		print(f"_____{chrom}_____")
		pos_df = org_df_by_chrom[org_df_by_chrom["label"]==1]
		pos_size = len(pos_df)
		target_neg_size = int(len(pos_df) * alpha)

		all_neg = get_all_neg(org_df_by_chrom, pos_df, dmin, dmax) # Note variable, "all_neg" is unchanged through the iteration.
		restricted_neg = get_all_neg(pos_df, pos_df, dmin, dmax)

		print(f"Possible negative size: {len(all_neg):8d}")
		print(f"Filtered negative size: {len(restricted_neg):8d}")
		print(f"Target   negative size: {target_neg_size:8d}")

		if target_neg_size >= len(all_neg):
			print(f"Target negative size > Possible negative size")
			# target_neg_size = len(all_neg)
			neg_cand_set = all_neg

			neg_training_df_by_chrom = make_neg_df(neg_cand_set)
			neg_training_df = pd.concat([neg_training_df, neg_training_df_by_chrom], axis=0)
			continue
		elif target_neg_size >= len(restricted_neg):
			print(f"Target negative size > Filtered negative size")

			neg_cand_set = set(random.sample(sorted(all_neg - restricted_neg), target_neg_size - len(restricted_neg)))	
			neg_cand_set = neg_cand_set.union(restricted_neg)

			neg_training_df_by_chrom = make_neg_df(neg_cand_set)
			neg_training_df = pd.concat([neg_training_df, neg_training_df_by_chrom], axis=0)
			continue

		else:
			neg_cand_set = set(random.sample(sorted(restricted_neg), target_neg_size))


		enh_freq, prm_freq = get_freq(pos_df, pos_df, neg_cand_set)

		core_score = 0
		for enh in list(enh_freq.keys()):
			core_score += calc_sub_score_by_freq(enh_freq[enh]["+"], enh_freq[enh]["-"])

		for prm in list(prm_freq.keys()):
			core_score += calc_sub_score_by_freq(prm_freq[prm]["+"], prm_freq[prm]["-"])

		score_log = np.zeros((iterations+1))
		score_log[0] = calc_score(core_score, alpha, beta, pos_size, len(neg_cand_set))

		min_socre = math.inf
		opt_neg_set = neg_cand_set

		for t in range(iterations):
			# pick one pair randomly
			# cand_neg = random.sample(sorted(all_neg), 1)[0]
			cand_neg = random.sample(sorted(restricted_neg), 1)[0]

			cand_enh, cand_prm = cand_neg.split("=")
			enh_freq_cand_enh = enh_freq[cand_enh]
			prm_freq_cand_prm = prm_freq[cand_prm]
			cur_neg_size = len(neg_cand_set)
			if cand_neg in neg_cand_set:
				offset = 0
				prob_1 = calc_prob(T, enh_freq_cand_enh, prm_freq_cand_prm, offset, alpha, beta, pos_size, cur_neg_size)
				offset = -1
				prob_0 = calc_prob(T, enh_freq_cand_enh, prm_freq_cand_prm, offset, alpha, beta, pos_size, cur_neg_size)
			else:
				offset = 1
				prob_1 = calc_prob(T, enh_freq_cand_enh, prm_freq_cand_prm, offset, alpha, beta, pos_size, cur_neg_size)
				offset = 0
				prob_0 = calc_prob(T, enh_freq_cand_enh, prm_freq_cand_prm, offset, alpha, beta, pos_size, cur_neg_size)

			try:
				prob = prob_1 / (prob_0 + prob_1)
			except:
				print(f"prob is set ot 0. prob_0, prob_1 = {prob_0}, {prob_1}")
				prob = 0.0


			if random.random() < prob:
				if cand_neg in neg_cand_set:
					pass
				else:
					neg_cand_set.add(cand_neg)
					core_score -= calc_sub_score_by_freq(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"])
					core_score -= calc_sub_score_by_freq(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"])
					enh_freq[cand_enh]["-"] += 1
					prm_freq[cand_prm]["-"] += 1
					core_score += calc_sub_score_by_freq(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"])
					core_score += calc_sub_score_by_freq(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"])
			else:
				if cand_neg in neg_cand_set:
					try:
						neg_cand_set.remove(cand_neg)
					except:
						print(f"{cand_neg} is not in neg_cand_set")
					core_score -= calc_sub_score_by_freq(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"])
					core_score -= calc_sub_score_by_freq(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"])
					enh_freq[cand_enh]["-"] -= 1
					prm_freq[cand_prm]["-"] -= 1
					core_score += calc_sub_score_by_freq(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"])
					core_score += calc_sub_score_by_freq(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"])
				else:
					pass
			cur_score = calc_score(core_score, alpha, beta, pos_size, len(neg_cand_set))
			score_log[t+1] = cur_score
			if cur_score < min_socre:
				min_socre = cur_score
				opt_neg_set = neg_cand_set.copy()

			if t % 1000 == 0:
				print(f"iteration: {t:6d}   {prob_0:.2e}  {prob_1:.2e}   {prob:.3f}     {score_log[t+1]:.1f}      {len(neg_cand_set):6d}  {target_neg_size:6d}  ({pos_size}) ")

		plt.plot(range(iterations+1), score_log, label=chrom)
		neg_training_df_by_chrom = make_neg_df(opt_neg_set)
		neg_training_df = pd.concat([neg_training_df, neg_training_df_by_chrom], axis=0)
		
	if concat:
		neg_training_df = pd.concat([org_df[org_df["label"]==1], neg_training_df], axis=0)
	neg_training_df.to_csv(os.path.join(outdir, cell_type + ".csv"), index=False)

	

	plt.xlabel("Sampling iteration")
	plt.ylabel("Log score")
	plt.legend(ncol=5, fontsize="small")

	if fig:
		fig = os.path.join(fig)
		os.makedirs(os.path.dirname(fig), exist_ok=True)
		plt.savefig(fig)

def parse_arguments():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	p.add_argument("-i", "--input", help="path to an input file")
	p.add_argument("-o", "--outdir", help="path to an output directory")
	p.add_argument("-c", "--cell_type", help="cell type name like GM12878 (used as a prefix of output file name)")
	p.add_argument("--dmin", type=int, default=0, help="minimum distance between enhancer and promoter")
	p.add_argument("--dmax", default=2500000, help="maximum distance between enhancer and promoter")
	p.add_argument("--alpha", type=float, default=1.0)
	p.add_argument("--beta", type=float, default=1.0)
	p.add_argument("-T", type=float, default=5.0, help="temperature parameter")
	p.add_argument("--iterations", type=float, default=40000, help="number of sampling iteration")
	p.add_argument("--concat", action="store_true", default=True, help="concatenate generated negatives and input positives")
	p.add_argument("--fig", default="", help="path to a log figure file")
	args = p.parse_args()
	return args

if __name__ == "__main__":
	# input = "input_to_EPI_predictor/BENGI-P_retainedBENGI-N/GM12878.csv"
	# outdir = "tmp"
	# cell_type = "GM12878"
	# dmin = 0                 
	# dmax = 2500000
	# alpha = 1.0             
	# beta = 1.0                    # 0.01               
	# T = 5.0           # 20     
	# iterations = 10000        # 500000 # 40000
	# concat = True
	# fig = "./tmp.png"
	# CBGS(input, outdir, cell_type, dmin, dmax, alpha, beta, T, iterations, concat, fig)

	args = parse_arguments()
	CBGS(input=args.input, 
	  outdir=args.outdir, 
	  cell_type=args.cell_type, 
	  dmin=args.dmin, 
	  dmax=args.dmax, 
	  alpha=args.alpha, 
	  beta=args.beta,
	  T=args.T, 
	  iterations=args.iterations, 
	  concat=args.concat, 
	  fig=args.fig)