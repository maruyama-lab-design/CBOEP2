from operator import index
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import argparse


def add_value_label(x_list,y_list):
    for i in range(1, len(x_list)+1):
        plt.text(i,y_list[i-1],y_list[i-1])


def set_fontsize(axis_Max):
	x1, y1, x2, y2 = 100, 9, 625, 2.5
	a = (y2 - y1) / (x2 - x1)
	x = axis_Max * axis_Max
	x = max(x, 100)
	x = min(x, 625)
	y = a * (x - x1) + y1
	return y

def make_heatMap(VV, size, outfile, regionType=""):
	fontsize = set_fontsize(size)
	# print(fontsize)
	fontsize = 1.3

	data = VV[:size+1, :size+1] 
	# The matrix, data, is divided into three parts: data_upper, data_lower, and data_middle. 
	# These three parts are separated by the lines y = 2x and y = 0.5x.
	data_upper = np.zeros_like(data)
	data_lower = np.zeros_like(data)
	data_middle = np.zeros_like(data)
	for j in range(size+1):
		for i in range(size+1):
			if j > 2 * i:
				data_upper[i][j] = data[i][j]
			elif j < 0.5 * i:
				data_lower[i][j] = data[i][j]
			else:
				data_middle[i][j] = data[i][j]

	# Making masks for the three parts. 
	mask_upper = np.full_like(data_upper, False, dtype=bool)
	mask_upper[np.where(data_upper==0)] = True   
	# By the above code, the following warning is raised:
	# /home/om/miniconda3/envs/py312/lib/python3.12/site-packages/seaborn/matrix.py:202: RuntimeWarning: All-NaN slice encountered
	#   vmin = np.nanmin(calc_data)
	# /home/om/miniconda3/envs/py312/lib/python3.12/site-packages/seaborn/matrix.py:207: RuntimeWarning: All-NaN slice encountered
	#   vmax = np.nanmax(calc_data)
	 
	mask_lower = np.full_like(data_lower, False, dtype=bool)
	mask_lower[np.where(data_lower==0)] = True

	mask_middle = np.full_like(data_middle, False, dtype=bool)
	mask_middle[np.where(data_middle==0)] = True

	# for j in range(size+1):
	# 	for i in range(size+1):
	# 		if not (j > 2 * i):
	# 			mask_upper[i][j] = True
	# 		if not (j < 0.5 * i):
	# 			mask_lower[i][j] = True
	# 		if j > 2 * i or j < 0.5 * i:
	# 			mask_middle[i][j] = True


	plt.figure(figsize=(21, 21))
	fig, ax = plt.subplots()
	# plt.plot([0, 48], [0, 24],color="gray", zorder=1, linestyle="dashed", linewidth=0.5)
	plt.plot([0, 96], [0, 48],color="gray", zorder=1, linestyle=":", linewidth=0.5)
	plt.plot([0, 60], [0, 120],color="gray", zorder=1, linestyle=":", linewidth=0.5)
	# sns.heatmap(
	# 	data, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"green"},
	# 	fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask, cbar = False, alpha=0
	# )
	sns.heatmap(
		data_upper, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"blue"},
		fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask_upper, cbar = False, alpha=0
	)
	sns.heatmap(
		data_lower, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"red"},
		fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask_lower, cbar = False, alpha=0
	)
	sns.heatmap(
		data_middle, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"green"},
		fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask_middle, cbar = False, alpha=0
	)
	for i in range(size+1): # ruled line
		plt.plot([i, i], [0, size+1], color="black", zorder=2, linewidth=0.1)
		plt.plot([0, size+1], [i, i], color="black", zorder=2, linewidth=0.1)

	ax.invert_yaxis()
	ax.set_xlabel(f"Positive {regionType} Interactions") 
	ax.set_ylabel(f"Negative {regionType} Interactions")
	plt.setp(ax.get_xticklabels(), fontsize=6, rotation=0)
	plt.setp(ax.get_yticklabels(), fontsize=6)
	os.makedirs(os.path.dirname(outfile), exist_ok=True)
	plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
	plt.close('all')


def make_PosNeg_matrix(indir, outdir, cell, size=24):

	df = pd.read_csv(os.path.join(indir, f"{cell}.csv"))
	for regionType in ["enhancer", "promoter"]:
		PosNeg_cnt = np.zeros((1000, 1000), dtype="int64") # 大きめに用意

		for chrom, subdf in df.groupby("enhancer_chrom"):
			PosNeg_cnt_by_chrom = np.zeros((1000, 1000), dtype="int64") # 大きめに用意
			for regionName, subsubdf in subdf.groupby(regionType + "_name"):
				posCnt = len(subsubdf[subsubdf["label"] == 1])
				negCnt = len(subsubdf[subsubdf["label"] == 0])

				PosNeg_cnt_by_chrom[negCnt][posCnt] += negCnt + posCnt

			PosNeg_cnt += PosNeg_cnt_by_chrom
			outfile = os.path.join(outdir,  f"{cell}_{regionType}.pdf")
		make_heatMap(PosNeg_cnt, size, outfile, regionType=regionType.capitalize())

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="NA")
	parser.add_argument("--indir", help="")
	parser.add_argument("--outdir", help="")
	parser.add_argument("--cell", help="cell type", default="GM12878")
	parser.add_argument("--size", type=int, default=24)
	args = parser.parse_args()

	cell_7 = ["GM12878", "HeLa-S3", "HMEC", "HUVEC", "IMR90", "K562", "NHEK"]
	cell_6 = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
	cell_5 = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]
	cell_1 = ["GM12878"]

	size = 24 # 68 # 24 

	for data in ['_BENGI-P_retainedBENGI-N', 'BENGI-P_retainedBENGI-N-1', 'BENGI-P_removedBENGI-N-1', 'BENGI-P_CBMF-N-1', 'BENGI-P_CBGS-N-1']:
		indir = os.path.join('..', '..', 'input_to_EPI_predictor', data)
		outdir = os.path.join('PosNeg_matrix', data)
		os.makedirs(outdir, exist_ok=True)
		print(indir)
		for cell_type in cell_1:
			make_PosNeg_matrix(indir, outdir, cell_type, size)

