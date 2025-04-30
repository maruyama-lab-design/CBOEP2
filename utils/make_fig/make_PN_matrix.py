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

import matplotlib
matplotlib.use('Agg')


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

def make_heatMap_0(VV, matrix_size, outfile, regionType=""):
	fontsize = set_fontsize(matrix_size)
	print(fontsize)
	# fontsize = 1.3

	data = VV[:matrix_size+1, :matrix_size+1] 
	# The matrix, data, is divided into three parts: data_upper, data_lower, and data_middle. 
	# These three parts are separated by the lines y = 2x and y = 0.5x.
	data_upper = np.zeros_like(data)
	data_lower = np.zeros_like(data)
	data_middle = np.zeros_like(data)
	for j in range(matrix_size+1):
		for i in range(matrix_size+1):
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
	for i in range(matrix_size+1): # ruled line
		plt.plot([i, i], [0, matrix_size+1], color="black", zorder=2, linewidth=0.1)
		plt.plot([0, matrix_size+1], [i, i], color="black", zorder=2, linewidth=0.1)

	ax.invert_yaxis()
	ax.set_xlabel(f"Positive {regionType} Interactions") 
	ax.set_ylabel(f"Negative {regionType} Interactions")
	plt.setp(ax.get_xticklabels(), fontsize=6, rotation=0)
	plt.setp(ax.get_yticklabels(), fontsize=6)
	os.makedirs(os.path.dirname(outfile), exist_ok=True)
	plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
	plt.close('all')

def make_heatMap(VV, size, outfile, regionType=""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np

    # フォントサイズ（自動調整または固定値）
    fontsize = set_fontsize(size)
    fontsize = 6.0  # 読みやすい固定値に調整

    # サイズ制限して必要な部分を抽出
    # data = VV[:size+1, :size+1]
    data = VV[:size+1, :size+1]
    data_for_display = data.copy()                    # 表示用（元の整数）
    log_data = np.log1p(data.astype(np.float64))     # 背景色用（log(1 + x)）


    # 値がすべて0だと seaborn が警告を出すので非ゼロ部分で vmin/vmax を指定
    # nonzero_data = data[data > 0]
    # if nonzero_data.size > 0:
    #     vmin = np.min(nonzero_data)
    #     vmax = np.max(nonzero_data)
    # else:
    #     vmin, vmax = 0, 1  # fallback
    nonzero_data = log_data[np.isfinite(log_data)]
    if nonzero_data.size > 0:
        vmin = np.min(nonzero_data)
        vmax = np.max(nonzero_data)
    else:
        vmin, vmax = 0, 1


    # 図の作成
    plt.figure(figsize=(12, 10))  # サイズは調整可能
    # ax = sns.heatmap(
    #     data, annot=True, square=True, annot_kws={"fontsize": fontsize},
    #     fmt="d", cmap="Blues", linewidths=0.1, linecolor='black',
    #     vmin=vmin, vmax=vmax, cbar=True
    # )
    ax = sns.heatmap(
		log_data, annot=data_for_display, square=True,
		annot_kws={"fontsize": fontsize},
		fmt="d", cmap="Blues", linewidths=0.1, linecolor='black',
		vmin=vmin, vmax=vmax, cbar=True
	)


    # 領域分割線 y = 2x と y = 0.5x の描画（補助線として）
    plt.plot([0, size], [0, 2 * size], linestyle=":", color="red", linewidth=1.5, label="y=2x")
    plt.plot([0, size+1], [0, 0.5 * (size+2)], linestyle=":", color="green", linewidth=1.5, label="y=0.5x")
    # plt.legend(loc="upper right", fontsize=fontsize, frameon=False)

    # 軸設定
    ax.invert_yaxis()
    ax.set_xlabel(f"Positive {regionType} Interactions")
    ax.set_ylabel(f"Negative {regionType} Interactions")
    plt.setp(ax.get_xticklabels(), fontsize=6, rotation=0)
    plt.setp(ax.get_yticklabels(), fontsize=6)

    # 保存処理
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close('all')



def make_PosNeg_matrix(indir, outdir, cell, matrix_size=24):

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
		make_heatMap(PosNeg_cnt, matrix_size, outfile, regionType=regionType.capitalize())

def compute_PosNeg_matrix(indir, cell, regionType="promoter"):
    df = pd.read_csv(os.path.join(indir, f"{cell}.csv"))
    PosNeg_cnt = np.zeros((1000, 1000), dtype="int64")
    for chrom, subdf in df.groupby("enhancer_chrom"):
        for regionName, subsubdf in subdf.groupby(regionType + "_name"):
            posCnt = len(subsubdf[subsubdf["label"] == 1])
            negCnt = len(subsubdf[subsubdf["label"] == 0])
            PosNeg_cnt[negCnt][posCnt] += negCnt + posCnt
    return PosNeg_cnt

def make_subplot_heatmap(VV_list, labels, matrix_size, outfile, regionType="Promoter"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    fontsize = set_fontsize(matrix_size)
    # fontsize = 6.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'wspace': -0.3, 'hspace': 0.3})  # 2x2 subplot
    axs = axes.flatten()

    # カラースケール共有のために vmax を事前に計算
    all_log_data = []
    for VV in VV_list:
        data = VV[:matrix_size+1, :matrix_size+1]
        log_data = np.log1p(data.astype(np.float64))
        all_log_data.append(log_data)
    all_values = np.concatenate([d[np.isfinite(d)] for d in all_log_data])
    vmin = np.min(all_values)
    vmax = np.max(all_values)

    # ヒートマップを (0), (1), (2) に配置
    for i, (log_data, label, VV) in enumerate(zip(all_log_data, labels, VV_list)):
        data_for_display = VV[:matrix_size+1, :matrix_size+1]
        ax = axs[i]
        sns.heatmap(
            log_data, annot=data_for_display, square=True,
            annot_kws={"fontsize": fontsize}, fmt="d", cmap="Blues",
            linewidths=0.1, linecolor='black',
            vmin=vmin, vmax=vmax, cbar=False, ax=ax
        )
        ax.plot([0, matrix_size], [0, 2 * matrix_size], linestyle=":", color="red", linewidth=1.5)
        ax.plot([0, matrix_size+1], [0, 0.5 * (matrix_size+2)], linestyle=":", color="green", linewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(f"Positive {regionType}")
        if i % 2 == 0:
            ax.set_ylabel(f"Negative {regionType}")
        else:
            ax.set_ylabel("")
        ax.invert_yaxis()
        ax.tick_params(labelsize=6)

    # # カラーバーだけ (3) に追加（空のヒートマップ＋色バー）
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # ax_cb = axs[3]
    # ax_cb.axis('off')  # 軸非表示
    # norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax_cb, orientation='vertical', fraction=0.8)

    # plt.tight_layout()
    # os.makedirs(os.path.dirname(outfile), exist_ok=True)
    # plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
    # plt.close()

	# カラーバーだけ (3) に追加（空のヒートマップ＋色バー）
    ax_cb = axs[3]
    ax_cb.axis('off')  # 軸非表示
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_cb, orientation='vertical', fraction=0.8)

    # tight_layout は使わず、明示的に調整
    plt.subplots_adjust(
        left=0.07, right=0.95,
        top=0.93, bottom=0.08,
        wspace=0.25, hspace=0.25  # 横・縦の間隔
    )

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()




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

	matrix_size = 24 # 68 # 24 

	for data in ['_BENGI-P_retainedBENGI-N', 'BENGI-P_retainedBENGI-N-1', 'BENGI-P_removedBENGI-N-1', 'BENGI-P_CBMF-N-1', 'BENGI-P_CBGS-N-1']:
		indir = os.path.join('..', '..', 'input_to_EPI_predictor', data)
		outdir = os.path.join('PosNeg_matrix', data)
		os.makedirs(outdir, exist_ok=True)
		print(indir)
		for cell_type in cell_1:
			make_PosNeg_matrix(indir, outdir, cell_type, matrix_size)

	# subplot 表示対象
	selected_data = ['_BENGI-P_retainedBENGI-N', 'BENGI-P_CBMF-N-1', 'BENGI-P_CBGS-N-1']
	VV_list = []
	for data in selected_data:
		indir = os.path.join('..', '..', 'input_to_EPI_predictor', data)
		VV = compute_PosNeg_matrix(indir, cell_type, regionType="promoter")
		VV_list.append(VV)

	# labels = ["BENGI-generated positive and negative sets", "BENGI-generated positive set and CBMF-generated negative set", "BENGI-generated positive set and CBGS-generated negative set"]
	labels = ["(a) Positive and negative sets generated by BENGI", "(b) Positive set from BENGI and negative set from CBMF", "(c) Positive set from BENGI and negative set from CBGS"]
	subplot_outfile = os.path.join('PosNeg_matrix', 'combined', f'{cell_type}_promoter_subplot.pdf')
	make_subplot_heatmap(VV_list, labels, matrix_size, subplot_outfile, regionType="Promoter")

