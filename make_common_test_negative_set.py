import os
import pandas as pd
import cbgs

def make(input, out_path, pos_data_label, cell_type, dmin, dmax, concat):
	if dmax != "INF": # if dmax is not INF, convert to int
		dmax = int(dmax)

	org_df = pd.read_csv(input)

	for _neg_label in ['retained', 'removed']:
		neg_label = _neg_label + 'CommonTest-N'
		neg_test_df = pd.DataFrame() # negative test instances.

		for chrom, org_df_by_chrom in org_df.groupby("enhancer_chrom"):
			pos_df = org_df_by_chrom[org_df_by_chrom["label"]==1]
			
			if _neg_label == 'retained':
				all_neg = cbgs.get_all_neg(org_df_by_chrom, pos_df, dmin, dmax) 
			elif _neg_label == 'removed':
				all_neg = cbgs.get_all_neg(pos_df, pos_df, dmin, dmax) 
			else:
				raise ValueError(f"Unknown neg_label: {_neg_label}")

			neg_test_df_by_chrom = cbgs.make_neg_df(all_neg) # all negative candidates are selected as negative. The argument should be a dictionary.
			neg_test_df = pd.concat([neg_test_df, neg_test_df_by_chrom], axis=0)	
		if concat:
			df = pd.concat([org_df[org_df["label"]==1], neg_test_df], axis=0)
		outdir=f"{out_path}/{pos_data_label}_{neg_label}"
		os.makedirs(os.path.join(outdir), exist_ok=True)
		df.to_csv(os.path.join(outdir, cell_type + ".csv"), index=False)


if __name__ == "__main__":
	input = "input_to_EPI_predictor/BENGI-P_BENGI-N/GM12878.csv"
	outdir = "tmp"
	cell_type = "GM12878"
	dmin = 0                 
	dmax = 2500000        
	concat = True
	make(input, outdir, cell_type, dmin, dmax, concat)