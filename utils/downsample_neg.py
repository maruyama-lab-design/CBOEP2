import os
import pandas as pd

cell_types = ["GM12878"] # , "HeLa-S3", "IMR90", "K562", "NHEK", "HMEC"]

for dataset in ["_BENGI-P_retainedBENGI-N", "_BENGI-P_removedBENGI-N"]: 

    for cell_type in cell_types:
        print(f"Processing {cell_type}")
        infile = os.path.join("..", "input_to_EPI_predictor", dataset, f"{cell_type}.csv")
        
        df = pd.read_csv(infile, sep=",")
        all_pos_df = df[df["label"]==1]
        all_neg_df = df[df["label"]==0]

        all_pos_size = len(all_pos_df)
        all_neg_size = len(all_neg_df)
        print(f"all_pos_size: {all_pos_size}, all_neg_size: {all_neg_size}")

        # alpha_max = all_neg_size//all_pos_size
        alpha_max = 1


        for alpha in range(alpha_max, 0, -1):
            print(f"alpha: {alpha}")
            print(f"all_pos_size: {all_pos_size}, all_neg_size: {all_neg_size}, sample_size: {all_pos_size*alpha}")
            sample_df = all_neg_df.sample(n=int(all_pos_size*alpha), random_state=2020)

            out_df = pd.concat([all_pos_df, sample_df], axis=0, ignore_index=True)

            outdir = os.path.join("..", "input_to_EPI_predictor", f"{dataset[1:]}-{alpha}")
            # outdir = os.path.join("..", "input_to_EPI_predictor", f"{dataset[1:]}")

            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, f"{cell_type}.csv")
            out_df.to_csv(outfile, index=False, sep=",")