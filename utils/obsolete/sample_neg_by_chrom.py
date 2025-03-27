import os
import pandas as pd

cell_types = ["GM12878"] # , "HeLa-S3", "IMR90", "K562", "NHEK", "HMEC"]
for cell_type in cell_types:
    print(f"Processing {cell_type}")
    infile = os.path.join("..", "input_to_EPI_predictor", "BENGI-P_BENGI-N", f"{cell_type}.csv")
    
    df = pd.read_csv(infile, sep=",")
    all_pos_df = df[df["label"]==1]
    all_neg_df = df[df["label"]==0]

    all_pos_size = len(all_pos_df)
    all_neg_size = len(all_neg_df)
    print(f"all_pos_size: {all_pos_size}, all_neg_size: {all_neg_size}")
    alpha_max = all_neg_size//all_pos_size


    for alpha in range(alpha_max, 0, -1):
        print(f"alpha: {alpha}")

        chrom_neg_dfs = []
        for chrom, df_by_chrom in df.groupby("enhancer_chrom"):
            print(f"_____{chrom}_____")
            pos_df = df_by_chrom[df_by_chrom["label"]==1]
            neg_df = df_by_chrom[df_by_chrom["label"]==0]
            chrom_pos_size = len(pos_df)
            # neg_size = len(neg_df)
            
            print(f"chrom: {chrom}, chrom_pos_size: {chrom_pos_size * alpha}, neg_size: {len(neg_df)}    {len(neg_df)  -  chrom_pos_size * alpha }")
            try:
                sample_df = neg_df.sample(n=chrom_pos_size*alpha, random_state=2020)
            except:
                sample_df = neg_df


            chrom_neg_dfs.append(sample_df)

        out_df = pd.concat([all_pos_df] + chrom_neg_dfs, axis=0, ignore_index=True)

        outdir = os.path.join("..", "input_to_EPI_predictor", f"BENGI-P_BENGI-N-{alpha}")
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"{cell_type}.csv")
        out_df.to_csv(outfile, index=False, sep=",")