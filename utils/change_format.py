import os
import pandas as pd

"""Change format, remove duplicated rows and sample negative pairs from all negative candidates"""

cell_types = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HMEC"]
outdir = os.path.join("..", "input_to_EPI_predictor", "_BENGI-P_retainedBENGI-N")
os.makedirs(outdir, exist_ok=True)

for cell_type in cell_types:
    print(f"Processing {cell_type}")
    infile = os.path.join("..", "input_to_EPI_predictor", "_BENGI-P_BENGI-N", f"{cell_type}.csv")
    
    df = pd.read_csv(infile, sep=",")
    # Before: chr11:450279-450280|HeLa|ENSG00000174915.7|ENST00000308020.5|+
    # After:  HeLa|chr11:450279-450280
    df["enhancer_name"] = df["enhancer_name"].apply(lambda x: "|".join([x.split("|")[1], x.split("|")[0]])) 
    df["promoter_name"] = df["promoter_name"].apply(lambda x: "|".join([x.split("|")[1], x.split("|")[0]]))
    
    print(f"duplication before: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"duplication after: {df.duplicated().sum()}")

    # Datasets of different experiments in the same cell type are merged. 
    # As a result, the same enhancer-promoter pairs have different labels. 
    # In this case, we keep the positive label. 
    columns_except_label = df.columns.difference(["label", "enhancer_distance_to_promoter"]).tolist() # The column names except for "label" are stored in the list.
    print(columns_except_label)
    # columns_except_label = df.columns.difference(["label"]).tolist() # The column names except for "label" are stored in the list.
    result_df = df.sort_values(by="label", ascending=False)  # label=1 is prioritized

    duplicated_rows = result_df[result_df.duplicated(subset=columns_except_label, keep="first")]

    # Display duplicated rows. 
    print("Duplicated the first 10 rows:")
    print(duplicated_rows.head(10))

    print(f"duplication before: {result_df.shape}")
    df = result_df.drop_duplicates(subset=columns_except_label, keep="first") 
    print(f"duplication after : {df.shape}")

    pos_size = len(df[df["label"]==1])
    neg_size = len(df[df["label"]==0])

    print(f"pos_size: {pos_size}, neg_size: {neg_size}, ratio: {neg_size/pos_size}")

    df = df.reset_index(drop=True) 


    outfile = os.path.join(outdir, f"{cell_type}.csv")
    df.to_csv(outfile, index=False, sep=",")