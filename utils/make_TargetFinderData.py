import os
import pandas as pd
from os.path import expanduser
home = expanduser("~")


# Specify the path to targetfinder-master directory
input_path = os.path.join(home, "c", "Downloads", "targetfinder-master", "targetfinder-master", "paper", "targetfinder")

cell_types = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HUVEC"]

selected_columns = ["label", "enhancer_distance_to_promoter", "enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name", "promoter_chrom", "promoter_start", "promoter_end", "promoter_name"]

out_dir = os.path.join(home, 'github', 'negative_interactor_generator', 'input_to_EPI_predictor', 'TargetFinderData-P_TargetFinderData-N')
os.makedirs(out_dir, exist_ok=True)

for cell_type in cell_types:
    print(cell_type)
    df = pd.read_csv(os.path.join(input_path, f"{cell_type}", "output-epw/training.csv.gz"), compression='gzip')
    print(df.shape)
    df = df.drop_duplicates()
    print(df.shape)
    df = df[selected_columns]
    df.to_csv(os.path.join(out_dir, f"{cell_type}.csv"), index=False, sep=",")   