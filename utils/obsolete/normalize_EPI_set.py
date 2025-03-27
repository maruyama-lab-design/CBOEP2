import pandas as pd 
import os

import random



def normalize_EPI_set(infile, outfile):
    df = pd.read_csv(infile)
    enh_set = set(df[df["label"] == 1]["enhancer_name"].to_list())
    prm_set = set(df[df["label"] == 1]["promoter_name"].to_list())

    drop_i = df.query("enhancer_name not in @enh_set | promoter_name not in @prm_set").index
    print(drop_i)
    out_df = df.drop(drop_i, axis=0)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    out_df.to_csv(outfile, index=False)

    print(f"positive: {len(out_df[out_df['label']==1])}")
    print(f"negative: {len(out_df[out_df['label']==0])}")



if __name__ == "__main__":
    infile = os.path.join(os.path.dirname(__file__), "..", "input_to_generator", "unfilteredBENGI", "HMEC.csv")
    outfile = os.path.join(os.path.dirname(__file__), "tmp", "normalized_BENGI", "test-HMEC.csv")
    normalize_EPI_set(infile, outfile)
