import pandas as pd 
import os
import sys

def filter_negative_EPIs(infile, outfile):
    print(infile)
    df = pd.read_csv(infile, sep=",")
    enh_set = set(df[df["label"] == 1]["enhancer_name"].to_list())
    prm_set = set(df[df["label"] == 1]["promoter_name"].to_list())
    print(f"positive: {len(enh_set)}")

    drop_i = df.query("enhancer_name not in @enh_set | promoter_name not in @prm_set").index
    print(drop_i)
    out_df = df.drop(drop_i, axis=0)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    out_df.to_csv(outfile, index=False, sep=",")

    print(f"positive: {len(out_df[out_df['label']==1])}")
    print(f"negative: {len(out_df[out_df['label']==0])}")

if __name__ == "__main__":
    args = sys.argv

    if args[1] == "BENGI":
        cell_types = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HMEC"]
        for cell_type in cell_types:
            infile = os.path.join(os.path.dirname(__file__), "..", "input_to_EPI_predictor", "_BENGI-P_retainedBENGI-N", f"{cell_type}.csv")
            outfile = os.path.join(os.path.dirname(__file__), "..", "input_to_EPI_predictor", "_BENGI-P_removedBENGI-N", f"{cell_type}.csv")
            filter_negative_EPIs(infile, outfile)

            # if cell_type == "GM12878":
            #     for alpha in [1, 2]:
            #         infile = os.path.join(os.path.dirname(__file__), "..", "input_to_EPI_predictor", "_BENGI-P_removedBENGI-N", f"{cell_type}.csv")
            #         outfile = os.path.join(os.path.dirname(__file__), "..", "input_to_EPI_predictor", f"BENGI-P_BENGI-N-{alpha}", f"{cell_type}.csv")
            #         down_sample(infile, outfile, alpha)

    elif args[1] == "TargetFinderData":
        cell_types = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HUVEC"]
        for cell_type in cell_types:
            infile = os.path.join(os.path.dirname(__file__), "..", "input_to_EPI_predictor", "TargetFinderData-P_retainedTargetFinderData-N", f"{cell_type}.csv")
            outfile = os.path.join(os.path.dirname(__file__), "..", "input_to_EPI_predictor", "TargetFinderData-P_removedTargetFinderData-N", f"{cell_type}.csv")
            filter_negative_EPIs(infile, outfile)
    else:
        print("Please specify the dataset to filter: BENGI or TargetFinderData")