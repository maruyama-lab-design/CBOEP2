import make_common_test_negative_set

datasets = ["_BENGI-P_retainedBENGI-N"] # , "TargetFinderData-P_retainedTargetFinderData-N"]

cell_types = [["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HMEC"], 
              ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HUVEC"]]

out_path = "input_to_EPI_predictor"

dmin =   0 # 10000
dmax = 2500000   
concat=True

for data_index, dataset in enumerate(datasets):
    print(f"--- {dataset} ---")

    pos_data_label = dataset.split("_")[1] # _BENGI-P_

    for cell_type in cell_types[data_index]:
        print(f"Processing {cell_type}")
        make_common_test_negative_set.make(
            input=f"input_to_EPI_predictor/{dataset}/{cell_type}.csv", 
            # outdir=f"{out_to_dir}/{pos_neg_pair}",
            out_path = out_path, 
            pos_data_label = pos_data_label,
            cell_type=cell_type,
            dmin=dmin,
            dmax=dmax,
            concat=concat)
            