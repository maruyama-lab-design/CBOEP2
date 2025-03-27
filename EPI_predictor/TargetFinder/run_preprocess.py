import preprocess

def make(pos_label, neg_labels, cell_types):
    for neg_label in neg_labels:
        for cell_type in cell_types:
            print(f"Processing {pos_label} {neg_label} {cell_type}")
            infile = f"../../input_to_EPI_predictor/{pos_label}_{neg_label}/{cell_type}.csv"
            outfile = f"./input_to_TargetFinder/{pos_label}_{neg_label}/{cell_type}.csv"
            preprocess.main(infile, outfile, cell_type, use_window=True, data_split_size=20)
            print(f"Finished {pos_label}_{neg_label}_{cell_type}")


pos_label = "BENGI-P"
neg_labels = [
    "retainedBENGI-N-1", 
    "removedBENGI-N-1", 
    "CBMF-N-1", 
    "CBGS-N-1"
    ]
cell_types = ["GM12878"]
make(pos_label, neg_labels, cell_types)


pos_label = "BENGI-P"
neg_labels = [ 
    "removedCommonTest-N", 
    "retainedCommonTest-N"]
cell_types = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"] # , "HMEC"]
make(pos_label, neg_labels, cell_types)

