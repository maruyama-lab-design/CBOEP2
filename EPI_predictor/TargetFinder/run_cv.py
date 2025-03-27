import cross_validation

pos_labels = ["BENGI-P"] # , "TargetFinderData-P"]
neg_labels = [["retainedBENGI-N-1", "removedBENGI-N-1", "CBMF-N-1", "CBGS-N-1"], 
              ["retainedTargetFinderData-N", "CBMF-N", "CBGS-N"]]
cell_types = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"] 

for pos_label_index, pos_label in enumerate(pos_labels):
    for neg_label in neg_labels[pos_label_index]:
        for test_label in ["removedCommonTest-N", "retainedCommonTest-N"]:
            for cell_type in cell_types:
                cross_validation.main(
                    train_EPI = f'input_to_TargetFinder/{pos_label}_{neg_label}/GM12878.csv', 
                    test_EPI = f'input_to_TargetFinder/{pos_label}_{test_label}/{cell_type}.csv', 
                    # pred_dir = f'output/{pos_label}_{neg_label}/{cell_type}.csv',
                    pred_dir = f'output/{test_label}/{pos_label}_{neg_label}/{cell_type}',
                    use_window=True)