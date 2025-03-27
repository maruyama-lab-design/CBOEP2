import cross_validation
import os, sys

# pos_labels = ["BENGI-P"] # , "TargetFinderData-P"]
# neg_labels = [ ["retainedBENGI-N-1", # "removedBENGI-N-1", "CBMF-N-1", "CBGS-N-1"
#                 ], 
#                # ["retainedTargetFinderData-N", "CBMF-N", "CBGS-N"]
#                ]

# cell_types = [["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HMEC"], 
#               ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]]  # , "HUVEC"

def main(pos_label, neg_label):
    config="opt.json"
    gpu=0
    seed=2020

    use_mask=False
    use_weighted_bce=False
    use_dist_loss=False

    training_output_top_dir = "output"
    test_output_top_dir = "output"    

    test_labels = ["removedCommonTest-N", "retainedCommonTest-N"]

    training_cell_type = "GM12878"

    flag_of_training = True
    flag_of_testing = True

    cell_type_list = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HMEC"]


    if flag_of_training:
        model_dir = f"{training_output_top_dir}/{pos_label}_{neg_label}/{training_cell_type}/model"

        tensorboard_dir = f"{training_output_top_dir}/{pos_label}_{neg_label}/{training_cell_type}/tensorboard"
        train_data_path = f"../../input_to_EPI_predictor/{pos_label}_{neg_label}/{training_cell_type}.csv"

        print(f'Training {pos_label}_{neg_label} on {training_cell_type}...')
        cross_validation.do_train(train_data_path, model_dir, tensorboard_dir, 
            use_mask, use_weighted_bce, use_dist_loss, 
            config, gpu, seed)

    if flag_of_testing:
        for cell_type in cell_type_list:
            model_dir = f"{training_output_top_dir}/{pos_label}_{neg_label}/{training_cell_type}/model"
            for test_label in test_labels:
                test_data_path = f"../../input_to_EPI_predictor/{pos_label}_{test_label}/{cell_type}.csv"
                pred_dir = f"{test_output_top_dir}/{test_label}/{pos_label}_{neg_label}/{cell_type}"

                print(f'Testing {pos_label}_{neg_label} on {cell_type}...')
                cross_validation.do_test(test_data_path, pred_dir, model_dir, 
                    use_mask, 
                    config, gpu, seed)
                
if __name__ == "__main__":
    args = sys.argv
    pos_label = args[1]
    neg_label = args[2]
    main(pos_label, neg_label)
    
    # run the nexts independently:
    # python run_cv.py BENGI-P retainedBENGI-N-1
    # python run_cv.py BENGI-P removedBENGI-N-1
    # python run_cv.py BENGI-P CBMF-N-1
    # python run_cv.py BENGI-P CBGS-N-1