import make_fig as ra
import os
import glob

if __name__ == "__main__":
    predictors = ["TransEPI", "TargetFinder"] 
    # predictors = ["TransEPI"] 

    cell_types = [["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"],
                  ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]]

    # pos_labels = ["BENGI-P", "TargetFinderData-P"]
    pos_labels = ["BENGI-P"]

    neg_labels = [ 
        [ "retainedBENGI-N-1", "removedBENGI-N-1", "CBMF-N-1", "CBGS-N-1"], 
               # ["retainedTargetFinderData-N", "CBMF-N", "CBGS-N"]
               ]
    

    labels = [
        [ "BENGI-N(retained)", "BENGI-N(removed)", "CBMF-N", "CBGS-N"], 
        ["TargetFinderData(retained)", "CBMF", "CBGS"]]
    
    test_labels = ["removedCommonTest-N", "retainedCommonTest-N"]

    # graph-specific part.
    legend = {
        "fontsize": 8,
        "loc": "upper center",
        "ncol": 3
    }


    prediction_dir = 'output'     

    for predictor_index, predictor in enumerate(predictors): 
        indirs = []
        for pos_label_index, pos_label in enumerate(pos_labels):  # ["BENGI-P"]
            for test_label in test_labels:                        
                print(f"predictor: {predictor}")
                print(f"pos_label: {pos_label}")
                print(f"test_label: {test_label}")

                # Make input paths. 
                for neg_label in neg_labels[pos_label_index]:      
                    indirs.append(os.path.join('..', '..', 'EPI_predictor', f'{predictor}', f'{prediction_dir}', f'{test_label}', f'{pos_label}_{neg_label}'))

 
                colors = [['#1E90FF',  # (Bright Blue)
                        '#FF8C00',  # (Vivid Orange)
                        '#32CD32',  # (Bright Green)
                        '#DC143C']]  # (Deep Red)


                # Here, task-specific. 
                for metric in ["Balanced accuracy", "Specificity", "Recall", "AUC", "AUPR", "MCC", "F1", "Precision"]:
                    outfile = f"{prediction_dir}_{predictor}_{test_label}/{pos_label}/bar/{metric.replace(' ', '-')}.pdf"
                    os.makedirs(os.path.dirname(outfile), exist_ok=True)
                    ra.make_result_bar(
                        indirs=indirs,
                        labels=labels[pos_label_index],
                        # title=f'{metric} on {pos_label} by {predictor}',
                        colors=colors[pos_label_index],
                        outfile=outfile,
                        metric=metric,
                        test_cells=cell_types[predictor_index], # cell_6,  
                        n_fold=5,
                        legend=legend)
                    

                ### PR curve.
                marker_by_threshold = {
                    0.05: "^", 0.5: "x", 0.95: "v",
                }
                ylim = (0.0, 1.0)
                folds = ["combined"]
                for test_cell in cell_types[predictor_index]: 
                    title = ""
                    
           
                    outfile = f"{prediction_dir}_{predictor}_{test_label}/{pos_label}/curve/{test_cell}.pdf"
                    ra.make_pr_curve(
                        indirs=indirs,
                        labels=labels[pos_label_index],
                        colors=colors[pos_label_index],
                        folds=folds,
                        title=title,
                        outfile=outfile,
                        test_cell=test_cell,
                        marker_by_threshold=marker_by_threshold,
                        ylim=ylim
                    )

                ### line graph.
                prob_start = 0
                prob_end = 1.0
                n_bins = 20
                accum = False
 
                markers = [
                    "o",
                    "^",
                    "s",
                    "x",
                ]
                ylim = [-1, 1]
                custom_bin_list = [0.0, 0.05, 0.5, 0.95, 1.0]
                for test_cell in cell_types[predictor_index]:

                    outfile = f"{prediction_dir}_{predictor}_{test_label}/{pos_label}/line/{test_cell}.pdf"

                    ra.make_prob_line_graph(
                        indirs = indirs, 
                        labels = labels[pos_label_index],
                        colors=colors[pos_label_index],
                        markers=markers,
                        prob_range=[prob_start, prob_end],
                        n_bins=n_bins,
                        custom_bin_list=custom_bin_list,
                        accumulate=accum,
                        ylim=ylim,
                        outfile= outfile,
                        folds=["combined"],
                        test_cell=test_cell
                    )
