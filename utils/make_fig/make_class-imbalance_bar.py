import make_fig as ra

if __name__ == "__main__":
    ra.make_dataset_score_bar(
        indirs=[
            '../../input_to_EPI_predictor/BENGI-P_unfilteredBENGI-N-1', 
            '../../input_to_EPI_predictor/BENGI-P_filteredBENGI-N-1', 
            '../../input_to_EPI_predictor/BENGI-P_CBMF-N-1', 
            '../../input_to_EPI_predictor/BENGI-P_CBGS-N-1',
        ],
        labels=[
            "unfilBENGI",
            "filBENGI",
            "CBMF",
            "CBGS"
        ],
        colors=[
        "#bf7fff", # 紫
        "#7fbfff", # 青
        "#ff7f7f", # 赤
        "#7fff7f" # 緑
        ],
        ylim=(0.0, 8.0),
        outfile=f"./class_imbalance_bar_graph.pdf",
    )