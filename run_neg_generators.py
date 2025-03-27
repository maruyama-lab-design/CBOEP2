import cbgs
import cbmf

datasets = ["BENGI-P_removedBENGI-N-1"] # , "TargetFinderData-P_unfilteredTargetFinderData-N"]

# cell_types = [["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HMEC"], 
#               ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK", "HUVEC"]]
cell_types = [["GM12878"], 
              ["GM12878"]]



out_to_dir = "input_to_EPI_predictor"

dmin =   0 # 10000
dmax = 2500000
beta = 1.0                    # 0.01               
iterations = 150000 # 500000

# alphas = [4.7, 4.0, 3.0, 2.0, 1.0]
# Ts = [20, 15, 10, 5, 5]
alphas = [1]
Ts = [5]

for alpha, T in zip(alphas, Ts):
    print(f"alpha: {alpha}, T: {T}")
    for data_index, dataset in enumerate(datasets):
        pos_data_label = dataset.split("_")[0]

        pos_neg_pair = f"{pos_data_label}_CBGS-N-{alpha}" 
        for cell_type in cell_types[data_index]:
            print(f"Processing {cell_type}")
            cbgs.CBGS(input=f"input_to_EPI_predictor/{dataset}/{cell_type}.csv", 
                    outdir=f"{out_to_dir}/{pos_neg_pair}",
                    cell_type=cell_type,
                    dmin=dmin,
                    dmax=dmax,
                    alpha=alpha,
                    beta=beta,
                    T=T, 
                    iterations=iterations,
                    concat=True,
                    fig=f"{out_to_dir}/cbgs_fig/{pos_neg_pair}/{cell_type}.png")
            
        
        pos_neg_pair = f"{pos_data_label}_CBMF-N-1" 
        for cell_type in cell_types[data_index]:
            cbmf.CBMF(input=f"input_to_EPI_predictor/{dataset}/{cell_type}.csv", 
                    outdir=f"{out_to_dir}/{pos_neg_pair}",
                    cell_type=cell_type,
                    dmin=dmin,
                    dmax=dmax,
                    alpha=alpha,
                    concat=True)