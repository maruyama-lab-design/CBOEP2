import torch
import matplotlib.pyplot as plt


# Load the file
# pt_file = torch.load("narrowPeak_GM12878_CTCF.500bp.pt")
histone_names = ["DNase", "H3K4me1", "H3K4me3", "H3K9me3", "H3K27ac", "H3K27me3", "H3K36me3"]
for histone_name in histone_names:
    # pt_file = torch.load(f"bigWig_GM12878_{histone_name}.500bp.pt")
    pt_file = torch.load("narrowPeak_GM12878_CTCF.500bp.pt")


    Y = pt_file["chr1"].tolist()
    minX = 0
    maxX = 99999999999
    # plt.figure()
    # plt.plot(list(range(minX, maxX)), Y[minX:maxX])
    # plt.savefig(f"CTCF_bin{minX}-{maxX}.png")
    with open(f"bedgraph/CTCF.bedgraph", "w") as f:
        for bin_i in range(minX, min(maxX, len(Y))):
            start = bin_i*500
            end = start+500
            f.write(
                f"chr1\t{start}\t{end}\t{Y[bin_i]}\n"
            )

