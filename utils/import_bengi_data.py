import os
import pandas as pd
import sys

from os.path import expanduser
home = expanduser("~")

def import_bengi_data(bengi_data_path):

    # Common settings:
    outdir = os.path.join('..', 'input_to_EPI_predictor', '_BENGI-P_BENGI-N')
    os.makedirs(outdir, exist_ok=True)
    columns = ["label", "enhancer_distance_to_promoter",
            "enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name",
            "promoter_chrom", "promoter_start", "promoter_end", "promoter_name"
            ]
    

    # cell types with multiple experiments
    cell_types = ['GM12878', 'HeLa-S3']
    list_list_filename = [['GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz', 'GM12878.HiC-Benchmark.v3.tsv.gz', 'GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv.gz'],
                ['HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz', 'HeLa.HiC-Benchmark.v3.tsv.gz', 'HeLa.RNAPII-ChIAPET-Benchmark.v3.tsv.gz']]
    for cell_type, filenames in zip(cell_types, list_list_filename):

        dfs = []
        for filename in filenames:
            dfs.append(pd.read_csv(os.path.join(bengi_data_path, filename), sep='\t', compression='gzip', header=None))
            print(dfs[-1].shape)
            # print(dfs[-1].head(1))

        df = pd.concat(dfs, axis=0, ignore_index=True)
        df.columns = columns

        print(f"duplication before: {df.duplicated().sum()}")
        df = df.drop_duplicates()
        print(f"duplication after: {df.duplicated().sum()}")

        # At this point, there can be multiple rows with the same enhancer-promoter pair but different labels (and distances).
        # change_format.py resolves the issue. 
        
        df = df.reset_index(drop=True) # This may be redundant because ignore_index=True of pd.concat. 

        # df_unique.sort_values(by=[2, 3, 4], inplace=True)
        df.to_csv(os.path.join(outdir, f'{cell_type}.csv'), sep=',', index=False)

    other_cell_types = ['HMEC.HiC-Benchmark.v3.tsv.gz',
        'IMR90.HiC-Benchmark.v3.tsv.gz',
        'K562.HiC-Benchmark.v3.tsv.gz',
        'NHEK.HiC-Benchmark.v3.tsv.gz']

    for filename in other_cell_types:
        df = pd.read_csv(os.path.join(bengi_data_path, filename), sep="\t", compression='gzip', header=None)
        print(df.shape)
        df.columns = columns

        print(f"duplication before: {df.duplicated().sum()}")
        df = df.drop_duplicates()
        print(f"duplication after: {df.duplicated().sum()}")    
        df = df.reset_index(drop=True) 

        cell_type = filename.split(".")[0]
        # df.to_csv(os.path.join(home, 'github', 'negative_interactor_generator', 'input_to_EPI_predictor', '_BENGI-P_BENGI-N', f"{cell_type}.csv"), index=False, sep=",")
        df.to_csv(os.path.join(outdir, f"{cell_type}.csv"), sep=',', index=False)


if __name__ == "__main__":
    # bengi_data_path = os.path.join(home, 'OneDrive', 'work', 'TransEPI', 'TransEPI-main', 'data', 'BENGI')
    args = sys.argv
    bengi_data_path = args[1]
    import_bengi_data(bengi_data_path)
    print("Done")