# Generators of Negative enhancer-promoter pairs, CBMF (Class Balanced negative set by Maximum-Flow) and CBGS (Class Balanced negative set by Gibbs Sampling)

This repository contains files related to our methods, CBMF and CBGS, a methods for generating sets of negative enhancer-promoter interactions (EPIs) from positive EPIs. 
The implementations of the methods are `cbmf.py` and `cbgs.py` in the top directory. 

The preliminary work has been published in the preceedings of ACM-BCB 2023 [1]. 
The codes are publicly available at 
https://github.com/maruyama-lab-design/CBOEP/tree/main. 
We recommend to use the codes in this site instead of those of the preliminary work. 


# Preprocessing input datasets
1. Download the BENGI datasets from https://github.com/chenkenbio/TransEPI.
2. The data are imported, processed and saved into the directory, input_to_EPI_predictor/_BENGI-P_BENGI-N. 
```
   cd utils
   python import_bengi_data.py path_to_TransEPI-main/data/BENGI
```
3. The datasets get format change and are saved into the directory, input_to_EPI_predictor/_BENGI-P_retainedBENGI-N.
```
   python change_format.py
```
4. Filtering is appled. The resulting files are saved into the directory, input_to_EPI_predictor/_BENGI-P_removedBENGI-N.
```
python filter_negative_EPIs.py BENGI
```
5. Downsamping is applied. The resulting files are saved into the directories, input_to_EPI_predictor/BENGI-P_removedBENGI-N-1 and input_to_EPI_predictor/BENGI-P_retainedBENGI-N-1. 
```
python downsample_neg.py
```

# Making CBMF- and CBGS-generated negative training datasets from the BENGI dataset of GM12878
The output files are saved as input_to_EPI_predictor/BENGI-P_CBMF-N-1 and input_to_EPI_predictor/BENGI-P_CBGS-N-1. 
```
cd ..
python run_neg_generators.py
```

# Making common test negative sets
The output files are saved in the directories, input_to_EPI_predictor/BENGI-P_retainedCommonTest-N and input_to_EPI_predictor/BENGI-P_removedCommonTest-N. 
```
python run_make_common_test_negative_set.py
```

# Run TargetFinder
First, download https://github.com/shwhalen/targetfinder 
and extract the archive.
Then, download the top-level directories of GM12878, HUVEC, HeLa-S3, IMR90, K562, and NHEK to our directory:
EPI_predictor/TargetFinder/input_features.
Furthermore, put the directory, chromatics, into 
our directory, EPI_predictor/TargetFinder. 
These directories, 
EPI_predictor/TargetFinder/input_features and 
EPI_predictor/TargetFinder/chromatics 
are not included in this repository due to the total amount 
of files. 


Next, execute the following command (Running run_preprocess.py taks a few hours):
```
cd EPI_predictor/TargetFinder
python run_preprocess.py 
python run_cv.py
```

# Run TransEPI
First, download 
the files and directories in 
https://github.com/chenkenbio/TransEPI/tree/main/data/genomic_data 
and save them in the directory, EPI_predictor/TransEPI/input_features.

```
cd ../TransEPI
python run_cv.py BENGI-P retainedBENGI-N-1
python run_cv.py BENGI-P removedBENGI-N-1
python run_cv.py BENGI-P CBMF-N-1
python run_cv.py BENGI-P CBGS-N-1
```
Note that, when you run "python run_cv.py X Y" the path "../../input_to_EPI_predictor/X_Y/GM12878.csv" should exist. 




# Details

# File format of positive EPIs, given to negative EPI generators as input

It is a csv file with columns:

| Column | Description |
| :---: | --- |
| ```label``` | Numeric ```1``` for positive EPI, ```0``` for negative EPI |
| ```enhancer_distance_to_promoter``` | Distance between the enhancer and the promoter |
| ```enhancer_chrom``` | Chromosome number of the enhancer |
| ```enhancer_start``` | Start position of the enhancer |
| ```enhancer_end``` | End position of the enhancer |
| ```enhancer_name``` | Name of the enhancer, such as `GM12878\|chr16:88874-88924` |
| ```promoter_chrom``` | Chromosome number of the promoter |
| ```promoter_start``` | Start position of the promoter |
| ```promoter_end``` | End position of the promoter |
| ```promoter_name``` | Name of the promoter, such as `GM12878\|chr16:103009-103010`|


## CBMF Requirements
We have tested CBMF in the following environments.

| Library | Version |
| :---: | :---: |
|```python```|3.8.0|
| ```numpy``` |1.18.1|
| ```pandas``` |1.0.1|
| ```pulp``` | 2.8.0 |


## CBMF Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```-i``` ||Path to an input EPI dataset file.|
| ```-o``` ||Output directory.|
| ```-c``` ||Cell type name like GM12878 (just used as a prefix of output file name).|
| ```--dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--dmax``` |2,500,000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--concat``` |False|Whether or not to concatenate the CBMF negative set with an input positive set. If not given, only the CBMF negative set will be output.|



## CBMF Execution example
```  
python cbmf.py \
-i ./input_to_EPI_predictor/BENGI-P_removedBENGI-N-1/GM12878.csv \
-o ./output \
--dmax 2500000 \
--dmin 0 \
--concat
```


## CBGS Requirements

We have tested the work in the following environments.

| Library | Version |
| :---: | :---: |
|```python```|3.8.0|
| ```numpy``` |1.18.1|
| ```pandas``` |1.0.1|
| ```matplotlib``` | 3.2.2 |

## CBGS Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```-i``` ||Path to an input EPI dataset file.|
| ```-o``` ||Output directory.|
| ```-c``` ||Cell type name like GM12878 (just used as a prefix of output file name).|
| ```--dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--dmax``` |2,500,000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--beta``` |1.0|Coefficient of scoring function.|
|```--T```|5.0|Temperature parameter.|
|```--iteration```|10,000|Number of sampling iteration.|
| ```--concat``` |True|If given, the CBGS negative set is concatenated with the positive set given as input. If not given, only the CBGS negative set will be output.|
|```--fig```||Path to a log figure of the score function of the Gibbs sampling.|


## CBGS Execution example
```  
python cbgs.py \
-i ./input_to_EPI_predictor/BENGI-P_removedBENGI-N-1/GM12878.csv \
-o ./output \
--dmax 2500000 \
--dmin 0 \
--concat \
--fig ./input_to_EPI_predictor/BENGI-P_CBGS-N_test/GM12878.png
```


# Reference
[1]
Tsukasa K, Osamu M.
CBOEP: Generating negative enhancer-promoter interactions to train classifiers.
In Proceedings of the 14th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB ’23).
2023;Article 27:1–6.






