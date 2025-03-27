# Modifications

In our script,
there are two major differences from the original TargetFinder.

One is that the script to add genomic features to EPI data is in ```generate_training.py``` in the TargetFinder repository, and we found a problem that some EPI data did not fit in memory, so we improved the script and replaced it with ```preprocess.py```.

Another is the addition of ```cross_validation.py``` to perform chromosome-wise cross validation using a gradient boosting decision tree.

# License

GNU General Public License v3.0 is used in https://github.com/shwhalen/targetfinder/blob/master/LICENSE.






