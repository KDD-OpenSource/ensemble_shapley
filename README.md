Code for our paper "On the efficient Explanation of Outlier Detection Ensembles through Shapley Values" (PAKDD 2024) introducing a quick way of calculating shapley values for outlier ensembles.

main.py trains many models. It requires a file training.npz containing a list of training samples and a file test.npz containing a list of testing samples (and their labels).
Here it trains isolation trees, while the file dean.py trains DEAN models.

Both can easily be parallelized, by running multiple instances of the the same code. And while not technically necesarry, adding different integers as arguments can help reduce disk usage.
The result are many files in a folder "results/". These are combined for further processing by onefile.py

The resulting file is either used by outlier_performance.py to calculate the anomaly score, or by get_shapleys.py to calculate shapley values, which are then used by draw to draw the heatmaps as shown in the paper



