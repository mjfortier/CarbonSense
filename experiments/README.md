# CarbonSense / EcoPerceiver Experiment Code

Here we provide the code for replicating the experiments from the CarbonSense paper. Much of the distributed code was modified from [Meta AI's MAE repository](https://github.com/facebookresearch/mae/) and therefore carries the same CC 4.0 non-commercial license.

## Replicating Experiments
The included `config.yml` contains the parameters used in our baseline model experiments. We ran our experiments on the Compute Canada [Narval](https://docs.alliancecan.ca/wiki/Narval/en) cluster which contains compute nodes with 4xA100 GPUs. We use a single node with 4 processes, each using a single GPU. Use of other clusters is possible, but may require some modification of the submission script (especially for non-SLURM clusters). We do not recommend reproducing our experiments on a local machine due to compute limitations, but have provided a script for researchers looking to run smaller scale experiments (such as single ecosystem type).

The following steps should reproduce our results:
1. Clone this repository and load requirements.txt into your environment manager of choice (CC uses `venv`)
2. Download the CarbonSense tar file and transfer it to the cluster
3. Create a symlink to the directory with the tar file using `ln -s path/to/directory/with/tar data`
4. Create a `runs` directory, or a symlink titled `runs` pointing to a run directory
5. Use the submission script with `python submit_distributed.py --n_nodes 1 --gpus_per_node 4 --hours 12 --prefix reproduction --seed 0`
6. Repeat the above experiment using seeds `[0, 10, 20, ..., 90]`

Depending on the architecture of the cluster you use, runs may need to be resubmitted until the models converge. We use tensorboard to examine validation batch losses for model selection. Note that while we attempt to control random number generators at every point in our training script, there may be inherent differences in the libraries of various clusters. Luckily, the variance of our results was relatively low across multiple experiments, so the results should be in the right ballpark.

## Evaluation
Everyone will have their own preferred methods for running evaluation suites, but we have included ours for posterity's sake.

### EcoPerceiver Inference
Once all runs have completed, use `inference_ecoperceiver.ipynb` to run the best model from each seed over the test set. The results will be stored as `.csv` files which can be used for further analysis. Note that you need to specify which checkpoint to use for each seed.

### XGBoost Inference
We have also included our notebook for running the XGBoost model `inference_xgboost.ipynb`. Using this requires already having an EcoPerceiver run directory; XGBoost will take the train/test split from the config and run its own analysis. The auxilliary functions called in this notebook will also reformat the data into tabular form and save it as a `.csv`, so note that this can create large files in your run directory.
