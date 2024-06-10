# CarbonSense / EcoPerceiver Experiment Code

Here we provide the code for replicating the experiments from the CarbonSense paper. Much of the distributed code was adapted from [Meta AI's MAE repository](https://github.com/facebookresearch/mae/) and therefore carries the same CC 4.0 non-commercial license.

## Replicating Experiments
Replication can take two forms:
1. Using our model checkpoints for inference (short)
2. Training the model from scratch using our configuration and seeds (long)

We will cover both in this document

## Running Inference

### EcoPerceiver
Inference requires a run directory with model checkpoints. If you are replicating the experiments of our paper, you can download our [run directory from Zenodo](https://zenodo.org/records/11493138). Then you can set up the run directory with:

```
mkdir runs
mv ~/Downloads/ecoperceiver_run_directory.tar runs/
cd runs && tar xvf ecoperceiver_run_directory.tar && cd ..
```

If you are running inference on your own experiments, then run directories will be created via our training scripts.

Inference can be carried out locally, or on a cluster with Jupyter access. It is performed by running the `inference_ecoperceiver.ipynb` notebook. In the second cell, it is important to specify the run name, and the seed subdirectories / checkpoints for each; the checkpoint to use will be the one with the lowest validation score for that seed. The default values in this notebook are set for the main experiment from our paper, and no further modification is needed.

This script can take several hours per seed if run locally (or less if your GPU supports bfloat16), but it is tractable. At the end, each seed subdirectory will have a `.csv` file containing all the inferred values for each site and timestep (as well as ground truth values).

### XGBoost
XGBoost training and inference all happens in the same notebook: `inference_xgboost.ipynb`. It notebook performs the following actions:
- Evaluates train / test split from a config file
- Formats train / test data for tabular processing and saves it to a `.csv` in the run directory
- Performs hyperparameter search via cross-validation on the train set
- Trains a model with optimal parameters
- Runs inference with that model on the test set and saves the inferred values to the run directory

Note that this requires a run directory as well as the CarbonSense dataset which can be downloaded [here](https://zenodo.org/records/11403428/files/carbonsense.tar?download=1). The script will shortcircuit if it detects that the model has already been trained, so if you wish to run this all from scratch, you can delete the `xgboost` subdirectory in the run directory, and then rerun the notebook.

**If you're replicating the results of the EcoPerceiver paper** you can opt to use the XGBoost inference values that are already present in our run directory. This way, you won't need to download the whole dataset if you so choose. In this case, move straight on to the evaluation script.

### Evaluation
We include `inference_analysis.ipynb` for analyzing model outputs from both EcoPerceiver and XGBoost. This notebook is more free-form than the others, and users are encouraged to play with the data in order to visualize it in a way that makes sense for them. Metric functions are provided for RMSE and NSE, and the code demonstrates how to partition results by ecosystem type for a more detailed model analysis.


## Training from scratch
The included `config.yml` contains the parameters used in our baseline model experiments. We ran our experiments on the Compute Canada [Narval](https://docs.alliancecan.ca/wiki/Narval/en) cluster which contains compute nodes with 4xA100 GPUs. We use a single node with 4 processes, each using a single GPU. Use of other clusters is possible, but may require some modification of the submission script (especially for non-SLURM clusters). We do not recommend reproducing the training portion of our experiments on a local machine due to compute limitations, but will provide the instructions anyway. Local training is best used for smaller scale experiments.

### Training - Distributed
The following steps should reproduce our training results:
1. Clone this repository and load `../requirements.txt` into your environment manager of choice (CC uses `venv`)
2. Download the [CarbonSense tar file from Zenodo](https://zenodo.org/records/11403428/files/carbonsense.tar?download=1) and transfer it to the cluster
3. Create a symlink to the directory with the tar file using `ln -s path/to/directory/with/tar data`
4. Create a `runs` directory, or a symlink titled `runs` pointing to a run directory
5. Use the submission script with `python submit_distributed.py --n_nodes 1 --gpus_per_node 4 --hours 12 --prefix reproduction --seed 0`. You may need to modify parts of the submission script depending on your particular cluster.
6. Repeat the above experiment using seeds `[0, 10, 20, ..., 90]`

Depending on the architecture of the cluster you use, runs may need to be resubmitted until the models converge. We use tensorboard to examine validation batch losses for model selection. Note that while we attempt to control random number generators at every point in our training script, there may be inherent differences in the libraries of various clusters. Luckily, the variance of our results was relatively low across multiple experiments, so the results should be in the right ballpark.

### Training - Local
1. Clone this repository and load add the dependencies to your environment with `pip install -r ../requirements.txt`
2. Download the [CarbonSense tar file from Zenodo](https://zenodo.org/records/11403428/files/carbonsense.tar?download=1) and place it in a data directory (can be anywhere on your computer)
3. Create a symlink to the directory with the tar file using `ln -s path/to/directory/with/tar data`
4. Unzip the archive:
```
cd data
tar xvf carbonsense.tar
cd ..
```

5. Run experiments with `python run_local.py --autoresume --seed 0`
6. Repeat the above experiment using seeds `[0, 10, 20, ..., 90]`

Note that this will require substantial resources if replicating the original experiment, due to all data being loaded into memory. High resource requirements can be mitigated by modifying the `batch_size` and `num_workers` fields in `config.yml`, at the cost of a much longer runtime. Again, if replicating the original EcoPerceiver experiment from our paper, it is best done on a cluster.

