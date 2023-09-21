#from __future__ import print_function
import os
import time
import datetime
import torch
import numpy as np
from utils import MLP5


"""
Script used for final evaluation on test set. Note that this script will produce
the predictions of the model; the results will later be assessed by the organizers
who have access to ground truth COT values for the corresponding predictions.
Also note that the loaded inputs are 12-dimensional (all spectral bands except B1).
"""

# Global vars
BASE_PATH_DATA = '../data/SDL2_SMHI_data'
BASE_PATH_LOG = '../log_smhi'
USE_GPU = False  # The code uses small-scale models, GPU doesn't seem to accelerate things actually
BATCH_SIZE = 100
MODEL_LOAD_PATH = None  # See examples on how to point to a specific model path below. None --> Model randomly initialized.

# 10 models trained on SMHI synthetic data, with 12 bands, 3% additive noise
MODEL_LOAD_PATH = ['../log_smhi/2023-08-10_10-33-44/model_it_2000000', '../log_smhi/2023-08-10_10-34-06/model_it_2000000', '../log_smhi/2023-08-10_10-34-18/model_it_2000000',
				   '../log_smhi/2023-08-10_10-34-28/model_it_2000000', '../log_smhi/2023-08-10_10-34-46/model_it_2000000', '../log_smhi/2023-08-10_10-34-58/model_it_2000000',
				   '../log_smhi/2023-08-10_10-35-09/model_it_2000000', '../log_smhi/2023-08-10_10-35-31/model_it_2000000', '../log_smhi/2023-08-10_10-35-52/model_it_2000000',
				   '../log_smhi/2023-08-10_10-36-12/model_it_2000000']

# MODEL_LOAD_PATH must be a list of model paths
if not isinstance(MODEL_LOAD_PATH, list):
	MODEL_LOAD_PATH = [MODEL_LOAD_PATH]

# Specify model input and output dimensions
input_dim = 12
output_dim = 1

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)

# Load test set
testset = np.load(os.path.join(BASE_PATH_DATA, 'testset_smhi_12bands.npy'))
nbr_examples = testset.shape[0]
assert nbr_examples % BATCH_SIZE == 0

# Setup prediction model
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
models = []
for model_load_path in MODEL_LOAD_PATH:  # Ensemble approach if len(MODEL_LOAD_PATH) > 1
	model = MLP5(input_dim, output_dim, apply_relu=True)
	if model_load_path is not None:
		model.load_state_dict(torch.load(model_load_path, map_location=device))
	model.to(device)
	model.eval()
	models.append(model)

# Normalize input data based on training set mean-std statistics
means = np.array([0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])
stds = np.array([0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])
inputs = testset
inputs = (inputs - means) / stds

# Convert to Torch
inputs = torch.Tensor(inputs).to(device)
	
print("Running evaluation on test set...")
it = 0
all_preds = torch.zeros(nbr_examples)
while True:
	
	# Break evaluation loop at end of epoch
	if (it + 1) * BATCH_SIZE >= nbr_examples:
		break
	
	# Do prediction for current batch
	preds = 0
	for model in models:
		curr_pred = model(inputs[it * BATCH_SIZE : (it + 1) * BATCH_SIZE, :]) / len(models)
		preds += curr_pred

	# Track statistics
	if it % 100 == 0:
		print("Batch: %d / %d" % (it, nbr_examples // BATCH_SIZE))
		
	all_preds[it * BATCH_SIZE : (it + 1) * BATCH_SIZE] = preds[:, 0]
	it += 1

# NOTE: This script must save a numpy array of shape (120000,) in the end. This is done already
# by default in this script, but in case you change anything, ensure the format is preserved.
np.save('smhi_preds.npy', all_preds.cpu().detach().numpy())
print("Done with test set evaluation! You can now provide the file 'smhi_preds.npy' to the organizers.")
