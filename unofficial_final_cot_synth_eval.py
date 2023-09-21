#from __future__ import print_function
import os
import sys
import time
import random
import datetime
from shutil import copyfile
import torch
import torch.nn as nn
import numpy as np
from utils import StatCollector, MLP5


"""
Unofficial script used for test set evaluation. Will not work for 
hackathon participants since they don't have access to test set ground
truth data.

Only for internal for the organizers!
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
copyfile("unofficial_final_cot_synth_eval.py", os.path.join(log_dir, "unofficial_final_cot_synth_eval.py"))

# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Load test set
testset = np.load(os.path.join(BASE_PATH_DATA, 'testset_smhi_12bands.npy'))
testset_with_gt = np.load(os.path.join(BASE_PATH_DATA, 'testset_smhi_12bands_with_gt.npy'))
gts = testset_with_gt[:, -1]
nbr_examples = len(gts)
assert nbr_examples % BATCH_SIZE == 0

# Normalize regressor data and convert to Torch
gt_max = 50  # COT provided in the [0, 50]-range
gts /= gt_max
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
gts = torch.Tensor(gts).to(device)

# Setup prediction model and loss
models = []
for model_load_path in MODEL_LOAD_PATH:  # Ensemble approach if len(MODEL_LOAD_PATH) > 1
	model = MLP5(input_dim, output_dim, apply_relu=True)
	if model_load_path is not None:
		model.load_state_dict(torch.load(model_load_path, map_location=device))
	model.to(device)
	model.eval()
	models.append(model)
criterion = nn.MSELoss().to(device)

# Setup statistics collector
sc = StatCollector(stat_train_dir, 9999999, 10)
sc.register('RMSE', {'type': 'avg', 'freq': 'step'})

# Normalize input data based on training set mean-std statistics
means = np.array([0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])
stds = np.array([0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])
inputs = testset
inputs = (inputs - means) / stds

# Convert to Torch
inputs = torch.Tensor(inputs).to(device)
	
print("Running evaluation on test set...")
it = 0
while True:
	
	# Break evaluation loop at end of epoch
	if (it + 1) * BATCH_SIZE >= nbr_examples:
		break
	
	# Compute a prediction and get the loss
	curr_gts = gts[it * BATCH_SIZE : (it + 1) * BATCH_SIZE]
	preds = 0
	for model in models:
		curr_pred = model(inputs[it * BATCH_SIZE : (it + 1) * BATCH_SIZE, :]) / len(models)
		preds += curr_pred
	loss = criterion(preds[:, 0], curr_gts)
	loss_to_sc = loss.cpu().detach().numpy()
	loss_to_sc *= (gt_max ** 2)  # MSE "denormalized" to the [0,50] COT regime
	sc.s('RMSE').collect(np.sqrt(loss_to_sc))

	# Track statistics
	if it % 100 == 0:
		sc.print()
		sc.save()
		print("Batch: %d / %d" % (it, nbr_examples // BATCH_SIZE))
	it += 1
print("Done with test set evaluation!")

print("DONE! Final stats (see 'tot' -- forget 'ma' and 'last' below):")
sc.print()
