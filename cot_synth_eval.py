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


# Global vars
BASE_PATH_DATA = '../data/SDL2_SMHI_data'
BASE_PATH_LOG = '../log_smhi'
USE_GPU = False  # The code uses small-scale models, GPU doesn't seem to accelerate things actually
SEED = 0
SPLIT_TO_USE = 'train'  # 'train' or 'val'
BATCH_SIZE = 100
INPUT_NOISE = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]  # Add 0-mean white noise with std being a fraction of the mean input of train, to data inputs
SKIP_BAND_10 = False  # True --> Skip band 10 as input (may be needed for Skogsstyrelsen data)
SKIP_BAND_1 = True  # True --> Skip band 1 (SKIP_BAND_1 should always be True, as the band currently does not make sense in the data; further work needed in that direction in future work)
PROPERTY_COLUMN_MAPPING = {'spec_bands': [i for i in range(1 + SKIP_BAND_1, 14)], 'angles': [14, 15, 16], 'thick': [17], 'type': [18], 'prof_id': [19], 'gas_vapour': [20, 21], 'surf_prof': [22]}
INPUTS = ['spec_bands']  # Add keys from PROPERTY_COLUMN_MAPPING to use those as inputs
REGRESSOR = 'thick'  # Can be set to any other key in PROPERTY_COLUMN_MAPPING to regress that instead
THRESHOLD_THICKNESS_IS_CLOUD = 0.025  # Cloud optical tickness (COT) above this --> seen as an 'opaque cloud' pixel
THRESHOLD_THICKNESS_IS_THIN_CLOUD = 0.015  # Cloud optical tickness (COT) above this --> seen as a 'thin cloud' pixel
MODEL_LOAD_PATH = None  # See examples on how to point to a specific model path below. None --> Model randomly initialized.

# 10 models trained on SMHI synthetic data, with 12 bands, 3% additive noise
MODEL_LOAD_PATH = ['../log_smhi/2023-08-10_10-33-44/model_it_2000000', '../log_smhi/2023-08-10_10-34-06/model_it_2000000', '../log_smhi/2023-08-10_10-34-18/model_it_2000000',
				   '../log_smhi/2023-08-10_10-34-28/model_it_2000000', '../log_smhi/2023-08-10_10-34-46/model_it_2000000', '../log_smhi/2023-08-10_10-34-58/model_it_2000000',
				   '../log_smhi/2023-08-10_10-35-09/model_it_2000000', '../log_smhi/2023-08-10_10-35-31/model_it_2000000', '../log_smhi/2023-08-10_10-35-52/model_it_2000000',
				   '../log_smhi/2023-08-10_10-36-12/model_it_2000000']

# As above, but omit band 10 also (for use in skogs data)
#MODEL_LOAD_PATH = ['../log_smhi/2023-08-10_11-49-01/model_it_2000000', '../log_smhi/2023-08-10_11-49-22/model_it_2000000', '../log_smhi/2023-08-10_11-49-49/model_it_2000000',
#				   '../log_smhi/2023-08-10_11-50-44/model_it_2000000', '../log_smhi/2023-08-10_11-51-11/model_it_2000000', '../log_smhi/2023-08-10_11-51-36/model_it_2000000',
#				   '../log_smhi/2023-08-10_11-51-49/model_it_2000000', '../log_smhi/2023-08-10_11-52-02/model_it_2000000', '../log_smhi/2023-08-10_11-52-24/model_it_2000000',
#				   '../log_smhi/2023-08-10_11-52-47/model_it_2000000']

# MODEL_LOAD_PATH must be a list of model paths
if not isinstance(MODEL_LOAD_PATH, list):
	MODEL_LOAD_PATH = [MODEL_LOAD_PATH]
# Same for INPUT_NOISE
if not isinstance(INPUT_NOISE, list):
	INPUT_NOISE = [INPUT_NOISE]

# Specify model input and output dimensions
input_dim = np.sum([len(PROPERTY_COLUMN_MAPPING[inp]) for inp in INPUTS]) - SKIP_BAND_10
output_dim = 1

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)
copyfile("cot_synth_eval.py", os.path.join(log_dir, "cot_synth_eval.py"))

# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(SEED)

# Read data
print("Reading data")
try:
	# Load pre-defined splits
	trainset = np.load(os.path.join(BASE_PATH_DATA, 'trainset_smhi.npy'))
	valset = np.load(os.path.join(BASE_PATH_DATA, 'valset_smhi.npy'))
	nbr_examples_train = trainset.shape[0]
	nbr_examples_val = valset.shape[0]
	nbr_examples = nbr_examples_train + nbr_examples_val
except:
	print("First run cot_synth_train.py, which will create the datasets in the right format!")
	sys.exit(0)
print("Done reading data")

# Separate input and regression variable
inputs_train = np.concatenate([trainset[:, PROPERTY_COLUMN_MAPPING[inp]] for inp in INPUTS], axis=1)
gts_train = np.squeeze(trainset[:, PROPERTY_COLUMN_MAPPING[REGRESSOR]])
inputs_val = np.concatenate([valset[:, PROPERTY_COLUMN_MAPPING[inp]] for inp in INPUTS], axis=1)
gts_val = np.squeeze(valset[:, PROPERTY_COLUMN_MAPPING[REGRESSOR]])

# Below are means and stds based on the synthetic training data, for all 13 bands (recall that band B1 is not reliable and should not be used, though)
means = np.array([0.64984976, 0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])
stds = np.array([0.3596485, 0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])

# Throw away unused input features
if SKIP_BAND_1:
	means = means[1:]
	stds = stds[1:]
if SKIP_BAND_10:
	if SKIP_BAND_1:
		inputs_train = inputs_train[:, [0,1,2,3,4,5,6,7,8,10,11]]
		inputs_val = inputs_val[:, [0,1,2,3,4,5,6,7,8,10,11]]
		means = means[[0,1,2,3,4,5,6,7,8,10,11]]
		stds = stds[[0,1,2,3,4,5,6,7,8,10,11]]
	else:
		inputs_train = inputs_train[:, [0,1,2,3,4,5,6,7,8,9,11,12]]
		inputs_val = inputs_val[:, [0,1,2,3,4,5,6,7,8,9,11,12]]
		means = means[[0,1,2,3,4,5,6,7,8,9,11,12]]
		stds = stds[[0,1,2,3,4,5,6,7,8,9,11,12]]

# Normalize regressor data and convert to Torch
gt_max = 50  # COT provided in the [0, 50]-range
gts_train /= gt_max
gts_val /= gt_max
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
gts_train = torch.Tensor(gts_train).to(device)
gts_val = torch.Tensor(gts_val).to(device)

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
sc_loss_string_base = 'MSE_loss'
sc.register(sc_loss_string_base + '_' + SPLIT_TO_USE, {'type': 'avg', 'freq': 'step'})
sc.register('RMSE_' + SPLIT_TO_USE, {'type': 'avg', 'freq': 'step'})

# Begin the evaluation!
if SPLIT_TO_USE == 'train':
	inputs = inputs_train
elif SPLIT_TO_USE == 'val':
	inputs = inputs_val
inputs_train_orig = inputs_train
inputs_orig = inputs
for noise_idx, noise in enumerate(INPUT_NOISE):
	
	# Add noise disturbances to data (if enabled)
	white_noise = np.random.randn(*inputs_orig.shape) * means[np.newaxis, :] * noise
	inputs = inputs_orig + white_noise
	
	# Normalize input data
	inputs = (inputs - means) / stds

	# Convert to Torch
	inputs = torch.Tensor(inputs).to(device)
	
	# Set things based on which split is used
	if SPLIT_TO_USE == 'train':
		nbr_examples = nbr_examples_train
		gts = gts_train
	elif SPLIT_TO_USE == 'val':
		nbr_examples = nbr_examples_val
		gts = gts_val
		
	print("Running evaluation (%s set)..." % SPLIT_TO_USE)
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
		sc.s(sc_loss_string_base + '_' + SPLIT_TO_USE).collect(loss_to_sc)
		sc.s('RMSE_' + SPLIT_TO_USE).collect(np.sqrt(loss_to_sc))

		# Track statistics
		if it % 100 == 0:
			sc.print()
			sc.save()
			print("Batch: %d / %d" % (it, nbr_examples // BATCH_SIZE))
		it += 1
	print("Done with %s set evaluation!" % SPLIT_TO_USE)

print("DONE! Final stats (see 'tot' -- forget 'ma' and 'last' below):")
sc.print()
