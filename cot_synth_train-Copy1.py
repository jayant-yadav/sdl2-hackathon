#from __future__ import print_function
import os
import sys
import time
import random
import datetime
from shutil import copyfile
import numpy as np
import torch
import torch.nn as nn
from utils import StatCollector, MLP5, MLP6, ANN_v1, ANN_v2
import wandb
wandb.login()

# Global vars
BASE_PATH_DATA = '../data/SDL2_SMHI_data'
BASE_PATH_LOG = '../log_smhi'
USE_GPU = True  # The code uses small-scale models, GPU doesn't seem to accelerate things actually
SEED = 0
BATCH_SIZE = 32
LR = 0.0003  # Learning rate
WEIGHT_DECAY = 0.0  # Parameter for ADAM optimizer
BETA1 = 0.9  # Parameter for ADAM optimizer
INPUT_NOISE_TRAIN = 0.03  # Add 0-mean white noise with std being a fraction of the mean input of train, to train inputs
INPUT_NOISE_VAL = 0.00  # Add 0-mean white noise with std being a fraction of the mean input of train, to val inputs
SKIP_BAND_10 = False  # True --> Skip band 10 as input (may be needed for Skogsstyrelsen data)
SKIP_BAND_1 = True  # True --> Skip band 1 (SKIP_BAND_1 should always be True, as the band currently does not make sense in the data; further work needed in that direction in future work)
PROPERTY_COLUMN_MAPPING = {'spec_bands': [i for i in range(1 + SKIP_BAND_1, 14)], 'angles': [14, 15, 16], 'thick': [17], 'type': [18], 'prof_id': [19], 'gas_vapour': [20, 21], 'surf_prof': [22]}
INPUTS = ['spec_bands']  # Add keys from PROPERTY_COLUMN_MAPPING to use those as inputs
REGRESSOR = 'thick'  # Can be set to any other key in PROPERTY_COLUMN_MAPPING to regress that instead
MODEL_LOAD_PATH = ['../log_smhi/2023-08-10_10-33-44/model_it_2000000', '../log_smhi/2023-08-10_10-34-06/model_it_2000000', '../log_smhi/2023-08-10_10-34-18/model_it_2000000',
				   '../log_smhi/2023-08-10_10-34-28/model_it_2000000', '../log_smhi/2023-08-10_10-34-46/model_it_2000000', '../log_smhi/2023-08-10_10-34-58/model_it_2000000',
				   '../log_smhi/2023-08-10_10-35-09/model_it_2000000', '../log_smhi/2023-08-10_10-35-31/model_it_2000000', '../log_smhi/2023-08-10_10-35-52/model_it_2000000',
				   '../log_smhi/2023-08-10_10-36-12/model_it_2000000']  # None --> Model randomly initialized. Set to string pointing to model weights to init with those weights.
UNIFORM_DIST_NO_CLOUD_THIN_CLOUD_REG_CLOUD = True  # Ensures that we train evenly in the 'clear', 'thin-cloud' and 'opaque-cloud' regimes
THRESHOLD_THICKNESS_IS_CLOUD = 0.025  # Cloud optical tickness (COT) above this --> seen as an 'opaque cloud' pixel
THRESHOLD_THICKNESS_IS_THIN_CLOUD = 0.015  # Cloud optical tickness (COT) above this --> seen as a 'thin cloud' pixel
SAVE_MODEL_WEIGHTS = True  # False --> No model weights are saved using this script
# NUM_TRAIN_ITER = 2_000_000  # For how many batches to train the model
NUM_TRAIN_ITER = 2_00_000  # For how many batches to train the model
USE_SC = True  # Recommend to turn it off if NUM_TRAIN_ITER > 1,000,000 - otherwise code gets very slow.

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="AI sweden hackathon",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": LR,
#     "batch size": BATCH_SIZE,
#     "weight decay": WEIGHT_DECAY,
#     "beta 1": BETA1,
#     "training itr": NUM_TRAIN_ITER,
#     }
# )

sweep_configuration = {
	'project': "AI sweden hackathon",
    'method': 'bayes',
    'name': 'MLP5_sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'RMSE'
        },
    'parameters': {
        'batch_size': {'values': [16, BATCH_SIZE, 64,128,256]},
        'training_itr': {'max': NUM_TRAIN_ITER, 'min': 50_000},
        'learning_rate': {'max': 0.001, 'min': LR-0.0001},
        'weight_decay': {'max': 0.01, 'min': WEIGHT_DECAY},
        "beta1": {'value': BETA1},
     }
}

def train(config = None):

	global MODEL_LOAD_PATH
	
	with wandb.init(config=config):
		config = wandb.config
		
		# MODEL_LOAD_PATH must be a list of model paths
		if not isinstance(MODEL_LOAD_PATH, list):
			MODEL_LOAD_PATH = [MODEL_LOAD_PATH]
		
		# Specify model input and output dimensions
		input_dim = np.sum([len(PROPERTY_COLUMN_MAPPING[inp]) for inp in INPUTS]) - SKIP_BAND_10
		output_dim = 1
		
		# Create directory in which to save current run
		timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
		log_dir = os.path.join(BASE_PATH_LOG, timestamp)
		stat_train_dir = os.path.join(log_dir, "train_stats")
		os.makedirs(stat_train_dir, exist_ok=False)
		copyfile("cot_synth_train.py", os.path.join(log_dir, "cot_synth_train.py"))
		
		# Set seeds
		random.seed(0)
		np.random.seed(0)
		torch.manual_seed(SEED)
		
		# Read data
		print("Reading data")
		trainset = np.load(os.path.join(BASE_PATH_DATA, 'trainset_smhi.npy'))
		valset = np.load(os.path.join(BASE_PATH_DATA, 'valset_smhi.npy'))
		nbr_examples_train = trainset.shape[0]
		nbr_examples_val = valset.shape[0]
		nbr_examples = nbr_examples_train + nbr_examples_val
		print("Done reading data")
		
		# Separate input and regression variable
		inputs_train = np.concatenate([trainset[:, PROPERTY_COLUMN_MAPPING[inp]] for inp in INPUTS], axis=1)
		gts_train = np.squeeze(trainset[:, PROPERTY_COLUMN_MAPPING[REGRESSOR]])
		gts_train_binary = np.squeeze(trainset[:, PROPERTY_COLUMN_MAPPING['type']]) > 0
		inputs_val = np.concatenate([valset[:, PROPERTY_COLUMN_MAPPING[inp]] for inp in INPUTS], axis=1)
		gts_val = np.squeeze(valset[:, PROPERTY_COLUMN_MAPPING[REGRESSOR]])
		gts_val_binary = np.squeeze(valset[:, PROPERTY_COLUMN_MAPPING['type']]) > 0
		if SKIP_BAND_10:
			if SKIP_BAND_1:
				inputs_train = inputs_train[:, [0,1,2,3,4,5,6,7,8,10,11]]
				inputs_val = inputs_val[:, [0,1,2,3,4,5,6,7,8,10,11]]
			else:
				inputs_train = inputs_train[:, [0,1,2,3,4,5,6,7,8,9,11,12]]
				inputs_val = inputs_val[:, [0,1,2,3,4,5,6,7,8,9,11,12]]
		
		# Normalize regressor data
		gt_max = max(np.max(gts_train), np.max(gts_val))
		gts_train /= gt_max
		gts_val /= gt_max
		
		# Convert to Torch
		device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
		inputs_train = torch.Tensor(inputs_train).to(device)
		inputs_val = torch.Tensor(inputs_val).to(device)
		gts_train = torch.Tensor(gts_train).to(device)
		gts_val = torch.Tensor(gts_val).to(device)
		
		# Setup prediction model, loss, optimizer and stat collector
		models = []
		for model_load_path in MODEL_LOAD_PATH:  # Ensemble approach if len(MODEL_LOAD_PATH) > 1
			# Setup MLP model for the cloud optical thickness (COT) prediction task
			# OBS: Not training with ReLU it often hurts model learning. It is however applied in the eval script.
			model = MLP5(input_dim, output_dim, apply_relu=False)
			# model = MLP6(input_dim, output_dim, apply_relu=False)
			# model = ANN_v1(input_dim, output_dim, apply_relu=False)
			# model = ANN_v2(input_dim, output_dim, apply_relu=False)
			if model_load_path is not None:
				model.load_state_dict(torch.load(model_load_path, map_location=device))
			model.to(device)
			models.append(model)
		criterion = nn.MSELoss().to(device)
		sc_loss_string_base = 'MSE_loss'
		optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.beta1, 0.999))
		
		# Setup statistics collector
		if USE_SC:
			sc = StatCollector(stat_train_dir, config.training_itr, 10)
			sc.register(sc_loss_string_base, {'type': 'avg', 'freq': 'step'})
			sc.register(sc_loss_string_base + '_val', {'type': 'avg', 'freq': 'step'})
			# Track also RMSE
			sc.register('RMSE', {'type': 'avg', 'freq': 'step'})
			sc.register('RMSE_val', {'type': 'avg', 'freq': 'step'})
		
		# Create copies of original entities (some modifications, e.g. adding
		# noise, is done after each epoch, and the starting point per modification
		# should always be the original data)
		inputs_train_orig = torch.clone(inputs_train)
		inputs_val_orig = torch.clone(inputs_val)
		gts_train_orig = torch.clone(gts_train)
		gts_train_binary_orig = np.copy(gts_train_binary)
		means_input_train = torch.mean(inputs_train_orig, axis=0)
		stds_input_train = torch.std(inputs_train_orig, axis=0)
		
		# Below we ensure equal distribution of no-cloud, thin-cloud, opaque-cloud
		if UNIFORM_DIST_NO_CLOUD_THIN_CLOUD_REG_CLOUD:
			all_0_idxs = torch.squeeze(torch.nonzero(gts_train_orig < THRESHOLD_THICKNESS_IS_THIN_CLOUD)).cpu().detach().numpy()
			all_thin_idxs = torch.squeeze(torch.nonzero(torch.logical_and(gts_train >= THRESHOLD_THICKNESS_IS_THIN_CLOUD, gts_train < THRESHOLD_THICKNESS_IS_CLOUD))).cpu().detach().numpy()
			all_reg_idxs = torch.squeeze(torch.nonzero(gts_train_orig >= THRESHOLD_THICKNESS_IS_CLOUD)).cpu().detach().numpy()
			nbr_thin = len(all_thin_idxs)
			nbr_examples_train = nbr_thin * 3
		
			perm = np.random.permutation(nbr_examples_train)
			curr_idxs = np.concatenate([np.random.choice(all_0_idxs, nbr_thin), np.random.choice(all_reg_idxs, nbr_thin), all_thin_idxs])
			inputs_train = torch.clone(inputs_train_orig[curr_idxs, :])
			inputs_train = inputs_train[perm, :]
			gts_train = gts_train_orig[curr_idxs]
			gts_train = gts_train[perm]
			gts_train_binary = gts_train_binary_orig[curr_idxs]
			gts_train_binary = gts_train_binary[perm]
		
		# Add noise disturbances to data (if enabled)
		white_noise_train = torch.randn(*inputs_train.shape).to(device) * means_input_train[np.newaxis, :] * INPUT_NOISE_TRAIN
		inputs_train += white_noise_train
		white_noise_val = torch.randn(*inputs_val.shape).to(device) * means_input_train[np.newaxis, :] * INPUT_NOISE_VAL
		inputs_val += white_noise_val
		inputs_train = (inputs_train - means_input_train) / stds_input_train
		inputs_val = (inputs_val - means_input_train) / stds_input_train
	
	
	
		print("Starting training loop...")
		train_ctr = 0
		val_ctr = 0


		for it in range(config.training_itr):
			# Re-shuffle data if reached end of epoch
			# print('hi0')
			if (train_ctr + 1) * config.batch_size >= nbr_examples_train:
				# print('hi1')
				if UNIFORM_DIST_NO_CLOUD_THIN_CLOUD_REG_CLOUD:
					# print('hi2')
					perm = np.random.permutation(nbr_examples_train)
					curr_idxs = np.concatenate([np.random.choice(all_0_idxs, nbr_thin), np.random.choice(all_reg_idxs, nbr_thin), all_thin_idxs])
					inputs_train = torch.clone(inputs_train_orig[curr_idxs, :])
					inputs_train = inputs_train[perm, :]
					gts_train = gts_train_orig[curr_idxs]
					gts_train = gts_train[perm]
					gts_train_binary = gts_train_binary_orig[curr_idxs]
					gts_train_binary = gts_train_binary[perm]
				else:
					# print('hi3')
					perm = np.random.permutation(nbr_examples_train)
					inputs_train = torch.clone(inputs_train_orig[perm, :])
					gts_train = gts_train_orig[perm]
					gts_train_binary = gts_train_binary_orig[perm]
					
				# Add noise disturbances to data (if enabled)
				white_noise_train = torch.randn(*inputs_train.shape).to(device) * means_input_train[np.newaxis, :] * INPUT_NOISE_TRAIN
				inputs_train += white_noise_train
				white_noise_val = torch.randn(*inputs_val.shape).to(device) * means_input_train[np.newaxis, :] * INPUT_NOISE_VAL
				inputs_val = torch.clone(inputs_val_orig)
				inputs_val += white_noise_val
				inputs_train = (inputs_train - means_input_train) / stds_input_train
				inputs_val = (inputs_val - means_input_train) / stds_input_train
				
				# Reset counter
				train_ctr = 0
			
			# Compute a prediction and get the loss
			curr_gts = gts_train[train_ctr * config.batch_size : (train_ctr + 1) * config.batch_size]
			curr_gts_binary = gts_train_binary[train_ctr * config.batch_size : (train_ctr + 1) * config.batch_size]
			preds = 0
			for model in models:
				curr_pred = model(inputs_train[train_ctr * config.batch_size : (train_ctr + 1) * config.batch_size, :]) / len(models)
				preds += curr_pred
			loss_mse = criterion(preds[:, 0], curr_gts)
			loss = loss_mse
		
			# Track some stats
			if USE_SC:
				loss_to_sc = loss_mse.cpu().detach().numpy()
				sc.s(sc_loss_string_base).collect(loss_to_sc)
				sc.s('RMSE').collect(np.sqrt(loss_to_sc))
					
			# Increment counter
			train_ctr += 1
		
			# Update model weights
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
			# Track training and validation statistics
			if it % 100 == 0 and USE_SC:
				# Reset val_ctr if necessary
				if (val_ctr + 1) * config.batch_size >= nbr_examples_val:
					val_ctr = 0
				# Compute a prediction and get the loss
				curr_gts = gts_val[val_ctr * config.batch_size : (val_ctr + 1) * config.batch_size]
				curr_gts_binary = gts_val_binary[val_ctr * config.batch_size : (val_ctr + 1) * config.batch_size]
				preds_val = 0
				for model in models:
					model.eval()
					curr_pred = model(inputs_val[val_ctr * config.batch_size : (val_ctr + 1) * config.batch_size, :]) / len(models)
					preds_val += curr_pred
					model.train()
				loss_val = criterion(preds_val[:, 0], curr_gts)
				loss_val.cpu().detach().numpy()
				sc.s(sc_loss_string_base + '_val').collect(loss_to_sc)
		
				sc.s('RMSE').collect(np.sqrt(loss_to_sc))
				val_ctr += 1
				sc.print()
				sc.save()
				wandb.log({"RMSE": np.sqrt(loss_to_sc)})
		
			if it % 100 == 0:
				print("Iter: %d / %d" % (it, config.training_itr))
		
		# After training, optionally save model weights
		if SAVE_MODEL_WEIGHTS:
			print("Saving model weights")
			torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % config.training_itr))
			wandb.run.summary["model path"] = os.path.join(log_dir, 'model_it_%d' % config.training_itr)
		
		print("DONE!")

sweep_id = wandb.sweep(sweep_configuration)
wandb.agent(sweep_id, train)