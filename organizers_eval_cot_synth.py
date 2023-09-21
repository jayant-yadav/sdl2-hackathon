import os
import torch
import torch.nn as nn
import numpy as np

"""
This script is NOT provided to any hackathon participants; it is only
seen and used by the organizers for evaluating the participant's results
on Track 1 (i.e. synthetic data test set COT estimation results).

When a group of participants has shared a file called 'smhi_preds.npy', 
simply run `python organizers_eval_cot_synth.py` to get their test set
RMSE result.
"""

# Global vars
BASE_PATH_DATA = '../data/SDL2_SMHI_data'

# Load ground truth and normalize it
testset_with_gt = np.load(os.path.join(BASE_PATH_DATA, 'testset_smhi_12bands_with_gt.npy'))
gts = testset_with_gt[:, -1]
gt_max = 50  # COT provided in the [0, 50]-range
gts /= gt_max

# Load predictions by participants
all_preds = np.load('smhi_preds.npy')

# Perform final model evaluation on test set
loss = nn.MSELoss()(torch.Tensor(all_preds), torch.Tensor(gts)).numpy()
loss *= (gt_max ** 2)  # MSE "denormalized" to the [0,50] COT regime
print("RMSE:", np.sqrt(loss))
