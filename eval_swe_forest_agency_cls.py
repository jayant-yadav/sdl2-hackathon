import os
import numpy as np

"""
Before running this script, see the script `swe_forest_agency_cls.py`, where
you will first run models to predict cloudy / clear on images provided by
the Swedish Forest Agency.

You can then use this script on either the train, val or trainval split
to see how well you did. You cannot do it for the test split; instead,
give the organizers the file `skogs_preds.npy` that you obtained by running
`swe_forest_agency_cls.py` on the test set.
"""

BASE_PATH_DATA = '../data/skogsstyrelsen/'
SPLIT_TO_USE = 'trainval'  # 'train', 'val', 'trainval' or 'test'

def _eval_swe_forest_cls(base_path, split='train'):
	all_binary_preds = np.load('skogs_preds.npy')
	assert len(all_binary_preds.shape) == 1
	if split == 'test':
		assert all_binary_preds.shape[0] == 100

	if split == 'train':
		all_binary_gts = np.load(os.path.join(base_path, 'skogs_gts_train.npy'))
	elif split == 'val':
		all_binary_gts = np.load(os.path.join(base_path, 'skogs_gts_val.npy'))
	elif split == 'trainval':
		all_binary_gts = np.load(os.path.join(base_path, 'skogs_gts_train.npy'))
		all_binary_gts = np.concatenate([all_binary_gts, np.load(os.path.join(base_path, 'skogs_gts_val.npy'))])
	elif split == 'test':
		all_binary_gts = np.load(os.path.join(base_path, 'skogs_gts_test.npy'))
	assert len(all_binary_preds) == len(all_binary_gts)

	# Print results
	print(all_binary_preds)
	print(all_binary_gts)
	print("Nbr cloudy gt, nbr images total: %d, %d" % (np.count_nonzero(all_binary_gts == 1), len(all_binary_gts)))
	print("Frac cloudy gt: %.4f" % (np.count_nonzero(all_binary_gts == 1) / len(all_binary_gts)))
	print("Accuracy: %.4f" % (np.count_nonzero(all_binary_preds == all_binary_gts) / len(all_binary_preds)))

	rec_0 = np.count_nonzero(all_binary_preds[all_binary_gts == 0] == all_binary_gts[all_binary_gts == 0]) / np.count_nonzero(all_binary_gts == 0)
	rec_1 = np.count_nonzero(all_binary_preds[all_binary_gts == 1] == all_binary_gts[all_binary_gts == 1]) / np.count_nonzero(all_binary_gts == 1)
	print("Recall (macro): %.4f" % (0.5 * (rec_0 + rec_1)))
	print("Recall (gt is clear (0)): %.4f" % (rec_0))
	print("Recall (gt is cloudy (1)): %.4f" % (rec_1))

	prec_0 = np.count_nonzero(all_binary_gts[all_binary_preds == 0] == 0) / np.count_nonzero(all_binary_preds == 0)
	prec_1 = np.count_nonzero(all_binary_gts[all_binary_preds == 1] == 1) / np.count_nonzero(all_binary_preds == 1)
	print("Precision (macro): %.4f" % (0.5 * (prec_0 + prec_1)))
	print("Precision (gt is clear (0)): %.4f" % (prec_0))
	print("Precision (gt is cloudy (1)): %.4f" % (prec_1))

	f1_0 = 2 / (1 / rec_0 + 1 / prec_0)
	f1_1 = 2 / (1 / rec_1 + 1 / prec_1)
	print("F1 score (macro): %.4f" % (0.5 * (f1_0 + f1_1)))
	print("F1 score (gt is clear (0)): %.4f" % (f1_0))
	print("F1 score (gt is cloudy (1)): %.4f" % (f1_1))

# Evaluate predictions. Note that only the organizers will be able to run
# this with SPLIT_TO_USE = 'test', since only they have access to the
# ground truth for this split.
_eval_swe_forest_cls(BASE_PATH_DATA, SPLIT_TO_USE)
