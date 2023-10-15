import os
import numpy as np
import time
import random
from collections import OrderedDict
from scipy.special import expit
import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def replace(string_in, replace_from, replace_to='_'):
    if not isinstance(replace_from, list):
        replace_from = [replace_from]
    string_out = string_in
    for replace_entry in replace_from:
        string_out = string_out.replace(replace_entry, replace_to)
    return string_out


class BaseStat():
    """
    Basic statistic from which all other statistic types inherit
    """
    def __init__(self, name):
        self.name = name
        self.ep_idx = 0
        self.stat_collector = None

    def collect(self, value):
        pass

    def get_data(self):
        return {}

    def next_step(self):
        pass

    def next_ep(self):
        self.ep_idx += 1

    def next_batch(self):
        pass

    def compute_mean(self, mean, value, counter):
        return (counter * mean + value) / (counter + 1)

    def compute_ma(self, ma, value, ma_weight):
        return (1 - ma_weight) * ma + ma_weight * value


class AvgStat(BaseStat):
    """
    Standard average statistic (can track total means, moving averages,
    exponential moving averages etcetera)
    """
    def __init__(self, name, coll_freq='ep', ma_weight=0.001):
        super(AvgStat, self).__init__(name=name)
        self.counter = 0
        self.mean = 0.0
        self.ma = 0.0
        self.last = None
        self.means = []
        self.mas = []
        self.values = []
        self.times = []
        self.coll_freq = coll_freq
        self.ma_weight = ma_weight

    def collect(self, value, delta_counter=1):
        self.counter += delta_counter

        self.values.append(value)
        self.times.append(self.counter)
        self.mean = self.compute_mean(self.mean, value, len(self.means))
        self.means.append(self.mean)
        if self.counter < 10:
            # Want the ma to be more stable early on
            self.ma = self.mean
        else:
            self.ma = self.compute_ma(self.ma, value, self.ma_weight)
        self.mas.append(self.ma)
        self.last = value

    def get_data(self):
        return {'times': self.times, 'means': self.means, 'mas': self.mas, 'values': self.values}

    def print(self, timestamp=None):
        if self.counter <= 0:
            return
        self._print_helper()

    def _print_helper(self, mean=None, ma=None, last=None):

        # Set defaults
        if mean is None:
            mean = self.mean
        if ma is None:
            ma = self.ma
        if last is None:
            last = self.last

        if isinstance(mean, float):
            print('Mean %-35s tot: %10.5f, ma: %10.5f, last: %10.5f' %
                  (self.name, mean, ma, last))
        else:
            print('Mean %-35s tot:  (%.5f' % (self.name, mean[0]), end='')
            for i in range(1, mean.size - 1):
                print(', %.5f' % mean[i], end='')
            print(', %.5f)' % mean[-1])
            print('%-40s ma:   (%.5f' % ('', ma[0]), end='')
            for i in range(1, ma.size - 1):
                print(', %.5f' % ma[i], end='')
            print(', %.5f)' % ma[-1])
            print('%-40s last: (%.5f' % ('', last[0]), end='')
            for i in range(1, last.size - 1):
                print(', %.5f' % last[i], end='')
            print(', %.5f)' % last[-1])

    def save(self, save_dir):
        file_name = replace(self.name, [' ', '(', ')', '/'], '-')
        file_name = replace(file_name, ['<', '>'], '')
        file_name += '.npz'
        np.savez(os.path.join(save_dir, file_name),
                 values=np.asarray(self.values), means=np.asarray(self.means),
                 mas=np.asarray(self.mas), times=np.asarray(self.times))

    def plot(self, times=None, values=None, means=None, mas=None, save_dir=None):
        # Set defaults
        if times is None:
            times = self.times
        if values is None:
            values = self.values
        if means is None:
            means = self.means
        if mas is None:
            mas = self.mas
        if save_dir is None:
            save_dir_given = None
            save_dir = os.path.join(self.log_dir, 'stats', 'data')
        else:
            save_dir_given = save_dir

        # Define x-label
        if self.coll_freq == 'ep':
            xlabel = 'episode'
        elif self.coll_freq == 'step':
            xlabel = 'step'

        if np.asarray(values).ndim > 1:
            # Plot all values
            self._plot(times, values, self.name + ' all', xlabel, 'y', None,
                       save_dir_given)

            # Plot total means
            self._plot(times, means, self.name + ' total mean', xlabel, 'y', None,
                       save_dir_given)

            # Plot moving averages
            self._plot(times, mas, self.name + ' total exp ma', xlabel, 'y', None,
                       save_dir_given)
        else:
            self._plot_in_same(times, [values, means, mas],
                               self.name, xlabel, 'y',
                               ['all-data', 'mean', 'ma'],
                               [None, '-.', '-'], [0.25, 1.0, 1.0],
                               save_dir_given)

        # Also save current data to file
        if save_dir_given is None:
            file_name = replace(self.name, [' ', '(', ')', '/'], '-')
            file_name = replace(file_name, ['<', '>'], '')
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, file_name), values)

    def _plot(self, x, y, title='plot', xlabel='x', ylabel='y', legend=None,
              log_dir=None):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend is None:
            plt.legend([str(k) for k in range(np.asarray(y).shape[1])])
        else:
            plt.legend(legend)
        title_to_save = replace(title, [' ', '(', ')', '/'], '-')
        title_to_save = replace(title_to_save, ['<', '>'], '')
        if log_dir is None:
            log_dir = os.path.join(self.log_dir, 'stats', 'plots')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        plt.savefig(os.path.join(log_dir, title_to_save + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()

    def _plot_in_same(self, x, ys, title='plot', xlabel='x', ylabel='y',
                      legend=None, line_styles=None, alphas=None,
                      log_dir=None):
        if alphas is None:
            alphas = [1.0 for _ in range(len(ys))]
        plt.figure()
        for i in range(len(ys)):
            if line_styles[i] is not None:
                plt.plot(x, ys[i],
                         linestyle=line_styles[i], alpha=alphas[i])
            else:
                plt.plot(x, ys[i], 'yo', alpha=alphas[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend is None:
            plt.legend([str(k) for k in range(np.asarray(y).shape[1])])
        else:
            plt.legend(legend)
        title_to_save = replace(title, [' ', '(', ')', '/'], '-')
        title_to_save = replace(title_to_save, ['<', '>'], '')
        if log_dir is None:
            log_dir = os.path.join(self.log_dir, 'stats', 'plots')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        plt.savefig(os.path.join(log_dir, title_to_save + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()


class StatCollector():
    """
    Statistics collector class
    """
    def __init__(self, log_dir, tot_nbr_steps, print_iter):
        self.stats = OrderedDict()
        self.log_dir = log_dir
        self.ep_idx = 0
        self.step_idx = 0
        self.epoch_idx = 0
        self.print_iter = print_iter
        self.tot_nbr_steps = tot_nbr_steps

    def has_stat(self, name):
        return name in self.stats

    def register(self, name, stat_info, ma_weight=0.001):
        if self.has_stat(name):
            sys.exit("Stat already exists")

        if stat_info['type'] == 'avg':
            stat_obj = AvgStat(name, stat_info['freq'], ma_weight=ma_weight)
        else:
            sys.exit("Stat type not supported")

        stat = {'obj': stat_obj, 'name': name, 'type': stat_info['type']}
        self.stats[name] = stat

    def s(self, name):
        return self.stats[name]['obj']

    def next_step(self):
        self.step_idx += 1

    def next_ep(self):
        self.ep_idx += 1
        for stat_name, stat in self.stats.items():
            stat['obj'].next_ep()
        if self.ep_idx % self.print_iter == 0:
            self.print()
            self._plot_to_hdock()

    def print(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].print()

    def plot(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].plot(save_dir=self.log_dir)

    def save(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].save(save_dir=self.log_dir)

def _mlp_post_filter(pred_map_binary_list, pred_map_binary_thin_list, pred_map, thresh_thin_cloud, post_filt_sz):
	if post_filt_sz == 1:
		return pred_map_binary_list, pred_map_binary_thin_list
	H, W = pred_map.shape
	for list_idx, pred_map_binary in enumerate(pred_map_binary_list):
		tmp_map = np.zeros_like(pred_map)
		tmp_map_thin = np.zeros_like(pred_map)
		count_map = np.zeros_like(pred_map)
		for i_start in range(post_filt_sz):
			for j_start in range(post_filt_sz):
				for i in range(i_start, H // post_filt_sz):
					for j in range(j_start, W // post_filt_sz):
						count_map[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
						curr_patch = pred_map_binary[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz]
						curr_patch_thin = pred_map_binary_thin_list[min(list_idx, len(thresh_thin_cloud) - 1)][i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz]
						if np.count_nonzero(curr_patch) >= np.prod(curr_patch.shape) // 2:
							tmp_map[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
						if np.count_nonzero(curr_patch_thin) >= np.prod(curr_patch_thin.shape) // 2:
							tmp_map_thin[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
		tmp_map[count_map == 0] = 0
		count_map[count_map == 0] = 1
		tmp_map /= count_map
		assert np.min(tmp_map) >= 0 and np.max(tmp_map) <= 1
		pred_map_binary = tmp_map >= 0.50
		pred_map_binary_list[list_idx] = pred_map_binary

		tmp_map_thin[count_map == 0] = 0
		tmp_map_thin /= count_map
		assert np.min(tmp_map_thin) >= 0 and np.max(tmp_map_thin) <= 1
		pred_map_binary_thin = tmp_map_thin >= 0.50
		pred_map_binary_thin_list[min(list_idx, len(thresh_thin_cloud) - 1)] = pred_map_binary_thin

		# 'Aliasing effect' after this filtering can cause BOTH thin and regular cloud to be active at the same time -- give prevalence to regular
		pred_map_binary_thin_list[0][pred_map_binary_list[0]] = 0

	return pred_map_binary_list, pred_map_binary_thin_list

# Setup MLP-computation function
def mlp_inference(img, means, stds, models, batch_size, thresh_cloud, thresh_thin_cloud, post_filt_sz, device='cpu', predict_also_cloud_binary=False):
	H, W, input_dim = img.shape
	img_torch = torch.reshape((torch.Tensor(img).to(device) - means) / stds, [H * W, input_dim])
	pred_map_tot = 0.0
	pred_map_binary_tot = 0.0
	for model in models:
		pred_map = np.zeros(H * W)
		pred_map_binary = np.zeros(H * W)
		for i in range(0, H * W, batch_size):
			curr_pred = model(img_torch[i : i + batch_size, :])
			pred_map[i : i + batch_size] = curr_pred[:, 0].cpu().detach().numpy()
			if predict_also_cloud_binary:
				pred_map_binary[i : i + batch_size] = curr_pred[:, 1].cpu().detach().numpy()
		pred_map = np.reshape(pred_map, [H, W])
		if predict_also_cloud_binary:
			pred_map_binary = np.reshape(expit(pred_map_binary), [H, W]) >= 0.5
		else:
			pred_map_binary = np.zeros_like(pred_map)#pred_map >= thresh_cloud[-1] <<--- overwritten anyway
			
		# Average model predictions
		pred_map_tot += pred_map / len(models)
		pred_map_binary_tot += pred_map_binary.astype(float) / len(models)
		
	# Return final predictions
	pred_map = pred_map_tot
	if predict_also_cloud_binary:
		pred_map_binary = pred_map_binary_tot >= 0.5
	else:
		pred_map_binary_list = []
		pred_map_binary_thin_list = []
		for thresh in thresh_cloud:
			pred_map_binary_list.append(pred_map_tot >= thresh)
		for thresh in thresh_thin_cloud:
			# Below: A thin cloud is a thin cloud only if it is above the thin thresh AND below the regular cloud thresh
			pred_map_binary_thin_list.append(np.logical_and(pred_map_tot >= thresh, pred_map_tot < thresh_cloud[0]))

	# Potentially do post-processing on the cloud/not cloud (binary)
	# prediction, so that it becomes more spatially coherent
	pred_map_binary_list, pred_map_binary_thin_list = _mlp_post_filter(pred_map_binary_list, pred_map_binary_thin_list, pred_map, thresh_thin_cloud, post_filt_sz)

	# Return
	return pred_map, pred_map_binary_list, pred_map_binary_thin_list

# Simple 5-layer MLP model
class MLP5(nn.Module):
	def __init__(self, input_dim, output_dim=1, hidden_dim=64, apply_relu=True):
		super(MLP5, self).__init__()
		self.lin1 = nn.Linear(input_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim)
		self.lin3 = nn.Linear(hidden_dim, hidden_dim)
		self.lin4 = nn.Linear(hidden_dim, hidden_dim)
		self.lin5 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
		self.apply_relu = apply_relu

	def forward(self, x):
		x1 = self.lin1(x)
		x1 = self.relu(x1)
		x2 = self.lin2(x1)
		x2 = self.relu(x2)
		x3 = self.lin3(x2)
		x3 = self.relu(x3)
		x4 = self.lin4(x3)
		x4 = self.relu(x4)
		x5 = self.lin5(x4)
		if self.apply_relu:
			x5[:, 0] = self.relu(x5[:, 0])  # NB: cloud optical thicknesses cannot be negative
		return x5
        
# Simple 6-layer MLP model
class MLP6(nn.Module):
	def __init__(self, input_dim, output_dim=1, hidden_dim=64, apply_relu=True):
		super(MLP6, self).__init__()
		self.lin1 = nn.Linear(input_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim)
		self.lin3 = nn.Linear(hidden_dim, hidden_dim)
		self.lin4 = nn.Linear(hidden_dim, hidden_dim)
		self.lin5 = nn.Linear(hidden_dim, hidden_dim)
		self.lin6 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
		self.apply_relu = apply_relu

	def forward(self, x):
		x1 = self.lin1(x)
		x1 = self.relu(x1)
		x2 = self.lin2(x1)
		x2 = self.relu(x2)
		x3 = self.lin3(x2)
		x3 = self.relu(x3)
		x4 = self.lin4(x3)
		x4 = self.relu(x4)
		x5 = self.lin5(x4)
		x5 = self.relu(x5)
		x6 = self.lin5(x5)
		if self.apply_relu:
			x6[:, 0] = self.relu(x6[:, 0])  # NB: cloud optical thicknesses cannot be negative
		return x6

# Simple 5-layer ANN model
class ANN_v1(nn.Module):
	def __init__(self, input_dim, output_dim=1, hidden_dim=64, apply_relu=True):
		super(ANN_v1, self).__init__()
		self.lin1 = nn.Linear(input_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, 2*64)
		self.lin3 = nn.Linear(2*64, 2*64)
		self.lin4 = nn.Linear(2*64, hidden_dim)
		self.dropout = nn.Dropout(0.2)
		self.lin5 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
		self.apply_relu = apply_relu

	def forward(self, x):
		x1 = self.lin1(x)
		x1 = self.relu(x1)
		x2 = self.lin2(x1)
		x2 = self.relu(x2)
		x3 = self.lin3(x2)
		x3 = self.relu(x3)
		x4 = self.lin4(x3)
		x4 = self.relu(x4)
		x4 = self.dropout(x4)
		x5 = self.lin5(x4)
		if self.apply_relu:
			x5[:, 0] = self.relu(x5[:, 0])  # NB: cloud optical thicknesses cannot be negative
		return x5

# Simple 7-layer ANN model
class ANN_v2(nn.Module):
	def __init__(self, input_dim, output_dim=1, hidden_dim=64, apply_relu=True):
		super(ANN_v2, self).__init__()
		self.lin1 = nn.Linear(input_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, 2*64)
		self.lin3 = nn.Linear(2*64, 2*2*64)
		self.lin4 = nn.Linear(2*2*64, 2*2*64)
		self.dropout = nn.Dropout(0.2)
		self.lin5 = nn.Linear(2*2*64, 2*64)
		self.dropout = nn.Dropout(0.2)
		self.lin6 = nn.Linear(2*64, hidden_dim)
		self.dropout = nn.Dropout(0.2)
		self.lin7 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
		self.apply_relu = apply_relu

	def forward(self, x):
		x1 = self.lin1(x)
		x1 = self.relu(x1)
		x2 = self.lin2(x1)
		x2 = self.relu(x2)
		x3 = self.lin3(x2)
		x3 = self.relu(x3)
		x4 = self.lin4(x3)
		x4 = self.relu(x4)
		x4 = self.dropout(x4)
		x5 = self.lin5(x4)
		x5 = self.relu(x5)
		x5 = self.dropout(x5)
		x6 = self.lin6(x5)
		x6 = self.relu(x6)
		x6 = self.dropout(x6)
		x7 = self.lin7(x6)
		if self.apply_relu:
			x7[:, 0] = self.relu(x7[:, 0])  # NB: cloud optical thicknesses cannot be negative
		return x7