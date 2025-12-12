import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import ceil
from utils.constants import ENV_NAMES
from utils.monitor import load_results

import seaborn # sets some style parameters automatically

# COLORS = [(57, 106, 177), (218, 124, 48)] 
COLORS = seaborn.color_palette('deep')

def switch_to_outer_plot(fig):
    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    return ax0

def ema(data_in, smoothing=0):
    data_out = np.zeros_like(data_in)
    curr = np.nan

    for i in range(len(data_in)):
        x = data_in[i]
        if np.isnan(curr):
            curr = x
        else:
            curr = (1 - smoothing) * x + smoothing * curr

        data_out[i] = curr

    return data_out

def plot_data_mean_std(ax, data_y, color_idx=0, data_x=None, x_scale=1, smoothing=0, first_valid=0, label=None):
    color = COLORS[color_idx]
    # hexcolor = '#%02x%02x%02x' % color

    data_y = data_y[:,first_valid:]
    nx, num_datapoint = np.shape(data_y)

    if smoothing > 0:
        for i in range(nx):
            data_y[i,...] = ema(data_y[i,...], smoothing)

    if data_x is None:
        data_x = (np.array(range(num_datapoint)) + first_valid) * x_scale

    data_mean = np.mean(data_y, axis=0)
    data_std = np.std(data_y, axis=0, ddof=1)

    ax.plot(data_x, data_mean, color=color, label=label, linestyle='solid', alpha=1, rasterized=True)
    ax.fill_between(data_x, data_mean - data_std, data_mean + data_std, color=color, alpha=.25, linewidth=0.0, rasterized=True)

    # ax.plot(data_x, data_mean, color=hexcolor, label=label, linestyle='solid', alpha=1, rasterized=True)
    # ax.fill_between(data_x, data_mean - data_std, data_mean + data_std, color=hexcolor, alpha=.25, linewidth=0.0, rasterized=True)

def read_csv(filename, key_name):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        key_index = -1

        values = []

        for line_num, row in enumerate(csv_reader):
            row = [x.lower() for x in row]
            if line_num == 0:
                idxs = [i for i, val in enumerate(row) if val == key_name]
                key_index = idxs[0]
            else:
                values.append(row[key_index])

    return np.array(values, dtype=np.float32)

def plot_values(ax, all_values, title=None, max_x=0, label=None, **kwargs):
    if max_x > 0:
        all_values = all_values[...,:max_x]

    if ax is not None:
        plot_data_mean_std(ax, all_values, label=label, **kwargs)
        ax.set_title(title)

    return all_values

def plot_progress_gen(env_list, log_folders, key_label, normalization_ranges=None, **kwargs):
    num_envs = len(env_list)
    will_normalize_and_reduce = normalization_ranges is not None

    if will_normalize_and_reduce:
        num_visible_plots = 1
        f, axarr = plt.subplots()
    else:
        num_visible_plots = num_envs
        dimy = 5
        dimx = dimy = ceil(np.sqrt(num_visible_plots))
        # dimx = ceil(num_visible_plots // dimy)
        f, axarr = plt.subplots(dimx, dimy, sharex=True)

    for key_idx, key_name in enumerate(key_label.keys()):
        all_values = []
        game_weights = [1] * num_envs

        for env_idx in range(num_envs):
            env_name = env_list[env_idx]
            label = key_label[key_name] if env_idx == 0 else None # only label the first graph to avoid legend duplicates
            print(f'loading logs from {env_name}...')
            env_log_folders = [f for f in log_folders if env_name in f]

            if num_visible_plots == 1:
                ax = axarr
            else:
                dimy = len(axarr[0]) if dimx > 1 else dimy
                ax = axarr[env_idx // dimy][env_idx % dimy] if dimx > 1 else axarr[env_idx % dimy]

            csv_files = [f"{resid}/progress.csv" for resid in env_log_folders]
            raw_data = np.array([read_csv(file, key_name) for file in csv_files])
            curr_ax = None if will_normalize_and_reduce else ax
            values = plot_values(curr_ax, raw_data, title=env_name, color_idx=key_idx, label=label, **kwargs)

            if will_normalize_and_reduce:
                game_range = normalization_ranges[env_name]
                game_min = game_range[0]
                game_max = game_range[1]
                game_delta = game_max - game_min
                sub_values = game_weights[env_idx] * (np.array(values) - game_min) / (game_delta)
                all_values.append(sub_values)

        if will_normalize_and_reduce:
            normalized_data = np.sum(all_values, axis=0)
            normalized_data = normalized_data / np.sum(game_weights)
            title = 'Mean Normalized Score'
            plot_values(ax, normalized_data, title=None, color_idx=key_idx, label=key_label[key_name], **kwargs)
        
    if len(key_label) > 1:
        if num_visible_plots == 1:
            ax.legend(loc='lower right')
        else:
            f.legend(loc='lower right', bbox_to_anchor=(.5, 0, .5, 1))

    return f, axarr

def plot_progress_compare(env_list, log_folders, key=None, normalization_ranges=None, **kwargs):
    num_envs = len(env_list)
    will_normalize_and_reduce = normalization_ranges is not None

    if will_normalize_and_reduce:
        num_visible_plots = 1
        f, axarr = plt.subplots()
    else:
        num_visible_plots = num_envs
        dimx = dimy = ceil(np.sqrt(num_visible_plots))
        f, axarr = plt.subplots(dimx, dimy, sharex=True)

    for log_idx, log_key in enumerate(log_folders.keys()):
        all_values = []
        game_weights = [1] * num_envs

        for env_idx in range(num_envs):
            env_name = env_list[env_idx]
            label = log_key if env_idx == 0 else None # only label the first graph to avoid legend duplicates
            print(f'loading logs from {env_name}...')
            env_log_folders = [f for f in log_folders[log_key] if env_name in f]

            if num_visible_plots == 1:
                ax = axarr
            else:
                dimy = len(axarr[0])
                ax = axarr[env_idx // dimy][env_idx % dimy]

            csv_files = [f"{resid}/progress.csv" for resid in env_log_folders]
            raw_data = np.array([read_csv(file, key) for file in csv_files])
            curr_ax = None if will_normalize_and_reduce else ax
            values = plot_values(curr_ax, raw_data, title=env_name, color_idx=log_idx, label=label, **kwargs)

            if will_normalize_and_reduce:
                game_range = normalization_ranges[env_name]
                game_min = game_range[0]
                game_max = game_range[1]
                game_delta = game_max - game_min
                sub_values = game_weights[env_idx] * (np.array(values) - game_min) / (game_delta)
                all_values.append(sub_values)

        if will_normalize_and_reduce:
            normalized_data = np.sum(all_values, axis=0)
            normalized_data = normalized_data / np.sum(game_weights)
            title = 'Mean Normalized Score'
            plot_values(ax, normalized_data, title=None, color_idx=log_idx, label=log_key, **kwargs)
        
    if len(log_folders) > 1:
        if num_visible_plots == 1:
            ax.legend(loc='lower right')
        else:
            f.legend(loc='lower right', bbox_to_anchor=(.5, 0, .5, 1))

    return f, axarr

def smooth(y, radius, mode='two_sided', valid_only=False):
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out

def monitor_process_fn(m):
    x = np.cumsum(m.l.values)
    y = smooth(m.r, radius=10)
    return (x, y)

def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))


    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0 # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys

def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys

def plot_monitor(env_list, log_folders, eval_mode, normalization_ranges=None, **kwargs):
    num_envs = len(env_list)
    will_normalize_and_reduce = normalization_ranges is not None

    if will_normalize_and_reduce:
        num_visible_plots = 1
        f, axarr = plt.subplots()
    else:
        num_visible_plots = num_envs
        dimx = dimy = ceil(np.sqrt(num_visible_plots))
        f, axarr = plt.subplots(dimx, dimy, sharex=True)

    all_values = []
    game_weights = [1] * num_envs
    for env_idx in range(num_envs):
        env_name = env_list[env_idx]
        print(f'loading logs from {env_name}...')
        env_log_folders = [f for f in log_folders if env_name in f]

        if num_visible_plots == 1:
            ax = axarr
        else:
            dimy = len(axarr[0])
            ax = axarr[env_idx // dimy][env_idx % dimy]

        csv_files = [load_results(resid) for resid in env_log_folders]
        xys = [monitor_process_fn(file) for file in csv_files]

        resample = 512
        origxs = [xy[0] for xy in xys]
        low  = max(x[0] for x in origxs)
        high = min(x[-1] for x in origxs)

        usex = np.linspace(low, high, resample)
        ys = []
        for (x, y) in xys:
            ys.append(symmetric_ema(x, y, low, high, resample, decay_steps=1.0)[1])

        kwargs['data_x']  = usex
        raw_data = np.array(ys)

        curr_ax = None if will_normalize_and_reduce else ax
        values = plot_values(curr_ax, raw_data, title=env_name, color_idx=0, **kwargs)

        if will_normalize_and_reduce:
            game_range = normalization_ranges[env_name]
            game_min = game_range[0]
            game_max = game_range[1]
            game_delta = game_max - game_min
            sub_values = game_weights[env_idx] * (np.array(values) - game_min) / (game_delta)
            all_values.append(sub_values)

    if will_normalize_and_reduce:
        normalized_data = np.sum(all_values, axis=0)
        normalized_data = normalized_data / np.sum(game_weights)
        title = 'Mean Normalized Score'
        plot_values(ax, normalized_data, title=None, color_idx=0, **kwargs)

    return f, axarr