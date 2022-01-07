import numpy as np
import pandas as pd
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
import tqdm as tqdm

def aggregate_as_time_series(df, freq, y_time, y_numeric={}, y_categoric=[], trim_edges=False, unpack=True):
    '''
    Converts a spatiotemporal dataset into a set of time series according to the given parameters.

    Parameters:
    df (pandas.DataFrame): the dataset
    freq (str): the offset alias representing the desired granularity to divide the time series
       (see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
    y_time (str): the name of the time column, column must be of type datetime
    y_numeric (object): the aggregations to be performed on each numerical observation.
        (e.g.: {'count': ('id', 'count'), 'avg_age': ('age', 'mean')})
    y_categoric (list): the list of column names to be used as categorical constraints
    trim_edges (bool, default=True): if True, first and last rows are discarded
       (used to disregard possibly incomplete data)

    Outputs:
    (1) a numpy matrix representing the time series observations, with each row representing
        a time interval and each column representing an observation and constraint combination
    (2) the time index
    (3) the list of constraints
    (4) the counts
    '''
    group = df.groupby([pd.Grouper(key=y_time, freq=freq, label='left', closed='left')] + y_categoric)
    aggregations = group.agg(**y_numeric if unpack else y_numeric).reset_index().set_index(y_time)
    counts = group.agg(count=(y_time, 'count')).reset_index().set_index(y_time)
    if len(y_categoric) > 0:
        aggregations = aggregations.pivot(columns=y_categoric, values=[i for i in y_numeric.keys()]).fillna(0)
        counts = counts.pivot(columns=y_categoric, values=['count']).fillna(0)    
    boundaries = aggregations.index
    constraints = aggregations.columns
    aggregations = aggregations.to_numpy()
    counts = counts.to_numpy()
    if trim_edges:
        aggregations = aggregations[1:-1]
        counts = aggregations[1:-1]
        boundaries = boundaries[1:-1]
    return aggregations, boundaries, constraints, counts

def slope_score_function(slope, max_slope):
    slope = np.clip(slope, -max_slope, max_slope)
    return slope / max_slope

def sliding_regression(mat, boundaries, constraints, count_mat,
                       window_lengths=[], slide_length=1, interval_starts=None,
                       min_r2=0, min_count=0):
    '''
    Performs candidate trend discovery from list of time series

    Parameters:
    mat, boundaries, constraints, count_mat: aggregation resulting from aggregation step
    window_lengths (list): length of windows (L) to find
    slide_length: the slide length (kappa)
    intervals_starts (list): alternatively, specify the index of the start of the intervals instead of the slide length
        (slide_length is ignored if this is not None)
    min_r2: r2_min parameter
    min_count: minimum count to consider (optional)

    Outputs:
    (1) dataframe of candidate trends
    '''
    boundary_step = boundaries[1] - boundaries[0]
    boundaries_numeric = np.array([(np.array((boundaries - boundaries[0]).days)).astype('int').astype('float'),] * mat.shape[1]).transpose()
    constraint_names = ['observation'] + ['group_' + i for i in constraints.names[1:]]
    if constraints.nlevels > 1:
        obs_count = len(list(dict.fromkeys([i[0] for i in constraints])))
    else:
        obs_count = len(list(dict.fromkeys([i for i in constraints])))
    properties = ['start_index', 'end_index', 'start', 'end',
        'slope', 'slope_norm', 'slope_norm_abs', 'intercept', 'support', 'r2'] + constraint_names
    data = {i: [] for i in properties}
    
    means = np.repeat(np.mean(count_mat, axis=0), obs_count)
    
    with tqdm.tqdm(total=len(window_lengths) * len(mat)) as pbar:
        for window_length in window_lengths:
            y = mat[0:window_length,:]
            x = boundaries_numeric[0:window_length,:]
            c = np.tile(count_mat[0:window_length,:], obs_count)
            xsum = np.sum(x, axis=0)
            ysum = np.sum(y, axis=0)
            xysum = np.sum(x*y, axis=0)
            x2sum = np.sum(np.power(x,2), axis=0)
            csum = np.sum(c, axis=0)
            n = window_length

            i = window_length
            pbar.update(i-1)
            while i <= len(mat):
                if i != window_length:
                    next_y = mat[i-window_length:i]
                    next_x = boundaries_numeric[i-window_length:i]
                    next_c = count_mat[i-window_length:i]
                    xsum = xsum - x[0,:] + next_x[-1,:]
                    ysum = ysum - y[0,:] + next_y[-1,:]
                    xysum = xysum - (x[0,:] * y[0,:]) + (next_x[-1,:] * next_y[-1,:])
                    x2sum = x2sum - (np.power(x[0,:],2)) + (np.power(next_x[-1,:],2))
                    csum = csum - c[0,:] + next_c[-1,:]
                    x = next_x
                    y = next_y
                    c = next_c

                start_index = i - window_length
                end_index = i - 1

                xbar = xsum/n
                ybar = ysum/n
                num = xysum - (1/n) * xsum * ysum
                den = x2sum - (1/n) * np.power(xsum,2)

                m = num/den
                m_norm = m/means
                m_norm_abs = np.abs(m_norm)
                #m_score = slope_score_function(m_norm, eta)
                #m_score_abs = np.abs(m_score)
                b = ybar - m * xbar

                ybarmat = np.array([ybar,] * len(x))
                mmat = np.array([m,] * len(x))
                bmat = np.array([b,] * len(x))

                f = mmat * x + bmat

                sstot = np.sum(np.power(y - ybarmat, 2),axis=0)
                ssres = np.sum(np.power(y - f, 2),axis=0)
                r2 = 1 - np.divide(ssres, sstot, out=np.ones(ssres.shape, dtype=float), where=sstot!=0)

                if (interval_starts is None and start_index % slide_length == 0) or (interval_starts is not None and start_index in interval_starts) or i == len(mat):
                    starts = np.repeat(boundaries[start_index], len(constraints))
                    ends = np.repeat(boundaries[end_index] + boundary_step, len(constraints))
                    start_indices = np.repeat(start_index, len(constraints))
                    end_indices = np.repeat(end_index, len(constraints))
                    current_constraints = np.array([i for i in constraints])

                    cond = (r2 >= min_r2) & (csum >= min_count)

                    stats = {'start_index': start_indices, 'end_index': end_indices,
                             'start': starts, 'end': ends,
                             'slope': m, 'slope_norm': m_norm, 'slope_norm_abs': m_norm_abs,
                             'intercept': b, 'support': csum, 'r2': r2}
                    for key in stats:
                        data[key].extend(stats[key][cond])

                    if constraints.nlevels == 1:
                        data[constraint_names[0]].extend(current_constraints[cond])
                    else:
                        for j in range(0,len(constraint_names)):
                            data[constraint_names[j]].extend([k[j] for k in current_constraints[cond]])

                i = i + 1
                pbar.update(1)
        
    result = pd.DataFrame(data)
    return result

def kde_clustering(slope, orig_len, h=0.075, a1=.57, a2=.43, plot=False):
    K = lambda x: np.power(np.e, -(np.power(x, 2) / 2)) / np.sqrt(2 * np.pi) 
    KDE = lambda x: (1 / (len(slope) * h)) * np.sum(K((x - slope) / h))
    x = np.linspace(-1.2, 1.2, 101)
    y = np.vectorize(KDE)(x)
    minima = argrelextrema(y, np.less_equal)[0]
    maxima = argrelextrema(y, np.greater_equal)[0]
    if len(minima) < len(maxima):
        minima = np.concatenate([[0], minima, [len(x) - 1]]) 
    minima = minima[minima - np.roll(minima,1) != 1]
    maxima = maxima[maxima - np.roll(maxima,1) != 1]
    minima_x = x[minima]
    maxima_x = x[maxima]
    
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(10,1.15))
        sns.scatterplot(x=np.array(slope), y=np.vectorize(KDE)(np.array(slope)), color='black', marker='o', ax=ax);
        sns.lineplot(x=x, y=y, color='gray', ax=ax);
        for i in minima_x:
            plt.axvline(i, 0, 1, color='red')

    clusters = maxima_x[np.digitize(slope, minima_x) - 1]
    cluster, count = np.unique(clusters, return_counts=True)
    d = np.array([[i] for i in cluster])
    #w = np.array([[i] for i in count / (np.sum(count))])
    w = np.array([0 if len(cluster) == 1 else np.repeat(1/(len(cluster)-1), len(cluster))])
    
    unusual = 1 - count / orig_len
    unusual = dict({c: u for c, u in zip(cluster, unusual)})
    unusual = np.array([unusual[i] for i in clusters])
    
    distance = (np.abs(d - d[:,None]) * w).sum(axis=1).flatten()
    distance = dict({c: d for c, d in zip(cluster, distance)})
    distance = np.array([distance[i] for i in clusters])

    #variance = [np.std(slope[clusters==i]) for i in cluster]
    #variance = dict({c: v for c, v in zip(cluster, variance)})
    #variance = np.array([variance[i] for i in clusters])

    return a1 * unusual + a2 * (distance/2), len(cluster)

def compute_deviation(trends, r0, dimensions=[], h=0.075, a1=.57, a2=.43, eta_q=100, eta=None): 
    '''
    Computes the deviation on a list of candidate trends on the specified dimension.

    Parameters:
    dimensions (list): desired dimension, either ['start', 'end'] to choose the interval or the name
        of a group as a single element of a list
    r0: number of relevant trends in the chosen dimension including those that do not satisfy r2 criteria
    h: bandwidth for KDE
    a1: weight
    a2: weight
    eta_q: the percentile to use to derive the eta scaling constant (100 means the max value)
    eta: alternatively, a fixed eta can be used (eta_q will be ignored if this is not None)

    Outputs:
    (1) series representing the deviation of each candidate trend
    '''
    if eta is None:
        eta = np.percentile(np.abs(trends['slope_norm']), q=eta_q)
    trends['slope_score'] = slope_score_function(trends['slope_norm'], eta)
    trends['slope_score_abs'] = np.abs(trends['slope_score'])

    tqdm.tqdm.pandas()
    
    dimensions = [('group_' + i if i != 'start' and i != 'end' else i) for i in dimensions]

    score_list = pd.Series(index=trends.index, dtype='float64')
    
    def apply_deviation_measure(group):
        a, b = kde_clustering(group['slope_score'].to_numpy(), r0, h, a1=a1, a2=a2, plot=False)
        current_scores = a
        score_list.iloc[group.index] = current_scores
        
    groups = ['start', 'end'] + [i for i in trends.columns if i.startswith('group_') or i.startswith('_temp_')]
    fixed = [i for i in groups if i not in dimensions]
    grouped = trends.groupby(fixed)
    grouped.progress_apply(apply_deviation_measure) 

    return score_list