import numpy as np

def mean_range(data, i, j, window_size):
    a_ts = data[:,i]/data[0,i]
    b_ts = data[:,j]/data[0,j]
    spread = a_ts - b_ts
    moving_average = np.convolve(spread, np.ones((window_size,))/window_size, mode='valid')
    ave_range = np.abs(np.max(moving_average) - np.min(moving_average))
    return ave_range

# Get pairs with most stable means over time.
# Assumes data is stored with each time series
# in its own column.
def best_mean_pairs(data, window_size=20):
    num_stocks = data.shape[1]
    mean_range_and_pair = [] 
    for i in range(num_stocks - 1):
        for j in range(i + 1, num_stocks):
            print("getting mean for: ({}, {})".format(i, j))
            ave_range = mean_range(data, i, j, window_size)
            mean_range_and_pair += [[ave_range, i, j]]
    mean_range_and_pair = np.array(mean_range_and_pair)
    mean_range_and_pair = mean_range_and_pair[mean_range_and_pair[:,0].argsort()]
    return mean_range_and_pair

def score_pairs(pairs, means):
    total = 0
    for pair in pairs:
        idx = np.argmin(np.abs(np.sum(means[:,1:]-[pair[0],pair[1]],axis=1)))
        print("Pair ({}, {}) idx = {}".format(pair[0], pair[1], idx))
        if idx <= 100:
            total += 1
            print("+1")
        elif idx <= 1000:
            total += 0.5
            print("+0.5")
        elif idx <= 5000:
            total += 0.25
            print("+0.25")
    print("score = {} / {}".format(total, pairs.shape[0]))
    return total/pairs.shape[0]
