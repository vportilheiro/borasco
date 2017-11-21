import numpy as np
import pandas 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

N = 20

def plot_ts(TS, labels, title, filename): 
    plt.figure()
    colors = ["r", "b"]
    for i ,ts in enumerate(TS): 
        plt.plot(range(len(ts)), ts, color = colors[i], label = labels[i])
    plt.legend()
    plt.title(title)
    plt.savefig(filename)


def top_n(M, N): 
    pairs = []
    n = M.shape[0]
    for i in range(n): 
        for j in range(i+1, n): 
            pairs.append((i, j, M[i, j]))
    pairs = sorted(pairs, key = lambda x: x[2])
    return pairs[:N]

def similarities(df): 
    stocks = list(df.columns[1:])
    n = len(stocks)
    D = np.zeros((n, n))
    for i in range(n): 
        for j in range(i+1, n):
            D[i, j] = dist(df[stocks[i]], df[stocks[j]])
    return stocks, D

def dist(A, B): 
    return np.linalg.norm(np.array(A) - np.array(B))

#divide time series by first price point
def normalize_prices(df): 
    for c in df.columns[1:]: 
        df[c] = [df[c][r]/df[c][0] for r in df.index]
    return df

if __name__ == "__main__": 
    SnP = pandas.read_csv("/Users/borauyumazturk/CS229/borasco/data/SP500_90day_ts.csv")
    SnP = normalize_prices(SnP)
    test = SnP[60:]
    SnP = SnP[0:60]
    stocks, scores = similarities(SnP)
    pairs = top_n(scores, N)
    pairs = [(stocks[x[0]], stocks[x[1]], x[2]) for x in pairs]

    #some info
    for x in pairs: 
        print("{} and {}: {}".format(x[0], x[1], x[2]))
    pair1 = pairs[0]
    plot_ts([SnP[pair1[0]], SnP[pair1[1]]], [pair1[0], pair1[1]], "Sample Pair", "sample_pair.png")

    #profit = backtest_baseline(pairs, test)
    #print(profit)
    


    
    

