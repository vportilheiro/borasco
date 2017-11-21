import numpy as np
import pandas 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

N = 20

def backtest_baseline(pairs, T): 
    in_trade = False
    profit = 0
    trade_start=0
    counter = 0
    for pair in pairs: 
        print(counter)
        counter+=1
        A = list(T[pair[0]])
        B = list(T[pair[1]]) 
        spread = np.absolute(np.array(A) - np.array(B))
        threshold = 2*np.std(spread)
        i = 0
        while(i < len(spread)):
            #enter trade
            if spread[i] >= threshold: 
                
                print("{}-{}, entered trade at {}, spread: {}".format(pair[0], pair[1], i, spread[i]))
                trade_start = i
                in_trade = True

                if A[i] > B[i]: 
                    #stay in trade until cross
                    while(i < len(spread) and A[i] > B[i]): 
                        i+=1
                    i-=1 
                    if i == len(spread)-1 and A[i] > B[i]: 
                        break
                    profit += A[trade_start] - B[trade_start]
                    in_trade = False

                elif B[i] >= A[i]: 
                    while(i < len(spread) and B[i] > A[i]): 
                        i+=1 
                    i-=1
                    if i == len(spread)-1 and B[i] > A[i]: 
                        break
                    profit += B[trade_start] - A[trade_start]
                    in_trade = False
                print("exit: {}".format(i+1))
            i+=1
        
        #if in trade then we just exit
        if in_trade: 
            print("loss!")
            profit += -A[len(spread)-1] + A[trade_start]
            profit += B[len(spread)-1] - B[trade_start] 
    return profit







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
    test = SnP[30:]
    SnP = SnP[0:30]
    stocks, scores = similarities(SnP)
    pairs = top_n(scores, N)
    pairs = [(stocks[x[0]], stocks[x[1]], x[2]) for x in pairs]

    #some info
    for x in pairs: 
        print("{} and {}: {}".format(x[0], x[1], x[2]))
    pair1 = pairs[6]
    plot_ts([SnP[pair1[0]], SnP[pair1[1]]], [pair1[0], pair1[1]], "Sample Pair_train", "sample_pair_train.png")
    plot_ts([test[pair1[0]], test[pair1[1]]], [pair1[0], pair1[1]], "Sample Pair_test", "sample_pair_test.png")


    profit = backtest_baseline(pairs, test)
    print(profit)
    


    
    

