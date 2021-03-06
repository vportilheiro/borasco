import numpy as np
import pandas 
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from random import choice

N = 20

def sample_for_regression(spread, N): 
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    for i in range(N): 
        j = choice(range(1, len(spread)))
        X[i] = [spread[j], spread[j] - spread[j-1]]
        length = 0
        for k in range(j, len(spread)): 
            if spread[j]*spread[k] < 0:
                length = k - j
                break
        if length == 0: 
            length = len(spread)-j
        Y[i] = length
    return X, Y

def backtest_linear(pairs, T, P): 
    in_trade = False
    profit = 0
    trade_start=0
    counter = 0
    for pair in pairs: 
        print(counter)
        counter+=1
        A = list(T[pair[0]])
        B = list(T[pair[1]]) 

        #create linear model for pair
        A_train = np.array(P[pair[0]])
        A_train = A_train/A_train[0]
        B_train = np.array(P[pair[1]])
        B_train = B_train/B_train[0]
        signed_spread = A_train - B_train
        X, Y = sample_for_regression(signed_spread, 20)
        reg = LinearRegression()
        reg.fit(X, Y)
        print("R^2 score for {} and {}: {}".format(pair[0], pair[1], reg.score(X, Y)))
        #if counter == 6:
        #    plt.figure()
        #    plt.plot(range(len(signed_spread)), signed_spread)
        #    plt.title("{} and {}".format(pair[0], pair[1]))
        #    plt.savefig("lin_regress.png")


        #calculate threshold based on past data
        spread = np.absolute(np.array(P[pair[0]]) - np.array(P[pair[1]]))
        cur_spread = np.absolute(np.array(A) - np.array(B))
        signed_spread = np.array(A) - np.array(B)
        threshold = np.mean(spread) + 2*np.std(spread)
        i = 1
        while(i < len(A)):
            #if spread is big, enter trade
            if cur_spread[i] >= threshold and reg.predict([[signed_spread[i], signed_spread[i] - signed_spread[i-1]]]) < 12: 
                
                print("{}-{}, entered trade at {}, spread: {}".format(pair[0], pair[1], i, cur_spread[i]))
                trade_start = i
                in_trade = True

                if A[i] > B[i]: 
                    #stay in trade until cross
                    while(i < len(A) and A[i] > B[i]): 
                        i+=1
                    i-=1 
                    if i == len(A)-1 and A[i] > B[i]: 
                        break
                    profit += A[trade_start] - B[trade_start]
                    in_trade = False

                elif B[i] >= A[i]: 
                    #stay in trade until cross
                    while(i < len(A) and B[i] > A[i]): 
                        i+=1 
                    i-=1
                    if i == len(A)-1 and B[i] > A[i]: 
                        break
                    profit += B[trade_start] - A[trade_start]
                    in_trade = False
                print("exit: {}".format(i+1))
            i+=1
        
        #if in trade then we just exit position and cut losses
        if in_trade: 
            print("loss!")
            profit += -A[len(A)-1] + A[trade_start]
            profit += B[len(A)-1] - B[trade_start] 
    return profit



def backtest_baseline(pairs, T, P): 
    in_trade = False
    profit = 0
    trade_start=0
    counter = 0
    for pair in pairs: 
        print(counter)
        counter+=1
        A = list(T[pair[0]])
        B = list(T[pair[1]]) 

        #calculate threshold based on past data
        spread = np.absolute(np.array(P[pair[0]]) - np.array(P[pair[1]]))
        cur_spread = np.absolute(np.array(A) - np.array(B))
        threshold = np.mean(spread) + 2*np.std(spread)
        i = 0
        while(i < len(A)):
            #if spread is big, enter trade
            if cur_spread[i] >= threshold: 
                
                print("{}-{}, entered trade at {}, spread: {}".format(pair[0], pair[1], i, cur_spread[i]))
                trade_start = i
                in_trade = True

                if A[i] > B[i]: 
                    #stay in trade until cross
                    while(i < len(A) and A[i] > B[i]): 
                        i+=1
                    i-=1 
                    if i == len(A)-1 and A[i] > B[i]: 
                        break
                    profit += A[trade_start] - B[trade_start]
                    in_trade = False

                elif B[i] >= A[i]: 
                    #stay in trade until cross
                    while(i < len(A) and B[i] > A[i]): 
                        i+=1 
                    i-=1
                    if i == len(A)-1 and B[i] > A[i]: 
                        break
                    profit += B[trade_start] - A[trade_start]
                    in_trade = False
                print("exit: {}".format(i+1))
            i+=1
        
        #if in trade then we just exit position and cut losses
        if in_trade: 
            print("loss!")
            profit += -A[len(A)-1] + A[trade_start]
            profit += B[len(A)-1] - B[trade_start] 
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
    test = SnP[35:]
    SnP = SnP[0:35]
    stocks, scores = similarities(SnP)
    pairs = top_n(scores, N)
    pairs = [(stocks[x[0]], stocks[x[1]], x[2]) for x in pairs]

    #some info
    for x in pairs: 
        print("{} and {}: {}".format(x[0], x[1], x[2]))
    pair1 = pairs[5]
    plot_ts([SnP[pair1[0]], SnP[pair1[1]]], [pair1[0], pair1[1]], "Sample Pair_train", "sample_pair_train.png")
    plot_ts([test[pair1[0]], test[pair1[1]]], [pair1[0], pair1[1]], "Sample Pair_test", "sample_pair_test.png")


    print("BASELINE")
    profit = backtest_baseline(pairs, test, SnP)
    print(profit)
    print("\n\nBASELINE + LINEAR REGRESSION") 
    profit = backtest_linear(pairs, test, SnP)
    print(profit)
    


    
    

