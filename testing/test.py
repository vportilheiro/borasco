import pandas as pd
import numpy as np

def profit(testData, pairs2trades, pairs):
    X = np.array(testData)
    total_profit = 0
    for (a, b) in pairs: 
        pair_profit = 0
        a_0 = testData[0][a]
        b_0 = testData[0][b]
        for (start, end, clean) in pairs2trades[(a, b)]: 
            if clean: 
                pair_profit += np.absolute((testData[start][a]/a_0) - (testData[start][b]/b_0)) 
            else: 
                if testData[start][a] > testData[start][b]: 
                    pair_profit += (1/a_0)*(testData[end][a] - testData[start][a]) - (1/b_0)*(testData[end][b] - testData[start][b]) 
                else: 
                    pair_profit += (1/b_0)*(testData[end][b] - testData[start][b]) - (1/a_0)*(testData[end][a] - testData[start][a]) 
        total_profit += pair_profit
    return total_profit / len(pairs)


def test(data, frac_train, model):   
    #split data
    (m, n) = data.shape
    train_rows = int(np.around(frac_train * m))
    train_data = data[0:train_rows]
    dev_data = data[train_rows:]
    M = model(train_data, dev_data)
    pairs = M.get_pairs()
    pairs2trades = M.get_trades(pairs)
    score = profit(data, pairs2trades, pairs)
       
    return score, pairs, pairs2trades 

if __name__ == "__main__": 
    print("testing module")



