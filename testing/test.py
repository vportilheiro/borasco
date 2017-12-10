import pandas as pd
import numpy as np

def profit_old(testData, pairs2trades, pairs):
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

def profit(testData, pairs2trades, pairs): 
    X = np.array(testData)
    total_profit = 0
    total_cost = 0
    pair_roi = []
    for pair in pairs: 
        A = pair[0]
        B = pair[1]
        pair_cost = 0
        pair_profit = 0
        for trade in pairs2trades[tuple(pair)]: 
            start, end, sharesA, sharesB = trade
            trade_cost = np.abs(sharesA)*testData[start, A] + np.abs(sharesB)*testData[start, B]
            trade_profit = sharesA*(testData[end, A] - testData[start, A]) + sharesB*(testData[end, B]- testData[start, B])
            pair_cost += trade_cost
            pair_profit += trade_profit
        

        total_profit += pair_profit 
        total_cost += pair_cost
        pair_roi.append(pair_profit/pair_cost)
    return total_profit, total_profit/total_cost, pair_roi
             
def visuals(M, pair, pairs2trades): 
    train_spread = M.get_spread(pair[0], pair[1])
    dev_spread = M.get_spread(pair[0], pair[1], dev=True)
    both = np.concatenate((train_spread, dev_spread), axis=0)
    trades = pairs2trades[pair]
    trades = [(a, b) for a, b, c, d in trades]
    trades = np.array(trades).flatten() + len(train_spread)
    c, phi = M.c_phi_matrix[M.pair_to_idx[pair]]
    mu = c / (1-phi)
    return both, trades, mu 

    


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
       
    return score, pairs, pairs2trades, M 

if __name__ == "__main__": 
    print("testing module")



