import numpy as np

class NaiveModel:
    N = 20

    def __init__(self, trainData, devData): 
        self.trainData = trainData
        self.devData = devData

    def get_pairs(self): 
        X = np.array(self.trainData)
        (m, n) = X.shape
        D = []
        for i in range(n): 
            for j in range(i+1, n):
                sim_ij = np.linalg.norm((X[:,i]/X[0, i]) - (X[:, j]/X[0, j])) 
                D.append((i, j, sim_ij))
        pairs = sorted(D, key= lambda x: x[2])
        pairs = [(x[0], x[1]) for x in pairs][0:self.N] 
        return pairs

    def get_trades(self, pairs): 
        pair2trades = {}
        for pair in pairs: 
            print("computing pair: {}".format(pair))
            pair2trades[pair] = self.get_pair_trades(self.trainData, self.devData, pair) 
        return pair2trades

    def get_pair_trades(self, X, Y, pair): 
        a = pair[0]
        b = pair[1]
        a_ts = X[:, a]/X[0, a]
        b_ts = X[:, b]/X[0, b]

        spread = np.absolute(a_ts - b_ts)
        threshold = np.mean(spread) + 2*np.std(spread)
        trades = []
        
        a_ts = Y[:, a]/X[0, a]
        b_ts = Y[:, b]/X[0, b]
        i = 0
        while(i < len(a_ts)): 
            #print("timestep: {}".format(i/len(a_ts)))
            if np.absolute(a_ts[i] - b_ts[i]) >= threshold:
                trade_start = i

                if a_ts[i] > b_ts[i]: 
                    while (i < len(a_ts) and a_ts[i] > b_ts[i]): 
                        i += 1
                    i -= 1

                elif a_ts[i] < b_ts[i]: 
                    while (i < len(a_ts) and a_ts[i] < b_ts[i]): 
                        i += 1
                    i-=1

                if a_ts[trade_start] > b_ts[trade_start]:
                    trades.append((trade_start, i, -1/X[0,a], 1/X[0,b]))
                else:
                    trades.append((trade_start, i, 1/X[0,a], -1/X[0,b]))
            i += 1
        return trades 
    
