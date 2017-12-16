import numpy as np
from scipy.stats import norm

class LinReg:

    def __init__(self, trainData, devData, order): 
        self.trainData = trainData.T
        self.devData = devData.T
        self.p = order 

    def get_variance(self, X, y, c, phis):
        x = X[:,1:]
        v = x.dot(phis) 
        #v = phis.dot(x) + c
        w = y - v
        result = np.inner(w, w)
        result = result/v.size
        return result
        
    def get_AR1_params(self):
        c_phi_list = []
        pair_list = []
        var_list = []
        for i in range(self.trainData.shape[0]-1):
            for j in range(i+1, self.trainData.shape[0]):
                #print("pair: {}".format((i, j)))
                a_ts = self.trainData[i]/self.trainData[i,0]
                b_ts = self.trainData[j]/self.trainData[j,0]

                # Build design matrix X based based on spread over time,
                # not including last spread
                spread = a_ts - b_ts
                X = np.ones((spread.size-self.p, self.p+1))
                (m, n) = X.shape
                #print((m, n))

                #X[:,1] = spread[:-1]
                for k in range(1, self.p+1): 
                    lagged = spread[k-1:]
                    X[:, k] = lagged[:len(X)] 


                # Label vector, based on spread over time, not including
                # first spread
                y = spread[self.p:]

                # Learn parameter theta by normal equation
                theta = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
                c, phi = theta[0], theta[1:]

                variance = self.get_variance(X, y, c, phi)
                #print(c, phi)
                c_phi_list += [ np.concatenate(([c], phi)) ]
                pair_list += [ [i, j] ]
                var_list += [ [variance] ]
                del theta 
                del a_ts
                del b_ts
                del y
        print(np.array(c_phi_list))
        return np.array(c_phi_list), np.array(pair_list), np.array(var_list)

    def get_spread(self, a, b, dev=False):
        if dev:
            return self.devData[a]/self.trainData[a,0]-self.devData[b]/self.trainData[b,0]
        return self.trainData[a]/self.trainData[a,0]-self.trainData[b]/self.trainData[b,0]

    def get_pairs(self, phi_threshold = 0.9): 
        c_phi_matrix, pair_matrix, var_matrix = self.get_AR1_params()
        phi_norms = [np.linalg.norm(x[1:]) for x in c_phi_matrix]
        ##phi_mask = np.argsort(c_phi_matrix[:,1])[:20]
        phi_mask = np.argsort(phi_norms)[:20]

        self.c_phi_matrix = c_phi_matrix[phi_mask]
        pair_matrix = pair_matrix[phi_mask]
        self.var_matrix = var_matrix[phi_mask]
        self.pair_to_idx = dict([ (tuple(pair_matrix[i]),i) for i in range(len(pair_matrix))])

        # TODO: get best performers
        return pair_matrix

    def get_trades(self, pairs): 
        pair2trades = {}
        for pair in pairs: 
            print("computing pair: {}".format(pair))
            pair2trades[tuple(pair)] = self.get_pair_trades(self.trainData, self.devData, pair) 
        print("finished get_trades")
        return pair2trades

    # Default to 5000 Monte-Carlo simulations, experimentally shown to be accurate to 1%.
    def get_convergence_prob(self, pair, spreads, time_limit, num_simulations=2500):
        idx = self.pair_to_idx[tuple(pair)]
        theta = self.c_phi_matrix[idx]
        c = theta[0]
        phis = theta[1:]
        std_dev = np.sqrt(self.var_matrix[idx])
        mean = c/(1-np.sum(phis))
        
        # Run Monte-Carlo simulations to approximate
        # probability of reversion to mean in time_limit steps
        num_reverted = 0
        x_0 = spreads
        for i in range(num_simulations):
            last_spread = x_0
            spread = x_0[-1]
            for t in range(1, time_limit):
                noise = np.random.normal(loc=0, scale=std_dev)[0]
                spread = c + np.inner(phis, last_spread) + noise 
                if spread <= mean <= last_spread[-1] or last_spread[-1] <= mean <= spread:
                    num_reverted += 1
                    break
                last_spread = np.concatenate((last_spread[1:], [spread]))
        return num_reverted/num_simulations


    def get_pair_trades(self, train, dev, pair, revert_threshold=0.5, grow_threshold=2): 
        trades = []
        time_horizon = dev.shape[1]
        spread = self.get_spread(pair[0], pair[1], dev=True)
        in_trade = False
        trade_start = -1

        # Get price normalizers
        X_0 = self.trainData[pair[0],0]
        Y_0 = self.trainData[pair[1],0]

        # Get model mean
        idx = self.pair_to_idx[tuple(pair)]
        theta = self.c_phi_matrix[idx]
        c = theta[0]
        phis = theta[1:]
        #c, phi = self.c_phi_matrix[idx]
        std_dev = np.sqrt(self.var_matrix[idx])
        mean = c/(1-np.sum(phis))

        for t in range(self.p, time_horizon):
            if not in_trade:
                revert_prob = self.get_convergence_prob(pair, spread[t-self.p:t], time_horizon-t)
                grow_prob = norm.cdf((spread[t]-c-np.inner(phis, spread[t-self.p:t]))/std_dev)
                if spread[t] > mean:
                    grow_prob = 1 - grow_prob
                if revert_prob > revert_threshold and grow_prob < grow_threshold:
                    in_trade = True
                    trade_start = t 
            else:
                if spread[t-1] <= mean <= spread[t] or spread[t] <= mean <= spread[t-1]:
                    if spread[trade_start] > mean:
                        trades += [(trade_start, t, -1/X_0, 1/Y_0)]
                    else:
                        trades += [(trade_start, t, 1/X_0, -1/Y_0)]
                    trade_start = -1
                    in_trade = False
        if in_trade:
            if spread[trade_start] > mean:
                trades += [(trade_start, t, -1/X_0, 1/Y_0)]
            else:
                trades += [(trade_start, t, 1/X_0, -1/Y_0)]

        return trades
