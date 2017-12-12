import numpy as np
from scipy.stats import norm

class AdjustedMean:

    def __init__(self, trainData, devData, order, window_size): 
        self.trainData = trainData.T
        self.devData = devData.T
        self.p = order 
        self.q = window_size

    def get_variance(self, X, y, phis):
        x = X[:,1:]
        v = x.dot(phis) 
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
                a_ts = self.trainData[i]/self.trainData[i,0]
                b_ts = self.trainData[j]/self.trainData[j,0]

                # Build design matrix X based based on spread over time,
                # not including last spread
                spread = a_ts - b_ts
                X = np.ones((spread.size-self.p, self.p+1))
                (m, n) = X.shape

                for k in range(1, self.p+1): 
                    lagged = spread[k-1:]
                    X[:, k] = lagged[:len(X)] 


                # Label vector, based on spread over time, not including
                # first spread
                y = spread[self.p:]

                # Learn parameter theta by normal equation
                theta = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
                c, phi = theta[0], theta[1:]

                variance = self.get_variance(X, y, phi)
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
    def get_convergence_prob(self, theta, std_dev, spreads, time_limit, num_simulations=2500):
        c = theta[0]
        phis = theta[1:]
        mean = c/(1-np.sum(phis))
        
        # Run Monte-Carlo simulations to approximate
        # probability of reversion to mean in time_limit steps,
        # as well as expected reversion time
        num_reverted = 0
        reversion_time = 0
        x_0 = spreads
        noise = np.random.normal(loc=0, scale=std_dev, size =(num_simulations, time_limit))
        for i in range(num_simulations):
            last_spread = x_0
            spread = x_0[-1]
            for t in range(1, time_limit):
                spread = c + np.inner(phis, last_spread) + noise[i, t]
                if spread <= mean <= last_spread[-1] or last_spread[-1] <= mean <= spread:
                    num_reverted += 1
                    reversion_time += t
                    break
                last_spread = np.concatenate((last_spread[1:], [spread]))
        return num_reverted/num_simulations, reversion_time//num_simulations

    def get_pair_trades(self, train, dev, pair, revert_threshold=0.5, grow_threshold=0.2): 
        trades = []
        time_horizon = dev.shape[1]
        spread = self.get_spread(pair[0], pair[1], dev=True)
        in_trade = False
        trade_start = -1

        # Get pair price normalizers
        X_0 = self.trainData[pair[0],0]
        Y_0 = self.trainData[pair[1],0]

        # For given pair, get model mean, variance, and std deviation of noise
        idx = self.pair_to_idx[tuple(pair)]
        theta = self.c_phi_matrix[idx].copy()
        c = theta[0]
        phis = theta[1:]
        mean = c/(1-np.sum(phis))
        mean_moving = False
        process_variance = self.var_matrix[idx]/(1-np.inner(phis,phis))
        std_dev = np.sqrt(self.var_matrix[idx])

        # make trade helper
        def make_trade(start, t, mean, status):
            nonlocal trades
            if spread[trade_start] > mean:
                trades += [(trade_start, t, -1/X_0, 1/Y_0)]
            else:
                trades += [(trade_start, t, 1/X_0, -1/Y_0)]
            print("\tTraded! " + status)

        for t in range(max(self.p, self.q), time_horizon):
            # check for trade exit conditions
            if in_trade:
                # check trade condition
                if spread[t-1] <= mean <= spread[t] or spread[t] <= mean <= spread[t-1]:
                    make_trade(trade_start, t, mean, "good")
                    trade_start = -1
                    in_trade = False
                # exit failed trade
                elif t == trade_stop_time \
                        and np.abs(spread[t] - mean) > np.abs(spread[trade_start]):
                    make_trade(trade_start, t, mean, "bad")
                    trade_start = -1
                    in_trade = False

            # check that mean is stable, recalculating c if instability occured
            window = spread[t-self.q:t]
            window_mean = np.mean(window)
            w = window-window_mean
            window_variance = np.inner(w,w)/(self.q - 1)
            if np.abs(window_variance - process_variance) <= process_variance:
                if mean_moving:
                    c = window_mean * (1 - np.sum(phis))
                    theta[0] = c
                    mean = c/(1-np.sum(phis))
                    print("Shifted mean")
                    mean_moving = False
                revert_prob, exp_rev_time = self.get_convergence_prob(theta, \
                        std_dev, spread[t-self.p:t], time_horizon-t)
                grow_prob = norm.cdf((spread[t]-c-np.inner(phis, spread[t-self.p:t]))/std_dev)
                if spread[t] > mean:
                    grow_prob = 1 - grow_prob
                if revert_prob > revert_threshold and grow_prob < grow_threshold:
                    in_trade = True
                    trade_start = t 
                    trade_stop_time = t + exp_rev_time
            else:
                mean_moving = True

        return trades
