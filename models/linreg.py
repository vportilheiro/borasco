import numpy as np

class LinReg:

    def __init__(self, trainData, devData): 
        self.trainData = trainData.T
        self.devData = devData.T

    def get_variance(self, X, y, c, phi):
        x = X[:,1]
        v = phi*x + c
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
                X = np.ones((spread.size-1,2))
                X[:,1] = spread[:-1]

                # Label vector, based on spread over time, not including
                # first spread
                y = spread[1:]

                # Learn parameter theta by normal equation
                theta = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
                c, phi = theta[0], theta[1]

                variance = self.get_variance(X, y, c, phi)
                if(np.absolute(phi) < 0.8):     
                    c_phi_list += [ [c, phi] ]
                    pair_list += [ [i, j] ]
                    var_list += [ [variance] ]
                del theta 
                del a_ts
                del b_ts
                del y
        return np.array(c_phi_list), np.array(pair_list), np.array(var_list)


    def get_pairs(self): 
        c_phi_matrix, pair_matrix, var_matrix = self.get_AR1_params()
        phi_mask = np.argsort(c_phi_matrix[:,1])[:20]

        self.c_phi_matrix = c_phi_matrix[phi_mask]
        pair_matrix = pair_matrix[phi_mask]
        self.var_matrix = var_matrix[phi_mask]

        # TODO: get best performers
        return pair_matrix

    def get_trades(self, pairs): 
        pair2trades = {}
        for pair in pairs: 
            print("computing pair: {}".format(pair))
            pair2trades[pair] = self.get_pair_trades(self.trainData, self.devData, pair) 
        return pair2trades

    def get_pair_trades(self, train, dev, pair): 
        a = pair[0]
        b = pair[1]
        a_ts = train[:, a]/train[0, a]
        b_ts = train[:, b]/train[0, b]

        # Build design matrix X based based on spread over time,
        # not including last spread
        spread = a_ts - b_ts
        X = np.ones((spread.size-1,2))
        X[:,1] = spread[:-1]

        # Label vector, based on spread over time, not including
        # first spread
        y = spread[1:]

        # Learn parameter theta by normal equation
        theta = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
        
        a_ts = Y[:, a]/X[0, a]
        b_ts = Y[:, b]/X[0, b]
        i = 0
        while(i < len(a_ts)): 
            #print("timestep: {}".format(i/len(a_ts)))
            if np.absolute(a_ts[i] - b_ts[i]) >= threshold:
                clean_exit = True 
                trade_start = i

                if a_ts[i] > b_ts[i]: 
                    while (i < len(a_ts) and a_ts[i] > b_ts[i]): 
                        i += 1
                    i-=1
                    if i == len(a_ts) - 1 and a_ts[i] > b_ts[i]: 
                        clean_exit = False

                elif a_ts[i] < b_ts[i]: 
                    while (i < len(a_ts) and a_ts[i] < b_ts[i]): 
                        i += 1
                    i-=1
                    if i == len(a_ts) - 1 and a_ts[i] < b_ts[i]: 
                        clean_exit = False

                trades.append((trade_start, i, clean_exit))
            i += 1
        return trades 
    
