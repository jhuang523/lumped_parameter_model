import pandas as pd
import numpy as np
from scipy import optimize
from sklearn.metrics import mean_squared_error
from utils import * 

class Model(): #simplest model.
    def __init__(self, **params):
        self.__dict__.update(**params)

    def __repr__(self):
        return(self.__dict__)
    def explicit_solve(self, ht, r, dt, verbose = False, **params):
        w = params.get('w', self.w)
        l = params.get('l', self.l)
        kbar = params.get('kbar', self.kbar)
        b = params.get('b', self.b)
        hs = params.get('hs', self.hs)
        sy = params.get('sy', self.sy)
        qs = 2. * kbar * w * b * (ht-hs) / l #TODO: figure out why 
        qin = r * w * l
        h = qin + qs
        # h += w * l * sy * ht / dt
        # h = h * dt / w / l / sy
        # qin = r * w * l
        h = ht + (qin - qs) * dt / w / l / sy
        if verbose:
            print (f"qs: {qs}, qin: {qin}, h: {h}")
        return h, qs

    def explicit_solve_over_time(self, n_timesteps, h0, r, **params):
        w = params.get('w', self.w)
        l = params.get('l', self.l)
        kbar = params.get('kbar', self.kbar)
        b = params.get('b', self.b)
        hs = params.get('hs', self.hs)
        sy = params.get('sy', self.sy)
        dt = params.get('dt', self.dt)
        verbose = params.get('verbose', False)
        df = pd.DataFrame(columns = ['timestep', 'discharge', 'head'])
        if isinstance(r, float) or isinstance(r, int):
            r = r * np.ones(n_timesteps)
        for t in range(n_timesteps):
            ts = t * dt
            ht = h0 if t == 0 else h
            r0 = r.iloc[t]
            h, qs = self.explicit_solve(ht, r0, dt, verbose = verbose, kbar = kbar, w =w, l = l, b = b, hs = hs, sy = sy)
            df.loc[len(df)] = [ts, qs, h]
        return df
    def calibrate (self, params : tuple, loss_function, **other_params): #TODO: be able in pass in variables to calibrate 
        # def objective(args, r_data, val_data, error_fun): #give recharge data and observed data (validation)
        #     K = args[0]
        #     sy = args[1]
        #     h0 = args[2]
        #     model_out = self.explicit_solve_over_time(len(r_data), h0 = h0, r =  -1 * r_data, kbar = K, sy = sy)
        #     model_dis = model_out['discharge']
        #     error = error_fun(val_data, model_dis)
        #     return error
        return

class VariableKModel(Model): #variable K model, where K can vary as a function of time, depth, groundwater level
    def __init__(self, Ks = {}, **params): #K dict should have format {K val : [elev range]}
        self.Ks = Ks
        self.kbar = None
        super().__init__(**params)
    
    def explicit_solve(self, ht, r, dt, verbose=False, **params):
        Ks = params.get('Ks', self.Ks)
        kbar = self.kbar
        for k, elev in Ks.items():
            if ht >= elev[0] and ht < elev[1]:
                kbar = k
        if verbose:
            print(f"k: {kbar}")
        return super().explicit_solve(ht, r, dt, verbose, kbar = kbar, **params)
    def explicit_solve_over_time(self, n_timesteps, h0, r, **params):
        w = params.get('w', self.w)
        l = params.get('l', self.l)
        Ks = params.get('Ks', self.Ks)
        b = params.get('b', self.b)
        hs = params.get('hs', self.hs)
        sy = params.get('sy', self.sy)
        dt = params.get('dt', self.dt)
        verbose = params.get('verbose', False)
        df = pd.DataFrame(columns = ['timestep', 'discharge', 'head'])
        if isinstance(r, float) or isinstance(r, int):
            r = r * np.ones(n_timesteps)
        for t in range(n_timesteps):
            ts = t * dt
            ht = h0 if t == 0 else h
            r0 = r.iloc[t]
            h, qs = self.explicit_solve(ht, r0, dt, verbose = verbose, Ks = Ks , w =w, l = l, b = b, hs = hs, sy = sy)
            df.loc[len(df)] = [ts, qs, h]
        return df
    def optimize_K_2layer(self, K1, K2, hk, ht, rt, dt, Q_val, h_val, Q_mean, Q_std, h_mean, h_std, corr_coeff, error_fun, verbose = False):
        def objective(args, ht, rt, dt, error_fun): #give recharge data and observed data (validation)
            # sy = args[1]
            Ks = {K2 : [0, hk], K1 : [hk, 1000]}
            h, Q = self.explicit_solve(ht, rt, dt, Ks)
            h_error = (error_fun([h_val], [h]) - h_mean)/h_std
            Q_error = (error_fun([Q_val], [Q]) - Q_mean)/Q_std
            total_error = np.sqrt(h_error**2 + Q_error**2) #+ 2*corr_coeff*h_error*Q_error)
            if verbose: 
                print(h_error, Q_error)
                print(total_error)
            return total_error
            return error
        params = (K1, K2, hk)
        result = optimize.minimize(objective, params, args = (ht, rt, dt, error_fun), bounds = [(0, None), (0, None), (0, None)])
        return result
    def optimize_K_2layer_by_timestep(self, K1_o, K2_o, hk_o, Q, h, r, verbose = False, **params):
        Ks = pd.DataFrame(columns = ['K1', 'K2', 'hk'])
        corr_coeff = np.corrcoef(Q, h)[0,1]
        Q_mean = Q.mean()
        Q_std = Q.std()
        h_mean = h.mean()
        h_std = h.std()
        for date in h.index[:-1]:
            ht = h.loc[date]
            try:
                next_date = add_to_timestamp(date, 1, 'day')
                print(next_date)
                h_val = h.loc[next_date]
                Q_val = Q.loc[next_date]
                rt = r.loc[date]
            except:
                print(f'skipping {date}')
                continue
            dt = subtract_timestamps(next_date, date, 'day')
            result = self.optimize_K_2layer(K1_o, K2_o, hk_o, ht, rt, dt, Q_val, h_val, Q_mean, Q_std, h_mean, h_std, corr_coeff, mean_squared_error,  verbose=verbose)
            if verbose:
                print(f"Q metrics: std = {Q_std}, mean = {Q_mean}")
                print(f"h metrics: std = {h_std}, mean = {h_mean}")
                print(result)
            K1= result.x[0]
            K2 = result.x[1]
            hk = result.x[2]
            Ks.loc[date] = [K1, K2, hk]
        return Ks

    
        

