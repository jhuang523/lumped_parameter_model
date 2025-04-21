import pandas as pd
import numpy as np
from scipy import optimize
from sklearn.metrics import mean_squared_error
from utils import * 
from functools import partial

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
        qs = max(2. * kbar * w * b * (ht-hs) / l, 0)#TODO: figure out why 
        qin = r * w * l
        h = ht + (qin - qs) * dt / w / l / sy
        if verbose:
            print (f"k = {kbar}, qs: {qs}, qin: {qin}, h: {h}")
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
    def optimize_K(self, K, ht, rt, dt, Q_val, h_val, Q_mean, Q_std, h_mean, h_std,  error_fun, weight_Q =.5, weight_h = 0.5, verbose = False):
        def objective(args, ht, rt, dt, error_fun): #give recharge data and observed data (validation)
            kbar = args[0]
            h, Q= self.explicit_solve(ht, rt, dt, kbar = kbar)
            h_error = (error_fun([h_val], [h]) - h_mean)/h_std
            Q_error = (error_fun([Q_val], [Q]) - Q_mean)/Q_std
            total_error = np.sqrt(weight_h * h_error**2 + weight_Q * Q_error**2) #+ 2*corr_coeff*h_error*Q_error)
            if verbose: 
                print(h_error, Q_error)
                print(total_error)
            return total_error
        params = (K)
        result = optimize.minimize(objective, params, args = (ht, rt, dt, error_fun), bounds = [(0, None)])
        return result
    def optimize_K_by_timestep(self, K_o, Q, h, r, verbose = False, **params):
        weight_Q = params.get('weight_Q', 0.5)
        weight_h = params.get('weight_h', 0.5)
        Ks = pd.DataFrame(columns = ['K'])
        Q_mean = Q.mean()
        Q_std = Q.std()
        h_mean = h.mean()
        h_std = h.std()
        for date in h.index[:-1]:
            try: 
                ht = h.loc[date][0]
            except:
                ht = h.loc[date]
            try:
                next_date = add_to_timestamp(date, 1, 'day')
                try:
                    h_val = h.loc[next_date][0]
                except:
                    h_val = h.loc[next_date]
                try:
                    Q_val = Q.loc[next_date][0]
                except:
                    Q_val = Q.loc[next_date]
                try:
                    rt = r.loc[date][0]
                except:
                    rt = r.loc[date]
            except Exception as e:
                print(e)
                print(f'skipping {date}')
                continue
            dt = subtract_timestamps(next_date, date, 'day')
            result = self.optimize_K(K_o, ht, rt, dt, Q_val, h_val, Q_mean, Q_std, h_mean, h_std, mean_squared_error, weight_Q = weight_Q, weight_h = weight_h, verbose=verbose)
            if verbose:
                print(f"Q metrics: std = {Q_std}, mean = {Q_mean}")
                print(f"h metrics: std = {h_std}, mean = {h_mean}")
            K= result.x[0]
            Ks.loc[date] = [K]
        return Ks

class TwoLayerKModel(Model): #variable K model, where K can vary as a function of time, depth, groundwater level
    def __init__(self, K1, K2, hk, **params): #K dict should have format {K val : [elev range]}
        self.K1 = K1
        self.K2 = K2
        self.hk = hk
        self.kbar = None
        super().__init__(**params)
    
    def explicit_solve(self, ht, r, dt, verbose=False, **params):
        K1 = params.get("K1", self.K1)
        K2 = params.get("K2", self.K2)
        hk = params.get("hk", self.hk)
        kbar = self.kbar
        if ht >= hk:
            kbar = K1 * (ht-hk)/(ht-self.hs) + K2 * (hk-self.hs)/(ht-self.hs)
        else:
            kbar = K2
        if verbose:
            print(f"k: {kbar}")
        h, qs = super().explicit_solve(ht, r, dt, verbose, kbar = kbar, **params)
        return h, qs, kbar
    def explicit_solve_over_time(self, n_timesteps, h0, r, **params):
        w = params.get('w', self.w)
        l = params.get('l', self.l)
        K1 = params.get('K1', self.K1)
        K2 = params.get('K2', self.K2)
        hk = params.get('hk', self.hk)
        b = params.get('b', self.b)
        hs = params.get('hs', self.hs)
        sy = params.get('sy', self.sy)
        dt = params.get('dt', self.dt)
        verbose = params.get('verbose', False)
        df = pd.DataFrame(columns = ['timestep', 'discharge', 'head', 'K'])
        if isinstance(r, float) or isinstance(r, int):
            r = r * np.ones(n_timesteps)
        for t in range(n_timesteps):
            ts = t * dt
            ht = h0 if t == 0 else h
            r0 = r.iloc[t]
            h, qs, k = self.explicit_solve(ht, r0, dt, verbose = verbose, K1 = K1, K2 = K2, hk = hk , w =w, l = l, b = b, hs = hs, sy = sy)
            df.loc[len(df)] = [ts, qs, h, k]
        return df
    def optimize_K_2layer(self, K1, K2, hk, ht, rt, dt, Q_val, h_val, Q_mean, Q_std, h_mean, h_std, corr_coeff, error_fun, verbose = False):
        def objective(args, ht, rt, dt, error_fun): #give recharge data and observed data (validation)
            K1 = args[0]
            K2 = args[1]
            hk = args[2]
            h, Q, k = self.explicit_solve(ht, rt, dt, K1 = K1, K2= K2, hk = hk)
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
    def optimize_K1(self, K1, K2, hk, ht, rt, dt, Q_val, h_val, Q_mean, Q_std, h_mean, h_std, corr_coeff, error_fun, verbose = False):
        def objective(args, K2, hk, ht, rt, dt, error_fun): #give recharge data and observed data (validation)
            K1 = args[0]
            h, Q, k = self.explicit_solve(ht, rt, dt, K1 = K1, K2= K2, hk = hk)
            h_error = (error_fun([h_val], [h]) - h_mean)/h_std
            Q_error = (error_fun([Q_val], [Q]) - Q_mean)/Q_std
            total_error = np.sqrt(h_error**2 + Q_error**2) #+ 2*corr_coeff*h_error*Q_error)
            if verbose: 
                print(h_error, Q_error)
                print(total_error)
            return total_error
        params = (K1)
        result = optimize.minimize(objective, params, args = (K2, hk, ht, rt, dt, error_fun), bounds = [(0, None)])
        return result
    def optimize_K1_by_timestep(self, K1_o, K2_o, hk_o, Q, h, r, verbose = False, **params):
        Ks = pd.DataFrame(columns = ['K1'])
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
            K1= result.x[0]
            Ks.loc[date] = [K1]
        return Ks
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
            K1= result.x[0]
            K2 = result.x[1]
            hk = result.x[2]
            Ks.loc[date] = [K1, K2, hk]
        return Ks

class ParametrizedKModel(Model):
    def __init__(self, func_K = None, **params):
        self.func_K = func_K #function that returns K (float) as a function Q
        self.kbar = None
        super().__init__(**params)
    
    def calculate_K(self, Qt, **params):
        func_K = params.get('func_K', self.func_K)
        return func_K(Qt)
    
    def explicit_solve(self, ht, Qt, r, dt, verbose=False, **params):
        func_K = params.get("func_K", self.func_K)
        kbar = self.calculate_K(Qt, func_K = func_K)
        h, qs = super().explicit_solve(ht, r, dt, verbose = verbose, kbar = kbar, **params)
        return h, max(qs, 0), kbar
    
    def explicit_solve_over_time(self, n_timesteps, h0, Q0, r, **params):
        w = params.get('w', self.w)
        l = params.get('l', self.l)
        b = params.get('b', self.b)
        hs = params.get('hs', self.hs)
        sy = params.get('sy', self.sy)
        dt = params.get('dt', self.dt)
        func_K = params.get('func_K', self.func_K)
        verbose = params.get('verbose', False)
        df = pd.DataFrame(columns = ['timestep', 'Q', 'h', 'k'])
        if isinstance(r, float) or isinstance(r, int):
            r = r * np.ones(n_timesteps)
        qs = Q0
        ts = 0
        for t in range(n_timesteps):
            if isinstance(dt, (list, np.ndarray, pd.DataFrame)):
                dt_t = dt[t]
            else:
                dt_t = dt
            ts += dt_t
            ht = h0 if t == 0 else h
            r0 = r.iloc[t]
            if verbose:
                print(r0)
            h, qs, k = self.explicit_solve(ht, qs, r0, dt_t, verbose = verbose, w =w, l = l, b = b, hs = hs, sy = sy, func_K = func_K)
            df.loc[len(df)] = [ts, qs, h, k]
        return df
    def calibrate_linear_param_K(self, slope_o, b_o, rt, Q_val, h_val, error_fun = mean_squared_error, weight_Q =.5, weight_h = 0.5, verbose = False, **params):
        def objective(args, error_fun): #give recharge data and observed data (validation)
            slope = args[0]
            b = args[1]
            self.linear_k_q(slope, b)
            dt = np.array(h_val.index.diff().total_seconds()/86400)[1:]
            Q_mean = Q_val.mean()
            Q_std = Q_val.std()
            h_mean = h_val.mean()
            h_std = h_val.std()
            pred = self.explicit_solve_over_time(len(dt), h_val.iloc[0], Q_val.iloc[0], rt, dt = dt, verbose = verbose)
            if np.any(np.isnan(pred['h'])) or np.any(np.isnan(pred['Q'])):
                return 1e10
            h_error = (error_fun(h_val.iloc[1:], pred['h']) - h_mean)/h_std
            Q_error = (error_fun(Q_val.iloc[1:], pred['Q']) - Q_mean)/Q_std
            total_error = np.sqrt(weight_h * h_error**2 + weight_Q * Q_error**2) #+ 2*corr_coeff*h_error*Q_error)
            if verbose: 
                print(f"h_error: {h_error}, Q error: {Q_error}")
                print(total_error)
            return total_error
        params = (slope_o, b_o)
        result = optimize.minimize(objective, params, args = (error_fun), bounds = [(0, None), (0, None)], method = 'Powell')
        return result
    def linear_k_q(self, slope, b):
        def linear_func(q, slope, b):
            return q * slope + b
        self.func_K = partial(linear_func, slope = slope, b = b)
        print(f'set linear K function with slope {slope} and intercept {b}')
