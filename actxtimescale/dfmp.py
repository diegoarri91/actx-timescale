from functools import partial
from inspect import signature
import multiprocessing as mp

import pandas as pd


def apply(row, metadata, kwargs, apply_fun, manager_dic):
    idx, cols = row[0], row[1]
    pars_names = signature(apply_fun).parameters.keys()
#     cols = {key: val for key, val in cols.items() if key in apply_fun.__code__.co_varnames}
    cols = {key: val for key, val in cols.items() if key in pars_names}
    metadata = {key: val for key, val in metadata.items() if key in pars_names}
    kwargs = {} if kwargs is None else kwargs
    manager_dic[idx] = apply_fun(**{**cols, **metadata, **kwargs})

    
def apply_method(idx, kwargs, apply_method, manager_dic):
    kwargs = {} if kwargs is None else kwargs
    manager_dic[idx] = apply_method(idx, **kwargs)

    
class DataFrameMP:
    
    def __init__(self, df=None, **metadata):
        self.df = df
        for key, val in metadata.items():
            setattr(self, key, val)
            
    def new(self, df, **metadata):
        return self.__class__(df, **metadata)
    
    def get_metadata(self):
        metadata = {key: val for key, val in vars(self).items() if key != 'df'}
        return metadata
    
    def join(self, other, on=None, sort=False):
        dfmp = self.new(self.df.join(other.df, on=on, sort=sort), **self.get_metadata())
        return dfmp
    
    def append(self, other, ignore_index=False):
        dfmp = self.new(self.df.append(other.df, ignore_index=ignore_index), **self.get_metadata())
        return dfmp
    
    def apply(self, fun, kwargs=None, columns=None, processes=1):
        
        manager_dic = mp.Manager().dict()
        metadata = self.get_metadata()

        with mp.Pool(processes=processes, initargs=(manager_dic,)) as pool:
            pool.map(partial(apply, metadata=metadata, kwargs=kwargs, apply_fun=fun, manager_dic=manager_dic), zip(self.df.index, self.df.to_dict('records')))
            pool.close()
            pool.join()
        
        df = pd.DataFrame.from_dict(manager_dic, orient='index', columns=columns)
        df = df.loc[self.df.index, :] # sort indexes
        dfmp = self.new(df, **self.get_metadata())
        return dfmp
    
    def apply_method(self, method, kwargs=None, columns=None, processes=1):
        
        manager_dic = mp.Manager().dict()

        with mp.Pool(processes=processes, initargs=(manager_dic,)) as pool:
            pool.map(partial(apply_method, kwargs=kwargs, apply_method=method, manager_dic=manager_dic), self.df.index)
            pool.close()
            pool.join()
        
        df = pd.DataFrame.from_dict(manager_dic, orient='index', columns=columns)
        df = df.loc[self.df.index, :] # sort indexes
        dfmp = self.new(df, **self.get_metadata())
        return dfmp
