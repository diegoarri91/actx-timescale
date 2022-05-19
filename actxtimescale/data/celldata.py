import os

import numpy as np
import pandas as pd
import scipy.io

from spiketrain.sptr import SpikeTrains

from actxtimescale.data.stimdata import StimData
from actxtimescale.utils import float_to_int


class CellData:
    
    def __init__(self, folder):
        self.folder = folder
        stimuli_mat, spikes_mat = self.load_mat_files()

        self.arg_spikes = spikes_mat['timestamps']
        self.dt = 1. / float(spikes_mat['param']['samplerate'][0]) * 1000
        self.n_samples = spikes_mat['param']['nsamples'][0]

        self.df = self.build_dataframe(stimuli_mat) if len(stimuli_mat['stimuli']) > 0 else None

    def build_dataframe(self, stimuli_mat):
        stim_type = np.array([stimuli_mat['stimuli']['type'][ii][0][0] \
                              for ii in range(len(stimuli_mat['stimuli']['type']))])

        stimlength = np.array([float(stimuli_mat['stimuli']['stimlength'][ii]) \
                               for ii in range(len(stimuli_mat['stimuli']['stimlength']))])

        trigger = np.array([int(stimuli_mat['stimuli']['trigger'][ii]) \
                            for ii in range(len(stimuli_mat['stimuli']['trigger']))]) * self.dt

        tone_params = ['amplitude', 'ramp', 'frequency']
        sweep_params = ['start_frequency', 'stop_frequency', 'speed', 'method', 'next']
        pulse_params = ['start', 'width', 'height', 'npulses', 'isi']
        click_params = ['clickduration', 'nclicks']
#         wn_params = ['amplitude', 'duration', 'ramp']

        df = pd.DataFrame(np.stack((stim_type, trigger, stimlength), axis=1),
                               columns=['type', 'trigger', 'duration'])
        df['trigger'] = df['trigger'].astype(float)

        for col in tone_params + sweep_params + pulse_params + click_params:
            df[col] = np.nan
        for ii, idx in enumerate(df.index):
            cols = list(stimuli_mat['stimuli'][ii][0][1].dtype.fields.keys())
            cols.remove('duration')
            vals = [stimuli_mat['stimuli'][ii][0][1][f][0][0][0][0] for f in cols]
            df.loc[idx, cols] = vals
        df['method'] = df['method'].astype(str)
        df.loc[df['method'] == 'l', 'method'] = 'logarithmic'

        df['speed'] =  df['speed'].abs()
        df['sweep_direction'] = df['stop_frequency'] - df['start_frequency']
        df['sweep_direction'] = df['sweep_direction'].map(lambda x: 'up' if x > 0 else 'down' if x < 0 else 'NaN')

        df['speed_abs'] = df['speed'].abs().copy()
        df['speed'] = df['speed'] * df['sweep_direction'].map(dict(up=1, down=-1))
        df[['speed', 'speed_abs']] = float_to_int(df.copy(), columns=['speed', 'speed_abs'])
        df['duration'] = df['duration'].astype(float)

        return df
        
    def get_mask_spikes_trials(self, dic_stim, t0=-300, tf=1800):
    
        if dic_stim is not None:
            df_trials = self.df[pd.concat([self.df[key] == val for key, val in dic_stim.items()], axis=1).all(1)].reset_index(drop=True)
        else:
            df_trials = self.df.copy()
            
        df_trials = df_trials.sort_values('trigger', ascending=True)
#         display(df_trials)

        arg0, argf = int(t0 / self.dt), int(tf / self.dt)
        
        if len(df_trials) == 0:
            return None
        
        mask_spikes = np.zeros((argf - arg0, len(df_trials)), dtype=bool)
#         print(mask_spikes.shape)
        for ii, idx in enumerate(df_trials.index):
            _arg_trigger = int(df_trials.loc[idx, 'trigger'] / self.dt)
            _arg0, _argf = _arg_trigger + np.array([arg0, argf])
            _arg_spikes = self.arg_spikes[(self.arg_spikes >= _arg0) & (self.arg_spikes < _argf)]
            mask_spikes[_arg_spikes - _arg_trigger - arg0, ii] = True

        return mask_spikes
    
    def get_mask_spikes_stim(self, dic_filter, list_grouping, t0=-300, tf=1800):
    
        df_stim = self.df[pd.concat([self.df[key] == val for key, val in dic_filter.items()], axis=1).all(1)]

        if len(df_stim) > 0:
            stim_cats = df_stim[list(dic_filter.keys()) + list_grouping + ['trigger']]
        #     stim_cats = stim_cats.drop_duplicates(l).sort_values(l).reset_index(drop=True)
            stim_cats = stim_cats.groupby(list(dic_filter.keys()) + list_grouping)['trigger']\
                .count().reset_index().sort_values(list_grouping)
            stim_cats = stim_cats.rename(dict(trigger='n_trials'), axis=1)

            arg0, argf = int(t0 / self.dt), int(tf / self.dt)
            t = np.arange(arg0, argf, 1) * self.dt
            n_trials = int(stim_cats['n_trials'].max())
            mask_spikes = np.zeros((len(t), n_trials, len(stim_cats))) * np.nan

            for ii, idx in enumerate(stim_cats.index):
                row = stim_cats.loc[idx, list_grouping]
                dic_stim = row.to_dict()            
                mask_spikes_trials = self.get_mask_spikes_trials(dic_stim, t0=t0, tf=tf)
                mask_spikes[:, :mask_spikes_trials.shape[1], ii] = mask_spikes_trials.copy()
        else:
            raise ValueError('There are no rows with the required fields')

        return stim_cats, mask_spikes

    def get_stimdata(self, dic_filter, list_grouping, t0=0, tf=200):
        stim_cats, mask_spikes = self.get_mask_spikes_stim(dic_filter, list_grouping, t0=t0, tf=tf)
#         t = np.arange(t0, tf, cd.dt)
        t = np.arange(0, mask_spikes.shape[0], 1) * self.dt + t0
        stim_data = StimData(st=SpikeTrains(t, mask_spikes), df=stim_cats, t0=t0, tf=tf, neuron=self.folder)
        return stim_data

    def get_mask_spikes_after_stimulus(self, idx, t0=50, tf=1600):

        arg0, argf = int(t0 / self.dt), int(tf / self.dt)
        t = np.arange(arg0, argf, 1) * self.dt
        mask_spikes = np.zeros((len(t), len(idx)), dtype=bool)

        for ii, _idx in enumerate(idx):
#             _arg_trigger = int(self.df.loc[_idx, 'trigger'] / self.dt)
            _arg_end_stimulus = int((self.df.loc[_idx, 'trigger'] + self.df.loc[_idx, 'duration']) / self.dt)
            _arg0, _argf = _arg_end_stimulus + np.array([arg0, argf])
            _arg_spikes = self.arg_spikes[(self.arg_spikes >= _arg0) & (self.arg_spikes < _argf)]
            mask_spikes[_arg_spikes - _arg_end_stimulus - arg0, ii] = True

        return mask_spikes
    
    def get_mask_spikes_from_idx(self):

        arg0, argf = int(t0 / self.dt), int(tf / self.dt)
        t = np.arange(arg0, argf, 1) * self.dt
        mask_spikes = np.zeros((len(t), len(idx)), dtype=bool)

        for ii, _idx in enumerate(idx):
            _arg_trigger = int(self.df.loc[_idx, 'trigger'] / self.dt)
#             print(_arg_trigger)
            _arg0, _argf = _arg_trigger + np.array([arg0, argf])
            _arg_spikes = self.arg_spikes[(self.arg_spikes >= _arg0) & (self.arg_spikes < _argf)]
            mask_spikes[_arg_spikes - _arg_trigger - arg0, ii] = True

        return mask_spikes
    
    def get_mask_spikes_from_triggers(self, triggers, t0=0, tf=1800):

        arg0, argf = int(t0 / self.dt), int(tf / self.dt)
        t = np.arange(arg0, argf, 1) * self.dt
        mask_spikes = np.zeros((len(t), len(triggers)), dtype=bool)

        for ii, _idx in enumerate(triggers):
            _arg_trigger = int(_idx / self.dt)
            _arg0, _argf = _arg_trigger + np.array([arg0, argf])
            _arg_spikes = self.arg_spikes[(self.arg_spikes >= _arg0) & (self.arg_spikes < _argf)]
            mask_spikes[_arg_spikes - _arg_trigger - arg0, ii] = True

        return mask_spikes

    def load_mat_files(self):
        list_files = os.listdir(self.folder + '/')
        for f in list_files:
            if 'stimuli' in f:
                stim_file = f
            elif 'tt_spikes' in f:
                spikes_file = f
            else:
                pass

        stimuli_mat = scipy.io.loadmat(self.folder + '/' + stim_file)
        spikes_mat = scipy.io.loadmat(self.folder + '/' + spikes_file)

        return stimuli_mat, spikes_mat