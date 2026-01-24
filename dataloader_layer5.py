from __future__ import print_function
import os
import copy
import pathlib
import h5py
import platform
import sys
import numpy as np
from scipy.stats import norm
from scipy import sparse
import pickle
import time
import argparse
import logging
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, explained_variance_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
# import confidenceinterval #? what is this? 
from torch.nn import functional as F
from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# import wandb #? what is this?

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

# from utils.roc_utils import window_roc_curve
# from utils.utils import setup_logger, str2bool, ArgumentSaver, AddDefaultInformationAction, AddOutFileAction, TeeAll
# from utils.slurm_job import get_job_args

logger = logging.getLogger(__name__)
INFINITE_VOLTAGE_CLIP = 9e9

class SimulationData(Dataset):
    def __init__(self, base_directory, window_size=700, v_clip=INFINITE_VOLTAGE_CLIP, v_offset=0,\
         start_t=500, overlap_size=None, normalized_test=False, test_name='test', remove_zero_simulations=True):
        super(SimulationData).__init__()

        self.v_clip = v_clip
        self.window_size = window_size
        self.base_directory = base_directory
        self.v_offset = v_offset
        self.start_t = start_t

        self.dataset_summary = pickle.load(open(f'{self.base_directory}/summary.pkl','rb'))

        if 'firing_rate_information' in self.dataset_summary and 'by_index' in self.dataset_summary['firing_rate_information']:
            bigger_than = 0 if remove_zero_simulations else -1
            self.simulation_indices = [k for k, v in self.dataset_summary['firing_rate_information']['by_index'].items() if v > bigger_than]
            available = set(os.listdir(base_directory))
            self.simulation_indices = [k for k in self.simulation_indices
                                        if f"simulation_{k}" in available]
                                        
            self.avg_firing_rate = np.mean([self.dataset_summary['firing_rate_information']['by_index'][k] for k in self.simulation_indices])
        else:
            # TODO: remove one day
            self.simulation_indices = list(range(self.dataset_summary['count_simulations']))
            self.avg_firing_rate = 1

        self.normalized_test = False
        if normalized_test:
            if 'firing_rate_information' in self.dataset_summary and f'normalized_{test_name}_set_by_index' in self.dataset_summary['firing_rate_information']:
                self.normalized_test = True
                self.simulation_indices = list(self.dataset_summary['firing_rate_information'][f'normalized_{test_name}_set_by_index'].keys())
                if f'normalized_{test_name}_average_firing_rate' in self.dataset_summary['firing_rate_information']:
                    self.avg_firing_rate = self.dataset_summary['firing_rate_information'][f'normalized_{test_name}_average_firing_rate']
            else:
                raise ValueError(f"Can't use normalized test set, no 'normalized_{test_name}_set_by_index' data in 'firing_rate_information'")      
                # logger.info(f"No 'normalized_{test_name}_set_by_index' data in 'firing_rate_information', normalized {test_name} will be regular {test_name}")

        self.count_simulations = len(self.simulation_indices)

        if isinstance(self.v_offset, tuple):
            if 'average_somatic_voltage_information' in self.dataset_summary:
                if self.v_clip < INFINITE_VOLTAGE_CLIP and 'average_clipped_somatic_voltage' in self.dataset_summary['average_somatic_voltage_information']:
                    self.v_offset = self.dataset_summary['average_somatic_voltage_information']['average_clipped_somatic_voltage']
                elif self.v_clip >= INFINITE_VOLTAGE_CLIP and 'average_somatic_voltage' in self.dataset_summary['average_somatic_voltage_information']:
                    self.v_offset = self.dataset_summary['average_somatic_voltage_information']['average_somatic_voltage']
                else:
                    self.v_offset = self.v_offset[0]    
            else:
                self.v_offset = self.v_offset[0]

        # test one file
        example_simulation = f'{self.base_directory}/simulation_0'
        voltage = h5py.File(f'{example_simulation}/voltage.h5', 'r')
        summary = pickle.load(open(f'{example_simulation}/summary.pkl','rb'))
        simulation_duration_in_ms = summary['simulation_duration_in_ms']
        example_simulation_spike_count = len(np.nonzero(summary['output_spike_times'])[0])
        logger.info(f'{example_simulation} has {example_simulation_spike_count} spikes')

        self.simulation_duration_in_seconds = self.dataset_summary['args'].simulation_duration_in_seconds
        self.simulation_duration_in_ms = self.dataset_summary['args'].simulation_initialization_duration_in_ms + self.simulation_duration_in_seconds * 1000
        if self.simulation_duration_in_ms != simulation_duration_in_ms:
            # TODO: remove one day
            logger.info(f"Simulation duration in ms {self.simulation_duration_in_ms} does not match the one in the file {simulation_duration_in_ms}")
            logger.info(f"Probably the simulations is an old one")
            logger.info(f"Using the one from the file")
            self.simulation_duration_in_ms = simulation_duration_in_ms

        self.overlap_size = int(self.window_size / 2) if overlap_size is None else overlap_size

        # remove the first window from the length and check how many overlays fit
        self.num_per_sim = int((self.simulation_duration_in_ms-start_t-window_size)/(window_size-self.overlap_size))+1

        logger.info(f"SimulationData({base_directory}) [normalized_test={self.normalized_test}] has {self.count_simulations*self.num_per_sim} samples and {(self.count_simulations*(self.simulation_duration_in_ms-start_t))/1000/60/60:.3f} hours of data with average firing rate {self.avg_firing_rate}")
        
    def __len__(self):
        return self.count_simulations * self.num_per_sim
        
    def __getitem__(self, idx, debug=False):
        simulation_n_orig = int(idx/self.num_per_sim)
        simulation_n = self.simulation_indices[simulation_n_orig]

        # # TODO: a hack, not really needed?
        # loaded_successfully = False
        # while not loaded_successfully:
        #     try:
        #         pickle.load(open(f'{self.base_directory}/simulation_{simulation_n}/summary.pkl','rb'))
        #         if os.path.exists(f'{sim_folder}/exc_weighted_spikes.npz'):
        #                 # TODO: remove one day
        #             sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/exc_weighted_spikes.npz').A
        #             sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/inh_weighted_spikes.npz').A
        #         else:
        #             sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/all_weighted_spikes.npz').A
        #         h5py.File(f'{self.base_directory}/simulation_{simulation_n}/voltage.h5','r')['somatic_voltage']
        #         loaded_successfully = True
        #     except Exception as e:
        #         logger.error(f"Error loading {self.base_directory}/simulation_{simulation_n}/summary.pkl")
        #         simulation_n_orig += 1
        #         simulation_n = self.simulation_indices[simulation_n_orig]

        st_pos = self.start_t + (self.window_size-self.overlap_size)*(idx%self.num_per_sim) 
        en_pos = st_pos + self.window_size

        sim_folder = f"{self.base_directory}/simulation_{simulation_n}"

        if os.path.exists(f'{sim_folder}/exc_weighted_spikes.npz'):
            # TODO: remove one day
            exc_weighted_spikes = sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/exc_weighted_spikes.npz').A
            inh_weighted_spikes = sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/inh_weighted_spikes.npz').A

            exc_weighted_spikes_for_window = exc_weighted_spikes[:,st_pos:en_pos]
            inh_weighted_spikes_for_window = inh_weighted_spikes[:,st_pos:en_pos]
                
            all_weighted_spikes_for_window = np.vstack((exc_weighted_spikes_for_window, inh_weighted_spikes_for_window))

            exc_weighted_spikes = None
            del exc_weighted_spikes
            inh_weighted_spikes = None
            del inh_weighted_spikes

        else:
            all_weighted_spikes = sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/all_weighted_spikes.npz').A
            all_weighted_spikes_for_window = all_weighted_spikes[:,st_pos:en_pos]

            all_weighted_spikes = None
            del all_weighted_spikes

        # get voltage
        somatic_voltage = h5py.File(f'{self.base_directory}/simulation_{simulation_n}/voltage.h5','r')['somatic_voltage']

        somatic_voltage_for_window = somatic_voltage[st_pos:en_pos]
        somatic_voltage_for_window[somatic_voltage_for_window>self.v_clip] = self.v_clip
        somatic_voltage_for_window = somatic_voltage_for_window - self.v_offset

        somatic_voltage = None
        del somatic_voltage

        # TODO: that should be a param
        # voltage_meaned = np.mean(somatic_voltage_for_window.reshape(-1,40),1)

        # get output spike times
        summary = pickle.load(open(f'{self.base_directory}/simulation_{simulation_n}/summary.pkl','rb'))

        output_spikes_for_window = np.zeros(self.window_size)
        spike_times = summary['output_spike_times'][(summary['output_spike_times']>=st_pos) & (summary['output_spike_times']<en_pos)]
        output_spikes_for_window[spike_times.astype(int)-st_pos] = 1

        r_item = {'sps_in':all_weighted_spikes_for_window,
                  'somatic_voltage_out':somatic_voltage_for_window,
                  'sps_out':output_spikes_for_window}
        return r_item
