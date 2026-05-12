
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from models.inr_decoder import INR_Decoder, LatentRegressor
from data_loading.dataset import Data
from utils import *


class AtlasBuilder:
    """
    Class to build an atlas from a training dataset.
    """
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        self.loss_criterion = Criterion(args).to(args['device'])
        self._init_atlas_training()
        self.train_on_data()

    def train_on_data(self):
        if len(self.args['load_model']['path']) > 0: self.generation(epoch_train=0) 
        loss_hist_epochs = []
        start_time = time.time()
        for epoch in range(self.args['epochs']['train']):
            if self.args['optimizer']['re_init_latents']: self.re_init_latents()
            loss = self.train_epoch(epoch, split='train')
            loss_hist_epochs.append(loss)
            print(f"Training: Epoch: {epoch}, Loss: {np.mean(loss_hist_epochs):.4f}, Total Time Epoch: {time.time() - start_time:.2f}s")
            self.generation(epoch) 
            self._update_scheduler(split='train')
        return np.mean(loss_hist_epochs)

    def train_epoch(self, epoch, split):
        self.inr_decoder[split].train() if split == 'train' else self.inr_decoder[split].eval()
        loss_hist_batches = []
        time_data_loader = time.time()
        for batch in self.dataloaders[split]:
            print(f"Split: {split}, Current Epoch: {epoch}, Time Loading Batch: {time.time() - time_data_loader:.2f}s")
            start_time = time.time()
            loss = self.train_batch(batch, epoch, split)
            loss_hist_batches.append(loss)
            print(f"Split: {split}, Current Epoch: {epoch}, Loss Batch: {loss:.4f}, Total Training Time Batch: {time.time() - start_time:.2f}s")
        return np.mean(loss_hist_batches)

    def train_batch(self, batch, epoch, split='train'):
        loss_hist_samples = []
        n_smpls = self.args['n_samples']
        seg_weight = self.args['optimizer']['seg_weight'] if split == 'train' else 0.0
        coords_batch, values_batch, conditions_batch, idx_df_batch, sample_values_batch, sample_weights_batch= to_device(batch)
        sample_iterator = range(0, idx_df_batch.shape[0], n_smpls)
        start_time = time.time()
        print(f"Split: {split}, Current Epoch: {epoch}, Starting Batch ...\n")
        for i, smpls in enumerate(sample_iterator):
            self.optimizers[split].zero_grad()
            coords = coords_batch[smpls:smpls + n_smpls]
            values = values_batch[smpls:smpls + n_smpls]
            idx_df = idx_df_batch[smpls:smpls + n_smpls].squeeze()
            conditions = conditions_batch[smpls:smpls + n_smpls] if split == 'train' else self.conditions_val[idx_df]
            sample_values = sample_values_batch[smpls:smpls + n_smpls]
            sample_weights = sample_weights_batch[smpls:smpls + n_smpls]
            N, C = coords.shape
            if epoch == 0:
                point_spread_size = 4
                point_spread_std = (0.02,0.02,0.02)
            elif epoch == 1:
                point_spread_size = 8
                point_spread_std = (0.02,0.02,0.02)
            else:
                point_spread_size = 16
                point_spread_std = (0.02,0.02,0.02)
            idx_df_ps = idx_df.repeat_interleave(point_spread_size)
            conditions_ps = conditions.repeat_interleave(point_spread_size, dim=0)
            trans = self.transformations[split][idx_df]
            trans_ps = trans.repeat_interleave(point_spread_size, dim=0)
            point_spread_std = torch.tensor(point_spread_std, device=coords.device, dtype=coords.dtype).view(1, 1, C)
            coords_tile = coords[..., None, :].tile(1, point_spread_size, 1)
            coords_tile_delta = torch.randn(coords_tile.shape, device=coords.device) * point_spread_std.to(
            coords.device)
            coords_voxel_tile = coords_tile + coords_tile_delta
            coords_voxel_tile_ = coords_voxel_tile.reshape((N*point_spread_size, C))
            with torch.autocast(device_type=self.device, enabled=self.args['amp']):
                values_p = self.inr_decoder[split](coords_voxel_tile_, self.latents[split], conditions_ps,
                                            trans_ps, idcs_df=idx_df_ps)
                _,C_P = values_p.shape #_:N*P
                values_p_ = values_p.reshape(N, point_spread_size,C_P).mean(1)
                loss = self.loss_criterion(values_p_, values, trans, sample_values = sample_values, sample_weights=sample_weights, 
                                           seg_weight=seg_weight)

            if self.args['amp']:    
                self.grad_scalers[split].scale(loss['total']).backward()
                self.grad_scalers[split].step(self.optimizers[split])
                self.grad_scalers[split].update()
            else:
                loss['total'].backward()
                self.optimizers[split].step()

            loss_hist_samples.append(loss['total'].item())
            if i % 100 == 0 or i == (len(sample_iterator) - 1):
                log_loss(loss, epoch, split, self.args['logging'])
                print(f"Split: {split}, Epoch: {epoch}, "
                      f"Elapsed Training Time Batch: {time.time() - start_time:.2f}s"
                      f"Progress: {i/len(sample_iterator):.2f},"
                      f"Loss: {np.mean(loss_hist_samples):.4f},")
        return np.mean(loss_hist_samples)
    
    def generation(self, epoch_train):
        """
        Generate the spatio-temporal atlas
        """
        if self.args['generate_cond_atlas']: self.generate_atlas(epoch_train, n_max=100)
        self.save_state(epoch_train)


    def generate_atlas(self, epoch=0, n_max=100):
        """
        Generate temporal atlas for each condition combination in self.args['atlas_gen']['conditions'].
        """
        print(f"Generating atlases (depending on resolution and number of atlases this may take some time) ...\n")
        self.inr_decoder['train'].eval()
        grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
        temp_steps = self.args['atlas_gen']['temporal_values']
        atlas_list = []
        with torch.no_grad():
            for temp_step in temp_steps:
                temp_step_normed = normalize_condition(self.args, 'scan_age', temp_step)
                mean_latent = self.get_mean_latent('scan_age', temp_step_normed, n_max=n_max)
                condition_vectors = generate_combinations(self.args, self.args['atlas_gen']['conditions'])
                cond_list = []
                for c_v in condition_vectors:
                    c_v = torch.tensor(c_v, dtype=torch.float32).to(self.device)
                    values_p = self.inr_decoder['train'].inference(grid_coords, mean_latent, c_v, 
                                                                   grid_shape, None)
                    seg = values_p[:, :, :, -1]
                    seg[seg==9] = 0
                    values_p[:, :, :, -1] = seg
                    cond_list.append(values_p.detach().cpu())
                    # free up GPU memory
                    torch.cuda.empty_cache()
                    
                atlas_list.append(torch.stack(cond_list, dim=-1))
        atlas_list = torch.stack(atlas_list, dim=-1) # [x, y, z, num_modalities, num_conditions, t]
        save_atlas(self.args, atlas_list, affine, temp_steps, condition_vectors, epoch=epoch)
        return atlas_list
    
    def get_mean_latent(self, condition_key, condition_mean, n_max=100, split='train'):
        """
        Regress gaussian weighted latent code from subjects weighted by distance to condition mean
        of the condition with condition_key. Weights are clipped to the closest n_max subjects.
        sigma is the standard deviation of the gaussian distribution used to weight the latents
        emperically we want +/- 2 stds (covering 95% of the weights) to span +/- "gaussian_span" weeks of scan age, e.g. 0.75 weeks.
        Therefore:
        - Full range of condition values is [-1, 1], i.e. 2. 
        - Full range of scan age is c_max - c_min = c_range, e.g. 46 - 37 = 9 for term neonates.
        - The ratio of condition values to weeks is 2 / c_range = c_ratio, e.g. 2 / 9 = 0.222 units per week.
        ==> 2 std = 0.75 weeks = 0.75 * c_ratio e.g. = 0.165 units.
        ==> sigma = 1 std = 0.5 * 0.75 weeks * c_ratio, e.g. = 0.0825 units for term neonates.
        # Finally, we scale the sigma by the condition scale factor in the config, as scan age is actually normalized to [-cond_scale, cond_scale]
        """
        c_ratio = 2 / (self.args['dataset']['constraints'][condition_key]['max'] - self.args['dataset']['constraints'][condition_key]['min'])
        span_weeks = self.args['atlas_gen']['gaussian_span']
        sigma = 0.5 * span_weeks * c_ratio
        sigma = sigma * self.args['atlas_gen']['cond_scale']

        latents = self.latents[split]
        condition_values, df_idcs = self.datasets[split].get_condition_values(condition_key, normed=True, device=self.device)
        assert len(condition_values) == len(latents), "Condition values (all entries from the dataframe) \
                                                       and latents must have the same length!"
        weights = torch.exp(-(condition_values - condition_mean)**2 / (2*(sigma**2)))
        n_max = min(n_max, len(weights))
        weights[torch.argsort(weights, descending=True)[n_max:]] = 0
        weights = weights / torch.sum(weights)
        weights = weights[:, None, None, None, None] # [n_subjects, *4D]
        mean_latent = torch.sum(latents * weights, dim=0, keepdim=True)
        return mean_latent
        

    def save_state(self, epoch, split='train'):
        if self.args['save_model']:
            log_dir = self.args['output_dir']
            torch.save({
                'epoch': epoch,
                'latents': self.latents[split].cpu(),
                'transformations': self.transformations[split].cpu(),
                'inr_decoder': self.inr_decoder[split].state_dict(),
                'tsv_file': self.datasets[split].tsv_file,
                'dataset_df': self.datasets[split].df,
                'args': self.args
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Saved model state to {os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")}')
        else:
            print(f'Not saving model state as save_model is set to False')

    def load_checkpoint(self, chkp_path=None, epoch=None):  
        chkp_path = os.path.join(chkp_path, f'checkpoint_epoch_{epoch}.pth')
        if not os.path.exists(chkp_path):
            raise FileNotFoundError(f'State file {chkp_path} not found!')
        chkp = torch.load(chkp_path, weights_only=False)
        # self.args = chkp['args']
        self._init_dataloading(chkp['tsv_file'], chkp['dataset_df'])
        self._init_inr(chkp['inr_decoder'], split='train')
        self._init_transformations(chkp['transformations'])
        self._init_latents(chkp['latents'])
        print(f'Loaded state from {chkp_path}')
    
    def _init_atlas_training(self):
        self.datasets, self.dataloaders = {}, {}
        self.inr_decoder, self.latents, self.transformations = {}, {}, {}
        self.optimizers, self.grad_scalers = {}, {}
        self.schedulers = {}
        chkp_path = self.args['load_model']['path']
        if len(chkp_path) > 0:
            self.load_checkpoint(chkp_path, self.args['load_model']['epoch'])
        else:
            self._init_dataloading(split='train')
            self._init_inr(split='train')
            self._init_transformations(split='train')
            self._init_latents(split='train')
        self._init_optimizer(split='train') # optimizer is not loaded from checkpoint


    def _init_dataloading(self, tsv_file=None, df_loaded=None, split='train'):
        shuffle = True if split == 'train' else False
        tsv_file =  pd.read_csv(self.args['dataset']['tsv_file'], sep='\t') if tsv_file is None else tsv_file
        self.datasets[split] = Data(self.args, tsv_file, split=split, df_loaded=df_loaded)
        self.dataloaders[split] = DataLoader(self.datasets[split], batch_size=self.args['batch_size'], 
                                             num_workers=self.args['num_workers'], shuffle=shuffle, 
                                             collate_fn=self.datasets[split].collate_fn, pin_memory=True)

        print(f"Initialized dataloader for {split} with {len(self.datasets[split])} subjects")

    def _init_inr(self, state_dict=None, split='train'):
        # get the number of active conditions
        self.args['inr_decoder']['cond_dims'] = sum([self.args['dataset']['conditions'][c] 
                                                     for c in self.args['dataset']['conditions']])
        self.inr_decoder[split] = INR_Decoder(self.args, self.device).to(self.device)
        if state_dict is not None:
            self.inr_decoder[split].load_state_dict(state_dict)

    def _init_transformations(self, tfs=None, split='train'):
        shape = (len(self.datasets[split]), max(self.args['inr_decoder']['tf_dim'], 6)) # at least 6 for rigid, 9 for rigid+scale
        tfs = torch.zeros(shape).to(self.device) if tfs is None else tfs.to(self.device)
        self.transformations[split] = nn.Parameter(tfs) if self.args['inr_decoder']['tf_dim'] > 0 else tfs # if tf_dim=0, set trafos to 0 and fix
        
    def _init_latents(self, lats=None, split='train'):
        shape = (len(self.datasets[split]), *self.args['inr_decoder']['latent_dim'])
        lats = torch.normal(0, 0.01, size=shape).to(self.device) if lats is None else lats.to(self.device)
        self.latents[split] = nn.Parameter(lats)

    def re_init_latents(self, split='train'):
        self.latents[split].data.normal_(0, 0.01)
        self.transformations[split].data.zero_()
        self.optimizers[split].zero_grad()
        
    def _init_optimizer(self, split='train'):
        params = [{'name': f'latents_{split}',
                   'params': self.latents[split],
                   'lr': self.args['optimizer']['lr_latent'],
                   'weight_decay': self.args['optimizer']['latent_weight_decay']}]
        
        if self.args['inr_decoder']['tf_dim'] > 0:
            params.append({'name': f'transformations_{split}',
                           'params': self.transformations[split],
                           'lr': self.args['optimizer']['lr_tf'],
                           'weight_decay': self.args['optimizer']['tf_weight_decay']})
        if split == 'train':
            params.append({'name': f'inr_decoder',
                           'params': self.inr_decoder[split].parameters(),
                           'lr': self.args['optimizer']['lr_inr'],
                           'weight_decay': self.args['optimizer']['inr_weight_decay']})
        self.optimizers[split] = optim.AdamW(params)
        self.grad_scalers[split] = GradScaler() if self.args['amp'] else None
        if self.args['optimizer']['scheduler']['type'] == 'cosine':
            self.schedulers[split] = CosineAnnealingLR(self.optimizers[split], T_max=self.args['epochs'][split], 
                                                       eta_min=self.args['optimizer']['scheduler']['eta_min'])
        else:
            self.schedulers[split] = None

    def _update_scheduler(self, split='train'):
        if self.schedulers[split] is not None:
            self.schedulers[split].step()

    def _seed(self):
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        np.random.seed(self.args['seed'])
    