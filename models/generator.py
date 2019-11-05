import math

from ops import upscale

import torch
import torch.nn as nn

_EPS = 1e-5

class GeneratorBuilder(nn.Module):
    def __init__(self, config):
        super(GeneratorBuilder, self).__init__()
        
        self.config=config
        self.ngf_base = self.config["model_params"]["ngf_base"]
        self.num_extra_layers = self.config["model_params"]["g_extra_layers"]
        self.micro_patch_size = self.config["data_params"]["micro_patch_size"]
        self.c_dim = self.config["data_params"]["c_dim"]
        self.spatial_dim = self.config["model_params"]["spatial_dim"]
        
        init_sp = 2
        init_ngf_mult = 16
        in_ch = self.config["model_params"]["z_dim"] 
        out_ch = self.ngf_base*init_ngf_mult
        self.initial_layer = nn.Linear(in_ch+self.spatial_dim, out_ch*init_sp*init_sp)
        self.residual_block_main = nn.ModuleList()
        self.residual_block_residue = nn.ModuleList()
        self.residual_block_cbn = nn.ModuleList()
        # Stacking residual blocks
        num_resize_layers = int(math.log(min(self.micro_patch_size), 2) - 1)
        num_total_layers  = num_resize_layers + self.num_extra_layers
        basic_layers = [8, 4, 2] 
        if num_total_layers>=len(basic_layers):
            num_replicate_layers = num_total_layers - len(basic_layers)
            ngf_mult_list = basic_layers + [1, ] * num_replicate_layers
        else:
            ngf_mult_list = basic_layers[:num_total_layers]

        for idx, ngf_mult in enumerate(ngf_mult_list):
            n_ch = self.ngf_base * ngf_mult
            self.residual_block_cbn.append(nn.Linear(in_ch+self.spatial_dim, out_ch))
            self.residual_block_cbn.append(nn.Linear(in_ch+self.spatial_dim, out_ch))
            self.residual_block_main.append(nn.Conv2d(out_ch, n_ch, 3, 1, padding=1))
            self.residual_block_cbn.append(nn.Linear(in_ch+self.spatial_dim, n_ch))
            self.residual_block_cbn.append(nn.Linear(in_ch+self.spatial_dim, n_ch))
            self.residual_block_main.append(nn.Conv2d(n_ch, n_ch, 3, 1, padding=1))
            self.residual_block_residue.append(nn.Conv2d(out_ch, n_ch, 1, 1, padding=0))
            out_ch = n_ch

        self.bn = nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True)
        self.final_conv = nn.Conv2d(out_ch, self.c_dim, 3, 1, padding=1)


    def _cbn(self, x, y, residual_main_cbn_idx, is_training): # Spectral Batch Normalization
        ch = list(x.size())[1]
        gamma = self.residual_block_cbn[residual_main_cbn_idx](y)
        beta = self.residual_block_cbn[residual_main_cbn_idx+1](y)

        mean_rec = torch.zeros(ch, requires_grad=False)
        var_rec  = torch.ones(ch, requires_grad=False)
        running_mean = torch.mean(x, [0, 2, 3])
        running_var = torch.var(x, [0, 2, 3])
        
        if is_training:
            new_mean_rec = 0.99 * mean_rec + 0.01 * running_mean
            new_var_rec  = 0.99 * var_rec  + 0.01 * running_var
            mean = new_mean_rec
            var  = new_var_rec
        else:
            mean = mean_rec
            var  = var_rec
            
        mean  = mean.view(1, ch, 1, 1)
        var   = var.view(1, ch, 1, 1)
        gamma = gamma.view(-1, ch, 1, 1)
        beta  = beta.view(-1, ch, 1, 1)
        
        out = (x-mean) / (var+_EPS) * gamma + beta
        return out 


    def forward(self, z, coord, is_training):

        init_sp = 2
        init_ngf_mult = 16
        cond = torch.cat((z.float(), torch.tensor(coord).float()), 1)
        h = self.initial_layer(cond)
        h = h.view(-1, self.ngf_base*init_ngf_mult, init_sp, init_sp)

        # Stacking residual blocks
        num_resize_layers = int(math.log(min(self.micro_patch_size), 2) - 1)
        num_total_layers  = num_resize_layers + self.num_extra_layers
        basic_layers = [8, 4, 2] 
        if num_total_layers>=len(basic_layers):
            num_replicate_layers = num_total_layers - len(basic_layers)
            ngf_mult_list = basic_layers + [1, ] * num_replicate_layers
        else:
            ngf_mult_list = basic_layers[:num_total_layers]
        
        X=h
        residual_main_idx = 0
        residual_residue_idx = 0
        residual_main_cbn_idx = 0
        for idx, ngf_mult in enumerate(ngf_mult_list):
            # Standard layers first
            if idx < num_resize_layers:
                resize = True
            # Extra layers do not resize spatial size
            else:
                resize = False
            
            h = X
            h = self._cbn(h, cond, residual_main_cbn_idx, is_training)
            residual_main_cbn_idx+=2
            
            h = nn.functional.relu(h)
            if resize:
                h = upscale(h, 2)
                
            h = self.residual_block_main[residual_main_idx](h)
            residual_main_idx+=1
            
            h = self._cbn(h, cond, residual_main_cbn_idx, is_training)
            residual_main_cbn_idx+=2
            
            h = nn.functional.relu(h)
            h = self.residual_block_main[residual_main_idx](h)
            residual_main_idx+=1
    
            if resize:
                sc = upscale(X, 2)
            else:
                sc = X
            sc = self.residual_block_residue[residual_residue_idx](sc)
            residual_residue_idx+=1
            
            X = h + sc

        X = self.bn(X)
        X = nn.functional.relu(X)
        X = self.final_conv(h)
        return torch.tanh(X)