import math
_EPS = 1e-5

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Learnable Function for Discriminator
class DiscriminatorBuilder(nn.Module):
    def __init__(self, config):
        super(DiscriminatorBuilder, self).__init__()
        
        self.config=config
        self.ndf_base = self.config["model_params"]["ndf_base"]
        self.num_extra_layers = self.config["model_params"]["d_extra_layers"]
        self.macro_patch_size = self.config["data_params"]["macro_patch_size"]
        
        self.residual_block_main = nn.ModuleList()
        self.residual_block_residue = nn.ModuleList()
        
        out_ch = self.config["data_params"]["c_dim"]
        num_resize_layers = int(math.log(min(self.macro_patch_size), 2) - 1)
        num_total_layers  = num_resize_layers + self.num_extra_layers
        basic_layers = [2, 4, 8, 8]
        if num_total_layers>=len(basic_layers):
            num_replicate_layers = num_total_layers - len(basic_layers)
            ndf_mult_list = [1, ] * num_replicate_layers + basic_layers
        else:
            ndf_mult_list = basic_layers[-num_total_layers:]

        # Residual Block
        for idx, ndf_mult in enumerate(ndf_mult_list):
            n_ch = self.ndf_base * ndf_mult
            # Head is fixed and goes first
            if idx==0:
                resize= True
            # Extra layers before standard layers
            elif idx<=self.num_extra_layers:
                resize= False
            # Last standard layer has no resize
            elif idx==len(ndf_mult_list)-1:
                resize= False
            # Standard layers
            else:
                resize= True
                
            self.residual_block_main.append(nn.Conv2d(out_ch, n_ch, 3, 1, padding=1))
            self.residual_block_main.append(nn.Conv2d(n_ch, n_ch, 3, 1, padding=1))
            if resize:
                self.residual_block_main.append(nn.MaxPool2d(2, 2))
                self.residual_block_residue.append(nn.MaxPool2d(2, 2))
            self.residual_block_residue.append(nn.Conv2d(out_ch, n_ch, 1, 1, padding=0))
            out_ch = n_ch
            
        proj_out_ch = self.ndf_base*ndf_mult_list[-1]
        proj_in_ch = self.config["model_params"]["spatial_dim"]
        self.projection = nn.Linear(proj_in_ch, proj_out_ch, bias=torch.zeros).double()
        stddev = np.sqrt(2. / (proj_in_ch))
        self.projection.weight.data.uniform_(-stddev,stddev)
        
        
        self.global_linear = nn.Linear(proj_out_ch, 1, bias=torch.zeros)
        stddev = np.sqrt(2. / (proj_out_ch))
        self.global_linear.weight.data.uniform_(-stddev,stddev)
           
    
    # forward method            
    def forward(self, x, y=None, is_training=True):
        
        num_resize_layers = int(math.log(min(self.macro_patch_size), 2) - 1)
        num_total_layers  = num_resize_layers + self.num_extra_layers
        basic_layers = [2, 4, 8, 8]
        if num_total_layers>=len(basic_layers):
            num_replicate_layers = num_total_layers - len(basic_layers)
            ndf_mult_list = [1, ] * num_replicate_layers + basic_layers
        else:
            ndf_mult_list = basic_layers[-num_total_layers:]
            
        # Stack extra layers without resize first
        X = x
        residual_main_idx = 0
        residual_residue_idx = 0
        for idx, ndf_mult in enumerate(ndf_mult_list):
            # Head is fixed and goes first
            if idx==0:
                is_head, resize = True, True
            # Extra layers before standard layers
            elif idx<=self.num_extra_layers:
                is_head, resize = False, False
            # Last standard layer has no resize
            elif idx==len(ndf_mult_list)-1:
                is_head, resize = False, False
            # Standard layers
            else:
                is_head, resize = False, True
            
            h = X
            if not is_head:
                h = F.relu(h)
            h = self.residual_block_main[residual_main_idx](h)
            residual_main_idx+=1
            h = F.relu(h)
            h = self.residual_block_main[residual_main_idx](h)
            residual_main_idx+=1
            if resize:
                h = self.residual_block_main[residual_main_idx](h)
                residual_main_idx+=1
    
            # Short cut
            s = X
            if resize:
                s = self.residual_block_residue[residual_residue_idx](s)
                residual_residue_idx+=1
            s = self.residual_block_residue[residual_residue_idx](s)
            residual_residue_idx+=1
            
            X = h + s

        X = F.relu(X)
        X = torch.sum(X, (2,3)) # Global pooling
        last_feature_map = X
        adv_out = self.global_linear(X).type(torch.float64)
        
        # Projection Discriminator
        if y is not None:
            Y = torch.tensor(np.expand_dims(y,1), dtype=torch.float64)
            y_emb = self.projection(Y)
            proj_out = torch.sum(y_emb*X.type(torch.float64),(1,2), keepdim=True).view(-1,1)
            out = adv_out + proj_out
        else:
            out = adv_out
        
        return out, last_feature_map

