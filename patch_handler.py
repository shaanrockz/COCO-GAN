import numpy as np
import torch 

class PatchHandler():

    def __init__(self, config):
        self.config = config

        self.batch_size = self.config["train_params"]["batch_size"]
        self.micro_patch_size = self.config["data_params"]["micro_patch_size"]
        self.macro_patch_size = self.config["data_params"]["macro_patch_size"]
        self.full_image_size = self.config["data_params"]["full_image_size"]
        self.coordinate_system = self.config["data_params"]["coordinate_system"]
        self.c_dim = self.config["data_params"]["c_dim"]

        self.num_micro_compose_macro = config["data_params"]["num_micro_compose_macro"]


    def reord_patches_cpu(self, x, batch_size, patch_count):
        # Reorganize image order from [a0, b0, c0, a1, b1, c1, ...] to [a0, a1, ..., b0, b1, ..., c0, c1, ...]
        select = np.hstack([[i*batch_size+j for i in range(patch_count)] for j in range(batch_size)])
        x_reord = np.take(x, select, axis=0)
        return x_reord

    def concat_micro_patches_cpu(self, generated_patches, ratio_over_micro):

        patch_count = ratio_over_micro[0] * ratio_over_micro[1]
        macro_patches = torch.zeros(int(generated_patches.shape[0]/patch_count), self.c_dim, self.macro_patch_size[0], self.macro_patch_size[1])
        
        idx  = list(range(0,generated_patches.shape[0],4))
        micro_patch = generated_patches[idx]
        macro_patches[:,:,0:self.micro_patch_size[0],0:self.micro_patch_size[1]] = micro_patch
        
        idx  = list(range(2,generated_patches.shape[0],4))
        micro_patch = generated_patches[idx]
        macro_patches[:,:,0:self.micro_patch_size[0],self.micro_patch_size[1]:2*self.micro_patch_size[1]] = micro_patch
        
        idx  = list(range(1,generated_patches.shape[0],4))
        micro_patch = generated_patches[idx]
        macro_patches[:,:,self.micro_patch_size[0]:2*self.micro_patch_size[0],0:self.micro_patch_size[1]] = micro_patch
        
        idx  = list(range(3,generated_patches.shape[0],4))
        micro_patch = generated_patches[idx]
        macro_patches[:,:,self.micro_patch_size[0]:2*self.micro_patch_size[0],self.micro_patch_size[1]:2*self.micro_patch_size[1]] = micro_patch

        return macro_patches


    def crop_micro_from_full_cpu(self, imgs, crop_pos_x, crop_pos_y):

        ps_x, ps_y = self.micro_patch_size # i.e. Patch-Size

        valid_area_x = self.full_image_size[0] - self.micro_patch_size[0]
        valid_area_y = self.full_image_size[1] - self.micro_patch_size[1]

        crop_result = []
        batch_size = imgs.shape[0]
        for i in range(batch_size*self.num_micro_compose_macro):
            i_idx = i // self.num_micro_compose_macro
            x_idx = np.round((crop_pos_x[i, 0]+1)/2*valid_area_x).astype(int)
            y_idx = np.round((crop_pos_y[i, 0]+1)/2*valid_area_y).astype(int)
            t = imgs[i_idx, :, x_idx:x_idx+ps_x, y_idx:y_idx+ps_y]
            crop_result.append(t)
        return torch.stack(crop_result)