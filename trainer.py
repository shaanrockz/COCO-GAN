
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class Trainer():
    def __init__(self, config, 
                 g_builder, d_builder, cp_builder, zp_builder, 
                 coord_handler, patch_handler):
        self.config = config
        self.g_builder = g_builder
        self.d_builder = d_builder
        self.cp_builder = cp_builder
        self.zp_builder = zp_builder
        self.coord_handler = coord_handler
        self.patch_handler = patch_handler

        # Vars for graph building
        self.batch_size = self.config["train_params"]["batch_size"]
        self.z_dim = self.config["model_params"]["z_dim"]
        self.spatial_dim = self.config["model_params"]["spatial_dim"]
        self.micro_patch_size = self.config["data_params"]["micro_patch_size"]
        self.macro_patch_size = self.config["data_params"]["macro_patch_size"]

        self.ratio_macro_to_micro = self.config["data_params"]["ratio_macro_to_micro"]
        self.ratio_full_to_micro = self.config["data_params"]["ratio_full_to_micro"]
        self.num_micro_compose_macro = self.config["data_params"]["num_micro_compose_macro"]

        # Vars for training loop
        self.exp_name = config["log_params"]["exp_name"]
        self.epochs = float(self.config["train_params"]["epochs"])
        self.num_batches = self.config["data_params"]["num_train_samples"] // self.batch_size
        self.coordinate_system = self.config["data_params"]["coordinate_system"]
        self.G_update_period = self.config["train_params"]["G_update_period"]
        self.D_update_period = self.config["train_params"]["D_update_period"]
        self.Q_update_period = self.config["train_params"]["Q_update_period"]

        # Loss weights
        self.code_loss_w = self.config["loss_params"]["code_loss_w"]
        self.coord_loss_w = self.config["loss_params"]["coord_loss_w"]
        self.gp_lambda = self.config["loss_params"]["gp_lambda"]

        # Extrapolation parameters handling
        self.num_extrap_steps = 0
        self.weight_cliping_limit = 0.01
        


    def _train_content_prediction_model(self):
        return (self.Q_update_period>0) and (self.config["train_params"]["qlr"]>0)


    def sample_prior(self):
        return np.random.uniform(-1., 1., [self.batch_size, self.z_dim]).astype(np.float32)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)
    
    def _dup_z_for_macro(self, z):
        # Duplicate with nearest neighbor, different to `tf.tile`.
        ch = z.shape[-1]
        repeat = self.num_micro_compose_macro
        extend = torch.unsqueeze(z,1)
        
        extend_dup = self.tile(extend, 1, repeat)
        return extend_dup.view(-1, ch)

    def calc_gradient_penalty(self):
        """ Gradient Penalty for patches D """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = torch.FloatTensor(self.real_macro.size(0),1,1,1).uniform_(0., 1.)
        alpha = alpha.expand(self.real_macro.size(0), self.real_macro.size(1), self.real_macro.size(2), self.real_macro.size(3))
        interpolates = alpha*self.real_macro + ((1-alpha) * self.gen_macro)
        
        disc_inter, _ = self.d_builder(interpolates, None, is_training=True)
        disc_inter.unsqueeze_(1)
        disc_inter.unsqueeze_(2)
        disc_inter = disc_inter.type(torch.DoubleTensor)
        
        interpolates.detach().type(torch.DoubleTensor).requires_grad_(True)
        
        gradients = torch.autograd.grad(outputs=disc_inter, inputs=interpolates, grad_outputs= torch.ones(interpolates.size(), dtype=torch.float64), create_graph=True, allow_unused=True, retain_graph=True)[0]
        slopes = torch.sqrt(torch.sum(torch.pow(gradients,2), 1))
        gradient_penalty = torch.mean((slopes - 1.) ** 2).type(torch.DoubleTensor) * self.config["loss_params"]["gp_lambda"]
        return gradient_penalty


    def generate_full_image_cpu(self, z):
        all_micro_patches = []
        all_micro_coord = []
        num_patches_x = self.ratio_full_to_micro[0] + self.num_extrap_steps * 2
        num_patches_y = self.ratio_full_to_micro[1] + self.num_extrap_steps * 2
        
        full_image = np.empty((z.shape[0],self.config["data_params"]["c_dim"],self.micro_patch_size[0]*num_patches_x,0))
        for yy in range(num_patches_y):
            rows_data = np.empty((z.shape[0],self.config["data_params"]["c_dim"],0,self.micro_patch_size[1]))
            for xx in range(num_patches_x):
                micro_coord_single = np.array([
                    self.coord_handler.euclidean_coord_int_full_to_float_micro(xx, num_patches_x, extrap_steps=self.num_extrap_steps),
                    self.coord_handler.euclidean_coord_int_full_to_float_micro(yy, num_patches_y, extrap_steps=self.num_extrap_steps),
                ])
                micro_coord = np.tile(np.expand_dims(micro_coord_single, 0), [z.shape[0], 1])
                generated_patch = self.g_builder(torch.tensor(z, dtype=torch.float64), torch.tensor(micro_coord, dtype=torch.float64), is_training=False)
                rows_data = np.concatenate((rows_data, generated_patch.detach().numpy()), axis=2)
                all_micro_patches.append(torch.Tensor.cpu(generated_patch).detach().numpy())
                all_micro_coord.append(micro_coord)
            full_image = np.concatenate((full_image, rows_data), axis=3)

        all_micro_patches = np.concatenate(all_micro_patches, 0)

        return all_micro_patches, full_image


    def train(self, logger, real_images):
        
        # Optimizers
        self.g_optim = optim.Adam(self.g_builder.parameters(), lr=self.config["train_params"]["glr"], betas=[self.config["train_params"]["beta1"], self.config["train_params"]["beta2"]])
        self.d_optim = optim.Adam(self.d_builder.parameters(), lr=self.config["train_params"]["dlr"], betas=[self.config["train_params"]["beta1"], self.config["train_params"]["beta2"]])
        self.q_optim = optim.Adam(self.zp_builder.parameters(), lr=self.config["train_params"]["qlr"], betas=[self.config["train_params"]["beta1"], self.config["train_params"]["beta2"]])
        

        z_fixed = self.sample_prior()
        global_step = 0
        start_time = time.time()
        self.g_loss, self.d_loss, self.q_loss = 0, 0, 0
        
        cur_epoch = int(global_step / self.num_batches)
        cur_iter  = global_step - cur_epoch * self.num_batches
        
        self.g_optim.zero_grad()
        self.d_optim.zero_grad()
        
        while cur_epoch < 1000:
            for cur_iter, data in enumerate(real_images):
                print('iter : '+str(cur_iter))
#                for p in self.d_builder.parameters():
#                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                
                image = data[0]
                
                # Create data
                z = torch.tensor(self.sample_prior())
                macro_coord, micro_coord, y_angle_ratio = self.coord_handler.sample_coord()
                
                micro_coord_fake = micro_coord
                macro_coord_fake = macro_coord
                micro_coord_real = micro_coord
                macro_coord_real = macro_coord
                
                
                # Crop real micro for visualization
                self.real_micro = self.patch_handler.crop_micro_from_full_cpu(image, micro_coord_real[:, 0:1], micro_coord_real[:, 1:2])
                self.real_macro = self.patch_handler.concat_micro_patches_cpu(self.real_micro, ratio_over_micro=self.ratio_macro_to_micro)
                
                (self.disc_real, disc_real_h) = self.d_builder(self.real_macro, macro_coord_real, is_training=True)
                self.c_real_pred = self.cp_builder(torch.unsqueeze(disc_real_h,1), is_training=True)
                self.z_real_pred = self.zp_builder(torch.unsqueeze(disc_real_h,1), is_training=True)
        
                # Fake part
                z_dup_macro = self._dup_z_for_macro(z).clone().detach()
                self.gen_micro = self.g_builder(z_dup_macro, micro_coord_fake, is_training=True)
                self.gen_macro = self.patch_handler.concat_micro_patches_cpu(self.gen_micro, ratio_over_micro=self.ratio_macro_to_micro)
                (self.disc_fake, disc_fake_h) = self.d_builder(self.gen_macro, macro_coord_fake, is_training=True)
                self.c_fake_pred = self.cp_builder(torch.unsqueeze(disc_fake_h,1), is_training=True)
                self.z_fake_pred = self.zp_builder(torch.unsqueeze(disc_fake_h,1), is_training=True)
                
                
                self.macro_error = nn.MSELoss()(self.real_macro.type(torch.float64), self.gen_macro.type(torch.float64))
                
                # Spatial consistency loss (reduce later)
                self.coord_mse_real = self.coord_loss_w * nn.MSELoss()(torch.squeeze(torch.tensor(macro_coord_real).double()), torch.squeeze(self.c_real_pred).double())
                self.coord_mse_fake = self.coord_loss_w * nn.MSELoss()(torch.squeeze(torch.tensor(macro_coord_fake).double()), torch.squeeze(self.c_fake_pred).double())
        
        
                self.coord_mse_real = torch.mean(self.coord_mse_real)
                self.coord_mse_fake = torch.mean(self.coord_mse_fake)
                self.coord_loss = self.coord_mse_real + self.coord_mse_fake
                
                # Content consistency loss
                z_real = z.clone().detach().requires_grad_(True).type(torch.DoubleTensor).unsqueeze(1)
                z_fake = self.z_fake_pred.clone().detach().requires_grad_(True).type(torch.DoubleTensor)
                self.code_loss = self.code_loss_w * torch.mean(nn.L1Loss()(z_real, z_fake))
                
                # Gradient penalty loss of WGAN-GP
                gradient_penalty = self.calc_gradient_penalty()
                self.gp_loss = gradient_penalty
                
                # WGAN loss
                self.adv_real = torch.mean(self.disc_real)
                self.adv_fake = torch.mean(self.disc_fake)
                
                self.d_adv_loss = -self.adv_real + self.adv_fake
                self.g_adv_loss = -self.adv_fake #+ self.macro_error.type(torch.float64)
                
                
                # Total loss
                self.d_loss = self.d_adv_loss + self.gp_loss + self.coord_loss + self.code_loss
                self.g_loss = self.g_adv_loss + self.coord_loss + self.code_loss
                self.q_loss = self.g_adv_loss + self.code_loss
        
                self.w_dist = torch.abs(self.adv_real - self.adv_fake)
                print('Wasserstein Distance '+ str(self.w_dist))
                
                self.d_loss.float().backward(retain_graph=True)
                self.g_loss.float().backward(retain_graph=True)
                
                self.d_optim.step()
                self.d_optim.zero_grad()
                
                self.g_optim.step()
                self.g_optim.zero_grad()
                
                # Log
                time_elapsed = time.time() - start_time
                print("[{}] [Epoch: {}; {:4d}/{:4d}; global_step:{}] elapsed: {:.4f}".format(
                    self.exp_name, cur_epoch, cur_iter, self.num_batches, global_step, time_elapsed))
                
                logger.log_iter(self, cur_epoch, cur_iter, global_step, z, z_fixed)

                cur_iter += 1
                global_step += 1
                
            cur_epoch += 1
            cur_iter = 0
