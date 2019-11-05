import os
import numpy as np

from utils import save_manifold_images


class Logger():
    def __init__(self, config, patch_handler):
        self.config = config

        self.batch_size = self.config["train_params"]["batch_size"]
        self.num_micro_compose_full = self.config["data_params"]["num_micro_compose_full"]

        
        self.full_shape = [
            None, 
            self.config["data_params"]["full_image_size"][0], 
            self.config["data_params"]["full_image_size"][1], 
            self.config["data_params"]["c_dim"], 
        ]


        self.exp_name = config["log_params"]["exp_name"]
        self.log_dir = self._check_folder(os.path.join(config["log_params"]["log_dir"], self.exp_name))
        self.img_dir = self._check_folder(os.path.join(self.log_dir, "images"))

        # Use float to parse "inf"
        self.img_step = float(config["log_params"]["img_step"])
        self.dump_img_step = float(config["log_params"]["dump_img_step"])


    def _check_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder




    def _check_step(self, step, step_config):
        if step_config is None:
            return False
        elif step==0:
            return False
        return (step % step_config) == 0


    def log_iter(self, trainer, epoch, iter_, global_step, 
                 z_iter, z_fixed):


        # We use a set of fixed z here to better monitor the changes through time.
        if self._check_step(global_step, self.dump_img_step):

            fixed_patch, fixed_full = trainer.generate_full_image_cpu(z_fixed)
            _, sampled_full = trainer.generate_full_image_cpu(z_iter)
                        

            num_full = self.batch_size
            num_patches = self.batch_size * self.num_micro_compose_full
            manifold_h_f, manifold_w_f = int(np.sqrt(num_full)), int(np.sqrt(num_full))
            manifold_h_p, manifold_w_p = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
            
            # Save fixed micro patches
            save_name = 'fixed_patch_{:02d}_{:04d}.png'.format(epoch, iter_)
            save_manifold_images(fixed_patch[:manifold_h_p * manifold_w_p, :, :, :], 
                                 [manifold_h_p, manifold_w_p], 
                                 os.path.join(self.img_dir, save_name))
            
            # Save fixed full images
            save_name = 'fixed_full_{:02d}_{:04d}.png'.format(epoch, iter_)
            save_manifold_images(fixed_full[:manifold_h_f * manifold_w_f, :, :, :], 
                                 [manifold_h_f, manifold_w_f], 
                                 os.path.join(self.img_dir, save_name))

            # Save sampled full images
            save_name = 'sampled_full_{:02d}_{:04d}.png'.format(epoch, iter_)
            save_manifold_images(sampled_full[:manifold_h_f * manifold_w_f, :, :, :], 
                                 [manifold_h_f, manifold_w_f], 
                                 os.path.join(self.img_dir, save_name))
