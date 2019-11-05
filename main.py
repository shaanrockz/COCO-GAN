import yaml


from models.generator import GeneratorBuilder
from models.discriminator import DiscriminatorBuilder
from models.spatial_prediction import SpatialPredictorBuilder
from models.content_predictor import ContentPredictorBuilder

from coord_handler import CoordHandler
from patch_handler import PatchHandler
from logger import Logger

from trainer import Trainer

import torchvision
import torchvision.datasets as dataset
import torch


def precompute_parameters(config):
    full_image_size = config["data_params"]["full_image_size"]
    micro_patch_size = config["data_params"]["micro_patch_size"]
    macro_patch_size = config["data_params"]["macro_patch_size"]

    # Let NxM micro matches to compose a macro patch,
    #    `ratio_macro_to_micro` is N or M
    ratio_macro_to_micro = [
        macro_patch_size[0] // micro_patch_size[0],
        macro_patch_size[1] // micro_patch_size[1],
    ]
    num_micro_compose_macro = ratio_macro_to_micro[0] * ratio_macro_to_micro[1]

    # Let NxM micro matches to compose a full image,
    #    `ratio_full_to_micro` is N or M
    ratio_full_to_micro = [
        full_image_size[0] // micro_patch_size[0],
        full_image_size[1] // micro_patch_size[1],
    ]
    num_micro_compose_full = ratio_full_to_micro[0] * ratio_full_to_micro[1]

    config["data_params"]["ratio_macro_to_micro"] = ratio_macro_to_micro
    config["data_params"]["ratio_full_to_micro"] = ratio_full_to_micro
    config["data_params"]["num_micro_compose_macro"] = num_micro_compose_macro
    config["data_params"]["num_micro_compose_full"] = num_micro_compose_full



def load_dataset(config):
#    data_path = 'mnist/'
#    
#    train_dataset = dataset.MNIST(root=data_path, train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((config["data_params"]["full_image_size"][0],config["data_params"]["full_image_size"][1])), torchvision.transforms.ToTensor()]))
    
    data_path = 'celeb_data/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((config["data_params"]["full_image_size"][0],config["data_params"]["full_image_size"][1])), torchvision.transforms.ToTensor()])
    )
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train_params']['batch_size'],
        num_workers=0,
        shuffle=True
    )
    return train_loader


with open('./configs/CelebA_64x64_N2M2S32.yaml') as f:
    config = yaml.load(f)
    micro_size = config["data_params"]['micro_patch_size']
    macro_size = config["data_params"]['macro_patch_size']
    full_size = config["data_params"]['full_image_size']
    assert macro_size[0] % micro_size[0] == 0
    assert macro_size[1] % micro_size[1] == 0
    assert full_size[0] % micro_size[0] == 0
    assert full_size[1] % micro_size[1] == 0

# Pre-compute some frequently used parameters
precompute_parameters(config)

# Create model builders
coord_handler = CoordHandler(config)
patch_handler = PatchHandler(config)

d_builder = DiscriminatorBuilder(config)
g_builder = GeneratorBuilder(config)
cp_builder = SpatialPredictorBuilder(config)
zp_builder = ContentPredictorBuilder(config)


real_images = load_dataset(config)

## Create controllers
logger = Logger(config, patch_handler)
trainer = Trainer(config, g_builder, d_builder, cp_builder, zp_builder, coord_handler, patch_handler)
trainer.train(logger, real_images)
        





        



    
